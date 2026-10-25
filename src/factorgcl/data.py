from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from rsr.data import RSRData, make_data_args as make_rsr_data_args


def make_data_args(data, **overrides) -> Namespace:
    overrides.setdefault("relation_name", "auto")
    overrides.setdefault("input_features", "stock_all")
    overrides.setdefault("prior_beta_name", "factor_prior_beta")
    overrides.setdefault("future_seq_len", overrides.get("seq_len", 16))
    overrides.setdefault("use_contrastive", True)
    args = make_rsr_data_args(data, **overrides)
    args.prior_beta_name = overrides["prior_beta_name"]
    args.future_seq_len = int(overrides["future_seq_len"])
    args.use_contrastive = bool(overrides["use_contrastive"])
    return args


class FactorGCLTrainWindowDataset(Dataset):
    def __init__(self, market_data, offsets):
        self.market_data = market_data
        self.offsets = np.asarray(offsets, dtype=np.int64)

    def __len__(self):
        return int(self.offsets.shape[0])

    def __getitem__(self, index):
        return self.market_data.get_train_batch(int(self.offsets[index]))


class FactorGCLData(RSRData):
    """Unified stock windows plus FactorGCL prior factor incidence beta."""

    def __init__(self, args):
        super().__init__(args)
        self.prior_beta_name = str(getattr(self.args, "prior_beta_name", "factor_prior_beta"))
        self.prior_beta_path = Path(self.args.relation_root) / self.market / f"{self.prior_beta_name}.npy"
        self.prior_beta = self._load_prior_beta()
        self.prior_factor_count = int(self.prior_beta.shape[1])
        self.future_seq_len = int(getattr(self.args, "future_seq_len", self.args.seq_len))
        self.use_contrastive = bool(getattr(self.args, "use_contrastive", True))
        if self.future_seq_len <= 0:
            self.use_contrastive = False

        if self.use_contrastive:
            end = self.valid_index - int(self.args.seq_len) - self.future_seq_len + 1
            self.train_end_offset = min(int(self.train_end_offset), int(end))
            self.train_offsets = np.arange(self.train_start_offset, self.train_end_offset, dtype=np.int64)
            if self.train_end_offset <= self.train_start_offset:
                raise ValueError("no valid FactorGCL train window offsets after reserving future_inputs")

        self.input_transform_metadata.update(
            {
                "prior_beta_name": self.prior_beta_name,
                "prior_beta_path": str(self.prior_beta_path),
                "prior_beta_shape": list(self.prior_beta.shape),
                "prior_factor_count": self.prior_factor_count,
                "future_seq_len": self.future_seq_len,
                "use_contrastive": self.use_contrastive,
                "paper_note": (
                    "FactorGCL uses stock windows plus a prior factor exposure matrix beta. "
                    "This baseline uses data/processed/relations/*/factor_prior_beta.npy and the "
                    "project's unified seq_len/steps target protocol."
                ),
            }
        )

    def _load_prior_beta(self):
        if not self.prior_beta_path.exists():
            raise FileNotFoundError(f"prior beta file not found: {self.prior_beta_path}")
        beta = np.asarray(np.load(self.prior_beta_path), dtype=np.float32)
        if beta.ndim != 2:
            raise ValueError(f"prior beta must be [N,K], got {beta.shape}")
        if beta.shape[0] != self.stock_num:
            raise ValueError(
                f"prior beta stock dimension {beta.shape[0]} does not match data stock count {self.stock_num}"
            )
        if beta.shape[1] <= 0:
            raise ValueError("prior beta must contain at least one factor")
        if not np.isfinite(beta).all():
            raise ValueError("prior beta contains NaN or Inf")
        return beta

    def get_train_batch(self, offset):
        batch = dict(super().get_batch(offset))
        if self.use_contrastive:
            seq_len = int(self.args.seq_len)
            start = int(offset) + seq_len
            end = start + self.future_seq_len
            if end > self.valid_index:
                raise IndexError(
                    f"future_inputs [{start},{end}) crosses train boundary valid_index={self.valid_index}"
                )
            future = self.eod_data[:, start:end, self.feature_indices]
            batch["future_inputs"] = future.astype(np.float32, copy=False)
        return batch

    def train_loader(self, epoch=0):
        offsets = self.train_offsets
        if getattr(self.args, "debug", False):
            max_windows = int(self.args.debug_train_steps) * int(self.args.batch_size)
            offsets = offsets[:max_windows]
        dataset = FactorGCLTrainWindowDataset(self, offsets)
        generator = torch.Generator()
        generator.manual_seed(int(self.args.random_seed) + int(epoch))
        num_workers = int(getattr(self.args, "num_workers", 0))
        kwargs = {
            "batch_size": int(self.args.batch_size),
            "shuffle": True,
            "drop_last": bool(getattr(self.args, "train_drop_last", True)),
            "num_workers": num_workers,
            "pin_memory": bool(getattr(self.args, "pin_memory", True)),
            "generator": generator,
        }
        if num_workers > 0:
            kwargs["prefetch_factor"] = int(getattr(self.args, "prefetch_factor", 2))
            kwargs["persistent_workers"] = bool(getattr(self.args, "persistent_workers", False))
        return DataLoader(dataset, **kwargs)


__all__ = ["FactorGCLData", "make_data_args"]
