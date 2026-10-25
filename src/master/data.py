import json
import os
import pickle
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT_PATH = PROJECT_ROOT / "data" / "processed"

DATASET_DEFAULTS = {
    "NASDAQ": {"market_num": 20, "valid_index": 871, "test_index": 996, "data_start_index": 0},
    "NYSE": {"market_num": 25, "valid_index": 871, "test_index": 996, "data_start_index": 0},
    "SP500": {"market_num": 8, "valid_index": 1127, "test_index": 1288, "data_start_index": 915},
}


def make_data_args(data, **overrides):
    if data not in DATASET_DEFAULTS:
        raise ValueError(f"data must be one of {sorted(DATASET_DEFAULTS)}; got {data!r}")
    defaults = DATASET_DEFAULTS[data]
    values = {
        "data": data,
        "root_path": str(DEFAULT_ROOT_PATH),
        "seq_len": 16,
        "steps": 1,
        "input_features": "master_stock_market",
        "target_mode": "raw_return",
        "batch_size": 1,
        "random_seed": 1,
        "debug": False,
        "debug_train_steps": 8,
        "debug_eval_steps": 8,
        "num_workers": 0,
        "pin_memory": False,
        "prefetch_factor": 2,
        "persistent_workers": False,
        "train_drop_last": True,
        "valid_drop_last": False,
        "market_num": defaults["market_num"],
        "valid_index": defaults["valid_index"],
        "test_index": defaults["test_index"],
        "data_start_index": defaults["data_start_index"],
    }
    values.update(overrides)
    values["root_path"] = str(values["root_path"])
    return Namespace(**values)


class MasterWindowDataset(Dataset):
    def __init__(self, market_data, offsets, return_offset=False):
        self.market_data = market_data
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.return_offset = return_offset

    def __len__(self):
        return int(self.offsets.shape[0])

    def __getitem__(self, index):
        offset = int(self.offsets[index])
        batch = self.market_data.get_batch(offset)
        if self.return_offset:
            batch = dict(batch)
            batch["offset"] = np.int64(offset)
        return batch


class MasterData:
    """MASTER-local loader with explicit stock and market-status tensors."""

    def __init__(self, args):
        self.args = args
        self.market = args.data
        self.dataset_path = os.path.join(args.root_path, f"{self.market}_ohlcv")
        self.eod_data = self._load_pickle("eod_data.pkl")
        self.mask_data = self._load_pickle("mask_data.pkl")
        self.gt_data = self._load_pickle("gt_data.pkl")
        self.price_data = self._load_pickle("price_data.pkl")
        self.feature_config = self._load_feature_config()
        self.full_feature_names = list(self.feature_config.get("feature_names", []))

        self.target_mode = str(getattr(self.args, "target_mode", "raw_return") or "raw_return")
        if self.target_mode != "raw_return":
            raise ValueError("MasterData only supports target_mode=raw_return")

        self.input_features = str(getattr(self.args, "input_features", "master_stock_market") or "master_stock_market")
        self.data_start_index = int(getattr(self.args, "data_start_index", 0) or 0)
        self.target_scale_data = np.ones_like(self.gt_data, dtype=np.float32)
        self.target_data = self.gt_data
        self.feature_groups = self._resolve_feature_groups()
        self.stock_feature_indices, self.market_feature_indices = self._resolve_master_features()
        self.stock_feature_names = [self.full_feature_names[index] for index in self.stock_feature_indices]
        self.market_feature_names = [self.full_feature_names[index] for index in self.market_feature_indices]
        self.feature_names = self.stock_feature_names + self.market_feature_names

        if self.data_start_index:
            start = self.data_start_index
            self.eod_data = self.eod_data[:, start:, :]
            self.mask_data = self.mask_data[:, start:]
            self.gt_data = self.gt_data[:, start:]
            self.price_data = self.price_data[:, start:]
            self.target_scale_data = self.target_scale_data[:, start:]
            self.target_data = self.target_data[:, start:]

        self.stock_num = self.eod_data.shape[0]
        self.trade_dates = self.eod_data.shape[1]
        self.stock_feature_num = len(self.stock_feature_indices)
        self.market_feature_num = len(self.market_feature_indices)
        self.feature_num = self.stock_feature_num + self.market_feature_num
        self.valid_index, self.test_index = self._resolve_splits()
        self.train_start_offset = 0
        self.train_end_offset = self.valid_index - self.args.seq_len - self.args.steps + 1
        self.train_offsets = np.arange(self.train_start_offset, self.train_end_offset, dtype=np.int64)
        self.input_transform_metadata = {
            "feature_set": self.input_features,
            "stock_feature_names": self.stock_feature_names,
            "market_feature_names": self.market_feature_names,
            "market_status_source": "feature_config.market_ctx masked cross-sectional mean",
        }
        self._validate()

    def _load_pickle(self, name):
        path = os.path.join(self.dataset_path, name)
        with open(path, "rb") as f:
            return np.asarray(pickle.load(f), dtype=np.float32)

    def _load_feature_config(self):
        path = os.path.join(self.dataset_path, "feature_config.json")
        if not os.path.exists(path):
            raise ValueError("MASTER requires feature_config.json with stock and market feature groups")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _indices_for_feature_names(self, names):
        name_to_index = {name: index for index, name in enumerate(self.full_feature_names)}
        missing = [name for name in names if name not in name_to_index]
        if missing:
            raise ValueError(f"Requested input features not found in feature_config.json: {missing}")
        return [name_to_index[name] for name in names]

    def _resolve_feature_groups(self):
        if not self.full_feature_names:
            raise ValueError("MASTER requires feature_config.json feature_names")
        config_groups = self.feature_config.get("feature_group_indices") or {}
        if config_groups:
            return {name: [int(index) for index in indices] for name, indices in config_groups.items()}

        feature_groups = self.feature_config.get("feature_groups") or {}
        resolved = {}
        for name, features in feature_groups.items():
            resolved[name] = self._indices_for_feature_names(features)
        return resolved

    def _resolve_master_features(self):
        required = {"stock_raw", "endo", "stock_derived", "market_ctx"}
        missing = sorted(required.difference(self.feature_groups))
        if missing:
            raise ValueError(f"MASTER input_features require feature groups {sorted(required)}; missing {missing}")

        if self.input_features in {"master_stock_market", "all"}:
            stock_indices = (
                list(self.feature_groups["stock_raw"])
                + list(self.feature_groups["endo"])
                + list(self.feature_groups["stock_derived"])
            )
            market_indices = list(self.feature_groups["market_ctx"])
        elif self.input_features == "raw_ohlcv_market":
            stock_indices = list(self.feature_groups["stock_raw"])
            market_indices = list(self.feature_groups["market_ctx"])
        else:
            raise ValueError(
                "MasterData supports input_features=master_stock_market, all, or raw_ohlcv_market"
            )

        if not stock_indices:
            raise ValueError("MASTER requires at least one stock feature")
        if not market_indices:
            raise ValueError("MASTER requires at least one market status feature")
        return stock_indices, market_indices

    def _resolve_splits(self):
        return int(self.args.valid_index), int(self.args.test_index)

    def _validate(self):
        if self.eod_data.ndim != 3:
            raise ValueError(f"eod_data must be 3-D, got {self.eod_data.shape}")
        if self.eod_data.shape[:2] != self.mask_data.shape:
            raise ValueError("mask_data shape does not match eod_data")
        if self.gt_data.shape != self.mask_data.shape:
            raise ValueError("gt_data shape does not match mask_data")
        if self.target_data.shape != self.mask_data.shape:
            raise ValueError("target_data shape does not match mask_data")
        if self.price_data.shape != self.mask_data.shape:
            raise ValueError("price_data shape does not match mask_data")
        if len(self.feature_names) != self.feature_num:
            raise ValueError("feature_names length does not match MASTER feature_num")
        if not np.isfinite(self.eod_data).all():
            raise ValueError("eod_data contains NaN or Inf")
        if not np.isfinite(self.gt_data).all():
            raise ValueError("gt_data contains NaN or Inf")
        if not np.isfinite(self.price_data).all():
            raise ValueError("price_data contains NaN or Inf")
        if self.valid_index <= self.args.seq_len + self.args.steps:
            raise ValueError("valid_index is too small for seq_len and steps")
        if self.test_index <= self.valid_index or self.test_index > self.trade_dates:
            raise ValueError("test_index must be in (valid_index, trade_dates]")
        if self.train_end_offset <= self.train_start_offset:
            raise ValueError("no valid train window offsets")

    def _build_market_status(self, offset, seq_len):
        market_by_stock = self.eod_data[:, offset : offset + seq_len, self.market_feature_indices]
        valid = (self.mask_data[:, offset : offset + seq_len] >= 0.5).astype(np.float32)
        denom = np.sum(valid, axis=0)
        weighted_sum = np.sum(market_by_stock * valid[:, :, None], axis=0)
        fallback = np.mean(market_by_stock, axis=0)
        market = np.where(denom[:, None] > 0.0, weighted_sum / np.maximum(denom[:, None], 1.0), fallback)
        return market.astype(np.float32, copy=False)

    def get_batch(self, offset):
        seq_len = int(self.args.seq_len)
        steps = int(self.args.steps)
        stock_inputs = self.eod_data[:, offset : offset + seq_len, self.stock_feature_indices]
        market_inputs = self._build_market_status(offset, seq_len)
        market_expanded = np.broadcast_to(
            market_inputs[None, :, :],
            (stock_inputs.shape[0], market_inputs.shape[0], market_inputs.shape[1]),
        )
        inputs = np.concatenate([stock_inputs, market_expanded], axis=2).astype(np.float32, copy=False)
        mask_batch = self.mask_data[:, offset : offset + seq_len + steps]
        mask_batch = np.min(mask_batch, axis=1)
        return {
            "inputs": inputs,
            "stock_inputs": stock_inputs.astype(np.float32, copy=False),
            "market_inputs": market_inputs,
            "mask": np.expand_dims(mask_batch, axis=1),
            "price": np.expand_dims(self.price_data[:, offset + seq_len - 1], axis=1),
            "y_endo": np.expand_dims(self.target_data[:, offset + seq_len + steps - 1], axis=1),
            "raw_gt": np.expand_dims(self.gt_data[:, offset + seq_len + steps - 1], axis=1),
            "target_scale": np.expand_dims(self.target_scale_data[:, offset + seq_len + steps - 1], axis=1),
        }

    def iter_eval_offsets(self, start_index, end_index):
        start = int(start_index) - int(self.args.seq_len) - int(self.args.steps) + 1
        end = int(end_index) - int(self.args.seq_len) - int(self.args.steps) + 1
        return range(start, end)

    def _loader(self, offsets, batch_size, shuffle, drop_last, return_offset, seed=None):
        dataset = MasterWindowDataset(self, offsets, return_offset=return_offset)
        generator = None
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(int(seed if seed is not None else self.args.random_seed))
        num_workers = int(getattr(self.args, "num_workers", 0))
        kwargs = {
            "batch_size": int(batch_size),
            "shuffle": bool(shuffle),
            "drop_last": bool(drop_last),
            "num_workers": num_workers,
            "pin_memory": bool(getattr(self.args, "pin_memory", True)),
            "generator": generator,
        }
        if num_workers > 0:
            kwargs["prefetch_factor"] = int(getattr(self.args, "prefetch_factor", 2))
            kwargs["persistent_workers"] = bool(getattr(self.args, "persistent_workers", False))
        return DataLoader(dataset, **kwargs)

    def train_loader(self, epoch=0):
        offsets = self.train_offsets
        if getattr(self.args, "debug", False):
            max_windows = int(self.args.debug_train_steps) * int(self.args.batch_size)
            offsets = offsets[:max_windows]
        return self._loader(
            offsets=offsets,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=bool(getattr(self.args, "train_drop_last", True)),
            return_offset=False,
            seed=int(self.args.random_seed) + int(epoch),
        )

    def eval_loader(self, start_index, end_index, drop_last=False):
        offsets = np.asarray(list(self.iter_eval_offsets(start_index, end_index)), dtype=np.int64)
        if getattr(self.args, "debug", False):
            offsets = offsets[: int(self.args.debug_eval_steps)]
        batch_size = int(self.args.batch_size)
        effective_drop_last = bool(drop_last) and int(offsets.shape[0]) >= batch_size
        return self._loader(
            offsets=offsets,
            batch_size=batch_size,
            shuffle=False,
            drop_last=effective_drop_last,
            return_offset=True,
            seed=None,
        )
