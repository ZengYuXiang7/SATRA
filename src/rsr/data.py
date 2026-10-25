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

RELATION_ALIASES = {
    "sector_industry": "rsr_industry",
    "industry": "rsr_industry",
    "wikidata": "rsr_wiki",
    "wiki": "rsr_wiki",
    "sp500_sector": "sp500_sector",
    "estimate_distance": "estimate_leadlag_distance",
    "estimate_pearson": "estimate_leadlag_pearson",
    "estimate_kendall": "estimate_leadlag_kendall",
    "estimate_mutual_information": "estimate_leadlag_mutual_information",
    "correlation": "correlation_top16",
    "corr": "correlation_top16",
    "correlation_top16": "correlation_top16",
}


def make_data_args(data, **overrides):
    if data not in DATASET_DEFAULTS:
        raise ValueError(f"data must be one of {sorted(DATASET_DEFAULTS)}; got {data!r}")
    defaults = DATASET_DEFAULTS[data]
    values = {
        "data": data,
        "root_path": str(DEFAULT_ROOT_PATH),
        "relation_root": str(DEFAULT_ROOT_PATH / "relations"),
        "relation_name": "auto",
        "seq_len": 16,
        "steps": 1,
        "input_features": "stock_all",
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
    values["relation_root"] = str(values["relation_root"])
    return Namespace(**values)


class RSRWindowDataset(Dataset):
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


class RSRData:
    """Project-local RSR loader: unified windows plus a static stock relation tensor."""

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
            raise ValueError("RSRData only supports target_mode=raw_return")

        self.input_features = str(getattr(self.args, "input_features", "stock_all") or "stock_all")
        self.data_start_index = int(getattr(self.args, "data_start_index", 0) or 0)
        self.target_scale_data = np.ones_like(self.gt_data, dtype=np.float32)
        self.target_data = self.gt_data
        self.feature_groups = self._resolve_feature_groups()
        self.feature_indices = self._resolve_input_features()
        self.feature_names = [self.full_feature_names[index] for index in self.feature_indices]

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
        self.feature_num = len(self.feature_indices)
        self.valid_index, self.test_index = self._resolve_splits()
        self.train_start_offset = 0
        self.train_end_offset = self.valid_index - self.args.seq_len - self.args.steps + 1
        self.train_offsets = np.arange(self.train_start_offset, self.train_end_offset, dtype=np.int64)

        self.relation_name = self._resolve_relation_name()
        self.relation_path = Path(self.args.relation_root) / self.market / f"{self.relation_name}_relation.npy"
        self.relation_encoding = self._load_relation_encoding()
        self.relation_row, self.relation_col, self.relation_type = np.nonzero(self.relation_encoding > 0)
        self.relation_mask = np.sum(self.relation_encoding, axis=2) > 0
        self.relation_channels = int(self.relation_encoding.shape[2])
        self.relation_nonzero = int(self.relation_row.shape[0])
        self.input_transform_metadata = {
            "feature_set": self.input_features,
            "feature_names": self.feature_names,
            "relation_name": self.relation_name,
            "relation_path": str(self.relation_path),
            "relation_shape": list(self.relation_encoding.shape),
            "relation_nonzero": self.relation_nonzero,
            "relation_source": self._relation_source_from_manifest(),
        }
        self._validate()

    def _load_pickle(self, name):
        path = os.path.join(self.dataset_path, name)
        with open(path, "rb") as f:
            return np.asarray(pickle.load(f), dtype=np.float32)

    def _load_feature_config(self):
        path = os.path.join(self.dataset_path, "feature_config.json")
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _resolve_feature_groups(self):
        if not self.full_feature_names:
            return {}
        config_groups = self.feature_config.get("feature_group_indices") or {}
        if config_groups:
            return {name: [int(index) for index in indices] for name, indices in config_groups.items()}
        name_to_index = {name: index for index, name in enumerate(self.full_feature_names)}
        resolved = {}
        for name, features in (self.feature_config.get("feature_groups") or {}).items():
            resolved[name] = [name_to_index[feature] for feature in features if feature in name_to_index]
        return resolved

    def _resolve_input_features(self):
        if not self.full_feature_names:
            return list(range(self.eod_data.shape[2]))
        if self.input_features == "all":
            return list(range(len(self.full_feature_names)))
        if self.input_features == "raw_ohlcv":
            raw_names = list(self.feature_config.get("original_feature_names", []))
            if raw_names:
                name_to_index = {name: index for index, name in enumerate(self.full_feature_names)}
                return [name_to_index[name] for name in raw_names]
            return list(range(min(5, len(self.full_feature_names))))
        if self.input_features == "stock_all":
            required = ["stock_raw", "endo", "stock_derived"]
            missing = [name for name in required if name not in self.feature_groups]
            if missing:
                raise ValueError(f"input_features=stock_all requires feature groups; missing {missing}")
            indices = []
            for name in required:
                indices.extend(self.feature_groups[name])
            return indices
        if self.input_features == "stock_raw_derived":
            required = ["stock_raw", "stock_derived"]
            missing = [name for name in required if name not in self.feature_groups]
            if missing:
                raise ValueError(f"input_features=stock_raw_derived requires feature groups; missing {missing}")
            return list(self.feature_groups["stock_raw"]) + list(self.feature_groups["stock_derived"])
        raise ValueError("RSRData supports input_features=stock_all, stock_raw_derived, raw_ohlcv, or all")

    def _resolve_relation_name(self):
        requested = str(getattr(self.args, "relation_name", "auto") or "auto")
        if requested != "auto":
            return RELATION_ALIASES.get(requested, requested)
        if self.market in {"NASDAQ", "NYSE"}:
            # Official RSR relation files align with the active interpolated
            # tensors when checked on observed_mask_data.pkl entries.
            return "rsr_wiki"
        return "sp500_sector"

    def _load_relation_encoding(self):
        if not self.relation_path.exists():
            raise FileNotFoundError(
                f"relation file not found: {self.relation_path}. "
                "Run script/baselines/tool_prepare_relation_data.py first "
                "or choose --relation_name."
            )
        relation = np.asarray(np.load(self.relation_path), dtype=np.uint8)
        if relation.ndim != 3:
            raise ValueError(f"relation encoding must be [N,N,R], got {relation.shape}")
        if relation.shape[0] != self.eod_data.shape[0] or relation.shape[1] != self.eod_data.shape[0]:
            raise ValueError(
                f"relation stock dimensions {relation.shape[:2]} do not match data stock count {self.eod_data.shape[0]}"
            )
        return relation

    def _relation_source_from_manifest(self):
        manifest_path = Path(self.args.relation_root) / "manifest.json"
        if not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return manifest["datasets"][self.market]["files"][f"{self.relation_name}_relation"].get("source")
        except Exception:
            return None

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
            raise ValueError("feature_names length does not match feature_num")
        if self.relation_mask.shape != (self.stock_num, self.stock_num):
            raise ValueError("relation mask shape does not match stock count")
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

    def get_batch(self, offset):
        seq_len = int(self.args.seq_len)
        steps = int(self.args.steps)
        inputs = self.eod_data[:, offset : offset + seq_len, self.feature_indices]
        mask_batch = self.mask_data[:, offset : offset + seq_len + steps]
        mask_batch = np.min(mask_batch, axis=1)
        return {
            "inputs": inputs.astype(np.float32, copy=False),
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
        dataset = RSRWindowDataset(self, offsets, return_offset=return_offset)
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
