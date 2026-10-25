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
        "input_features": "stockmixer_ma",
        "stockmixer_ma_windows": "5,10,20,30",
        "target_mode": "raw_return",
        "batch_size": 32,
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


class StockMixerWindowDataset(Dataset):
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


class StockMixerData:
    def __init__(self, args):
        self.args = args
        self.market = args.data
        self.dataset_path = os.path.join(args.root_path, f"{self.market}_ohlcv")
        self.mask_data = self._load_pickle("mask_data.pkl")
        self.gt_data = self._load_pickle("gt_data.pkl")
        self.price_data = self._load_pickle("price_data.pkl")
        self.feature_config = self._load_feature_config()
        self.full_feature_names = list(self.feature_config.get("feature_names", []))
        self.target_mode = str(getattr(self.args, "target_mode", "raw_return") or "raw_return")
        if self.target_mode != "raw_return":
            raise ValueError("StockMixerData only supports target_mode=raw_return")

        self.input_features = str(getattr(self.args, "input_features", "stockmixer_ma") or "stockmixer_ma")
        if self.input_features != "stockmixer_ma":
            raise ValueError("StockMixerData only supports input_features=stockmixer_ma")

        self.data_start_index = int(getattr(self.args, "data_start_index", 0) or 0)
        self.target_scale_data = np.ones_like(self.gt_data, dtype=np.float32)
        self.stockmixer_ma_windows = self._parse_stockmixer_ma_windows()
        self.stockmixer_price_data = self.price_data.astype(np.float64, copy=False)
        prefix = np.zeros((self.stockmixer_price_data.shape[0], 1), dtype=np.float64)
        cumsum = np.cumsum(self.stockmixer_price_data, axis=1, dtype=np.float64)
        self.stockmixer_price_cumsum = np.concatenate([prefix, cumsum], axis=1)

        if self.data_start_index:
            start = self.data_start_index
            self.mask_data = self.mask_data[:, start:]
            self.gt_data = self.gt_data[:, start:]
            self.price_data = self.price_data[:, start:]
            self.target_scale_data = self.target_scale_data[:, start:]

        self.target_data = self.gt_data
        self.feature_names = ["MA5", "MA10", "MA20", "MA30", "Close"]
        self.stock_num = self.price_data.shape[0]
        self.trade_dates = self.price_data.shape[1]
        self.feature_num = len(self.feature_names)
        self.valid_index, self.test_index = self._resolve_splits()
        self.train_start_offset = 0
        self.train_end_offset = self.valid_index - self.args.seq_len - self.args.steps + 1
        self.train_offsets = np.arange(self.train_start_offset, self.train_end_offset, dtype=np.int64)
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

    def _parse_stockmixer_ma_windows(self):
        value = getattr(self.args, "stockmixer_ma_windows", "5,10,20,30")
        if isinstance(value, str):
            windows = tuple(int(item.strip()) for item in value.split(",") if item.strip())
        else:
            windows = tuple(int(item) for item in value)
        if windows != (5, 10, 20, 30):
            raise ValueError("StockMixer paper input expects moving-average windows 5,10,20,30")
        return windows

    def _build_stockmixer_ma_inputs(self, offset, seq_len):
        date_index = self.data_start_index + np.arange(offset, offset + seq_len, dtype=np.int64)
        end_index = date_index + 1
        columns = []
        for window in self.stockmixer_ma_windows:
            start_index = np.maximum(end_index - int(window), 0)
            total = self.stockmixer_price_cumsum[:, end_index] - self.stockmixer_price_cumsum[:, start_index]
            count = (end_index - start_index).astype(np.float64)
            columns.append(total / count[None, :])
        columns.append(self.stockmixer_price_data[:, date_index])
        return np.stack(columns, axis=2).astype(np.float32)

    def _resolve_splits(self):
        return int(self.args.valid_index), int(self.args.test_index)

    def _validate(self):
        if self.mask_data.ndim != 2:
            raise ValueError(f"mask_data must be 2-D, got {self.mask_data.shape}")
        if self.gt_data.shape != self.mask_data.shape:
            raise ValueError("gt_data shape does not match mask_data")
        if self.target_data.shape != self.mask_data.shape:
            raise ValueError("target_data shape does not match mask_data")
        if self.target_scale_data.shape != self.mask_data.shape:
            raise ValueError("target_scale_data shape does not match mask_data")
        if self.price_data.shape != self.mask_data.shape:
            raise ValueError("price_data shape does not match mask_data")
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
        inputs = self._build_stockmixer_ma_inputs(offset, seq_len)
        mask_batch = self.mask_data[:, offset : offset + seq_len + steps]
        mask_batch = np.min(mask_batch, axis=1)
        return {
            "inputs": inputs,
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
        dataset = StockMixerWindowDataset(self, offsets, return_offset=return_offset)
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

    def train_loader(self, epoch=0, seed=None):
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
            seed=int(self.args.random_seed) + int(epoch) if seed is None else int(seed),
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
