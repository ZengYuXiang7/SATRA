"""Leakage-aware market data loading and rolling-window construction.

Arrays are stored stock first: features use ``[N, D, F]`` and masks, returns,
and prices use ``[N, D]``. ``valid_index`` and ``test_index`` define the
chronological train/validation/test boundaries. Every sample is constructed
from a past window and a future return target without shuffling dates across
splits.
"""

import os
import pickle
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .alpha_features import FACTORVAE_ALPHA20_FEATURE_NAMES, build_factorvae_alpha20_features


REQUIRED_FEATURE_GROUPS = ("endo", "stock_raw", "stock_derived", "market_ctx")
OPTIONAL_FEATURE_GROUPS = ("alpha158",)


class StockWindowDataset(Dataset):
    """Expose selected rolling-window offsets through the PyTorch Dataset API."""

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


class StockMarketData:
    """Load one market and enforce the common feature and split contract.

    ``get_batch`` converts one window offset into model inputs ``[N, T, F]``
    and one-step targets ``[N, 1]``. DataLoader batching adds the leading batch
    axis. Training offsets end before ``valid_index``; validation and test
    offsets are generated from their respective chronological boundaries.
    """

    def __init__(self, args):
        self.args = args
        self.market = args.data
        self.dataset_path = os.path.join(args.root_path, f"{self.market}_ohlcv")

        self.eod_data = self._load_pickle("eod_data.pkl")
        self.mask_data = self._load_pickle("mask_data.pkl")
        self.gt_data = self._load_pickle("gt_data.pkl")
        self.price_data = self._load_pickle("price_data.pkl")
        self.feature_config = self._load_feature_config()
        self.full_feature_names = list(self.feature_config["feature_names"])
        self.target_mode = str(self.args.target_mode)
        if self.target_mode != "raw_return":
            raise ValueError(f"Unsupported target_mode: {self.target_mode}; data-level return normalization was removed")
        self.target_scale_data = self._build_target_scale_data()
        self.data_start_index = int(self.args.data_start_index)

        self.input_features = str(self.args.input_features)
        self.input_transform = None
        self.input_transform_metadata = {}
        self.stockmixer_ma_windows = ()
        self.stockmixer_price_cumsum = None
        self.stockmixer_price_data = None
        self.factorvae_alpha20_data = None
        self.selected_feature_indices, self.feature_names = self._resolve_input_features()

        if self.input_transform == "stockmixer_ma":
            self._prepare_stockmixer_ma_transform()
        if self.input_transform in {"factorvae_alpha20", "ohlcv_alpha20"}:
            self._prepare_factorvae_alpha20_transform()

        if self.data_start_index:
            start = self.data_start_index
            self.eod_data = self.eod_data[:, start:, :]
            self.mask_data = self.mask_data[:, start:]
            self.gt_data = self.gt_data[:, start:]
            self.price_data = self.price_data[:, start:]
            self.target_scale_data = self.target_scale_data[:, start:]
            if self.factorvae_alpha20_data is not None:
                self.factorvae_alpha20_data = self.factorvae_alpha20_data[:, start:, :]

        self.target_data = self.gt_data

        if self.selected_feature_indices is not None:
            self.eod_data = self.eod_data[:, :, self.selected_feature_indices]
        self.use_grouped_inputs = bool(int(self.args.use_grouped_inputs))
        self.feature_group_indices = self._resolve_feature_group_indices() if self.use_grouped_inputs else {}

        self.stock_num = self.eod_data.shape[0]
        self.trade_dates = self.eod_data.shape[1]
        self.feature_num = len(self.feature_names) if self.input_transform else self.eod_data.shape[2]
        self.args.features = self.feature_num
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
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _resolve_input_features(self):
        mode = self.input_features.strip()
        full_names = list(self.full_feature_names)
        if mode == "" or mode == "all":
            return None, full_names

        if mode == "raw_ohlcv":
            raw_names = list(self.feature_config["original_feature_names"])
            return self._indices_for_feature_names(raw_names), raw_names

        if mode == "stockmixer_ma":
            self.input_transform = "stockmixer_ma"
            return None, ["MA5", "MA10", "MA20", "MA30", "Close"]

        if mode == "factorvae_alpha20":
            self.input_transform = "factorvae_alpha20"
            return None, list(FACTORVAE_ALPHA20_FEATURE_NAMES)

        if mode in {"ohlcv_alpha20", "raw_ohlcv_alpha20", "factorvae_ohlcv_alpha20"}:
            self.input_transform = "ohlcv_alpha20"
            raw_names = list(self.feature_config["original_feature_names"])
            return None, raw_names + list(FACTORVAE_ALPHA20_FEATURE_NAMES)

        if not full_names:
            raise ValueError(
                "--input_features requires feature_config.json with feature_names when using a named subset"
            )

        feature_groups = self.feature_config["feature_groups"]
        if mode in feature_groups:
            requested = list(feature_groups[mode])
            return self._indices_for_feature_names(requested), requested

        requested = [item.strip() for item in mode.split(",") if item.strip()]
        if not requested:
            raise ValueError("--input_features must be all, raw_ohlcv, a feature-group name, or comma-separated names")
        return self._indices_for_feature_names(requested), requested

    def _indices_for_feature_names(self, names):
        name_to_index = {name: index for index, name in enumerate(self.full_feature_names)}
        missing = [name for name in names if name not in name_to_index]
        if missing:
            raise ValueError(f"Requested input features not found in feature_config.json: {missing}")
        return [name_to_index[name] for name in names]

    def _parse_stockmixer_ma_windows(self):
        value = self.args.stockmixer_ma_windows
        if isinstance(value, str):
            windows = tuple(int(item.strip()) for item in value.split(",") if item.strip())
        else:
            windows = tuple(int(item) for item in value)
        if windows != (5, 10, 20, 30):
            raise ValueError("StockMixer paper input expects moving-average windows 5,10,20,30")
        return windows

    def _prepare_stockmixer_ma_transform(self):
        self.stockmixer_ma_windows = self._parse_stockmixer_ma_windows()
        self.stockmixer_price_data = self.price_data.astype(np.float64, copy=False)
        prefix = np.zeros((self.stockmixer_price_data.shape[0], 1), dtype=np.float64)
        cumsum = np.cumsum(self.stockmixer_price_data, axis=1, dtype=np.float64)
        self.stockmixer_price_cumsum = np.concatenate([prefix, cumsum], axis=1)

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

    def _prepare_factorvae_alpha20_transform(self):
        train_start = self.data_start_index
        train_end = train_start + int(self.args.valid_index)
        alpha20_data, metadata = build_factorvae_alpha20_features(
            self.eod_data,
            self.price_data,
            self.full_feature_names,
            self.mask_data,
            train_start=train_start,
            train_end=train_end,
        )
        if self.input_transform == "ohlcv_alpha20":
            raw_names = list(self.feature_config["original_feature_names"])
            raw_indices = self._indices_for_feature_names(raw_names)
            raw_data = self.eod_data[:, :, raw_indices].astype(np.float32, copy=False)
            self.factorvae_alpha20_data = np.concatenate([raw_data, alpha20_data], axis=2).astype(np.float32, copy=False)
            metadata = dict(metadata)
            metadata["feature_set"] = "raw_ohlcv_plus_qlib_selected_alpha20"
            metadata["raw_feature_names"] = raw_names
        else:
            self.factorvae_alpha20_data = alpha20_data
        self.input_transform_metadata = metadata

    def _derive_feature_group_names(self):
        if "Open" in self.full_feature_names:
            stock_raw = ["Open", "High", "Low", "Volume", "Close"]
        else:
            stock_raw = ["open", "high", "low", "volume", "close"]
        stock_derived = [
            "price_range",
            "open_close",
            "volume_change",
            "overnight_ret",
            "close_to_MA5",
            "close_to_MA10",
            "close_to_MA15",
            "return_momentum_5",
            "return_momentum_10",
            "return_momentum_15",
        ]
        return {
            "endo": ["ret"],
            "stock_raw": stock_raw,
            "stock_derived": stock_derived,
            "market_ctx": [
                "market_return",
                "breadth",
                "ADI",
                "dispersion",
                "up_volume_ratio",
                "volume_surprise",
                "market_momentum_5",
                "market_drawdown_16",
            ],
            "alpha158": [name for name in self.full_feature_names if name.startswith("alpha158_")],
        }

    def _resolve_feature_group_indices(self):
        if self.input_transform is not None:
            raise ValueError(f"grouped inputs are not available for derived input_features={self.input_features!r}")
        if not self.full_feature_names:
            raise ValueError("grouped inputs require feature_config.json with feature_names")

        config_groups = self.feature_config["feature_group_indices"]
        full_group_indices = {
            name: [int(index) for index in indices]
            for name, indices in config_groups.items()
        }

        missing_groups = [name for name in REQUIRED_FEATURE_GROUPS if name not in full_group_indices]
        if missing_groups:
            missing_detail = self.feature_config["missing_feature_groups"]
            raise ValueError(
                "grouped inputs require feature groups {}; missing groups {}; detail={}".format(
                    list(REQUIRED_FEATURE_GROUPS),
                    missing_groups,
                    missing_detail,
                )
            )

        if self.selected_feature_indices is None:
            return {
                name: full_group_indices[name]
                for name in (*REQUIRED_FEATURE_GROUPS, *OPTIONAL_FEATURE_GROUPS)
                if name in full_group_indices
            }

        selected_map = {full_index: index for index, full_index in enumerate(self.selected_feature_indices)}
        resolved = {}
        for group_name in REQUIRED_FEATURE_GROUPS:
            missing = [index for index in full_group_indices[group_name] if index not in selected_map]
            if missing:
                missing_names = [self.full_feature_names[index] for index in missing]
                raise ValueError(
                    f"--input_features={self.input_features!r} excludes required grouped input features "
                    f"for {group_name}: {missing_names}"
                )
            resolved[group_name] = [selected_map[index] for index in full_group_indices[group_name]]
        for group_name in OPTIONAL_FEATURE_GROUPS:
            if group_name not in full_group_indices:
                continue
            missing = [index for index in full_group_indices[group_name] if index not in selected_map]
            if not missing:
                resolved[group_name] = [selected_map[index] for index in full_group_indices[group_name]]
        return resolved

    def _build_target_scale_data(self):
        return np.ones_like(self.gt_data, dtype=np.float32)

    def _resolve_splits(self):
        return self.args.valid_index, self.args.test_index

    def _validate(self):
        if self.eod_data.ndim != 3:
            raise ValueError(f"eod_data must be 3-D, got {self.eod_data.shape}")
        if self.eod_data.shape[:2] != self.mask_data.shape:
            raise ValueError("mask_data shape does not match eod_data")
        if self.gt_data.shape != self.mask_data.shape:
            raise ValueError("gt_data shape does not match mask_data")
        if self.target_data.shape != self.mask_data.shape:
            raise ValueError("target_data shape does not match mask_data")
        if self.target_scale_data.shape != self.mask_data.shape:
            raise ValueError("target_scale_data shape does not match mask_data")
        if self.price_data.shape != self.mask_data.shape:
            raise ValueError("price_data shape does not match mask_data")
        if self.feature_names and len(self.feature_names) != self.feature_num:
            raise ValueError(
                f"feature_names length {len(self.feature_names)} does not match input feature dim {self.feature_num}"
            )
        if not np.isfinite(self.eod_data).all():
            raise ValueError("eod_data contains NaN or Inf")
        if not np.isfinite(self.gt_data).all():
            raise ValueError("gt_data contains NaN or Inf")
        if not np.isfinite(self.target_data).all():
            raise ValueError("target_data contains NaN or Inf")
        if not np.isfinite(self.price_data).all():
            raise ValueError("price_data contains NaN or Inf")
        if self.factorvae_alpha20_data is not None:
            if self.factorvae_alpha20_data.shape[:2] != self.mask_data.shape:
                raise ValueError("factorvae_alpha20_data shape does not match mask_data")
            if self.factorvae_alpha20_data.shape[2] != self.feature_num:
                raise ValueError("factorvae_alpha20_data feature dim does not match feature_num")
            if not np.isfinite(self.factorvae_alpha20_data).all():
                raise ValueError("factorvae_alpha20_data contains NaN or Inf")
        if self.valid_index <= self.args.seq_len + self.args.steps:
            raise ValueError("valid_index is too small for seq_len and steps")
        if self.test_index <= self.valid_index or self.test_index > self.trade_dates:
            raise ValueError("test_index must be in (valid_index, trade_dates]")
        if self.train_end_offset <= self.train_start_offset:
            raise ValueError("no valid train window offsets")

    def get_batch(self, offset):
        """Build one past-window/future-target sample at ``offset``."""

        seq_len = self.args.seq_len
        steps = self.args.steps
        if self.input_transform == "stockmixer_ma":
            inputs = self._build_stockmixer_ma_inputs(offset, seq_len)
        elif self.input_transform in {"factorvae_alpha20", "ohlcv_alpha20"}:
            inputs = self.factorvae_alpha20_data[:, offset : offset + seq_len, :]
        else:
            inputs = self.eod_data[:, offset : offset + seq_len, :]
        mask_batch = self.mask_data[:, offset : offset + seq_len + steps]
        mask_batch = np.min(mask_batch, axis=1)
        batch = {
            "inputs": inputs,
            "mask": np.expand_dims(mask_batch, axis=1),
            "price": np.expand_dims(self.price_data[:, offset + seq_len - 1], axis=1),
            "y_endo": np.expand_dims(self.target_data[:, offset + seq_len + steps - 1], axis=1),
            "raw_gt": np.expand_dims(self.gt_data[:, offset + seq_len + steps - 1], axis=1),
            "target_scale": np.expand_dims(self.target_scale_data[:, offset + seq_len + steps - 1], axis=1),
        }
        if self.use_grouped_inputs:
            batch.update(
                {
                    "x_endo": inputs[:, :, self.feature_group_indices["endo"]],
                    "x_stock_raw": inputs[:, :, self.feature_group_indices["stock_raw"]],
                    "x_stock_derived": inputs[:, :, self.feature_group_indices["stock_derived"]],
                    "x_market_ctx": inputs[:, :, self.feature_group_indices["market_ctx"]],
                }
            )
            if "alpha158" in self.feature_group_indices:
                batch["x_alpha158"] = inputs[:, :, self.feature_group_indices["alpha158"]]
        return batch

    def iter_eval_offsets(self, start_index, end_index):
        """Translate target-date boundaries into valid input-window offsets."""

        start = start_index - self.args.seq_len - self.args.steps + 1
        end = end_index - self.args.seq_len - self.args.steps + 1
        return range(start, end)

    def _loader(self, offsets, batch_size, shuffle, drop_last, return_offset, seed=None):
        dataset = StockWindowDataset(self, offsets, return_offset=return_offset)
        generator = None
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(int(seed if seed is not None else self.args.random_seed))
        num_workers = int(self.args.num_workers)
        kwargs = {
            "batch_size": int(batch_size),
            "shuffle": bool(shuffle),
            "drop_last": bool(drop_last),
            "num_workers": num_workers,
            "pin_memory": bool(self.args.pin_memory),
            "generator": generator,
        }
        if num_workers > 0:
            kwargs["prefetch_factor"] = int(self.args.prefetch_factor)
            kwargs["persistent_workers"] = bool(self.args.persistent_workers)
        return DataLoader(dataset, **kwargs)

    def train_loader(self, epoch=0):
        offsets = self.train_offsets
        if self.args.debug:
            max_windows = int(self.args.debug_train_steps) * int(self.args.batch_size)
            offsets = offsets[:max_windows]
        return self._loader(
            offsets=offsets,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=bool(self.args.train_drop_last),
            return_offset=False,
            seed=int(self.args.random_seed) + int(epoch),
        )

    def eval_loader(self, start_index, end_index, drop_last=False):
        offsets = np.asarray(list(self.iter_eval_offsets(start_index, end_index)), dtype=np.int64)
        if self.args.debug:
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
