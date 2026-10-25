from argparse import Namespace
from pathlib import Path

from .stock_data import StockMarketData


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT_PATH = PROJECT_ROOT / "dataset_norm" / "global_price_volume_standard"

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
        "features": 5,
        "input_features": "all",
        "use_grouped_inputs": False,
        "stockmixer_ma_windows": "5,10,20,30",
        "require_feature_groups": False,
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


def build_market_data(data, **overrides):
    return StockMarketData(make_data_args(data, **overrides))
