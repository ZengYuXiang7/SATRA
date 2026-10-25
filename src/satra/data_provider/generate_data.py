import argparse
import filecmp
import os
import pickle
import shutil

import numpy as np
import pandas as pd


EPS = 1e-6
ALPHA158_ROLLING_WINDOWS = [5, 10, 20, 30, 60]
ALPHA158_ROLLING_OPERATORS = [
    "roc",
    "ma",
    "std",
    "beta",
    "rsqr",
    "resi",
    "max",
    "min",
    "qtlu",
    "qtld",
    "rank",
    "rsv",
    "imax",
    "imin",
    "imxd",
    "corr",
    "cord",
    "cntp",
    "cntn",
    "cntd",
    "sump",
    "sumn",
    "sumd",
    "vma",
    "vstd",
    "wvma",
    "vsump",
    "vsumn",
    "vsumd",
]
STOCK_DERIVED_FEATURES = [
    "ret",
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
MARKET_DERIVED_FEATURES = [
    "market_return",
    "breadth",
    "ADI",
    "dispersion",
    "up_volume_ratio",
    "volume_surprise",
    "market_momentum_5",
    "market_drawdown_16",
]
DATASET_SPLIT_DEFAULTS = {
    "NASDAQ": {"data_start_index": 0, "valid_index": 871},
    "NYSE": {"data_start_index": 0, "valid_index": 871},
    "SP500": {"data_start_index": 915, "valid_index": 1127},
}
INPUT_NORM_METHOD = "standard"
INPUT_NORM_SCOPE = "global"
RAW_NORM_LAYOUT = "price_volume"


def _require_csv_content(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if os.path.getsize(path) == 0:
        raise ValueError(f"CSV file is empty: {path}")
    with open(path, "r", encoding="utf-8") as f:
        if any(line.strip() for line in f):
            return
    raise ValueError(f"CSV file has no non-empty rows: {path}")


def _indices_for_feature_names(feature_names, names):
    name_to_index = {name: index for index, name in enumerate(feature_names)}
    return [name_to_index[name] for name in names]


def _safe_scale(scale):
    scale = float(scale)
    if not np.isfinite(scale) or abs(scale) < EPS:
        return 1.0
    return scale


def _fit_standard(values, train_values):
    center = np.nanmean(train_values)
    scale = _safe_scale(np.nanstd(train_values))
    return np.clip((values - center) / scale, -5.0, 5.0)


def normalize_eod_features(eod_data, feature_names, original_feature_names, market, feature_args):
    split = DATASET_SPLIT_DEFAULTS[market]
    train_start = int(split["data_start_index"])
    train_end = train_start + int(split["valid_index"])
    train_end = min(train_end, eod_data.shape[1])
    raw_dim = len(original_feature_names)
    price_indices = []
    volume_indices = []
    for index, name in enumerate(original_feature_names):
        lower = str(name).lower()
        if lower == "volume":
            volume_indices.append(index)
        else:
            price_indices.append(index)
    groups = []
    if price_indices:
        groups.append(price_indices)
    if volume_indices:
        groups.append(volume_indices)
    groups.extend([[index] for index in range(raw_dim, eod_data.shape[2])])

    normalized = np.empty_like(eod_data, dtype=np.float32)
    for indices in groups:
        values = eod_data[:, :, indices].astype(np.float64, copy=False)
        train_values = eod_data[:, train_start:train_end, indices].reshape(-1, len(indices)).astype(np.float64, copy=False)
        normalized[:, :, indices] = _fit_standard(values.reshape(-1, len(indices)), train_values).reshape(values.shape)

    normalized = np.nan_to_num(normalized, nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32)
    return normalized, {
        "method": INPUT_NORM_METHOD,
        "scope": INPUT_NORM_SCOPE,
        "raw_layout": RAW_NORM_LAYOUT,
        "train_start": train_start,
        "train_end": train_end,
        "fit": "train_only",
        "applied_to": "all_eod_features",
    }


def build_feature_group_metadata(
    feature_names,
    original_feature_names,
    ma_windows=(5, 10, 15),
    return_momentum_windows=(5, 10, 15),
    use_alpha158_features=False,
):
    stock_derived = [
        "price_range",
        "open_close",
        "volume_change",
        "overnight_ret",
        *[f"close_to_MA{int(window)}" for window in ma_windows],
        *[f"return_momentum_{int(window)}" for window in return_momentum_windows],
    ]
    feature_groups = {
        "endo": ["ret"],
        "stock_raw": list(original_feature_names),
        "stock_derived": stock_derived,
        "market_ctx": list(MARKET_DERIVED_FEATURES),
    }
    if use_alpha158_features:
        feature_groups["alpha158"] = [name for name in feature_names if name.startswith("alpha158_")]
    available_groups = {}
    feature_group_indices = {}
    missing_feature_groups = {}
    for group_name, names in feature_groups.items():
        missing = [name for name in names if name not in feature_names]
        if missing:
            missing_feature_groups[group_name] = missing
            continue
        available_groups[group_name] = names
        feature_group_indices[group_name] = _indices_for_feature_names(feature_names, names)
    return available_groups, feature_group_indices, missing_feature_groups


MARKET_CONFIG = {
    "NASDAQ": {
        "kind": "relational_csv",
        "tickers": "NASDAQ_tickers_qualify_dr-0.98_min-5_smooth.csv",
        "drop_last": True,
        "columns": ["Open", "High", "Low", "Volume", "Close"],
        "close_column": "Close",
    },
    "NYSE": {
        "kind": "relational_csv",
        "tickers": "NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv",
        "drop_last": False,
        "columns": ["Open", "High", "Low", "Volume", "Close"],
        "close_column": "Close",
    },
    "SP500": {
        "kind": "estimate_baseline",
        "columns": ["open", "high", "low", "volume", "close"],
        "close_column": "close",
    },
}


def dump_pickle(path, array):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(array.astype(np.float32), f, protocol=pickle.HIGHEST_PROTOCOL)


def dump_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import json

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_tickers(path):
    tickers = np.genfromtxt(path, dtype=str, delimiter="\t", skip_header=False)
    if tickers.ndim > 1:
        tickers = tickers[:, 0]
    return list(tickers)


def resolve_trsr_data_root(predata_root):
    google_finance_root = os.path.join(predata_root, "google_finance")
    if not os.path.isdir(google_finance_root):
        raise FileNotFoundError(f"Expected google_finance directory under predata_root: {google_finance_root}")
    return predata_root


def resolve_estimate_sp500_path(predata_root):
    path = os.path.join(predata_root, "baseline_data_sp500.npy")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def resolve_processed_market_root(predata_root, market):
    path = os.path.join(predata_root, market, "processed")
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    return path


def resolve_trsr_metadata_root(predata_root):
    metadata_path = os.path.join(predata_root, "NASDAQ_aver_line_dates.csv")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(metadata_path)
    return predata_root


def collect_valid_rows(raw, trade_date_to_index, pad_begin, trade_days):
    date_col = raw.columns[0]
    raw[date_col] = raw[date_col].str.replace("-05:00", "", regex=False)
    raw[date_col] = raw[date_col].str.replace("-04:00", "", regex=False)

    dates = []
    source_rows = []
    for row_index, date in enumerate(raw[date_col]):
        if date not in trade_date_to_index:
            continue

        date_index = trade_date_to_index[date]

        if date_index < pad_begin:
            continue

        out_index = date_index - pad_begin
        if out_index >= trade_days:
            continue

        dates.append(out_index)
        source_rows.append(row_index)
    return dates, source_rows


def validate_columns(available_columns, required_columns, source):
    missing = [column for column in required_columns if column not in available_columns]
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")


def _feature_index(columns, name):
    return columns.index(name) if name in columns else None


def _safe_div(numerator, denominator, default=0.0):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    out = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), default, dtype=np.float64)
    return np.divide(numerator, denominator, out=out, where=np.abs(denominator) > EPS)


def _clean_feature(array, clip=True):
    array = np.nan_to_num(array, nan=0.0, posinf=5.0, neginf=-5.0)
    if clip:
        array = np.clip(array, -5.0, 5.0)
    return array.astype(np.float32)


def _past_value(values):
    prev = np.empty_like(values, dtype=np.float64)
    prev[:, 0] = values[:, 0]
    prev[:, 1:] = values[:, :-1]
    return prev


def _lag_value(values, periods):
    periods = int(periods)
    if periods <= 0:
        raise ValueError(f"periods must be positive, got {periods}")
    lagged = np.empty_like(values, dtype=np.float64)
    lagged[:, :periods] = values[:, :1]
    lagged[:, periods:] = values[:, :-periods]
    return lagged


def _rolling_mean_current(values, window):
    frame = pd.DataFrame(values.T)
    rolled = frame.rolling(window=window, min_periods=1).mean().to_numpy(dtype=np.float64).T
    return rolled


def _rolling_std_current(values, window):
    frame = pd.DataFrame(values.T)
    rolled = frame.rolling(window=window, min_periods=1).std().to_numpy(dtype=np.float64).T
    return rolled


def _rolling_min_current(values, window):
    frame = pd.DataFrame(values.T)
    rolled = frame.rolling(window=window, min_periods=1).min().to_numpy(dtype=np.float64).T
    return rolled


def _rolling_max_current(values, window):
    frame = pd.DataFrame(values.T)
    rolled = frame.rolling(window=window, min_periods=1).max().to_numpy(dtype=np.float64).T
    return rolled


def _rolling_quantile_current(values, window, q):
    frame = pd.DataFrame(values.T)
    rolled = frame.rolling(window=window, min_periods=1).quantile(q).to_numpy(dtype=np.float64).T
    return rolled


def _rolling_sum_current(values, window):
    frame = pd.DataFrame(values.T)
    rolled = frame.rolling(window=window, min_periods=1).sum().to_numpy(dtype=np.float64).T
    return rolled


def _rolling_rank_current(values, window):
    rolling = pd.DataFrame(values.T).rolling(window=window, min_periods=1)
    if hasattr(rolling, "rank"):
        return rolling.rank(pct=True).to_numpy(dtype=np.float64).T

    def rank_last(x):
        x = x[np.isfinite(x)]
        if x.size == 0:
            return np.nan
        return float((x <= x[-1]).sum()) / float(x.size)

    return rolling.apply(rank_last, raw=True).to_numpy(dtype=np.float64).T


def _rolling_idx_current(values, window, reducer):
    if reducer == "max":
        func = lambda x: float(np.nanargmax(x) + 1) if np.isfinite(x).any() else np.nan
    elif reducer == "min":
        func = lambda x: float(np.nanargmin(x) + 1) if np.isfinite(x).any() else np.nan
    else:
        raise ValueError(f"unsupported rolling index reducer={reducer!r}")
    frame = pd.DataFrame(values.T)
    return frame.rolling(window=window, min_periods=1).apply(func, raw=True).to_numpy(dtype=np.float64).T


def _rolling_corr_current(left, right, window):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    sum_x = _rolling_sum_current(left, window)
    sum_y = _rolling_sum_current(right, window)
    sum_x2 = _rolling_sum_current(left * left, window)
    sum_y2 = _rolling_sum_current(right * right, window)
    sum_xy = _rolling_sum_current(left * right, window)
    days = np.arange(left.shape[1], dtype=np.float64)
    count = np.minimum(days + 1.0, float(window))[None, :]
    numerator = count * sum_xy - sum_x * sum_y
    denom_x = count * sum_x2 - sum_x * sum_x
    denom_y = count * sum_y2 - sum_y * sum_y
    denom = np.sqrt(np.maximum(denom_x * denom_y, 0.0))
    return _safe_div(numerator, denom)


def _rolling_linear_stats(values, window):
    values = np.asarray(values, dtype=np.float64)
    stock_num, trade_days = values.shape
    absolute_index = np.arange(1, trade_days + 1, dtype=np.float64)[None, :]
    sum_y = _rolling_sum_current(values, window)
    sum_y2 = _rolling_sum_current(values * values, window)
    sum_abs_xy = _rolling_sum_current(values * absolute_index, window)
    n = np.minimum(np.arange(trade_days, dtype=np.float64) + 1.0, float(window))[None, :]
    start = np.maximum(np.arange(trade_days, dtype=np.float64) - float(window) + 1.0, 0.0)[None, :]
    sum_xy = sum_abs_xy - start * sum_y
    sum_x = n * (n + 1.0) / 2.0
    sum_x2 = n * (n + 1.0) * (2.0 * n + 1.0) / 6.0
    denom_x = n * sum_x2 - sum_x * sum_x
    numerator = n * sum_xy - sum_x * sum_y
    slope = _safe_div(numerator, denom_x)
    intercept = _safe_div(sum_y - slope * sum_x, n)
    current_x = n
    fitted_current = intercept + slope * current_x
    residual = values - fitted_current
    denom_y = n * sum_y2 - sum_y * sum_y
    rsquare = _safe_div(numerator * numerator, denom_x * denom_y)
    rsquare = np.where(n >= 2.0, rsquare, 0.0)
    slope = np.where(n >= 2.0, slope, 0.0)
    residual = np.where(n >= 2.0, residual, 0.0)
    return (
        np.broadcast_to(slope, (stock_num, trade_days)).astype(np.float64, copy=False),
        np.broadcast_to(rsquare, (stock_num, trade_days)).astype(np.float64, copy=False),
        np.broadcast_to(residual, (stock_num, trade_days)).astype(np.float64, copy=False),
    )


def _rolling_mean_abs_current(values, window):
    frame = pd.DataFrame(np.abs(values).T)
    rolled = frame.rolling(window=window, min_periods=1).mean().to_numpy(dtype=np.float64).T
    return rolled


def _rolling_past_zscore(values, window):
    series = pd.Series(values)
    hist_mean = series.shift(1).rolling(window=window, min_periods=1).mean()
    hist_std = series.shift(1).rolling(window=window, min_periods=2).std(ddof=0)
    z = (series - hist_mean) / (hist_std + EPS)
    return z.fillna(0.0).to_numpy(dtype=np.float64)


def _market_index_from_return(market_return):
    index = np.ones_like(market_return, dtype=np.float64)
    for t in range(1, market_return.shape[0]):
        index[t] = index[t - 1] * (1.0 + market_return[t])
    return index


def _market_momentum(index, window):
    out = np.zeros_like(index, dtype=np.float64)
    for t in range(index.shape[0]):
        start = max(0, t - window)
        if t > start:
            out[t] = _safe_div(index[t], index[start], default=1.0) - 1.0
    return out


def _market_drawdown(index, window):
    out = np.zeros_like(index, dtype=np.float64)
    for t in range(index.shape[0]):
        start = max(0, t - window + 1)
        peak = np.max(index[start : t + 1])
        out[t] = _safe_div(index[t], peak, default=1.0) - 1.0
    return out


def _fill_raw_history(raw_values):
    filled = np.empty_like(raw_values, dtype=np.float64)
    for stock_idx in range(raw_values.shape[0]):
        frame = pd.DataFrame(raw_values[stock_idx])
        filled[stock_idx] = frame.ffill().bfill().fillna(0.0).to_numpy(dtype=np.float64)
    return filled


def build_alpha158_features(open_, high, low, close, volume, windows=ALPHA158_ROLLING_WINDOWS):
    windows = [int(window) for window in windows]
    high_low = high - low
    max_oc = np.maximum(open_, close)
    min_oc = np.minimum(open_, close)

    features = [
        _safe_div(close - open_, open_),
        _safe_div(high_low, open_),
        _safe_div(close - open_, high_low),
        _safe_div(high - max_oc, open_),
        _safe_div(high - max_oc, high_low),
        _safe_div(min_oc - low, open_),
        _safe_div(min_oc - low, high_low),
        _safe_div(2.0 * close - high - low, open_),
        _safe_div(2.0 * close - high - low, high_low),
        _safe_div(open_, close, default=1.0),
        _safe_div(high, close, default=1.0),
        _safe_div(low, close, default=1.0),
    ]
    names = [
        "alpha158_kmid",
        "alpha158_klen",
        "alpha158_kmid2",
        "alpha158_kup",
        "alpha158_kup2",
        "alpha158_klow",
        "alpha158_klow2",
        "alpha158_ksft",
        "alpha158_ksft2",
        "alpha158_open0",
        "alpha158_high0",
        "alpha158_low0",
    ]

    close_ref1 = _past_value(close)
    volume_ref1 = _past_value(volume)
    close_ratio = _safe_div(close, close_ref1, default=1.0)
    volume_ratio = _safe_div(volume, volume_ref1, default=1.0)
    close_delta = close - close_ref1
    volume_delta = volume - volume_ref1
    close_gain = np.maximum(close_delta, 0.0)
    close_loss = np.maximum(-close_delta, 0.0)
    close_abs_delta = np.abs(close_delta)
    volume_gain = np.maximum(volume_delta, 0.0)
    volume_loss = np.maximum(-volume_delta, 0.0)
    volume_abs_delta = np.abs(volume_delta)
    close_up = (close > close_ref1).astype(np.float64)
    close_down = (close < close_ref1).astype(np.float64)
    weighted_abs_return_volume = np.abs(close_ratio - 1.0) * volume
    corr_volume = np.log(np.maximum(volume, 0.0) + 1.0)
    cord_volume = np.log(np.maximum(volume_ratio, 0.0) + 1.0)

    for window in windows:
        slope, rsquare, residual = _rolling_linear_stats(close, window)
        rolling_max_high = _rolling_max_current(high, window)
        rolling_min_low = _rolling_min_current(low, window)
        idx_max = _rolling_idx_current(high, window, "max")
        idx_min = _rolling_idx_current(low, window, "min")
        close_abs_sum = _rolling_sum_current(close_abs_delta, window)
        volume_abs_sum = _rolling_sum_current(volume_abs_delta, window)
        wvma_mean = _rolling_mean_current(weighted_abs_return_volume, window)

        window_features = {
            "roc": _safe_div(_lag_value(close, window), close, default=1.0),
            "ma": _safe_div(_rolling_mean_current(close, window), close, default=1.0),
            "std": _safe_div(_rolling_std_current(close, window), close),
            "beta": _safe_div(slope, close),
            "rsqr": rsquare,
            "resi": _safe_div(residual, close),
            "max": _safe_div(rolling_max_high, close, default=1.0),
            "min": _safe_div(rolling_min_low, close, default=1.0),
            "qtlu": _safe_div(_rolling_quantile_current(close, window, 0.8), close, default=1.0),
            "qtld": _safe_div(_rolling_quantile_current(close, window, 0.2), close, default=1.0),
            "rank": _rolling_rank_current(close, window),
            "rsv": _safe_div(close - rolling_min_low, rolling_max_high - rolling_min_low),
            "imax": _safe_div(idx_max, float(window)),
            "imin": _safe_div(idx_min, float(window)),
            "imxd": _safe_div(idx_max - idx_min, float(window)),
            "corr": _rolling_corr_current(close, corr_volume, window),
            "cord": _rolling_corr_current(close_ratio, cord_volume, window),
            "cntp": _rolling_mean_current(close_up, window),
            "cntn": _rolling_mean_current(close_down, window),
            "cntd": _rolling_mean_current(close_up, window) - _rolling_mean_current(close_down, window),
            "sump": _safe_div(_rolling_sum_current(close_gain, window), close_abs_sum),
            "sumn": _safe_div(_rolling_sum_current(close_loss, window), close_abs_sum),
            "sumd": _safe_div(
                _rolling_sum_current(close_gain, window) - _rolling_sum_current(close_loss, window),
                close_abs_sum,
            ),
            "vma": _safe_div(_rolling_mean_current(volume, window), volume),
            "vstd": _safe_div(_rolling_std_current(volume, window), volume),
            "wvma": _safe_div(_rolling_std_current(weighted_abs_return_volume, window), wvma_mean),
            "vsump": _safe_div(_rolling_sum_current(volume_gain, window), volume_abs_sum),
            "vsumn": _safe_div(_rolling_sum_current(volume_loss, window), volume_abs_sum),
            "vsumd": _safe_div(
                _rolling_sum_current(volume_gain, window) - _rolling_sum_current(volume_loss, window),
                volume_abs_sum,
            ),
        }
        for op_name in ALPHA158_ROLLING_OPERATORS:
            features.append(window_features[op_name])
            names.append(f"alpha158_{op_name}{window}")

    return np.stack([_clean_feature(feature) for feature in features], axis=-1), names


def _linear_fill_1d(values, valid):
    values = np.asarray(values)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if valid.all():
        return values.copy()
    if not valid.any():
        return np.full(values.shape, np.nan, dtype=np.float64)
    index = np.arange(values.shape[0])
    filled = values.astype(np.float64, copy=True)
    missing = ~valid
    filled[missing] = np.interp(index[missing], index[valid], values[valid])
    return filled


def _fill_with_day_median(values, valid_mask):
    values = values.astype(np.float64, copy=True)
    valid_mask = valid_mask.astype(bool)
    for feature_idx in range(values.shape[-1]):
        feature = values[:, :, feature_idx]
        masked = feature.copy()
        masked[~valid_mask] = np.nan
        day_median = np.nanmedian(masked, axis=0)
        global_median = float(np.nanmedian(masked)) if np.isfinite(masked).any() else 0.0
        day_median = np.where(np.isfinite(day_median), day_median, global_median)
        missing = ~np.isfinite(feature)
        if missing.any():
            stock_idx, day_idx = np.where(missing)
            feature[stock_idx, day_idx] = day_median[day_idx]
        values[:, :, feature_idx] = feature
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def impute_ohlcv_data(normalized_data, raw_values, masks, close_index, method):
    method = str(method).lower()
    observed_mask = masks.astype(np.float32, copy=True)
    if method == "none":
        base_price = normalized_data[:, :, close_index].astype(np.float32, copy=True)
        ground_truth = np.zeros(masks.shape, dtype=np.float32)
        close = raw_values[:, :, close_index]
        valid = (masks > 0.0) & np.isfinite(close) & (close > EPS)
        gt_valid = valid[:, 1:] & valid[:, :-1]
        ground_truth[:, 1:] = np.where(gt_valid, _safe_div(close[:, 1:], close[:, :-1]) - 1.0, 0.0)
        return normalized_data, raw_values, masks, ground_truth, base_price, observed_mask, {}
    if method != "linear":
        raise ValueError(f"unsupported --impute_missing={method!r}; supported values are none, linear")

    filled_norm = normalized_data.astype(np.float64, copy=True)
    filled_raw = raw_values.astype(np.float64, copy=True)
    valid_mask = (masks > 0.0) & np.all(np.isfinite(filled_norm), axis=-1) & np.all(np.isfinite(filled_raw), axis=-1)
    for stock_idx in range(filled_norm.shape[0]):
        valid = valid_mask[stock_idx]
        if valid.all():
            continue
        for feature_idx in range(filled_norm.shape[-1]):
            filled_norm[stock_idx, :, feature_idx] = _linear_fill_1d(filled_norm[stock_idx, :, feature_idx], valid)
            filled_raw[stock_idx, :, feature_idx] = _linear_fill_1d(filled_raw[stock_idx, :, feature_idx], valid)

    filled_norm = _fill_with_day_median(filled_norm, valid_mask).astype(np.float32)
    filled_raw = _fill_with_day_median(filled_raw, valid_mask).astype(np.float64)
    filled_mask = np.ones_like(masks, dtype=np.float32)
    base_price = filled_norm[:, :, close_index].astype(np.float32, copy=True)

    close = filled_raw[:, :, close_index]
    close_valid = np.isfinite(close) & (np.abs(close) > EPS)
    ground_truth = np.zeros(masks.shape, dtype=np.float32)
    gt_valid = close_valid[:, 1:] & close_valid[:, :-1]
    ground_truth[:, 1:] = np.where(gt_valid, _safe_div(close[:, 1:], close[:, :-1]) - 1.0, 0.0).astype(np.float32)
    stats = {
        "original_missing_slots": int((observed_mask < 0.5).sum()),
        "original_valid_ratio": float((observed_mask >= 0.5).mean()),
        "final_missing_slots": int((filled_mask < 0.5).sum()),
        "method": method,
    }
    return filled_norm, filled_raw, filled_mask, ground_truth, base_price, observed_mask, stats


def augment_ohlcv_features(
    normalized_data,
    raw_values,
    masks,
    columns,
    close_column,
    use_ohlcv_features=True,
    use_market_features=True,
    ma_windows=(5, 10, 15),
    return_momentum_windows=(5, 10, 15),
    market_momentum_window=5,
    market_drawdown_window=16,
    volume_surprise_window=5,
    use_alpha158_features=False,
    alpha158_windows=ALPHA158_ROLLING_WINDOWS,
):
    feature_names = list(columns)
    blocks = [normalized_data.astype(np.float32)]
    raw = _fill_raw_history(raw_values)

    open_idx = _feature_index(columns, "Open")
    high_idx = _feature_index(columns, "High")
    low_idx = _feature_index(columns, "Low")
    volume_idx = _feature_index(columns, "Volume")
    close_idx = _feature_index(columns, close_column)
    pre_close_idx = _feature_index(columns, "pre_close")
    if open_idx is None:
        open_idx = _feature_index(columns, "open")
    if high_idx is None:
        high_idx = _feature_index(columns, "high")
    if low_idx is None:
        low_idx = _feature_index(columns, "low")
    if volume_idx is None:
        volume_idx = _feature_index(columns, "volume")
    if pre_close_idx is None:
        pre_close_idx = _feature_index(columns, "preclose")
    required = {
        "open": open_idx,
        "high": high_idx,
        "low": low_idx,
        "close": close_idx,
        "volume": volume_idx,
    }
    missing = [name for name, idx in required.items() if idx is None]
    if missing:
        raise ValueError(f"Cannot augment OHLCV features, missing columns: {missing}")

    open_ = raw[:, :, open_idx]
    high = raw[:, :, high_idx]
    low = raw[:, :, low_idx]
    close = raw[:, :, close_idx]
    volume = raw[:, :, volume_idx]
    
    pre_close = raw[:, :, pre_close_idx] if pre_close_idx is not None else _past_value(close)
    pre_volume = _past_value(volume)

    ret = _safe_div(close, pre_close, default=1.0) - 1.0
    ret[:, 0] = 0.0 if pre_close_idx is None else ret[:, 0]
    valid = (
        (masks > 0.0)
        & np.isfinite(close)
        & (close > EPS)
        & np.isfinite(volume)
        & (volume >= 0.0)
        & np.isfinite(ret)
    )
    ret = np.where(valid, ret, 0.0)

    stock_features = []
    
    if use_ohlcv_features:
        price_range = _safe_div(high - low, close)
        open_close = _safe_div(close, open_, default=1.0) - 1.0
        volume_change = _safe_div(volume, pre_volume, default=1.0) - 1.0
        volume_change[:, 0] = 0.0
        overnight_ret = _safe_div(open_, pre_close, default=1.0) - 1.0
        overnight_ret[:, 0] = 0.0 if pre_close_idx is None else overnight_ret[:, 0]
        ma_features = []
        for window in ma_windows:
            ma = _rolling_mean_current(close, int(window))
            ma_features.append(_safe_div(close, ma, default=1.0) - 1.0)
        return_momentum_features = []
        for window in return_momentum_windows:
            window = int(window)
            lagged_close = _lag_value(close, window)
            momentum = _safe_div(close, lagged_close, default=1.0) - 1.0
            momentum[:, :window] = 0.0
            return_momentum_features.append(momentum)

        stock_feature_names = [
            "ret",
            "price_range",
            "open_close",
            "volume_change",
            "overnight_ret",
            *[f"close_to_MA{int(window)}" for window in ma_windows],
            *[f"return_momentum_{int(window)}" for window in return_momentum_windows],
        ]
        stock_features = [
            ret,
            price_range,
            open_close,
            volume_change,
            overnight_ret,
            *ma_features,
            *return_momentum_features,
        ]
        blocks.append(np.stack([_clean_feature(x) for x in stock_features], axis=-1))
        feature_names.extend(stock_feature_names)

    if use_market_features:
        valid_float = valid.astype(np.float64)
        valid_count = np.maximum(valid_float.sum(axis=0), 1.0)
        market_return = (ret * valid_float).sum(axis=0) / valid_count
        breadth = ((ret > 0.0) & valid).sum(axis=0) / valid_count
        n_up = ((ret > 0.0) & valid).sum(axis=0)
        n_down = ((ret < 0.0) & valid).sum(axis=0)
        adi = (n_up - n_down) / valid_count
        dispersion = np.zeros(ret.shape[1], dtype=np.float64)
        for t in range(ret.shape[1]):
            day_ret = ret[valid[:, t], t]
            dispersion[t] = float(np.std(day_ret)) if day_ret.size else 0.0
        up_volume = (volume * ((ret > 0.0) & valid)).sum(axis=0)
        total_volume = (volume * valid_float).sum(axis=0)
        up_volume_ratio = _safe_div(up_volume, total_volume, default=0.5)
        market_volume = total_volume
        volume_surprise = _rolling_past_zscore(market_volume, int(volume_surprise_window))
        market_index = _market_index_from_return(market_return)
        market_momentum = _market_momentum(market_index, int(market_momentum_window))
        market_drawdown = _market_drawdown(market_index, int(market_drawdown_window))
        market_features = [
            market_return,
            breadth,
            adi,
            dispersion,
            up_volume_ratio,
            volume_surprise,
            market_momentum,
            market_drawdown,
        ]
        market_block = np.stack([_clean_feature(x) for x in market_features], axis=-1)
        blocks.append(np.broadcast_to(market_block[None, :, :], (normalized_data.shape[0], market_block.shape[0], market_block.shape[1])).copy())
        feature_names.extend(MARKET_DERIVED_FEATURES)

    if use_alpha158_features:
        alpha_block, alpha_names = build_alpha158_features(open_, high, low, close, volume, windows=alpha158_windows)
        blocks.append(alpha_block)
        feature_names.extend(alpha_names)

    augmented = np.concatenate(blocks, axis=-1).astype(np.float32)
    augmented = np.nan_to_num(augmented, nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32)
    return augmented, feature_names


def compute_market_max_values(csv_root, market, tickers, trade_date_to_index, pad_begin, trade_days, columns):
    market_max = np.zeros(len(columns), dtype=np.float64)
    for ticker in tickers:
        path = os.path.join(csv_root, f"{market}_{ticker}_30Y.csv")
        _require_csv_content(path)
        raw = pd.read_csv(path)
        validate_columns(raw.columns, columns, path)
        _, source_rows = collect_valid_rows(raw, trade_date_to_index, pad_begin, trade_days)
        if not source_rows:
            raise ValueError(f"{path} has no rows matching the trading calendar")
        values = raw.iloc[source_rows][columns].to_numpy(dtype=np.float64)
        market_max = np.maximum(market_max, values.max(axis=0))
    market_max[market_max == 0.0] = 1.0
    return market_max


def compute_stock_max_values(values):
    stock_max = np.max(values, axis=0).astype(np.float64)
    stock_max[~np.isfinite(stock_max)] = 1.0
    stock_max[stock_max == 0.0] = 1.0
    return stock_max


def convert_relational_csv(predata_root, market, feature_args):
    cfg = MARKET_CONFIG[market]
    columns = cfg["columns"]
    close_index = columns.index(cfg["close_column"])
    data_root = resolve_trsr_data_root(predata_root)
    csv_root = os.path.join(data_root, "google_finance")
    tickers = read_tickers(os.path.join(data_root, cfg["tickers"]))

    trading_dates = np.genfromtxt(
        os.path.join(data_root, f"{market}_aver_line_dates.csv"),
        dtype=str,
        delimiter=",",
        skip_header=False,
    )
    trade_date_to_index = {date: i for i, date in enumerate(trading_dates)}
    pad_begin = 29
    trade_days = len(trading_dates) - pad_begin
    if cfg["drop_last"]:
        trade_days -= 1
    normalization_scope = feature_args["normalization_scope"]
    market_max_values = None
    if normalization_scope == "market":
        market_max_values = compute_market_max_values(
            csv_root, market, tickers, trade_date_to_index, pad_begin, trade_days, columns
        )

    normalized_data = np.full((len(tickers), trade_days, len(columns)), np.nan, dtype=np.float32)
    raw_values = np.full((len(tickers), trade_days, len(columns)), np.nan, dtype=np.float64)
    masks = np.zeros((len(tickers), trade_days), dtype=np.float32)
    ground_truth = np.zeros((len(tickers), trade_days), dtype=np.float32)
    base_price = np.full((len(tickers), trade_days), np.nan, dtype=np.float32)

    for stock_index, ticker in enumerate(tickers):
        
        path = os.path.join(csv_root, f"{market}_{ticker}_30Y.csv")
        _require_csv_content(path)

        raw = pd.read_csv(path)
        validate_columns(raw.columns, columns, path)
        dates, source_rows = collect_valid_rows(raw, trade_date_to_index, pad_begin, trade_days)
        if not source_rows:
            raise ValueError(f"{path} has no rows matching the trading calendar")

        values = raw.iloc[source_rows][columns].to_numpy(dtype=np.float64)
        if normalization_scope == "stock":
            max_values = compute_stock_max_values(values)
        else:
            max_values = market_max_values
        norm_values = values / max_values



        for i, out_index in enumerate(dates):
            
            normalized_data[stock_index, out_index, :] = norm_values[i]
            raw_values[stock_index, out_index, :] = values[i]
            base_price[stock_index, out_index] = norm_values[i, close_index]
            
            if np.all(values[i] > 0.0):
                masks[stock_index, out_index] = 1.0
            if (
                i > 0
                and dates[i] - dates[i - 1] == 1
                and masks[stock_index, out_index] == 1.0
                and masks[stock_index, dates[i - 1]] == 1.0
                and norm_values[i - 1, close_index] != 0.0
            ):
                ground_truth[stock_index, out_index] = (
                    (norm_values[i, close_index] - norm_values[i - 1, close_index])
                    / norm_values[i - 1, close_index]
                )

    normalized_data, raw_values, masks, ground_truth, base_price, observed_mask, imputation_stats = impute_ohlcv_data(
        normalized_data,
        raw_values,
        masks,
        close_index,
        feature_args["impute_missing"],
    )

    eod_data, feature_names = augment_ohlcv_features(
        normalized_data,
        raw_values,
        masks,
        columns,
        cfg["close_column"],
        use_ohlcv_features=feature_args["use_ohlcv_features"],
        use_market_features=feature_args["use_market_features"],
        ma_windows=feature_args["ma_windows"],
        return_momentum_windows=feature_args["return_momentum_windows"],
        market_momentum_window=feature_args["market_momentum_window"],
        market_drawdown_window=feature_args["market_drawdown_window"],
        volume_surprise_window=feature_args["volume_surprise_window"],
        use_alpha158_features=feature_args["use_alpha158_features"],
        alpha158_windows=feature_args["alpha158_windows"],
    )

    return eod_data, masks, ground_truth, base_price, feature_names, observed_mask, imputation_stats


def convert_processed_csv(predata_root, market, feature_args):
    cfg = MARKET_CONFIG[market]
    columns = cfg["columns"]
    close_index = columns.index(cfg["close_column"])
    processed_root = resolve_processed_market_root(predata_root, market)
    if processed_root is None:
        raise FileNotFoundError(f"Cannot find processed source directory for {market} under {predata_root}")

    metadata_root = resolve_trsr_metadata_root(predata_root)
    tickers = read_tickers(os.path.join(metadata_root, cfg["tickers"]))
    trading_dates = np.genfromtxt(
        os.path.join(metadata_root, f"{market}_aver_line_dates.csv"),
        dtype=str,
        delimiter=",",
        skip_header=False,
    )
    trade_days = len(trading_dates) - 29
    if cfg["drop_last"]:
        trade_days -= 1

    normalized_data = np.zeros((len(tickers), trade_days, len(columns)), dtype=np.float32)
    raw_values = np.zeros((len(tickers), trade_days, len(columns)), dtype=np.float64)
    masks = np.zeros((len(tickers), trade_days), dtype=np.float32)
    ground_truth = np.zeros((len(tickers), trade_days), dtype=np.float32)
    base_price = np.zeros((len(tickers), trade_days), dtype=np.float32)

    for stock_index, ticker in enumerate(tickers):
        path = os.path.join(processed_root, f"{market}_{ticker}_1.csv")
        _require_csv_content(path)
        frame = pd.read_csv(path, header=None)
        values = frame.iloc[:, 1 : 1 + len(columns)].to_numpy(dtype=np.float64)
        if values.shape[1] != len(columns):
            raise ValueError(f"{path} expected {len(columns)} OHLCV columns after index column, got {values.shape[1]}")
        days = min(trade_days, values.shape[0])
        values = values[:days]
        normalized_data[stock_index, :days] = values.astype(np.float32)
        raw_values[stock_index, :days] = values
        base_price[stock_index, :days] = values[:, close_index].astype(np.float32)
        valid = np.all(np.isfinite(values), axis=1) & np.all(values > 0.0, axis=1)
        masks[stock_index, :days] = valid.astype(np.float32)
        close = values[:, close_index]
        gt_valid = valid[1:] & valid[:-1] & (np.abs(close[:-1]) > EPS)
        ground_truth[stock_index, 1:days] = np.where(gt_valid, (close[1:] - close[:-1]) / close[:-1], 0.0)

    normalized_data, raw_values, masks, ground_truth, base_price, observed_mask, imputation_stats = impute_ohlcv_data(
        normalized_data,
        raw_values,
        masks,
        close_index,
        feature_args["impute_missing"],
    )

    eod_data, feature_names = augment_ohlcv_features(
        normalized_data,
        raw_values,
        masks,
        columns,
        cfg["close_column"],
        use_ohlcv_features=feature_args["use_ohlcv_features"],
        use_market_features=feature_args["use_market_features"],
        ma_windows=feature_args["ma_windows"],
        return_momentum_windows=feature_args["return_momentum_windows"],
        market_momentum_window=feature_args["market_momentum_window"],
        market_drawdown_window=feature_args["market_drawdown_window"],
        volume_surprise_window=feature_args["volume_surprise_window"],
        use_alpha158_features=feature_args["use_alpha158_features"],
        alpha158_windows=feature_args["alpha158_windows"],
    )
    return eod_data, masks, ground_truth, base_price, feature_names, observed_mask, imputation_stats


def convert_sp500(predata_root, feature_args):
    cfg = MARKET_CONFIG["SP500"]
    columns = cfg["columns"]
    close_index = columns.index(cfg["close_column"])
    path = resolve_estimate_sp500_path(predata_root)
    baseline = np.load(path, allow_pickle=True).item()
    symbols = list(baseline.keys())
    first = baseline[symbols[0]]
    trade_days = len(first)

    normalized_data = np.zeros((len(symbols), trade_days, len(columns)), dtype=np.float32)
    raw_values = np.zeros((len(symbols), trade_days, len(columns)), dtype=np.float64)
    masks = np.ones((len(symbols), trade_days), dtype=np.float32)
    ground_truth = np.zeros((len(symbols), trade_days), dtype=np.float32)
    base_price = np.zeros((len(symbols), trade_days), dtype=np.float32)

    for symbol in symbols:
        validate_columns(baseline[symbol].columns, columns, f"SP500:{symbol}")
    normalization_scope = feature_args["normalization_scope"]
    market_max_values = None
    if normalization_scope == "market":
        market_max_values = pd.concat([baseline[symbol][columns] for symbol in symbols], axis=0).max()
        market_max_values[market_max_values == 0.0] = 1.0

    for stock_index, symbol in enumerate(symbols):
        df = baseline[symbol]
        raw_values[stock_index] = df[columns].to_numpy(dtype=np.float64)
        if normalization_scope == "stock":
            max_values = df[columns].max()
            max_values[max_values == 0.0] = 1.0
        else:
            max_values = market_max_values
        features = df[columns].div(max_values, axis=1).to_numpy(dtype=np.float32)
        normalized_data[stock_index] = features
        base_price[stock_index] = features[:, close_index]

        ground_truth[stock_index, 1:] = (
            (base_price[stock_index, 1:] - base_price[stock_index, :-1])
            / base_price[stock_index, :-1]
        )

    normalized_data, raw_values, masks, ground_truth, base_price, observed_mask, imputation_stats = impute_ohlcv_data(
        normalized_data,
        raw_values,
        masks,
        close_index,
        feature_args["impute_missing"],
    )

    eod_data, feature_names = augment_ohlcv_features(
        normalized_data,
        raw_values,
        masks,
        columns,
        cfg["close_column"],
        use_ohlcv_features=feature_args["use_ohlcv_features"],
        use_market_features=feature_args["use_market_features"],
        ma_windows=feature_args["ma_windows"],
        return_momentum_windows=feature_args["return_momentum_windows"],
        market_momentum_window=feature_args["market_momentum_window"],
        market_drawdown_window=feature_args["market_drawdown_window"],
        volume_surprise_window=feature_args["volume_surprise_window"],
        use_alpha158_features=feature_args["use_alpha158_features"],
        alpha158_windows=feature_args["alpha158_windows"],
    )

    return eod_data, masks, ground_truth, base_price, feature_names, observed_mask, imputation_stats


def convert_market(predata_root, output_root, market, feature_args):
    cfg = MARKET_CONFIG[market]
    if cfg["kind"] == "relational_csv":
        source_layout = feature_args["source_layout"]
        if source_layout == "processed_csv":
            eod_data, masks, ground_truth, base_price, feature_names, observed_mask, imputation_stats = convert_processed_csv(
                predata_root, market, feature_args
            )
        elif source_layout == "relational_csv":
            eod_data, masks, ground_truth, base_price, feature_names, observed_mask, imputation_stats = convert_relational_csv(
                predata_root, market, feature_args
            )
        else:
            raise ValueError(f"Unsupported source_layout={source_layout!r}")
    elif cfg["kind"] == "estimate_baseline":
        eod_data, masks, ground_truth, base_price, feature_names, observed_mask, imputation_stats = convert_sp500(
            predata_root, feature_args
        )
    else:
        raise ValueError(f"Unsupported market kind: {cfg['kind']}")

    out_dir = os.path.join(output_root, f"{market}_ohlcv")
    original_feature_names = MARKET_CONFIG[market]["columns"]
    eod_data, input_normalization = normalize_eod_features(eod_data, feature_names, original_feature_names, market, feature_args)
    feature_groups, feature_group_indices, missing_feature_groups = build_feature_group_metadata(
        feature_names,
        original_feature_names,
        ma_windows=feature_args["ma_windows"],
        return_momentum_windows=feature_args["return_momentum_windows"],
        use_alpha158_features=feature_args["use_alpha158_features"],
    )

    dump_pickle(os.path.join(out_dir, "eod_data.pkl"), eod_data)
    dump_pickle(os.path.join(out_dir, "mask_data.pkl"), masks)
    dump_pickle(os.path.join(out_dir, "observed_mask_data.pkl"), observed_mask)
    dump_pickle(os.path.join(out_dir, "gt_data.pkl"), ground_truth)
    dump_pickle(os.path.join(out_dir, "price_data.pkl"), base_price)
    if feature_args["use_alpha158_features"]:
        if "alpha158" not in feature_groups:
            raise ValueError("alpha158 feature group is required when use_alpha158_features is enabled")
        alpha158_feature_count = len(feature_groups["alpha158"])
    else:
        alpha158_feature_count = 0

    dump_json(
        os.path.join(out_dir, "feature_config.json"),
        {
            "market": market,
            "feature_names": feature_names,
            "feature_dim": len(feature_names),
            "original_feature_names": original_feature_names,
            "derived_feature_names": feature_names[len(original_feature_names) :],
            "feature_groups": feature_groups,
            "feature_group_indices": feature_group_indices,
            "missing_feature_groups": missing_feature_groups,
            "imputation": imputation_stats,
            "input_normalization": input_normalization,
            "alpha158": {
                "enabled": feature_args["use_alpha158_features"],
                "windows": feature_args["alpha158_windows"],
                "feature_count": alpha158_feature_count,
                "source": "raw OHLCV",
                "omitted_fields": ["alpha158_vwap0"],
                "omitted_reason": "raw OHLCV does not include amount or true VWAP",
            },
            **feature_args,
        },
    )

    return {
        "eod_data": eod_data.shape,
        "mask_data": masks.shape,
        "observed_mask_data": observed_mask.shape,
        "gt_data": ground_truth.shape,
        "price_data": base_price.shape,
        "feature_dim": len(feature_names),
        "imputation": imputation_stats,
    }


def verify_arrays(output_root, reference_root, market):
    names = ["eod_data.pkl", "mask_data.pkl", "gt_data.pkl", "price_data.pkl"]
    results = []
    for name in names:
        generated_path = os.path.join(output_root, f"{market}_ohlcv", name)
        reference_path = os.path.join(reference_root, f"{market}_ohlcv", name)
        with open(generated_path, "rb") as f:
            generated = np.asarray(pickle.load(f), dtype=np.float32)
        with open(reference_path, "rb") as f:
            reference = np.asarray(pickle.load(f), dtype=np.float32)
        same_shape = generated.shape == reference.shape
        max_abs_diff = 0.0 if not same_shape else float(np.max(np.abs(generated - reference)))
        array_equal = same_shape and np.array_equal(generated, reference)
        
        allclose = same_shape and np.allclose(generated, reference, rtol=0.0, atol=1e-7)
        
        results.append((name, same_shape, array_equal, allclose, max_abs_diff))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert source files under raw_data into the *_ohlcv pickle datasets used by this project."
    )
    parser.add_argument("--predata-root", default="raw_data")
    parser.add_argument("--output-root", default="data/processed")
    parser.add_argument("--reference-root", default="data/processed")
    parser.add_argument("--source_layout", choices=["processed_csv", "relational_csv"], default="processed_csv")
    parser.add_argument("--markets", nargs="+", default=["NASDAQ", "NYSE", "SP500"], choices=list(MARKET_CONFIG))
    parser.add_argument(
        "--feature_set",
        type=str,
        default="stock_market_aug",
        choices=["ohlcv_only", "raw", "stock_aug", "stock_market_aug"],
        help=(
            "Feature ablation preset. ohlcv_only/raw=original OHLCV columns only; "
            "stock_aug adds stock-level features; stock_market_aug adds stock and market features."
        ),
    )
    parser.add_argument(
        "--ohlcv_only",
        type=int,
        default=0,
        help="If 1, override feature_set/use_* flags and write only original OHLCV features.",
    )
    parser.add_argument("--use_ohlcv_features", type=int, default=-1)
    parser.add_argument("--use_market_features", type=int, default=-1)
    parser.add_argument("--ma_windows", nargs="+", type=int, default=[5, 10, 15])
    parser.add_argument("--return_momentum_windows", nargs="+", type=int, default=[5, 10, 15])
    parser.add_argument("--market_momentum_window", type=int, default=5)
    parser.add_argument("--market_drawdown_window", type=int, default=16)
    parser.add_argument("--volume_surprise_window", type=int, default=5)
    parser.add_argument(
        "--use_alpha158_features",
        type=int,
        default=0,
        help="If 1, append Qlib-style Alpha158 features computed from raw OHLCV after existing features.",
    )
    parser.add_argument("--alpha158_windows", nargs="+", type=int, default=ALPHA158_ROLLING_WINDOWS)
    parser.add_argument(
        "--normalization_scope",
        type=str,
        default="stock",
        choices=["market", "stock"],
        help="Legacy pre-augmentation OHLCV normalization scope for raw-source conversion.",
    )
    parser.add_argument(
        "--impute_missing",
        type=str,
        default="linear",
        choices=["none", "linear"],
        help="Fill missing stock-day OHLCV before feature augmentation. linear uses per-stock time interpolation followed by day-median fill.",
    )
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    preset = {
        "ohlcv_only": (False, False),
        "raw": (False, False),
        "stock_aug": (True, False),
        "stock_market_aug": (True, True),
    }[args.feature_set]
    ohlcv_only = bool(args.ohlcv_only) or args.feature_set == "ohlcv_only"
    if ohlcv_only:
        use_ohlcv_features = False
        use_market_features = False
        use_alpha158_features = False
        feature_set = "ohlcv_only"
    else:
        use_ohlcv_features = bool(preset[0] if args.use_ohlcv_features < 0 else args.use_ohlcv_features)
        use_market_features = bool(preset[1] if args.use_market_features < 0 else args.use_market_features)
        use_alpha158_features = bool(args.use_alpha158_features)
        feature_set = args.feature_set
    feature_args = {
        "feature_set": feature_set,
        "ohlcv_only": ohlcv_only,
        "use_ohlcv_features": use_ohlcv_features,
        "use_market_features": use_market_features,
        "ma_windows": [int(window) for window in args.ma_windows],
        "return_momentum_windows": [int(window) for window in args.return_momentum_windows],
        "market_momentum_window": int(args.market_momentum_window),
        "market_drawdown_window": int(args.market_drawdown_window),
        "volume_surprise_window": int(args.volume_surprise_window),
        "use_alpha158_features": use_alpha158_features,
        "alpha158_windows": [int(window) for window in args.alpha158_windows],
        "normalization_scope": args.normalization_scope,
        "input_norm_method": INPUT_NORM_METHOD,
        "input_norm_scope": INPUT_NORM_SCOPE,
        "raw_norm_layout": RAW_NORM_LAYOUT,
        "impute_missing": args.impute_missing,
        "source_layout": args.source_layout,
    }

    for market in args.markets:
        if args.clean_output:
            market_dir = os.path.join(args.output_root, f"{market}_ohlcv")
            if os.path.exists(market_dir):
                shutil.rmtree(market_dir)
        shapes = convert_market(args.predata_root, args.output_root, market, feature_args)
        print(f"{market} written to {os.path.join(args.output_root, market + '_ohlcv')}: {shapes}")
        feature_config_path = os.path.join(args.output_root, market + "_ohlcv", "feature_config.json")
        print(f"  feature_config: {feature_config_path}")
        if args.verify:
            for name, same_shape, array_equal, allclose, max_abs_diff in verify_arrays(
                args.output_root, args.reference_root, market
            ):
                status = "OK" if array_equal else ("CLOSE" if allclose else "DIFF")
                print(
                    f"  {status} {name}: same_shape={same_shape}, "
                    f"array_equal={array_equal}, allclose_1e-7={allclose}, "
                    f"max_abs_diff={max_abs_diff:.10g}"
                )


if __name__ == "__main__":
    main()
