import numpy as np
import pandas as pd


EPS = 1e-12

FACTORVAE_ALPHA20_FEATURE_NAMES = [
    "RESI5",
    "WVMA5",
    "RSQR5",
    "KLEN",
    "RSQR10",
    "CORR5",
    "CORD5",
    "CORR10",
    "ROC60",
    "RESI10",
    "VSTD5",
    "RSQR60",
    "CORR60",
    "WVMA60",
    "STD5",
    "RSQR20",
    "CORD60",
    "CORD10",
    "CORR20",
    "KLOW",
]


def _safe_div(numerator, denominator, default=0.0):
    numerator, denominator = np.broadcast_arrays(
        np.asarray(numerator, dtype=np.float64),
        np.asarray(denominator, dtype=np.float64),
    )
    out = np.full(numerator.shape, default, dtype=np.float64)
    return np.divide(numerator, denominator, out=out, where=np.abs(denominator) > EPS)


def _clean_feature(array, clip=5.0):
    array = np.nan_to_num(array, nan=0.0, posinf=clip, neginf=-clip)
    return np.clip(array, -clip, clip).astype(np.float32)


def _feature_index(feature_names, name, fallback=None):
    lower_to_index = {str(feature).lower(): index for index, feature in enumerate(feature_names)}
    return lower_to_index.get(name.lower(), fallback)


def _past_value(values):
    prev = np.empty_like(values, dtype=np.float64)
    prev[:, 0] = values[:, 0]
    prev[:, 1:] = values[:, :-1]
    return prev


def _lag_value(values, periods):
    periods = int(periods)
    lagged = np.empty_like(values, dtype=np.float64)
    lagged[:, :periods] = values[:, :1]
    lagged[:, periods:] = values[:, :-periods]
    return lagged


def _rolling_mean_current(values, window):
    return pd.DataFrame(values.T).rolling(window=window, min_periods=1).mean().to_numpy(dtype=np.float64).T


def _rolling_std_current(values, window):
    return pd.DataFrame(values.T).rolling(window=window, min_periods=1).std().to_numpy(dtype=np.float64).T


def _rolling_sum_current(values, window):
    return pd.DataFrame(values.T).rolling(window=window, min_periods=1).sum().to_numpy(dtype=np.float64).T


def _rolling_corr_current(left, right, window):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    sum_x = _rolling_sum_current(left, window)
    sum_y = _rolling_sum_current(right, window)
    sum_x2 = _rolling_sum_current(left * left, window)
    sum_y2 = _rolling_sum_current(right * right, window)
    sum_xy = _rolling_sum_current(left * right, window)
    count = np.minimum(np.arange(left.shape[1], dtype=np.float64) + 1.0, float(window))[None, :]
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
    fitted_current = intercept + slope * n
    residual = values - fitted_current
    denom_y = n * sum_y2 - sum_y * sum_y
    rsquare = _safe_div(numerator * numerator, denom_x * denom_y)
    rsquare = np.where(n >= 2.0, rsquare, 0.0)
    residual = np.where(n >= 2.0, residual, 0.0)
    return (
        np.broadcast_to(rsquare, (stock_num, trade_days)).astype(np.float64, copy=False),
        np.broadcast_to(residual, (stock_num, trade_days)).astype(np.float64, copy=False),
    )


def _fit_price_reconstruction(eod_close, price_data, mask_data):
    finite = np.isfinite(eod_close) & np.isfinite(price_data)
    if mask_data is not None:
        finite &= np.asarray(mask_data).astype(bool)
    x = eod_close[finite].astype(np.float64, copy=False)
    y = price_data[finite].astype(np.float64, copy=False)
    if x.size < 2:
        return None
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_var = float(np.mean((x - x_mean) ** 2))
    if x_var <= EPS:
        return None
    scale = float(np.mean((x - x_mean) * (y - y_mean)) / x_var)
    center = float(y_mean - scale * x_mean)
    if not np.isfinite(scale) or scale <= EPS or not np.isfinite(center):
        return None
    pred = center + scale * eod_close
    rel_mae = float(np.nanmean(np.abs(pred[finite] - price_data[finite])) / (np.nanmean(np.abs(price_data[finite])) + EPS))
    return {"center": center, "scale": scale, "close_rel_mae": rel_mae}


def _extract_ohlcv(eod_data, price_data, feature_names, mask_data):
    open_idx = _feature_index(feature_names, "open", 0)
    high_idx = _feature_index(feature_names, "high", 1)
    low_idx = _feature_index(feature_names, "low", 2)
    volume_idx = _feature_index(feature_names, "volume", 3)
    close_idx = _feature_index(feature_names, "close", 4)

    eod_data = np.asarray(eod_data, dtype=np.float64)
    price_data = np.asarray(price_data, dtype=np.float64)
    fit = _fit_price_reconstruction(eod_data[:, :, close_idx], price_data, mask_data)
    if fit is None:
        open_ = eod_data[:, :, open_idx]
        high = eod_data[:, :, high_idx]
        low = eod_data[:, :, low_idx]
        close = eod_data[:, :, close_idx]
        price_meta = {"method": "eod_price_columns", "reason": "linear_close_fit_unavailable"}
    else:
        open_ = fit["center"] + fit["scale"] * eod_data[:, :, open_idx]
        high = fit["center"] + fit["scale"] * eod_data[:, :, high_idx]
        low = fit["center"] + fit["scale"] * eod_data[:, :, low_idx]
        close = price_data
        high = np.maximum.reduce([high, open_, close])
        low = np.minimum.reduce([low, open_, close])
        price_meta = {"method": "linear_reconstruction_from_price_data", **fit}

    open_ = np.maximum(open_, EPS)
    high = np.maximum(high, EPS)
    low = np.maximum(low, EPS)
    close = np.maximum(close, EPS)

    volume_source = eod_data[:, :, volume_idx]
    if np.nanmin(volume_source) <= 0.0:
        volume = np.exp(np.clip(volume_source, -20.0, 20.0))
        volume_meta = {"method": "exp_standardized_volume_proxy"}
    else:
        volume = volume_source
        volume_meta = {"method": "as_is_positive_volume"}
    volume = np.maximum(volume, EPS)
    return open_, high, low, close, volume, {"price": price_meta, "volume": volume_meta}


def _standardize_features(features, mask_data, train_start, train_end):
    features = np.asarray(features, dtype=np.float64)
    mask = np.asarray(mask_data).astype(bool) if mask_data is not None else np.ones(features.shape[:2], dtype=bool)
    train_start = max(0, int(train_start))
    train_end = min(int(train_end), features.shape[1])
    normalized = np.empty_like(features, dtype=np.float32)
    means = []
    scales = []
    for feature_idx in range(features.shape[2]):
        train_values = features[:, train_start:train_end, feature_idx]
        train_mask = mask[:, train_start:train_end] & np.isfinite(train_values)
        values = train_values[train_mask]
        if values.size == 0:
            mean = 0.0
            scale = 1.0
        else:
            mean = float(np.mean(values))
            scale = float(np.std(values))
            if not np.isfinite(scale) or scale <= EPS:
                scale = 1.0
        normalized[:, :, feature_idx] = _clean_feature((features[:, :, feature_idx] - mean) / scale)
        means.append(mean)
        scales.append(scale)
    return normalized, {
        "method": "standard",
        "fit": "train_only",
        "train_start": train_start,
        "train_end": train_end,
        "clip": 5.0,
        "mean": means,
        "scale": scales,
    }


def build_factorvae_alpha20_features(eod_data, price_data, feature_names, mask_data, train_start, train_end):
    open_, high, low, close, volume, source_meta = _extract_ohlcv(eod_data, price_data, feature_names, mask_data)
    high_low = high - low
    close_ref1 = _past_value(close)
    volume_ref1 = _past_value(volume)
    close_ratio = _safe_div(close, close_ref1, default=1.0)
    volume_ratio = _safe_div(volume, volume_ref1, default=1.0)
    weighted_abs_return_volume = np.abs(close_ratio - 1.0) * volume
    corr_volume = np.log(volume + 1.0)
    cord_volume = np.log(volume_ratio + 1.0)

    raw = {
        "KLEN": _safe_div(high_low, open_),
        "KLOW": _safe_div(np.minimum(open_, close) - low, open_),
        "ROC60": _safe_div(_lag_value(close, 60), close, default=1.0),
    }
    for window in (5, 10, 20, 60):
        rsquare, residual = _rolling_linear_stats(close, window)
        raw[f"RSQR{window}"] = rsquare
        raw[f"RESI{window}"] = _safe_div(residual, close)
        raw[f"CORR{window}"] = _rolling_corr_current(close, corr_volume, window)
        raw[f"CORD{window}"] = _rolling_corr_current(close_ratio, cord_volume, window)
        if window in (5,):
            raw[f"STD{window}"] = _safe_div(_rolling_std_current(close, window), close)
            raw[f"VSTD{window}"] = _safe_div(_rolling_std_current(volume, window), volume)
        if window in (5, 60):
            wvma_mean = _rolling_mean_current(weighted_abs_return_volume, window)
            raw[f"WVMA{window}"] = _safe_div(_rolling_std_current(weighted_abs_return_volume, window), wvma_mean)

    selected = np.stack(
        [_clean_feature(raw[name], clip=50.0) for name in FACTORVAE_ALPHA20_FEATURE_NAMES],
        axis=-1,
    )
    normalized, norm_meta = _standardize_features(selected, mask_data, train_start, train_end)
    meta = {
        "feature_set": "qlib_selected_alpha20_from_ohlcv",
        "feature_names": list(FACTORVAE_ALPHA20_FEATURE_NAMES),
        "source": source_meta,
        "normalization": norm_meta,
    }
    return normalized.astype(np.float32, copy=False), meta
