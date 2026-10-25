import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


TRADING_DAYS_PER_YEAR = 252.0
SR_ANNUALIZATION = float(np.sqrt(TRADING_DAYS_PER_YEAR))
MAPE_DENOM_EPS = 1e-8
MONITOR_METRIC = "RankIC"
MONITOR_MODE = "max"


def _safe_mean(values):
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    if values.size == 0 or not np.any(valid):
        return np.nan
    return float(np.mean(values[valid]))


def _safe_sharpe(values, scale=1.0):
    values = np.asarray(values, dtype=np.float64)
    valid_values = values[np.isfinite(values)]
    if valid_values.size == 0:
        return np.nan
    std = float(np.std(valid_values))
    mean = float(np.mean(valid_values))
    return float((mean / std) * scale) if std > 0.0 else np.nan


def _portfolio_return_stats(daily_returns):
    daily_returns = np.asarray(daily_returns, dtype=np.float64)
    daily_returns = daily_returns[np.isfinite(daily_returns)]
    if daily_returns.size == 0:
        return np.nan, np.nan, np.nan
    cumulative = float(np.prod(1.0 + daily_returns))
    irr_10 = cumulative - 1.0
    ar_10 = (
        float(cumulative ** (TRADING_DAYS_PER_YEAR / daily_returns.size) - 1.0)
        if cumulative > 0.0
        else np.nan
    )
    sr_10 = _safe_sharpe(daily_returns, scale=SR_ANNUALIZATION)
    return irr_10, ar_10, sr_10


def _normalize_portfolio_sizes(portfolio_sizes):
    """Validate requested portfolio sizes while preserving the legacy K=10 outputs."""
    if portfolio_sizes is None:
        return (10,)

    normalized = []
    for size in portfolio_sizes:
        if isinstance(size, bool):
            raise ValueError(f"portfolio size must be a positive integer, got {size!r}")
        try:
            parsed = int(size)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"portfolio size must be a positive integer, got {size!r}") from exc
        if parsed <= 0:
            raise ValueError(f"portfolio size must be positive, got {parsed}")
        normalized.append(parsed)

    if not normalized:
        raise ValueError("portfolio_sizes must contain at least one positive integer")
    # The existing public metrics contract always exposes IRR@10 and SR@10.
    return tuple(sorted(set(normalized + [10])))


def _as_stock_date_matrix(value, name, stock_num=None):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim == 2:
        if stock_num is None:
            return value
        if value.shape[0] == stock_num:
            return value
        if value.shape[1] == stock_num:
            return value.T
        raise ValueError(
            f"{name} has shape {value.shape}, but neither dimension matches stock_num={stock_num}"
        )
    if value.ndim == 3:
        if stock_num is not None and value.shape[1] != stock_num:
            raise ValueError(
                f"{name} expected [batch, stocks, dates], got {value.shape}; "
                f"stock dimension should be {stock_num}"
            )
        return value.transpose(1, 0, 2).reshape(value.shape[1], value.shape[0] * value.shape[2])
    raise ValueError(
        f"{name} must be [stocks,dates], [dates,stocks], or [batch,stocks,dates], got {value.shape}"
    )


def _masked_corr(pred_day, gt_day):
    if pred_day.size < 2:
        return np.nan
    if np.nanstd(pred_day) <= 0.0 or np.nanstd(gt_day) <= 0.0:
        return np.nan
    return float(pd.Series(pred_day).corr(pd.Series(gt_day)))


def _masked_kendall_tau(pred_day, gt_day):
    if pred_day.size < 2:
        return np.nan
    if np.nanstd(pred_day) <= 0.0 or np.nanstd(gt_day) <= 0.0:
        return np.nan
    result = kendalltau(pred_day, gt_day, nan_policy="omit")
    return float(result.statistic) if np.isfinite(result.statistic) else np.nan


def _masked_rank_ic(pred_day, gt_day):
    if pred_day.size < 2:
        return np.nan
    if np.nanstd(pred_day) <= 0.0 or np.nanstd(gt_day) <= 0.0:
        return np.nan
    result = spearmanr(pred_day, gt_day, nan_policy="omit")
    return float(result.statistic) if np.isfinite(result.statistic) else np.nan


def evaluate(prediction, ground_truth, mask, stock_num=None, report=False, portfolio_sizes=None):
    prediction = _as_stock_date_matrix(prediction, "prediction", stock_num=stock_num)
    ground_truth = _as_stock_date_matrix(ground_truth, "ground_truth", stock_num=stock_num)
    mask = _as_stock_date_matrix(mask, "mask", stock_num=stock_num)
    if ground_truth.shape != prediction.shape or prediction.shape != mask.shape:
        raise ValueError(
            "shape mis-match: "
            f"prediction={prediction.shape}, ground_truth={ground_truth.shape}, mask={mask.shape}"
        )

    valid_mask = (mask >= 0.5) & np.isfinite(prediction) & np.isfinite(ground_truth)
    valid_count = int(np.sum(valid_mask))
    if valid_count <= 0:
        raise ValueError("mask has no valid entries")

    diff = prediction[valid_mask] - ground_truth[valid_mask]
    mae = float(np.mean(np.abs(diff)))
    mape_valid = valid_mask & (np.abs(ground_truth) > MAPE_DENOM_EPS)
    mape = (
        float(np.mean(np.abs((prediction[mape_valid] - ground_truth[mape_valid]) / ground_truth[mape_valid])))
        if np.any(mape_valid)
        else np.nan
    )

    ic = []
    rank_ic = []
    kt = []
    portfolio_sizes = _normalize_portfolio_sizes(portfolio_sizes)
    prec_10 = []
    portfolio_returns = {size: [] for size in portfolio_sizes}

    for i in range(prediction.shape[1]):
        valid = valid_mask[:, i]
        if not np.any(valid):
            ic.append(np.nan)
            rank_ic.append(np.nan)
            kt.append(np.nan)
            prec_10.append(np.nan)
            for size in portfolio_sizes:
                portfolio_returns[size].append(np.nan)
            continue

        pred_day = prediction[valid, i]
        gt_day = ground_truth[valid, i]

        ic.append(_masked_corr(pred_day, gt_day))
        rank_ic.append(_masked_rank_ic(pred_day, gt_day))
        kt.append(_masked_kendall_tau(pred_day, gt_day))

        for size in portfolio_sizes:
            k = min(size, pred_day.shape[0])
            pred_topk_order = np.argsort(pred_day, kind="mergesort")[-k:][::-1]
            selected_returns = gt_day[pred_topk_order]
            if size == 10:
                prec_10.append(float(np.mean(selected_returns >= 0.0)))
            portfolio_returns[size].append(float(np.mean(selected_returns)))

    ic = np.asarray(ic, dtype=np.float64)
    mean_ic = _safe_mean(ic)
    std_ic = float(np.nanstd(ic))
    rank_ic = np.asarray(rank_ic, dtype=np.float64)
    mean_rank_ic = _safe_mean(rank_ic)
    std_rank_ic = float(np.nanstd(rank_ic))
    portfolio_stats = {
        size: _portfolio_return_stats(portfolio_returns[size]) for size in portfolio_sizes
    }
    irr_10, ar_10, sr_10 = portfolio_stats[10]

    metrics = {
        "mae": mae,
        "MAE": mae,
        "RankIC": mean_rank_ic,
        "IRR": irr_10,
        "IRR@10": irr_10,
        "SR@10": sr_10,
        "RankICIR": float(mean_rank_ic / std_rank_ic)
        if std_rank_ic > 0.0 and np.isfinite(mean_rank_ic)
        else np.nan,
        "AR@10": ar_10,
        "SR": sr_10,
        "KT": _safe_mean(kt),
        "mape": mape,
        "prec_10": _safe_mean(prec_10),
        "IC": mean_ic,
        "ICIR": float(mean_ic / std_ic) if std_ic > 0.0 and np.isfinite(mean_ic) else np.nan,
    }
    for size, (irr, annualized_return, sr) in portfolio_stats.items():
        metrics[f"IRR@{size}"] = irr
        metrics[f"AR@{size}"] = annualized_return
        metrics[f"SR@{size}"] = sr
    return metrics
