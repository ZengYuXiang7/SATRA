"""Build a training-only Dynamic Time Warping stock relation prior.

The relation graph is computed exclusively from dates before
``market_data.valid_index``. A Euclidean prefilter limits expensive DTW
comparisons, after which the nearest DTW neighbors are symmetrized into a hard
attention-support matrix. Validation and test observations never enter prior
construction.
"""

from pathlib import Path

import numpy as np


def _standardize(series, mask):
    values = np.asarray(series, dtype=np.float32).copy()
    valid = np.asarray(mask, dtype=np.float32) > 0.5
    values[~valid] = np.nan
    med = np.nanmedian(values, axis=1, keepdims=True)
    values = np.where(np.isfinite(values), values, med)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    return (values - mean) / np.maximum(std, 1e-6)


def _downsample(series, length):
    stock_num, time_len = series.shape
    if time_len <= length:
        return series.astype(np.float32)
    edges = np.linspace(0, time_len, int(length) + 1).round().astype(np.int64)
    out = np.zeros((stock_num, int(length)), dtype=np.float32)
    for index in range(int(length)):
        start, end = int(edges[index]), int(edges[index + 1])
        if end <= start:
            end = min(time_len, start + 1)
        out[:, index] = series[:, start:end].mean(axis=1)
    return out


def _dtw_distance(x, y, band):
    length = int(x.shape[0])
    band = max(int(band), abs(length - length))
    inf = np.float32(1e20)
    prev = np.full(length + 1, inf, dtype=np.float32)
    curr = np.full(length + 1, inf, dtype=np.float32)
    prev[0] = 0.0
    for i in range(1, length + 1):
        curr.fill(inf)
        start = max(1, i - band)
        end = min(length, i + band)
        xi = x[i - 1]
        for j in range(start, end + 1):
            cost = (xi - y[j - 1]) ** 2
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return float(np.sqrt(prev[length] / max(length, 1)))


def build_dtw_stock_prior(
    market_data,
    cache_dir,
    *,
    topk=32,
    candidate_k=96,
    downsample=96,
    band=8,
    feature="raw_return",
    force=False,
):
    """Cache a symmetric top-k DTW neighborhood graph for one market.

    Args:
        market_data: Loaded market data with chronological split indices.
        cache_dir: Directory for the compressed relation artifact.
        topk: Number of final DTW neighbors selected per stock before
            symmetrization.
        candidate_k: Number of Euclidean candidates evaluated with DTW.
        downsample: Temporal length used for distance computation.
        band: Sakoe-Chiba half-band width for DTW.
        feature: Training-period series used to construct the relation.
        force: Recompute even when the exact cache key already exists.

    Returns:
        The path to an ``.npz`` file containing attention bias and distances.
    """

    # ``valid_index`` is the first validation date, hence this slice is
    # training-only and remains fixed throughout model selection and testing.
    train_end = int(market_data.valid_index)
    topk = int(topk)
    candidate_k = int(candidate_k)
    downsample = int(downsample)
    band = int(band)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = (
        f"stock_prior_dtw_{market_data.market}_train{train_end}_"
        f"{feature}_ds{downsample}_band{band}_cand{candidate_k}_top{topk}_hardmask.npz"
    )
    cache_path = cache_dir / cache_name
    if cache_path.exists() and not force:
        return str(cache_path)

    if feature == "raw_return":
        series = market_data.gt_data[:, :train_end]
    elif feature == "close":
        series = market_data.price_data[:, :train_end]
    elif feature in market_data.full_feature_names:
        feature_index = market_data.full_feature_names.index(feature)
        series = market_data.eod_data[:, :train_end, feature_index]
    else:
        raise ValueError(f"Unsupported stock_prior_feature={feature}")

    mask = market_data.mask_data[:, :train_end]
    trend = _downsample(_standardize(series, mask), downsample)
    stock_num = int(trend.shape[0])
    candidate_k = min(max(candidate_k, topk + 1), stock_num)
    topk = min(topk, stock_num - 1)

    dist = np.sum((trend[:, None, :] - trend[None, :, :]) ** 2, axis=-1)
    candidate_indices = np.argsort(dist, axis=1)[:, :candidate_k]

    prior = np.full((stock_num, stock_num), -1e4, dtype=np.float32)
    distance = np.full((stock_num, stock_num), np.inf, dtype=np.float32)
    np.fill_diagonal(prior, 0.0)
    np.fill_diagonal(distance, 0.0)

    for i in range(stock_num):
        scored = []
        for j in candidate_indices[i]:
            j = int(j)
            if j == i:
                continue
            d = _dtw_distance(trend[i], trend[j], band)
            scored.append((d, j))
        scored.sort(key=lambda item: item[0])
        if not scored:
            continue
        for d, j in scored[:topk]:
            prior[i, j] = 0.0
            distance[i, j] = d

    prior = np.maximum(prior, prior.T).astype(np.float32)
    distance = np.minimum(distance, distance.T).astype(np.float32)
    np.fill_diagonal(prior, 0.0)
    np.fill_diagonal(distance, 0.0)
    np.savez_compressed(
        cache_path,
        attention_bias=prior,
        distance=distance,
        feature=np.asarray(feature),
        train_end=np.asarray(train_end),
        topk=np.asarray(topk),
        candidate_k=np.asarray(candidate_k),
        downsample=np.asarray(downsample),
        band=np.asarray(band),
        relation_type=np.asarray("hard_dtw_neighborhood_mask"),
    )
    return str(cache_path)
