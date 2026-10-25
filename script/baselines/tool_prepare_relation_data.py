from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT_PATH = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT = DEFAULT_ROOT_PATH / "relations"
ESTIMATE_START_TRAIN_DATE = "2016-01-01"
ESTIMATE_LEAD_LAG_FILES = {
    "distance": ("lead_lag_edges_distance", "sp500_leadlag_{date}.csv"),
    "pearson": ("lead_lag_edges_pearson", "sp500_leadlag_pearson_{date}.csv"),
    "kendall": ("lead_lag_edges_kendall", "sp500_leadlag_kendall_{date}.csv"),
    "mutual_information": (
        "lead_lag_edges_mutual_information",
        "sp500_leadlag_mutual_information_{date}.csv",
    ),
}

DATASET_DEFAULTS = {
    "NASDAQ": {"valid_index": 871, "data_start_index": 0},
    "NYSE": {"valid_index": 871, "data_start_index": 0},
    "SP500": {"valid_index": 1127, "data_start_index": 915},
}

LOCAL_TICKER_CANDIDATES = {
    "NASDAQ": [
        PROJECT_ROOT / "external_data/rsr/NASDAQ/tickers.txt",
    ],
    "NYSE": [
        PROJECT_ROOT / "external_data/rsr/NYSE/tickers.txt",
    ],
}

SP500_DATA_CANDIDATES = [
    PROJECT_ROOT / "external_data/SP500/baseline_data_sp500.npy",
]

SP500_SECTOR_CANDIDATES = [
    PROJECT_ROOT / "external_data/SP500/sp500_ticker.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare static and proxy stock relation files for graph baselines."
    )
    parser.add_argument("--root-path", type=Path, default=DEFAULT_ROOT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", nargs="+", default=["NASDAQ", "NYSE", "SP500"])
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--rsr-relation-url",
        help="Optional URL for relation.tar.gz when no cached archive is available.",
    )
    parser.add_argument(
        "--sp500-source-base-url",
        help="Optional base URL for S&P500 lead-lag CSV files.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pickle_array(path: Path, dtype=np.float32) -> np.ndarray:
    with path.open("rb") as f:
        return np.asarray(pickle.load(f), dtype=dtype)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_tickers(path: Path, tickers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(tickers) + "\n", encoding="utf-8")


def cached_url_text(url: str | None, cache_path: Path) -> str:
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    if url is None:
        raise FileNotFoundError(
            f"{cache_path} does not exist; provide --sp500-source-base-url "
            "or place the file in the cache directory"
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    request = urllib.request.Request(url, headers={"User-Agent": "stock-baselines-relation-prep"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                text = response.read().decode("utf-8")
            cache_path.write_text(text, encoding="utf-8")
            return text
        except Exception as exc:  # Retry transient network failures.
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to download {url}") from last_error


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def safe_extract_tar(tar_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir.resolve()
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            target = (out_dir / member.name).resolve()
            if not str(target).startswith(str(base)):
                raise RuntimeError(f"unsafe tar member path: {member.name}")
        tar.extractall(out_dir, filter="data")


def ensure_rsr_relation(output: Path, skip_download: bool, relation_url: str | None) -> Path:
    downloads = output / "_downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    tar_path = downloads / "rsr_relation.tar.gz"
    extract_dir = downloads / "rsr_relation"
    relation_dir = extract_dir / "relation"

    if not tar_path.exists():
        if skip_download or relation_url is None:
            raise FileNotFoundError(
                f"{tar_path} does not exist; provide --rsr-relation-url "
                "or place the archive in the cache directory"
            )
        else:
            urllib.request.urlretrieve(relation_url, tar_path)

    if not relation_dir.exists():
        safe_extract_tar(tar_path, extract_dir)

    return relation_dir


def load_plain_tickers(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_sp500_tickers() -> tuple[list[str], str | None]:
    path = first_existing(SP500_DATA_CANDIDATES)
    if path is None:
        return [], None
    obj = np.load(path, allow_pickle=True).item()
    return list(obj.keys()), str(path)


def load_sp500_sector_map() -> tuple[dict[str, str], str | None]:
    path = first_existing(SP500_SECTOR_CANDIDATES)
    if path is None:
        return {}, None
    sector_map: dict[str, str] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            symbol = row.get("Symbol", "").strip()
            sector = row.get("Sector", "").strip()
            if symbol and sector:
                sector_map[symbol] = sector
    return sector_map, str(path)


def relation_to_adjacency(relation: np.ndarray) -> np.ndarray:
    adjacency = np.any(relation > 0, axis=2).astype(np.uint8)
    np.fill_diagonal(adjacency, 0)
    return adjacency


def diagonal_incidence(relation: np.ndarray, drop_self_column: bool = True) -> np.ndarray:
    n = relation.shape[0]
    diag = relation[np.arange(n), np.arange(n), :].astype(np.uint8, copy=False)
    if drop_self_column and diag.shape[1] > 1:
        diag = diag[:, :-1]
    keep = diag.sum(axis=0) > 0
    return diag[:, keep].astype(np.uint8, copy=False)


def incidence_to_hyperedge_index(incidence: np.ndarray) -> np.ndarray:
    node_idx, edge_idx = np.nonzero(incidence > 0)
    return np.stack([node_idx, edge_idx], axis=0).astype(np.int64)


def factor_prior_from_incidence(incidence: np.ndarray) -> tuple[np.ndarray, int]:
    prior = incidence.astype(np.float32, copy=False)
    zero_rows = prior.sum(axis=1) <= 0
    missing = int(zero_rows.sum())
    if missing:
        unknown = zero_rows.astype(np.float32)[:, None]
        prior = np.concatenate([prior, unknown], axis=1)
    return prior, missing


def save_relation_outputs(
    dataset_dir: Path,
    name: str,
    relation: np.ndarray,
    source: str,
    usage: list[str],
    manifest_files: dict,
    make_factor_prior: bool = False,
) -> None:
    relation = np.asarray(relation > 0, dtype=np.uint8)
    rel_path = dataset_dir / f"{name}_relation.npy"
    adj_path = dataset_dir / f"{name}_adjacency.npy"
    np.save(rel_path, relation)
    adjacency = relation_to_adjacency(relation)
    np.save(adj_path, adjacency)
    manifest_files[f"{name}_relation"] = {
        "path": str(rel_path),
        "shape": list(relation.shape),
        "dtype": str(relation.dtype),
        "nonzero": int(relation.sum()),
        "source": source,
        "usage": usage,
    }
    manifest_files[f"{name}_adjacency"] = {
        "path": str(adj_path),
        "shape": list(adjacency.shape),
        "dtype": str(adjacency.dtype),
        "nonzero": int(adjacency.sum()),
        "source": source,
        "usage": ["pairwise graph adjacency"],
    }

    neighbor_incidence = (adjacency | np.eye(adjacency.shape[0], dtype=np.uint8)).astype(np.uint8)
    neighbor_inc_path = dataset_dir / f"{name}_neighbor_incidence.npy"
    neighbor_edge_path = dataset_dir / f"{name}_neighbor_hyperedge_index.npy"
    neighbor_edge_index = incidence_to_hyperedge_index(neighbor_incidence)
    np.save(neighbor_inc_path, neighbor_incidence)
    np.save(neighbor_edge_path, neighbor_edge_index)
    manifest_files[f"{name}_neighbor_incidence"] = {
        "path": str(neighbor_inc_path),
        "shape": list(neighbor_incidence.shape),
        "dtype": str(neighbor_incidence.dtype),
        "nonzero": int(neighbor_incidence.sum()),
        "source": source,
        "usage": ["pairwise graph converted to neighborhood hyperedges"],
    }
    manifest_files[f"{name}_neighbor_hyperedge_index"] = {
        "path": str(neighbor_edge_path),
        "shape": list(neighbor_edge_index.shape),
        "dtype": "int64",
        "source": source,
        "usage": ["CI-STHPAN hyperedge_index from pairwise graph"],
    }

    incidence = diagonal_incidence(relation)
    if incidence.shape[1] > 0:
        inc_path = dataset_dir / f"{name}_incidence.npy"
        edge_path = dataset_dir / f"{name}_hyperedge_index.npy"
        np.save(inc_path, incidence)
        np.save(edge_path, incidence_to_hyperedge_index(incidence))
        manifest_files[f"{name}_incidence"] = {
            "path": str(inc_path),
            "shape": list(incidence.shape),
            "dtype": str(incidence.dtype),
            "nonzero": int(incidence.sum()),
            "source": source,
            "usage": ["hypergraph incidence"],
        }
        manifest_files[f"{name}_hyperedge_index"] = {
            "path": str(edge_path),
            "shape": list(np.load(edge_path, mmap_mode="r").shape),
            "dtype": "int64",
            "source": source,
            "usage": ["CI-STHPAN hyperedge_index"],
        }
        if make_factor_prior:
            prior, missing = factor_prior_from_incidence(incidence)
            prior_path = dataset_dir / "factor_prior_beta.npy"
            np.save(prior_path, prior)
            manifest_files["factor_prior_beta"] = {
                "path": str(prior_path),
                "shape": list(prior.shape),
                "dtype": "float32",
                "source": source,
                "usage": ["FactorGCL prior_beta"],
                "zero_row_unknown_column_added": missing,
            }


def prepare_rsr_dataset(
    dataset: str,
    dataset_dir: Path,
    relation_dir: Path,
    expected_stocks: int,
    manifest_files: dict,
) -> None:
    for rel_name, subdir in [("rsr_industry", "sector_industry"), ("rsr_wiki", "wikidata")]:
        src = relation_dir / subdir / f"{dataset}_{rel_name.split('_')[1]}_relation.npy"
        if not src.exists():
            continue
        relation = np.load(src, mmap_mode="r")
        if relation.shape[0] != expected_stocks or relation.shape[1] != expected_stocks:
            raise ValueError(f"{src} shape {relation.shape} does not match {expected_stocks} stocks")
        save_relation_outputs(
            dataset_dir=dataset_dir,
            name=rel_name,
            relation=relation,
            source=f"RSR official relation.tar.gz::{subdir}/{src.name}",
            usage=["RSR", "HyperStockGAT"],
            manifest_files=manifest_files,
            make_factor_prior=(rel_name == "rsr_industry"),
        )


def build_sp500_sector_relation(
    tickers: list[str],
    sector_map: dict[str, str],
) -> tuple[np.ndarray, list[str]]:
    sectors = sorted({sector_map[ticker] for ticker in tickers if ticker in sector_map})
    sector_index = {sector: i for i, sector in enumerate(sectors)}
    n = len(tickers)
    relation = np.zeros((n, n, len(sectors) + 1), dtype=np.uint8)
    for i, ticker_i in enumerate(tickers):
        sector = sector_map.get(ticker_i)
        if sector is None:
            relation[i, i, -1] = 1
            continue
        rel_idx = sector_index[sector]
        for j, ticker_j in enumerate(tickers):
            if sector_map.get(ticker_j) == sector:
                relation[i, j, rel_idx] = 1
        relation[i, i, -1] = 1
    return relation, sectors


def leadlag_relation_from_csv(
    url: str | None,
    cache_path: Path,
    tickers: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    text = cached_url_text(url, cache_path)
    assignments: dict[str, int] = {}
    for row in csv.reader(text.splitlines()):
        if len(row) < 2 or not row[0].strip():
            continue
        assignments[row[0].strip()] = int(row[1])

    missing = [ticker for ticker in tickers if ticker not in assignments]
    if missing:
        raise ValueError(f"ESTIMATE lead-lag file missing tickers: {missing[:10]}")

    labels = np.asarray([assignments[ticker] for ticker in tickers], dtype=np.int64)
    num_clusters = int(labels.max()) + 1
    incidence = np.zeros((len(tickers), num_clusters), dtype=np.uint8)
    incidence[np.arange(len(tickers)), labels] = 1

    relation = np.zeros((len(tickers), len(tickers), num_clusters + 1), dtype=np.uint8)
    for cluster_idx in range(num_clusters):
        members = incidence[:, cluster_idx].astype(bool)
        relation[np.ix_(members, members, [cluster_idx])] = 1
    relation[np.arange(len(tickers)), np.arange(len(tickers)), -1] = 1
    return relation, incidence


def prepare_sp500_estimate_leadlag(
    dataset_dir: Path,
    tickers: list[str],
    manifest_files: dict,
    source_base_url: str | None,
) -> None:
    for method, (directory, pattern) in ESTIMATE_LEAD_LAG_FILES.items():
        filename = pattern.format(date=ESTIMATE_START_TRAIN_DATE)
        url = (
            f"{source_base_url.rstrip('/')}/{directory}/{filename}"
            if source_base_url
            else None
        )
        cache_path = dataset_dir.parent / "_downloads" / "estimate_sp500" / directory / filename
        relation, incidence = leadlag_relation_from_csv(url, cache_path, tickers)
        source_label = "user-provided URL" if url else "local cache"
        save_relation_outputs(
            dataset_dir=dataset_dir,
            name=f"estimate_leadlag_{method}",
            relation=relation,
            source=source_label,
            usage=[
                "ESTIMATE S&P500 lead-lag hypergraph",
                "SP500 static relation fallback",
                "CI-STHPAN hyperedge_index",
            ],
            manifest_files=manifest_files,
            make_factor_prior=False,
        )
        inc_path = dataset_dir / f"estimate_leadlag_{method}_original_incidence.npy"
        edge_path = dataset_dir / f"estimate_leadlag_{method}_original_hyperedge_index.npy"
        np.save(inc_path, incidence)
        edge_index = incidence_to_hyperedge_index(incidence)
        np.save(edge_path, edge_index)
        manifest_files[f"estimate_leadlag_{method}_original_incidence"] = {
            "path": str(inc_path),
            "shape": list(incidence.shape),
            "dtype": str(incidence.dtype),
            "nonzero": int(incidence.sum()),
            "source": source_label,
            "usage": ["ESTIMATE original cluster incidence"],
        }
        manifest_files[f"estimate_leadlag_{method}_original_hyperedge_index"] = {
            "path": str(edge_path),
            "shape": list(edge_index.shape),
            "dtype": "int64",
            "source": source_label,
            "usage": ["CI-STHPAN hyperedge_index from ESTIMATE cluster incidence"],
        }


def save_sector_metadata(dataset_dir: Path, tickers: list[str], sector_map: dict[str, str]) -> None:
    path = dataset_dir / "sp500_sector_metadata.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "ticker", "sector"])
        writer.writeheader()
        for idx, ticker in enumerate(tickers):
            writer.writerow({"index": idx, "ticker": ticker, "sector": sector_map.get(ticker, "")})


def build_correlation_graph(
    root_path: Path,
    dataset: str,
    dataset_dir: Path,
    topk: int,
    manifest_files: dict,
) -> None:
    defaults = DATASET_DEFAULTS[dataset]
    market_dir = root_path / f"{dataset}_ohlcv"
    gt = load_pickle_array(market_dir / "gt_data.pkl", dtype=np.float32)
    mask = load_pickle_array(market_dir / "mask_data.pkl", dtype=np.float32) > 0.5

    start = int(defaults["data_start_index"])
    end = start + int(defaults["valid_index"])
    returns = gt[:, start:end].astype(np.float32, copy=False)
    valid = mask[:, start:end]

    returns = np.where(valid, returns, np.nan)
    mean = np.nanmean(returns, axis=1, keepdims=True)
    std = np.nanstd(returns, axis=1, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    z = np.where(valid, (returns - mean) / std, 0.0).astype(np.float32)
    counts = valid.astype(np.float32).dot(valid.astype(np.float32).T)
    corr = z.dot(z.T) / np.maximum(counts - 1.0, 1.0)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.clip(corr, -1.0, 1.0).astype(np.float32)
    np.fill_diagonal(corr, -np.inf)

    n = corr.shape[0]
    k = min(topk, max(1, n - 1))
    adjacency = np.zeros((n, n), dtype=np.uint8)
    top_idx = np.argpartition(-corr, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(n)[:, None]
    keep = corr[rows, top_idx] > 0
    row_idx = np.repeat(np.arange(n), k)[keep.reshape(-1)]
    col_idx = top_idx.reshape(-1)[keep.reshape(-1)]
    adjacency[row_idx, col_idx] = 1
    adjacency = np.maximum(adjacency, adjacency.T).astype(np.uint8)
    np.fill_diagonal(adjacency, 0)

    relation = adjacency[:, :, None]
    incidence = (adjacency | np.eye(n, dtype=np.uint8)).astype(np.uint8)
    attention_bias = np.where(incidence > 0, 0.0, -10000.0).astype(np.float32)

    corr_path = dataset_dir / "correlation_train_similarity.npy"
    adj_path = dataset_dir / f"correlation_top{topk}_adjacency.npy"
    rel_path = dataset_dir / f"correlation_top{topk}_relation.npy"
    inc_path = dataset_dir / f"correlation_top{topk}_incidence.npy"
    edge_path = dataset_dir / f"correlation_top{topk}_hyperedge_index.npy"
    bias_path = dataset_dir / f"correlation_top{topk}_attention_bias.npy"

    np.save(corr_path, np.nan_to_num(corr, neginf=0.0))
    np.save(adj_path, adjacency)
    np.save(rel_path, relation)
    np.save(inc_path, incidence)
    np.save(edge_path, incidence_to_hyperedge_index(incidence))
    np.save(bias_path, attention_bias)

    source = (
        f"{dataset}_ohlcv/gt_data.pkl training segment "
        f"[{start}:{end}], positive Pearson top-{topk}, symmetrized"
    )
    for key, path, usage in [
        ("correlation_train_similarity", corr_path, ["diagnostic similarity matrix"]),
        (f"correlation_top{topk}_adjacency", adj_path, ["fallback pairwise graph"]),
        (f"correlation_top{topk}_relation", rel_path, ["fallback RSR/HyperStockGAT relation"]),
        (f"correlation_top{topk}_incidence", inc_path, ["FactorGCL proxy prior_beta", "hypergraph incidence"]),
        (f"correlation_top{topk}_hyperedge_index", edge_path, ["CI-STHPAN hyperedge_index"]),
        (f"correlation_top{topk}_attention_bias", bias_path, ["attention mask/bias"]),
    ]:
        arr = np.load(path, mmap_mode="r")
        item = {
            "path": str(path),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "source": source,
            "usage": usage,
        }
        if arr.dtype != np.float32 or "bias" not in key:
            item["nonzero"] = int(np.count_nonzero(arr))
        manifest_files[key] = item


def prepare_dataset(args: argparse.Namespace, dataset: str, relation_dir: Path, manifest: dict) -> None:
    if dataset not in DATASET_DEFAULTS:
        raise ValueError(f"unsupported dataset: {dataset}")

    market_dir = args.root_path / f"{dataset}_ohlcv"
    if not market_dir.exists():
        raise FileNotFoundError(market_dir)
    price_data = load_pickle_array(market_dir / "price_data.pkl", dtype=np.float32)
    num_stocks = int(price_data.shape[0])

    dataset_dir = args.output / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest_files: dict = {}
    ticker_info: dict = {"available": False, "source": None, "count": 0}

    if dataset in {"NASDAQ", "NYSE"}:
        ticker_path = first_existing(LOCAL_TICKER_CANDIDATES[dataset])
        if ticker_path is not None:
            tickers = load_plain_tickers(ticker_path)
            if len(tickers) != num_stocks:
                raise ValueError(f"{ticker_path} has {len(tickers)} tickers, expected {num_stocks}")
            write_tickers(dataset_dir / "tickers.txt", tickers)
            ticker_info = {"available": True, "source": str(ticker_path), "count": len(tickers)}
        prepare_rsr_dataset(dataset, dataset_dir, relation_dir, num_stocks, manifest_files)

    if dataset == "SP500":
        tickers, ticker_source = load_sp500_tickers()
        sector_map, sector_source = load_sp500_sector_map()
        if tickers:
            if len(tickers) != num_stocks:
                raise ValueError(f"SP500 ticker count {len(tickers)} != current stocks {num_stocks}")
            write_tickers(dataset_dir / "tickers.txt", tickers)
            ticker_info = {"available": True, "source": ticker_source, "count": len(tickers)}
        if tickers and sector_map:
            missing = [ticker for ticker in tickers if ticker not in sector_map]
            if missing:
                raise ValueError(f"SP500 tickers missing sector metadata: {missing[:10]}")
            save_sector_metadata(dataset_dir, tickers, sector_map)
            relation, sectors = build_sp500_sector_relation(tickers, sector_map)
            save_relation_outputs(
                dataset_dir=dataset_dir,
                name="sp500_sector",
                relation=relation,
                source=f"{sector_source}, aligned to {ticker_source}",
                usage=["SP500 static sector fallback", "RSR/HyperStockGAT fallback"],
                manifest_files=manifest_files,
                make_factor_prior=True,
            )
            write_json(dataset_dir / "sp500_sector_names.json", {"sectors": sectors})
            prepare_sp500_estimate_leadlag(
                dataset_dir,
                tickers,
                manifest_files,
                args.sp500_source_base_url,
            )

    build_correlation_graph(args.root_path, dataset, dataset_dir, args.topk, manifest_files)

    manifest["datasets"][dataset] = {
        "num_stocks": num_stocks,
        "ticker_alignment": ticker_info,
        "files": manifest_files,
    }


def main() -> None:
    args = parse_args()
    args.root_path = args.root_path.resolve()
    args.output = args.output.resolve()

    if args.output.exists() and args.force:
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    relation_dir = ensure_rsr_relation(
        args.output,
        args.skip_download,
        args.rsr_relation_url,
    )
    manifest = {
        "created_by": "script/baselines/tool_prepare_relation_data.py",
        "root_path": str(args.root_path),
        "output": str(args.output),
        "rsr_relation_source": (
            "user-provided URL" if args.rsr_relation_url else "local cache"
        ),
        "rsr_relation_sha256": sha256_file(args.output / "_downloads/rsr_relation.tar.gz"),
        "notes": [
            "NASDAQ/NYSE RSR industry/wiki relations come from the public RSR relation archive.",
            "HyperStockGAT uses the same NASDAQ/NYSE relations as the closest source-compatible graph data.",
            "SP500 uses sector metadata plus lead-lag cluster hypergraphs from a user-supplied public source or local cache.",
            "SP500 has no official RSR/HyperStockGAT relation file here; ESTIMATE sector/lead-lag and correlation files are reproducible alternatives.",
            "Correlation graphs are generated only from each dataset's training segment to avoid validation/test leakage.",
        ],
        "datasets": {},
    }

    for dataset in args.datasets:
        prepare_dataset(args, dataset, relation_dir, manifest)

    write_json(args.output / "manifest.json", manifest)
    extracted_rsr_dir = args.output / "_downloads" / "rsr_relation"
    if extracted_rsr_dir.exists():
        shutil.rmtree(extracted_rsr_dir)
    print(f"relation data ready: {args.output}")
    for dataset, info in manifest["datasets"].items():
        print(f"{dataset}: {info['num_stocks']} stocks, {len(info['files'])} files")


if __name__ == "__main__":
    main()
