"""Download public stock datasets and build SATRA-ready tensors.

The repository intentionally excludes raw and processed market data. This
entry point downloads the public source repositories into ignored directories,
creates the input layout expected by the project preprocessor, and writes
model-ready arrays under ``data/processed``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed"

TRSR_REPOSITORY_ID = 157677083
ESTIMATE_REPOSITORY_ID = 523661816
TRSR_DIRNAME = "Temporal_Relational_Stock_Ranking"
ESTIMATE_DIRNAME = "estimate"

NASDAQ_PRIOR_NAME = (
    "dtw_binary_imputed_inputstandard_NASDAQ_"
    "train0-871_close_ds64_band6_cand48_top32.npz"
)
NASDAQ_PRIOR_SOURCE = PROJECT_ROOT / "data" / "priors" / NASDAQ_PRIOR_NAME

MARKETS = ("NASDAQ", "NYSE", "SP500")
TRSR_METADATA = (
    "NASDAQ_aver_line_dates.csv",
    "NASDAQ_tickers_qualify_dr-0.98_min-5_smooth.csv",
    "NYSE_aver_line_dates.csv",
    "NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv",
)
ESTIMATE_LEAD_LAG_FILES = (
    ("lead_lag_edges_distance", "sp500_leadlag_2016-01-01.csv"),
    ("lead_lag_edges_pearson", "sp500_leadlag_pearson_2016-01-01.csv"),
    ("lead_lag_edges_kendall", "sp500_leadlag_kendall_2016-01-01.csv"),
    (
        "lead_lag_edges_mutual_information",
        "sp500_leadlag_mutual_information_2016-01-01.csv",
    ),
)


def absolute_project_path(path: Path) -> Path:
    """Resolve a command-line path relative to the repository root."""

    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def run_checked(command: list[str], *, env: dict[str, str] | None = None) -> None:
    """Run one external command and stop immediately when it fails."""

    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def resolve_github_clone_url(repository_id: int, expected_name: str) -> str:
    """Resolve a public source without embedding a personal account name."""

    api_url = f"https://api.github.com/repositories/{int(repository_id)}"
    request = urllib.request.Request(
        api_url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "satra-data-preparation"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    if payload.get("name") != expected_name:
        raise RuntimeError(
            f"GitHub repository {repository_id} resolved to an unexpected name: "
            f"{payload.get('name')!r}"
        )
    clone_url = payload.get("clone_url")
    if not isinstance(clone_url, str) or not clone_url.startswith("https://github.com/"):
        raise RuntimeError(f"GitHub repository {repository_id} has no valid HTTPS clone URL")
    return clone_url


def clone_public_source(repository_id: int, expected_name: str, target: Path) -> None:
    """Create a shallow source clone without overwriting an existing directory."""

    if (target / ".git").is_dir():
        print(f"source already present: {target}")
        return
    if target.exists():
        raise FileExistsError(
            f"{target} exists but is not a Git checkout; move it aside or choose another --raw-root"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    clone_url = resolve_github_clone_url(repository_id, expected_name)
    run_checked(["git", "clone", "--depth", "1", clone_url, str(target)])


def require_file(path: Path) -> Path:
    """Return an existing file or raise a path-specific error."""

    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def require_directory(path: Path) -> Path:
    """Return an existing directory or raise a path-specific error."""

    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def replace_generated_symlink(link: Path, target: Path) -> None:
    """Create a relative symlink while refusing to replace user-managed files."""

    target = target.resolve()
    if not target.exists():
        raise FileNotFoundError(target)
    if link.is_symlink():
        if link.resolve() == target:
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"refusing to replace non-symlink input: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(os.path.relpath(target, start=link.parent.resolve()))


def source_paths(raw_root: Path) -> tuple[Path, Path]:
    """Return the two expected public source checkout paths."""

    source_root = raw_root / "sources"
    return source_root / TRSR_DIRNAME, source_root / ESTIMATE_DIRNAME


def download_sources(raw_root: Path, markets: list[str]) -> None:
    """Download only the repositories required by the selected markets."""

    trsr_root, estimate_root = source_paths(raw_root)
    if set(markets) & {"NASDAQ", "NYSE"}:
        clone_public_source(TRSR_REPOSITORY_ID, TRSR_DIRNAME, trsr_root)
    if "SP500" in markets:
        clone_public_source(ESTIMATE_REPOSITORY_ID, ESTIMATE_DIRNAME, estimate_root)


def stage_preprocessor_inputs(raw_root: Path, markets: list[str]) -> Path:
    """Build a small symlink layout consumed by ``generate_data.py``."""

    trsr_root, estimate_root = source_paths(raw_root)
    input_root = raw_root / "input"

    if set(markets) & {"NASDAQ", "NYSE"}:
        trsr_data = require_directory(trsr_root / "data")
        replace_generated_symlink(
            input_root / "google_finance",
            require_directory(trsr_data / "google_finance"),
        )
        for filename in TRSR_METADATA:
            replace_generated_symlink(
                input_root / filename,
                require_file(trsr_data / filename),
            )

    if "SP500" in markets:
        sp500_root = require_directory(estimate_root / "src" / "data" / "US" / "sp500")
        replace_generated_symlink(
            input_root / "baseline_data_sp500.npy",
            require_file(sp500_root / "baseline_data_sp500.npy"),
        )

    return input_root


def install_nasdaq_prior(output_root: Path) -> None:
    """Copy the single bundled NASDAQ DTW graph into the generated data tree."""

    source = require_file(NASDAQ_PRIOR_SOURCE)
    target = output_root / "NASDAQ_ohlcv" / "priors" / NASDAQ_PRIOR_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    print(f"NASDAQ DTW prior installed: {target}")


def preprocess_markets(
    raw_root: Path,
    output_root: Path,
    markets: list[str],
    *,
    clean_output: bool,
    include_alpha158: bool,
) -> None:
    """Generate leakage-aware features and chronological targets."""

    input_root = stage_preprocessor_inputs(raw_root, markets)
    environment = os.environ.copy()
    old_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(SRC_ROOT)
        if not old_pythonpath
        else os.pathsep.join((str(SRC_ROOT), old_pythonpath))
    )

    command = [
        sys.executable,
        "-m",
        "satra.data_provider.generate_data",
        "--predata-root",
        str(input_root),
        "--output-root",
        str(output_root),
        "--source_layout",
        "relational_csv",
        "--markets",
        *markets,
        "--feature_set",
        "stock_market_aug",
        "--normalization_scope",
        "stock",
        "--impute_missing",
        "linear",
        "--use_alpha158_features",
        "1" if include_alpha158 else "0",
    ]
    if clean_output:
        command.append("--clean-output")
    run_checked(command, env=environment)

    if "NASDAQ" in markets:
        install_nasdaq_prior(output_root)


def stage_baseline_sources(raw_root: Path, output_root: Path) -> None:
    """Expose public relation inputs to the baseline relation builder."""

    trsr_root, estimate_root = source_paths(raw_root)
    trsr_data = require_directory(trsr_root / "data")
    sp500_root = require_directory(estimate_root / "src" / "data" / "US" / "sp500")

    for market in ("NASDAQ", "NYSE"):
        replace_generated_symlink(
            PROJECT_ROOT / "external_data" / "rsr" / market / "tickers.txt",
            require_file(
                trsr_data
                / f"{market}_tickers_qualify_dr-0.98_min-5_smooth.csv"
            ),
        )
    replace_generated_symlink(
        PROJECT_ROOT / "external_data" / "SP500" / "baseline_data_sp500.npy",
        require_file(sp500_root / "baseline_data_sp500.npy"),
    )
    replace_generated_symlink(
        PROJECT_ROOT / "external_data" / "SP500" / "sp500_ticker.csv",
        require_file(sp500_root / "sp500_ticker.csv"),
    )

    downloads = output_root / "relations" / "_downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        require_file(trsr_data / "relation.tar.gz"),
        downloads / "rsr_relation.tar.gz",
    )
    for directory, filename in ESTIMATE_LEAD_LAG_FILES:
        source = require_file(sp500_root / directory / filename)
        target = downloads / "estimate_sp500" / directory / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def prepare_baseline_relations(
    raw_root: Path,
    output_root: Path,
    markets: list[str],
) -> None:
    """Generate optional baseline relation arrays from the downloaded sources."""

    missing_sources = []
    trsr_root, estimate_root = source_paths(raw_root)
    if set(markets) & {"NASDAQ", "NYSE"} and not trsr_root.is_dir():
        missing_sources.append(str(trsr_root))
    if "SP500" in markets and not estimate_root.is_dir():
        missing_sources.append(str(estimate_root))
    if missing_sources:
        raise FileNotFoundError(
            "baseline relation generation needs both public source trees; "
            f"missing: {', '.join(missing_sources)}"
        )

    stage_baseline_sources(raw_root, output_root)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "script" / "baselines" / "tool_prepare_relation_data.py"),
        "--root-path",
        str(output_root),
        "--output",
        str(output_root / "relations"),
        "--datasets",
        *markets,
        "--skip-download",
    ]
    run_checked(command)


def build_parser() -> argparse.ArgumentParser:
    """Define the public data-preparation command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Download public market sources and generate the data/processed tree. "
            "Raw and processed files remain outside the submission package."
        )
    )
    parser.add_argument(
        "action",
        nargs="?",
        choices=("all", "download", "preprocess"),
        default="all",
        help="all downloads and preprocesses; preprocess reuses existing source checkouts",
    )
    parser.add_argument(
        "--markets",
        nargs="+",
        choices=MARKETS,
        default=list(MARKETS),
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="replace the selected generated market directories before preprocessing",
    )
    parser.add_argument(
        "--alpha158",
        action="store_true",
        help="append the 157 OHLCV-derived Alpha158-compatible factors",
    )
    parser.add_argument(
        "--baseline-relations",
        action="store_true",
        help="also generate the larger relation files required by graph baselines",
    )
    return parser


def main() -> None:
    """Execute the selected download and preprocessing stages."""

    args = build_parser().parse_args()
    markets = list(dict.fromkeys(args.markets))
    raw_root = absolute_project_path(args.raw_root)
    output_root = absolute_project_path(args.output_root)

    if args.action in {"all", "download"}:
        download_sources(raw_root, markets)
    if args.action in {"all", "preprocess"}:
        preprocess_markets(
            raw_root,
            output_root,
            markets,
            clean_output=args.clean_output,
            include_alpha158=args.alpha158,
        )
        if args.baseline_relations:
            # Relation generation shares sources across markets, so make both
            # official repositories available before staging relation inputs.
            download_sources(raw_root, list(MARKETS))
            prepare_baseline_relations(raw_root, output_root, markets)


if __name__ == "__main__":
    main()
