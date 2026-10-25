#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS = [
    "stockmixer",
    "factorvae",
    "master",
    "enhancer",
    "tra",
    "rsr",
    "hstockgat",
    "factorgcl",
    "ci_sthpan",
]
DATASETS = ["NASDAQ", "NYSE", "SP500"]
MONITOR_METRIC = "RankIC"
MONITOR_MODE = "max"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check RankIC-monitor rerun summaries and per-run checkpoints."
    )
    parser.add_argument("--results", default="results_rankic")
    parser.add_argument("--expected-runs", type=int, default=5)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def latest_summary(results_root, dataset, model):
    model_dir = results_root / dataset / model
    if not model_dir.exists():
        return None
    summaries = sorted(model_dir.glob("*/summary_all_runs.json"))
    if not summaries:
        return None
    return summaries[-1]


def checkpoint_path(summary_path, run):
    if "checkpoint" in run:
        path = Path(run["checkpoint"])
        if path.exists():
            return path
        relative_to_root = summary_path.parents[4] / path
        if relative_to_root.exists():
            return relative_to_root
    return summary_path.parent / "checkpoints" / f"best_model_run{int(run['run_id'])}.pt"


def check_summary(summary_path, args):
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    runs = payload["runs"]
    errors = []
    if len(runs) != args.expected_runs:
        errors.append(f"run_count={len(runs)} expected={args.expected_runs}")
    for index, run in enumerate(runs):
        run_id = int(run["run_id"])
        if run_id != index:
            errors.append(f"run index mismatch at {index}: run_id={run_id}")
        if run["monitor_metric"] != MONITOR_METRIC:
            errors.append(f"run{run_id} monitor_metric={run['monitor_metric']}")
        if run["monitor_mode"] != MONITOR_MODE:
            errors.append(f"run{run_id} monitor_mode={run['monitor_mode']}")
        best_valid = run["best_valid"]
        if MONITOR_METRIC not in best_valid:
            errors.append(f"run{run_id} best_valid missing {MONITOR_METRIC}")
        test_metrics = run["test_metrics"]
        if MONITOR_METRIC not in test_metrics:
            errors.append(f"run{run_id} test_metrics missing {MONITOR_METRIC}")
        checkpoint = checkpoint_path(summary_path, run)
        if not checkpoint.exists():
            errors.append(f"run{run_id} checkpoint missing: {checkpoint}")
    return errors


def main():
    args = parse_args()
    results_root = Path(args.results)
    if not results_root.is_absolute():
        results_root = PROJECT_ROOT / results_root
    failures = []
    checked = 0

    for dataset in DATASETS:
        for model in MODELS:
            summary_path = latest_summary(results_root, dataset, model)
            if summary_path is None:
                message = f"{dataset}/{model}: missing summary_all_runs.json"
                if args.allow_incomplete:
                    print(f"[pending] {message}")
                    continue
                failures.append(message)
                continue
            errors = check_summary(summary_path, args)
            if errors:
                failures.append(f"{dataset}/{model}: {summary_path}: {'; '.join(errors)}")
                continue
            checked += 1
            print(f"[ok] {dataset}/{model}: {summary_path}")

    if failures:
        for failure in failures:
            print(f"[fail] {failure}")
        raise SystemExit(1)

    print(f"[ok] checked={checked} metric={MONITOR_METRIC} mode={MONITOR_MODE}")


if __name__ == "__main__":
    main()
