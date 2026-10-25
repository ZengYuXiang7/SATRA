#!/usr/bin/env python3
"""Evaluate portfolios from checkpoints selected by validation RankIC.

Formal runs under ``results_rankic`` select checkpoints by validation
RankIC/max. All replayable baselines are then measured under the current
evaluation environment. MAE remains a reported metric, but it is never used
for checkpoint selection.

TRA is retained in the five-metric catalog. Its Top-K replay is explicitly
marked unavailable because its test-time temporal error memory is not saved.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from metrics import _as_stock_date_matrix, evaluate as canonical_evaluate


DATASETS = ("NASDAQ", "NYSE", "SP500")
MODELS = (
    "stockmixer",
    "factorvae",
    "master",
    "enhancer",
    "tra",
    "rsr",
    "hstockgat",
    "factorgcl",
    "ci_sthpan",
)
FORMAL_METRICS = ("mae", "RankIC", "RankICIR", "IRR@10", "SR@10")
TOPKS = (5, 10, 15, 20, 25)
RUNTIME_CONFIG_FIELDS = ("precision", "torch_compile", "compile_mode", "fused_optimizer")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
        help="Datasets to process; defaults to all three formal datasets.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=list(MODELS),
        help="Baselines to process; defaults to all formal baselines.",
    )
    parser.add_argument(
        "--rankic-result-dirs",
        nargs="+",
        type=Path,
        default=None,
        metavar="RESULT_DIR",
        help=(
            "Explicit completed RankIC result directories. Requires --monitor rankic and "
            "must supply exactly one directory for every requested dataset/model pair."
        ),
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the RankIC-selected JSON and Markdown catalogs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only this script's existing generated catalog files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Generated catalog directory; defaults to results_rankic.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def finite_float(value: Any, context: str) -> float:
    if not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{context}: expected numeric value, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context}: non-finite value {result}")
    return result


def optional_metric(value: Any, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"nan", "none", "null", ""}:
        return None
    if not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{context}: expected numeric value or an explicit missing marker, got {type(value).__name__}")
    result = float(value)
    return result if math.isfinite(result) else None


def validate_completed_result(config_path: Path, monitor_metric: str, monitor_mode: str) -> dict[str, Any]:
    config = read_json(config_path)
    result_dir = config_path.parent
    if config.get("monitor_metric") != monitor_metric:
        raise ValueError(f"{result_dir}: expected monitor_metric={monitor_metric!r}")
    if config.get("monitor_mode") not in {monitor_mode, "auto"}:
        raise ValueError(f"{result_dir}: unexpected monitor_mode={config.get('monitor_mode')!r}")
    if int(config.get("num_runs", -1)) != 5 or bool(config.get("debug", 0)):
        raise ValueError(f"{result_dir}: not a formal five-run non-debug result")

    summary_path = result_dir / "summary_all_runs.json"
    summary = read_json(summary_path)
    runs = summary.get("runs")
    if not isinstance(runs, list) or len(runs) != 5:
        raise ValueError(f"{summary_path}: expected exactly five run records")
    aggregate = summary.get("aggregate")
    if not isinstance(aggregate, dict):
        raise TypeError(f"{summary_path}: aggregate is missing")
    for metric in FORMAL_METRICS:
        if metric not in aggregate:
            raise KeyError(f"{summary_path}: aggregate missing {metric}")
        metric_values = aggregate[metric]
        if not isinstance(metric_values, dict):
            raise TypeError(f"{summary_path}: malformed aggregate for {metric}")
        finite_float(metric_values.get("mean"), f"{summary_path}:{metric}.mean")
        finite_float(metric_values.get("std"), f"{summary_path}:{metric}.std")
    for run in runs:
        if not isinstance(run, dict):
            raise TypeError(f"{summary_path}: malformed run record")
        test_metrics = run.get("test_metrics")
        if not isinstance(test_metrics, dict):
            raise TypeError(f"{summary_path}: test_metrics is missing")
        for metric in FORMAL_METRICS:
            if metric not in test_metrics:
                raise KeyError(f"{summary_path}:run{run.get('run_id')} missing {metric}")
            optional_metric(test_metrics[metric], f"{summary_path}:run{run.get('run_id')}:{metric}")
        checkpoint = Path(str(run.get("checkpoint", "")))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{summary_path}: missing checkpoint {checkpoint}")
    return {"config": config, "summary": summary, "result_dir": result_dir}


def candidates_for(
    root: Path,
    dataset: str,
    model: str,
    monitor_metric: str,
    monitor_mode: str,
) -> list[dict[str, Any]]:
    base = root / dataset / model
    if not base.is_dir():
        raise FileNotFoundError(base)
    candidates: list[dict[str, Any]] = []
    for config_path in sorted(base.glob("*/config.json")):
        try:
            candidate = validate_completed_result(config_path, monitor_metric, monitor_mode)
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            continue
        candidates.append(candidate)
    return candidates


def select_rankic_catalog(datasets: tuple[str, ...], models: tuple[str, ...]) -> dict[tuple[str, str], dict[str, Any]]:
    root = PROJECT_ROOT / "results_rankic"
    if not root.is_dir():
        raise FileNotFoundError(root)
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset in datasets:
        for model in models:
            candidates = candidates_for(root, dataset, model, "RankIC", "max")
            if len(candidates) != 1:
                paths = [str(candidate["result_dir"]) for candidate in candidates]
                raise RuntimeError(f"RankIC source is ambiguous for {dataset}/{model}: {paths}")
            selected[(dataset, model)] = candidates[0]
    return selected


def select_explicit_rankic_catalog(
    result_dirs: tuple[Path, ...],
    datasets: tuple[str, ...],
    models: tuple[str, ...],
) -> dict[tuple[str, str], dict[str, Any]]:
    expected = {(dataset, model) for dataset in datasets for model in models}
    if len(result_dirs) != len(expected):
        raise ValueError(
            f"--rankic-result-dirs requires {len(expected)} directories for the requested dataset/model pairs; "
            f"got {len(result_dirs)}"
        )

    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for supplied_path in result_dirs:
        result_dir = supplied_path if supplied_path.is_absolute() else PROJECT_ROOT / supplied_path
        candidate = validate_completed_result(result_dir / "config.json", "RankIC", "max")
        config = candidate["config"]
        dataset = str(config["data"])
        model = str(config.get("model_id", config.get("model", "")))
        key = (dataset, model)
        if key not in expected:
            raise ValueError(f"{result_dir}: unexpected dataset/model pair {key}; expected one of {sorted(expected)}")
        if key in selected:
            raise ValueError(f"duplicate explicit RankIC result for {dataset}/{model}: {result_dir}")
        candidate["source_selection_note"] = "explicit RankIC result directory supplied for this evaluation"
        selected[key] = candidate

    missing = expected.difference(selected)
    if missing:
        raise ValueError(f"missing explicit RankIC result directories for {sorted(missing)}")
    return selected


def args_from_config(module: Any, config: dict[str, Any], model_name: str) -> argparse.Namespace:
    original_argv = list(sys.argv)
    try:
        sys.argv = ["checkpoint_topk_replay", "--data", str(config["data"])]
        args = module.parse_args()
    finally:
        sys.argv = original_argv
    for key, value in config.items():
        if key != "runtime_protocol":
            setattr(args, key, value)
    if str(args.data) not in DATASETS:
        raise ValueError(f"invalid dataset in config: {args.data!r}")
    if bool(args.debug):
        raise ValueError("formal checkpoint replay refuses debug configs")
    args.model = model_name
    return args


def construct_data(module: Any, model_name: str, args: argparse.Namespace) -> Any:
    common = {
        "root_path": args.root_path,
        "seq_len": args.seq_len,
        "steps": args.steps,
        "input_features": args.input_features,
        "batch_size": args.batch_size,
        "random_seed": args.random_seed,
        "debug": False,
        "debug_train_steps": args.debug_train_steps,
        "debug_eval_steps": args.debug_eval_steps,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "train_drop_last": True,
        "valid_drop_last": True,
    }
    if model_name == "ci_sthpan":
        data_args = module.make_data_args(
            args.data,
            relation_root=args.relation_root,
            relation_name=args.relation_name,
            hypergraph_name=args.hypergraph_name,
            **common,
        )
        data = module.CISTHPANData(data_args)
        args.relation_name_resolved = data.relation_name
        args.relation_channels = data.relation_channels
        args.hypergraph_name_resolved = data.hypergraph_name
        args.hyperedge_count = data.hyperedge_count
    elif model_name == "factorgcl":
        data_args = module.make_data_args(
            args.data,
            relation_root=args.relation_root,
            relation_name=args.relation_name,
            prior_beta_name=args.prior_beta_name,
            future_seq_len=args.future_seq_len,
            use_contrastive=bool(args.use_contrastive),
            **common,
        )
        data = module.FactorGCLData(data_args)
        args.relation_name_resolved = data.relation_name
        args.prior_beta_path = str(data.prior_beta_path)
        args.prior_beta_shape = list(data.prior_beta.shape)
        args.prior_factor_count = data.prior_factor_count
    elif model_name in {"rsr", "hstockgat"}:
        data_args = module.make_data_args(
            args.data,
            relation_root=args.relation_root,
            relation_name=args.relation_name,
            **common,
        )
        data = module.RSRData(data_args)
        args.relation_name_resolved = data.relation_name
        args.relation_channels = data.relation_channels
        args.relation_nonzero = data.relation_nonzero
    elif model_name == "stockmixer":
        data_args = module.make_data_args(
            args.data,
            stockmixer_ma_windows=args.stockmixer_ma_windows,
            require_feature_groups=False,
            **common,
        )
        data = module.StockMixerData(data_args)
    elif model_name == "factorvae":
        data_args = module.make_data_args(args.data, **common)
        data = module.FactorVAEData(data_args)
    elif model_name == "master":
        data_args = module.make_data_args(args.data, **common)
        data = module.MasterData(data_args)
        args.stock_feature_num = data.stock_feature_num
        args.market_feature_num = data.market_feature_num
    elif model_name == "enhancer":
        data_args = module.make_data_args(args.data, **common)
        data = module.EnhancerData(data_args)
    else:
        raise ValueError(f"unsupported model construction: {model_name}")
    args.features = data.feature_num
    args.stock_num = data.stock_num
    args.market_num = int(getattr(data.args, "market_num", data.stock_num))
    if model_name == "stockmixer" and args.stock_hidden is None:
        args.stock_hidden = args.market_num
    return data


def portfolio_topk_metrics(prediction: Any, ground_truth: Any, mask: Any, stock_num: int) -> dict[str, float]:
    prediction = _as_stock_date_matrix(prediction, "prediction", stock_num=stock_num)
    ground_truth = _as_stock_date_matrix(ground_truth, "ground_truth", stock_num=stock_num)
    mask = _as_stock_date_matrix(mask, "mask", stock_num=stock_num)
    if prediction.shape != ground_truth.shape or prediction.shape != mask.shape:
        raise ValueError(
            f"Top-K shape mismatch: prediction={prediction.shape}, ground_truth={ground_truth.shape}, mask={mask.shape}"
        )
    valid_mask = (mask >= 0.5) & np.isfinite(prediction) & np.isfinite(ground_truth)
    result: dict[str, float] = {}
    for k in TOPKS:
        daily_returns: list[float] = []
        for day_idx in range(prediction.shape[1]):
            valid = valid_mask[:, day_idx]
            if not np.any(valid):
                daily_returns.append(float("nan"))
                continue
            pred_day = prediction[valid, day_idx]
            gt_day = ground_truth[valid, day_idx]
            selected = np.argsort(pred_day, kind="mergesort")[-min(k, pred_day.size) :][::-1]
            daily_returns.append(float(np.mean(gt_day[selected])))
        valid_daily = np.asarray(daily_returns, dtype=np.float64)
        valid_daily = valid_daily[np.isfinite(valid_daily)]
        if valid_daily.size == 0:
            raise ValueError(f"no valid daily return for K={k}")
        irr = float(np.prod(1.0 + valid_daily) - 1.0)
        std = float(np.std(valid_daily))
        sr = float(math.sqrt(252.0) * np.mean(valid_daily) / std) if std > 0.0 else float("nan")
        if not math.isfinite(irr) or not math.isfinite(sr):
            raise ValueError(f"non-finite Top-K metric for K={k}: IRR={irr}, SR={sr}")
        result[f"IRR@{k}"] = irr
        result[f"SR@{k}"] = sr
    return result


def measured_formal_metrics(replay: dict[str, Any], topk: dict[str, float], context: str) -> dict[str, float | None]:
    """Return the five report metrics from this checkpoint replay.

    IRR@10 and SR@10 come from the same captured portfolio calculation as the
    K sensitivity table, keeping the two reported locations internally
    consistent.
    """

    result = {
        "mae": finite_float(replay.get("mae"), f"{context}:replay:mae"),
        "RankIC": optional_metric(replay.get("RankIC"), f"{context}:replay:RankIC"),
        "RankICIR": optional_metric(replay.get("RankICIR"), f"{context}:replay:RankICIR"),
        "IRR@10": finite_float(topk.get("IRR@10"), f"{context}:topk:IRR@10"),
        "SR@10": finite_float(topk.get("SR@10"), f"{context}:topk:SR@10"),
    }
    return result


def load_checkpoint_strict(model: torch.nn.Module, checkpoint: Path, device: torch.device) -> str:
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint is not a state dictionary: {checkpoint}")
    target_keys = set(model.state_dict())
    source_keys = set(state)
    transform = "identity"
    if source_keys != target_keys:
        if {f"core.{key}" for key in source_keys} == target_keys:
            state = {f"core.{key}": value for key, value in state.items()}
            transform = "add_core_prefix"
        elif {key.removeprefix("core.") for key in source_keys} == target_keys and all(
            key.startswith("core.") for key in source_keys
        ):
            state = {key.removeprefix("core."): value for key, value in state.items()}
            transform = "remove_core_prefix"
        else:
            missing = sorted(target_keys - source_keys)[:8]
            unexpected = sorted(source_keys - target_keys)[:8]
            raise RuntimeError(
                f"checkpoint/model state keys differ for {checkpoint}; missing_sample={missing}, unexpected_sample={unexpected}"
            )
    model.load_state_dict(state, strict=True)
    return transform


def invoke_eval(module: Any, model_name: str, model: Any, loader: Any, device: torch.device, args: argparse.Namespace, data: Any) -> tuple[dict[str, Any], dict[str, float]]:
    captured: dict[str, dict[str, float]] = {}
    original_evaluate = module.evaluate

    def capture_evaluate(prediction: Any, ground_truth: Any, mask: Any, stock_num: int | None = None, report: bool = False) -> dict[str, Any]:
        if "topk" in captured:
            raise RuntimeError(f"{model_name}: evaluator called more than once")
        if stock_num is None:
            raise ValueError(f"{model_name}: evaluator did not receive stock_num")
        captured["topk"] = portfolio_topk_metrics(prediction, ground_truth, mask, stock_num)
        return canonical_evaluate(prediction, ground_truth, mask, stock_num=stock_num, report=report)

    module.evaluate = capture_evaluate
    try:
        if model_name == "factorvae":
            replay = module.eval_model(model, loader, device, data.stock_num, args.runtime_protocol)
        else:
            replay = module.eval_model(model, loader, device, args, data.stock_num, args.runtime_protocol)
    finally:
        module.evaluate = original_evaluate
    if "topk" not in captured:
        raise RuntimeError(f"{model_name}: Top-K evaluator capture did not run")
    if not isinstance(replay, dict):
        raise TypeError(f"{model_name}: eval_model did not return a metric dictionary")
    return replay, captured["topk"]


def invoke_legacy_stockmixer_eval(
    module: Any, model: Any, loader: Any, device: torch.device, args: argparse.Namespace, data: Any
) -> tuple[dict[str, Any], dict[str, float]]:
    captured: dict[str, dict[str, float]] = {}
    original_evaluate = module.evaluate

    def capture_evaluate(prediction: Any, ground_truth: Any, mask: Any, stock_num: int | None = None, report: bool = False) -> dict[str, Any]:
        if "topk" in captured:
            raise RuntimeError("legacy StockMixer evaluator called more than once")
        if stock_num is None:
            raise ValueError("legacy StockMixer evaluator did not receive stock_num")
        captured["topk"] = portfolio_topk_metrics(prediction, ground_truth, mask, stock_num)
        return canonical_evaluate(prediction, ground_truth, mask, stock_num=stock_num, report=report)

    module.evaluate = capture_evaluate
    try:
        replay = module.eval_model(model, loader, device, args, data.stock_num)
    finally:
        module.evaluate = original_evaluate
    if "topk" not in captured:
        raise RuntimeError("legacy StockMixer Top-K evaluator capture did not run")
    if not isinstance(replay, dict):
        raise TypeError("legacy StockMixer eval_model did not return a metric dictionary")
    return replay, captured["topk"]


def replay_legacy_stockmixer_configuration(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    legacy_root = PROJECT_ROOT / "legacy_src_after_top_level_refactor"
    if not legacy_root.is_dir():
        raise FileNotFoundError(f"legacy StockMixer source snapshot is missing: {legacy_root}")
    if str(legacy_root) not in sys.path:
        sys.path.insert(0, str(legacy_root))
    module = importlib.import_module("stockmixer.run")
    config = candidate["config"]
    summary = candidate["summary"]
    module.set_seed(int(summary["runs"][0]["seed"]))
    args = args_from_config(module, config, "stockmixer")
    device = module.setup_device(args)
    data_args = module.make_data_args(
        args.data,
        root_path=args.root_path,
        seq_len=args.seq_len,
        steps=args.steps,
        batch_size=args.batch_size,
        random_seed=args.random_seed,
        debug=False,
        debug_train_steps=args.debug_train_steps,
        debug_eval_steps=args.debug_eval_steps,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        train_drop_last=True,
        input_features=args.input_features,
        stockmixer_ma_windows=args.stockmixer_ma_windows,
        require_feature_groups=False,
    )
    data = module.StockMarketData(data_args)
    args.features = data.feature_num
    args.stock_num = data.stock_num
    args.market_num = int(getattr(data.args, "market_num", data.stock_num))
    if args.stock_hidden is None:
        args.stock_hidden = args.market_num
    raw_model = module.build_model(args, data).to(device)
    loader = data.eval_loader(data.test_index, data.trade_dates, drop_last=False)

    rows: list[dict[str, Any]] = []
    try:
        for run in summary["runs"]:
            run_id = int(run["run_id"])
            seed = int(run["seed"])
            checkpoint = Path(str(run["checkpoint"]))
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            module.set_seed(seed)
            checkpoint_key_transform = load_checkpoint_strict(raw_model, checkpoint, device)
            replay, topk = invoke_legacy_stockmixer_eval(module, raw_model, loader, device, args, data)
            context = f"{candidate['result_dir']} run={run_id}"
            rows.append(
                {
                    "run_id": run_id,
                    "seed": seed,
                    "best_epoch": int(run["best_epoch"]),
                    "checkpoint_key_transform": checkpoint_key_transform,
                    "formal_metrics": measured_formal_metrics(replay, topk, context),
                    "saved_source_metrics": {
                        metric: optional_metric(run["test_metrics"].get(metric), f"{context}:saved:{metric}")
                        for metric in FORMAL_METRICS
                    },
                    "formal_metric_origin": "checkpoint_replay",
                    "topk_metrics": topk,
                    "replay_status": "measured",
                    "replay_runtime": "legacy_source_snapshot",
                    "replay_reason": None,
                }
            )
    finally:
        del raw_model
        del data
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def replay_configuration(candidate: dict[str, Any], model_name: str) -> list[dict[str, Any]]:
    if model_name == "tra":
        reason = (
            "TRA test-time temporal error memory is not persisted in its checkpoint or history artifact; "
            "replaying only model weights would change the model input."
        )
        rows = []
        for run in candidate["summary"]["runs"]:
            rows.append(
                {
                    "run_id": int(run["run_id"]),
                    "seed": int(run["seed"]),
                    "best_epoch": int(run["best_epoch"]),
                    "formal_metrics": {metric: optional_metric(run["test_metrics"][metric], metric) for metric in FORMAL_METRICS},
                    "saved_source_metrics": {metric: optional_metric(run["test_metrics"][metric], metric) for metric in FORMAL_METRICS},
                    "formal_metric_origin": "saved_source_unreplayable",
                    "topk_metrics": None,
                    "replay_status": "unavailable",
                    "replay_runtime": "unavailable",
                    "replay_reason": reason,
                }
            )
        return rows

    if model_name == "stockmixer" and not all(key in candidate["config"] for key in RUNTIME_CONFIG_FIELDS):
        return replay_legacy_stockmixer_configuration(candidate)

    module = importlib.import_module(f"{model_name}.run")
    config = candidate["config"]
    summary = candidate["summary"]
    first_seed = int(summary["runs"][0]["seed"])
    module.set_seed(first_seed)
    args = args_from_config(module, config, model_name)
    device = module.setup_device(args)
    args.runtime_protocol = module.make_runtime_protocol(args, device)
    data = construct_data(module, model_name, args)
    raw_model = module.build_model(args, data).to(device)
    model = module.prepare_model(raw_model, args, device, args.runtime_protocol)
    loader = data.eval_loader(data.test_index, data.trade_dates, drop_last=False)

    rows: list[dict[str, Any]] = []
    try:
        for run in summary["runs"]:
            run_id = int(run["run_id"])
            seed = int(run["seed"])
            checkpoint = Path(str(run["checkpoint"]))
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            module.set_seed(seed)
            checkpoint_key_transform = load_checkpoint_strict(raw_model, checkpoint, device)
            replay, topk = invoke_eval(module, model_name, model, loader, device, args, data)
            context = f"{candidate['result_dir']} run={run_id}"
            rows.append(
                {
                    "run_id": run_id,
                    "seed": seed,
                    "best_epoch": int(run["best_epoch"]),
                    "checkpoint_key_transform": checkpoint_key_transform,
                    "formal_metrics": measured_formal_metrics(replay, topk, context),
                    "saved_source_metrics": {
                        metric: optional_metric(run["test_metrics"].get(metric), f"{context}:saved:{metric}")
                        for metric in FORMAL_METRICS
                    },
                    "formal_metric_origin": "checkpoint_replay",
                    "topk_metrics": topk,
                    "replay_status": "measured",
                    "replay_runtime": "current_source",
                    "replay_reason": None,
                }
            )
    finally:
        del model
        del raw_model
        del data
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def sample_aggregate(values: list[float], context: str) -> dict[str, Any]:
    if len(values) != 5:
        raise ValueError(f"{context}: expected five values, got {len(values)}")
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{context}: non-finite values")
    return {"mean": float(np.mean(array)), "std": float(np.std(array, ddof=1)), "values": array.tolist()}


def formal_replay_aggregate(runs: list[dict[str, Any]], metric: str, context: str) -> dict[str, Any]:
    """Aggregate a newly measured formal metric, retaining a valid-run count."""

    values = [
        optional_metric(run["formal_metrics"].get(metric), f"{context}:run{run['run_id']}:{metric}")
        for run in runs
    ]
    finite_values = [value for value in values if value is not None]
    if not finite_values:
        raise ValueError(f"{context}:{metric}: no finite replayed values")
    array = np.asarray(finite_values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "values": array.tolist(),
        "num_valid_runs": int(array.size),
        "num_declared_runs": len(runs),
    }


def result_entry(candidate: dict[str, Any], model_name: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    measured_formal = all(run["formal_metric_origin"] == "checkpoint_replay" for run in runs)
    if measured_formal:
        formal_aggregate = {
            metric: formal_replay_aggregate(runs, metric, str(candidate["result_dir"]))
            for metric in FORMAL_METRICS
        }
        formal_metric_origin = "checkpoint_replay"
    else:
        aggregate = candidate["summary"]["aggregate"]
        formal_aggregate = {
            metric: {
                "mean": finite_float(aggregate[metric]["mean"], f"{candidate['result_dir']}:{metric}.mean"),
                "std": finite_float(aggregate[metric]["std"], f"{candidate['result_dir']}:{metric}.std"),
                "values": [finite_float(value, f"{candidate['result_dir']}:{metric}.value") for value in aggregate[metric]["values"]],
                "num_valid_runs": len(aggregate[metric]["values"]),
                "num_declared_runs": len(runs),
            }
            for metric in FORMAL_METRICS
        }
        formal_metric_origin = "saved_source_unreplayable"
    if any(run["topk_metrics"] is None for run in runs):
        topk_status = "unavailable"
        topk_reason = runs[0]["replay_reason"]
        topk_aggregate = None
    else:
        topk_status = "measured"
        topk_reason = None
        topk_aggregate = {
            metric: sample_aggregate([float(run["topk_metrics"][metric]) for run in runs], f"{candidate['result_dir']}:{metric}")
            for metric in (f"{name}@{k}" for name in ("IRR", "SR") for k in TOPKS)
        }
    return {
        "dataset": candidate["config"]["data"],
        "model": model_name,
        "monitor_metric": candidate["config"]["monitor_metric"],
        "monitor_mode": candidate["config"]["monitor_mode"],
        "num_runs": len(runs),
        "source_result_dir": str(candidate["result_dir"].relative_to(PROJECT_ROOT)),
        "source_selection_note": candidate.get("source_selection_note", "formal RankIC catalog result"),
        "formal_metric_origin": formal_metric_origin,
        "formal_aggregate": formal_aggregate,
        "portfolio_topk_status": topk_status,
        "portfolio_topk_reason": topk_reason,
        "portfolio_topk_aggregate": topk_aggregate,
        "runs": runs,
    }


def stat_text(stats: dict[str, Any] | None) -> str:
    if stats is None:
        return "N/A"
    text = f"{float(stats['mean']):.3f} ± {float(stats['std']):.3f}"
    valid_runs = stats.get("num_valid_runs")
    declared_runs = stats.get("num_declared_runs")
    if valid_runs is not None and declared_runs is not None and int(valid_runs) != int(declared_runs):
        text += f" (n={int(valid_runs)}/{int(declared_runs)})"
    return text


def render_markdown(monitor: str, entries: list[dict[str, Any]], elapsed_seconds: float) -> str:
    title = "RankIC/max"
    lines = [
        f"# Financial Baseline Results: {title} Checkpoint Selection",
        "",
        "- Checkpoints are selected by this catalog's monitor criterion, then measured anew under the current evaluation environment.",
        "- Measured metrics: MAE, RankIC, RankICIR, IRR@10, SR@10, plus portfolio sensitivity at K = 5, 10, 15, 20, and 25.",
        "- No model is retrained. TRA alone retains saved formal metrics and has no Top-K replay because its test-time temporal memory was not persisted.",
        "- Values are means ± sample standard deviations over five runs; `n=a/b` marks a saved historical metric with fewer than five finite runs.",
        f"- Evaluation wall time: {elapsed_seconds:.1f} s.",
        "",
        "## Formal Metrics",
        "",
        "| Dataset | Model | MAE | RankIC | RankICIR | IRR@10 | SR@10 | Metric Source | Top-K Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for entry in entries:
        formal = entry["formal_aggregate"]
        lines.append(
            f"| {entry['dataset']} | {entry['model']} | {stat_text(formal['mae'])} | {stat_text(formal['RankIC'])} | {stat_text(formal['RankICIR'])} | {stat_text(formal['IRR@10'])} | {stat_text(formal['SR@10'])} | {entry['formal_metric_origin']} | {entry['portfolio_topk_status']} |"
        )

    lines.extend(
        [
            "",
            "## Portfolio Size Sensitivity: IRR@K",
            "",
            "| Dataset | Model | IRR@5 | IRR@10 | IRR@15 | IRR@20 | IRR@25 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for entry in entries:
        topk = entry["portfolio_topk_aggregate"]
        cells = [stat_text(topk.get(f"IRR@{k}") if topk else None) for k in TOPKS]
        lines.append(f"| {entry['dataset']} | {entry['model']} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Portfolio Size Sensitivity: SR@K",
            "",
            "| Dataset | Model | SR@5 | SR@10 | SR@15 | SR@20 | SR@25 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for entry in entries:
        topk = entry["portfolio_topk_aggregate"]
        cells = [stat_text(topk.get(f"SR@{k}") if topk else None) for k in TOPKS]
        lines.append(f"| {entry['dataset']} | {entry['model']} | " + " | ".join(cells) + " |")

    nonfull = [entry for entry in entries if entry["portfolio_topk_status"] != "measured"]
    lines.extend(["", "## Source Mapping", ""])
    for entry in entries:
        lines.append(
            f"- `{entry['dataset']}/{entry['model']}`: `{entry['source_result_dir']}`; {entry['source_selection_note']}"
        )
    if nonfull:
        lines.extend(["", "## Non-full Top-K Replay Status", ""])
        for entry in nonfull:
            lines.append(
                f"- `{entry['dataset']}/{entry['model']}` ({entry['portfolio_topk_status']}): {entry['portfolio_topk_reason']}"
            )
    lines.append("")
    return "\n".join(lines)


def atomic_write(path: Path, content: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite generated artifact without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def process_monitor(
    monitor: str,
    catalog: dict[tuple[str, str], dict[str, Any]],
    datasets: tuple[str, ...],
    models: tuple[str, ...],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    total = len(datasets) * len(models)
    for index, (dataset, model) in enumerate(((d, m) for d in datasets for m in models), start=1):
        candidate = catalog[(dataset, model)]
        print(f"[{monitor}] {index}/{total} {dataset}/{model}: replaying saved checkpoints", flush=True)
        started = time.monotonic()
        runs = replay_configuration(candidate, model)
        measured = sum(run["replay_status"] == "measured" for run in runs)
        unavailable = sum(run["replay_status"] == "unavailable" for run in runs)
        print(
            f"[{monitor}] {index}/{total} {dataset}/{model}: measured={measured}/5 unavailable={unavailable}/5 in {time.monotonic() - started:.1f}s",
            flush=True,
        )
        entries.append(result_entry(candidate, model, runs))
    return entries


def main() -> None:
    args = parse_args()
    datasets = tuple(args.datasets)
    models = tuple(args.models)
    if args.rankic_result_dirs is None:
        rankic_catalog = select_rankic_catalog(datasets, models)
    else:
        rankic_catalog = select_explicit_rankic_catalog(tuple(args.rankic_result_dirs), datasets, models)
    started = time.monotonic()
    entries = process_monitor("rankic", rankic_catalog, datasets, models)
    elapsed = time.monotonic() - started

    if args.write:
        output_root = args.output_root or PROJECT_ROOT / "results_rankic"
        if not output_root.is_absolute():
            output_root = PROJECT_ROOT / output_root
        payload = {
            "monitor_catalog": "rankic",
            "monitor_selection": "RankIC/max",
            "formal_metrics": list(FORMAL_METRICS),
            "portfolio_sizes": list(TOPKS),
            "measurement_protocol": {
                "checkpoint_selection": "validation RankIC selects checkpoints",
                "replayable_models": "all formal metrics and IRR/SR@K values are measured from current checkpoint replay",
                "unreplayable_model": "TRA retains saved formal metrics because test-time temporal error memory is absent from checkpoints",
            },
            "entries": entries,
        }
        markdown = render_markdown("rankic", entries, elapsed)
        atomic_write(output_root / "portfolio_topk_sensitivity.json", json.dumps(payload, indent=2, allow_nan=False) + "\n", args.overwrite)
        atomic_write(output_root / "portfolio_topk_sensitivity.md", markdown, args.overwrite)
        atomic_write(output_root / "RESULTS_MANIFEST.md", markdown, args.overwrite)
        print(f"[rankic] wrote {output_root / 'portfolio_topk_sensitivity.json'}", flush=True)

    measured_runs = sum(sum(run["replay_status"] == "measured" for run in entry["runs"]) for entry in entries)
    unavailable_runs = sum(sum(run["replay_status"] == "unavailable" for run in entry["runs"]) for entry in entries)
    print(
        f"[rankic] complete: entries={len(entries)}, measured_runs={measured_runs}, unavailable_runs={unavailable_runs}, wall={elapsed:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
