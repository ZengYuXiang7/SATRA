import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

import numpy as np

from satra.utils.metrics import (
    AR_KEY,
    IRR_KEY,
    PORTFOLIO_SIZE,
    PRECISION_KEY,
    SR_KEY,
)

PRIMARY_METRIC_ORDER = ["mae", "RankIC", "RankICIR", IRR_KEY, SR_KEY]
AUXILIARY_METRIC_ORDER = [AR_KEY, "KT", "mape", PRECISION_KEY, "IC", "ICIR"]
METRIC_DISPLAY_NAMES = {
    "mae": "MAE",
    "RankIC": "RankIC",
    IRR_KEY: f"IRR@{PORTFOLIO_SIZE}",
    SR_KEY: f"SR@{PORTFOLIO_SIZE}",
    "RankICIR": "RankICIR",
    AR_KEY: f"AR@{PORTFOLIO_SIZE}",
    "SR": "SR",
    "KT": "KT",
    "mape": "MAPE",
    PRECISION_KEY: f"P@{PORTFOLIO_SIZE}",
    "IC": "IC",
    "ICIR": "ICIR",
}


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_config(args, paths):
    skip = {"obs_logger", "obs_paths", "obs_run_paths"}
    payload = {k: v for k, v in vars(args).items() if k not in skip}
    payload["runtime_protocol_requested"] = {
        "precision": args.precision,
        "torch_compile": bool(args.torch_compile),
        "torch_compile_mode": args.torch_compile_mode,
        "max_grad_norm": args.max_grad_norm,
        "train_drop_last": bool(args.train_drop_last),
        "valid_drop_last": bool(args.valid_drop_last),
        "test_drop_last": False,
        "early_stop": True,
        "patience": args.patience_requested,
        "min_delta": float(args.min_delta),
    }
    payload["runtime_protocol"] = {
        "precision": {
            "requested": args.precision,
            "effective": "pending_until_run_start",
            "reason": "full effective state is recorded after runtime initialization",
        },
        "compile": {
            "requested": bool(args.torch_compile),
            "effective": "pending_until_run_start",
            "mode": args.torch_compile_mode,
            "reason": "debug and non-cuda runs disable compile before model construction",
        },
        "optimizer": {
            "name": "AdamW",
            "fused_requested": args.device_type == "cuda",
            "fused_effective": "pending_until_optimizer_creation",
            "disabled_reason": None,
        },
        "dataloader": {
            "pin_memory_requested": bool(args.pin_memory_requested),
            "pin_memory_effective": bool(args.pin_memory),
        },
        "grad_clip": {
            "max_grad_norm": float(args.max_grad_norm),
            "clip_ratio": "pending_until_epoch",
            "max_grad_norm_seen": "pending_until_epoch",
        },
        "drop_last": {
            "train_requested": bool(args.train_drop_last),
            "train_effective": bool(args.train_drop_last),
            "valid_effective": bool(args.valid_drop_last),
            "test_effective": False,
            "semantics": "stock time-window iteration; train/valid may drop tail batches for compile-stable shapes, test keeps all eval offsets",
        },
    }
    with open(paths["config_path"], "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False)


def update_config_runtime(paths, runtime_info, early_stop=None):
    with open(paths["config_path"], "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["runtime_protocol"] = runtime_info["runtime_protocol"]
    if early_stop is not None:
        payload["early_stop"] = early_stop
    with open(paths["config_path"], "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False)


def save_history(history, path, early_stop):
    payload = {"epochs": history, "early_stop": early_stop}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False)


def save_training_curve(history, path):
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train"]["loss"] for row in history]
    valid_loss = [row["valid"]["loss"] for row in history]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, valid_loss, label="valid_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_summary_all_runs(paths, payload):
    with open(paths["summary_path"], "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False)


def _metric_stats(run_summaries):
    values = {}
    for run in run_summaries:
        for name, value in run["test_metrics"].items():
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                values.setdefault(name, []).append(float(value))
    stats = {}
    for name, vals in values.items():
        arr = np.asarray(vals, dtype=np.float64)
        stats[name] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "values": vals,
        }
    return stats


def _format_metric_value(value):
    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
        return f"{float(value):.6f}"
    return str(value)


def _format_fixed_float(value, width=10, digits=6):
    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
        return f"{float(value):>{width}.{digits}f}"
    return f"{str(value):>{width}}"


def _round_train_seconds(run):
    value = run["train_seconds"]
    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
        return float(value)
    raise ValueError(f"train_seconds must be finite, got {value!r}")


def _format_rounds_table(args, run_summaries):
    metric = str(args.early_stop_metric)
    valid_col = f"Valid{metric}"[:12]
    test_evaluated = all(bool(run.get("test_evaluated", True)) for run in run_summaries)
    if not test_evaluated:
        headers = [
            ("Time", 19),
            ("Rd", 3),
            ("Seed", 5),
            ("BestEp", 6),
            (valid_col, 12),
            ("Time(s)", 7),
        ]

        def validation_row(values):
            return "| " + " | ".join(
                f"{str(value):>{width}}" if idx != 0 else f"{str(value):<{width}}"
                for idx, ((_, width), value) in enumerate(zip(headers, values))
            ) + " |"

        header = "| " + " | ".join(f"{name:<{width}}" for name, width in headers) + " |"
        rule = "| " + " | ".join("-" * width for _, width in headers) + " |"
        lines = ["```text", header, rule]
        for run in run_summaries:
            best_valid = run["best_valid"]
            lines.append(
                validation_row(
                    [
                        str(run["round_finished_at"]),
                        int(run["run_id"]) + 1,
                        run["seed"],
                        run["best_epoch"],
                        _format_fixed_float(best_valid[metric], width=12),
                        f"{_round_train_seconds(run):>7.1f}",
                    ]
                )
            )
        lines.append("```")
        return lines

    headers = [
        ("Time", 19),
        ("Rd", 3),
        ("Seed", 5),
        ("BestEp", 6),
        (valid_col, 12),
        ("MAE", 10),
        ("RankIC", 10),
        ("RankICIR", 10),
        (f"IRR@{PORTFOLIO_SIZE}", 10),
        (f"SR@{PORTFOLIO_SIZE}", 10),
        ("Time(s)", 7),
    ]

    def row(values):
        return "| " + " | ".join(
            f"{str(value):>{width}}" if idx not in {0} else f"{str(value):<{width}}"
            for idx, ((_, width), value) in enumerate(zip(headers, values))
        ) + " |"

    header = "| " + " | ".join(f"{name:<{width}}" for name, width in headers) + " |"
    rule = "| " + " | ".join("-" * width for _, width in headers) + " |"
    lines = ["```text", header, rule]
    for run in run_summaries:
        test = run["test_metrics"]
        best_valid = run["best_valid"]
        lines.append(
            row(
                [
                    str(run["round_finished_at"]),
                    int(run["run_id"]) + 1,
                    run["seed"],
                    run["best_epoch"],
                    _format_fixed_float(best_valid[metric], width=12),
                    _format_fixed_float(test["mae"]),
                    _format_fixed_float(test["RankIC"]),
                    _format_fixed_float(test["RankICIR"]),
                    _format_fixed_float(test[IRR_KEY]),
                    _format_fixed_float(test[SR_KEY]),
                    f"{_round_train_seconds(run):>7.1f}",
                ]
            )
        )
    lines.append("```")
    return lines


def format_aggregate_report(args, paths, run_summaries):
    if not run_summaries:
        raise ValueError("run_summaries must contain at least one run")
    stats = _metric_stats(run_summaries)
    if stats:
        metric_lines = [
            "```text",
            "| Metric     | Mean       | Std        |",
            "| ---------- | ---------- | ---------- |",
        ]
        ordered_names = [name for name in PRIMARY_METRIC_ORDER + AUXILIARY_METRIC_ORDER if name in stats]
        ordered_names.extend(name for name in stats.keys() if name not in ordered_names and name != "SR")
        for name in ordered_names:
            item = stats[name]
            mean = _format_fixed_float(item["mean"], width=10)
            std = _format_fixed_float(item["std"], width=10)
            metric_lines.append(f"| {METRIC_DISPLAY_NAMES[name]:<10} | {mean} | {std} |")
        metric_lines.append("```")
    else:
        metric_lines = ["Test evaluation: skipped for validation-only model selection."]
    round_lines = _format_rounds_table(args, run_summaries)
    return "\n".join(
        [
            "# Experiment Results",
            "",
            "## Summary",
            f"primary: valid {args.early_stop_metric} ({args.early_stop_mode})",
            "",
            "Run",
            f"- model       : {args.model}",
            f"- dataset     : {args.data}",
            f"- rounds      : {len(run_summaries)}",
            f"- base_seed   : {args.random_seed}",
            f"- precision   : {args.precision}",
            f"- compile     : {bool(args.torch_compile)} mode={args.torch_compile_mode}",
            f"- max_norm    : {args.max_grad_norm}",
            "",
            "Artifacts",
            "- summary     : summary_all_runs.json",
            "- detail      : model_detail.md",
            "- logs        : details/run*.log",
            "- curves      : curves/training_curve_run*.png",
            "- checkpoints : checkpoints/best_model_run*.pt",
            "",
            "Metrics",
            *metric_lines,
            "",
            "## Rounds",
            *round_lines,
            "",
        ]
    )


def save_run_md(paths, report):
    with open(paths["run_md_path"], "w", encoding="utf-8") as f:
        f.write(report)


def append_project_run_md(paths, report):
    with open(paths["project_run_md_path"], "a", encoding="utf-8") as f:
        f.write("\n\n" + report)


def save_model_detail(args, paths, model_text, params, runtime_info, run_summaries):
    if not run_summaries:
        raise ValueError("run_summaries must contain at least one run")
    runtime_protocol = runtime_info["runtime_protocol"]
    precision = runtime_protocol["precision"]
    compile_info = runtime_protocol["compile"]
    optimizer = runtime_protocol["optimizer"]
    grad_clip = runtime_protocol["grad_clip"]
    drop_last = runtime_protocol["drop_last"]
    ckpts = [
        "run{run_id} best: checkpoints/best_model_run{run_id}.pt\nrun{run_id} last: checkpoints/last_model_run{run_id}.pt".format(
            run_id=run["run_id"]
        )
        for run in run_summaries
    ]
    first_early_stop = run_summaries[0]["early_stop"]
    content = "\n".join(
        [
            "# Model Detail",
            "",
            "## Computation Cost",
            "",
            "| Method | Params (M) | FLOPs/MACs | Latency (ms) | Peak Memory (GB) | Training Time (h) |",
            "|---|---:|---:|---:|---:|---:|",
            f"| {args.model} | {params['trainable_m']:.3f} | N/A | N/A | N/A | N/A |",
            "",
            f"total_params: {params['total']}",
            f"trainable_params: {params['trainable']}",
            "",
            "## Measurement Protocol",
            "",
            f"- device: {runtime_info['device']}",
            f"- gpu_name: {runtime_info['gpu_name']}",
            f"- torch: {runtime_info['torch']}",
            f"- cuda: {runtime_info['cuda']}",
            f"- precision: {precision['effective']} (requested={precision['requested']}, reason={precision['reason']})",
            f"- amp: {precision['amp_effective']} (requested={precision['amp_requested']})",
            f"- tf32: {precision['tf32_effective']} (requested={precision['tf32_requested']})",
            f"- torch_compile: {compile_info['effective']} (requested={compile_info['requested']}, mode={compile_info['mode']}, reason={compile_info['reason']})",
            f"- optimizer: {optimizer['name']} (fused_requested={optimizer['fused_requested']}, fused_effective={optimizer['fused_effective']}, disabled_reason={optimizer['disabled_reason']})",
            f"- max_grad_norm: {grad_clip['max_grad_norm']}",
            f"- drop_last: train={drop_last['train_effective']}, valid={drop_last['valid_effective']}, test={drop_last['test_effective']} ({drop_last['semantics']})",
            f"- early_stop: {first_early_stop['effective']} (requested={first_early_stop['requested']})",
            f"- patience: {first_early_stop['patience_effective']} (requested={first_early_stop['patience_requested']})",
            f"- min_delta: {first_early_stop['min_delta']}",
            f"- monitor: valid.{args.early_stop_metric} ({args.early_stop_mode})",
            f"- seed: {args.random_seed}",
            "",
            "## Architecture",
            "",
            "```text",
            model_text,
            "```",
            "",
            "## Checkpoints",
            "",
            "```text",
            *ckpts,
            "```",
            "",
        ]
    )
    with open(paths["model_detail_path"], "w", encoding="utf-8") as f:
        f.write(content)
