import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def make_output_dir(results, data, model_id):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out = Path(results) / data / model_id / timestamp
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "details").mkdir(parents=True, exist_ok=True)
    return out


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, ensure_ascii=False)


def aggregate_runs(run_summaries):
    values = {}
    for run in run_summaries:
        for key, value in run["test_metrics"].items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                values.setdefault(key, []).append(float(value))
    return {
        key: {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "values": vals,
        }
        for key, vals in values.items()
    }


def _fmt_metric(value):
    if isinstance(value, (int, float, np.generic)) and np.isfinite(value):
        return f"{float(value):.6f}"
    return "nan"


def write_run_md(path, title, args, run_summaries, aggregate, extra_config_lines=None):
    monitor_metric = getattr(args, "monitor_metric", "RankIC")
    lines = [
        f"# {title}",
        "",
        f"- root_path: `{args.root_path}`",
        f"- num_runs: `{args.num_runs}`",
        f"- seq_len: `{args.seq_len}`",
        f"- batch_size: `{args.batch_size}`",
    ]
    if extra_config_lines:
        lines.extend(extra_config_lines)

    lines.extend(
        [
            "",
            "## Runs",
            "",
            f"| run | seed | best_epoch | best_valid_{monitor_metric} | MAE | RankIC | RankICIR | IRR | SR |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in run_summaries:
        test = row["test_metrics"]
        lines.append(
            "| {run} | {seed} | {best_epoch} | {best_valid} | {mae} | {rankic} | {rankicir} | {irr} | {sr} |".format(
                run=row["run_id"] + 1,
                seed=row["seed"],
                best_epoch=row["best_epoch"],
                best_valid=_fmt_metric(row["best_valid"].get(monitor_metric, np.nan)),
                mae=_fmt_metric(test.get("MAE", test.get("mae", np.nan))),
                rankic=_fmt_metric(test.get("RankIC", np.nan)),
                rankicir=_fmt_metric(test.get("RankICIR", np.nan)),
                irr=_fmt_metric(test.get("IRR", test.get("IRR@10", np.nan))),
                sr=_fmt_metric(test.get("SR", np.nan)),
            )
        )
    lines.extend(["", "## Aggregate", "", "```json", json.dumps(json_safe(aggregate), indent=2), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")

