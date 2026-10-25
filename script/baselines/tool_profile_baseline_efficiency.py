#!/usr/bin/env python3
"""Profile one native financial stock baseline without changing its runner.

The script reuses the baseline's saved formal configuration, overrides only the
output location, run length, and measurement batch size, and instruments the
existing ``run.py`` functions. Debug controls restrict the runner to one native
training batch and one native test batch at batch size one; formal total
training time is extracted from the original training logs. It deliberately
does not alter the baseline model, loss, precision, or compilation settings.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


def _primitive(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _format_value(action: argparse.Action, value: Any) -> str:
    if isinstance(value, bool) and getattr(action, "type", None) is int:
        return "1" if value else "0"
    return str(value)


def _capture_parser(module: Any, config: dict[str, Any]) -> argparse.ArgumentParser:
    """Extract the native runner's parser without depending on private code."""

    captured: dict[str, argparse.ArgumentParser] = {}
    original_parse_args = argparse.ArgumentParser.parse_args

    def capture_parse_args(
        parser: argparse.ArgumentParser,
        args: list[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        captured["parser"] = parser
        required_args: list[str] = []
        for action in parser._actions:
            if not action.required:
                continue
            if action.dest not in config:
                raise KeyError(
                    f"Formal config does not provide required argument {action.dest!r}"
                )
            option = next((item for item in action.option_strings if item.startswith("--")), None)
            if option is None:
                raise ValueError(f"No long option found for required argument {action.dest!r}")
            required_args.extend([option, _format_value(action, config[action.dest])])
        return original_parse_args(parser, required_args, namespace)

    argparse.ArgumentParser.parse_args = capture_parse_args
    try:
        module.parse_args()
    finally:
        argparse.ArgumentParser.parse_args = original_parse_args
    if "parser" not in captured:
        raise RuntimeError("Unable to capture the native runner argument parser")
    return captured["parser"]


def _native_argv(
    parser: argparse.ArgumentParser,
    config: dict[str, Any],
    overrides: dict[str, Any],
) -> list[str]:
    values = {**config, **overrides}
    argv: list[str] = []
    for action in parser._actions:
        if action.dest == "help" or not action.option_strings or action.dest not in values:
            continue
        value = values[action.dest]
        if value is None or not _primitive(value):
            continue
        long_option = next((item for item in action.option_strings if item.startswith("--")), None)
        if long_option is None:
            continue
        if isinstance(action, argparse._StoreTrueAction):
            if bool(value):
                argv.append(long_option)
            continue
        if isinstance(action, argparse._StoreFalseAction):
            if not bool(value):
                argv.append(long_option)
            continue
        argv.extend([long_option, _format_value(action, value)])
    return argv


def _cuda_snapshot() -> dict[str, float] | None:
    if not torch.cuda.is_available():
        return None
    return {
        "peak_allocated_gib": float(torch.cuda.max_memory_allocated() / (1024**3)),
        "peak_reserved_gib": float(torch.cuda.max_memory_reserved() / (1024**3)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a native financial stock baseline")
    parser.add_argument("--module", required=True, help="Native runner module, e.g. master.run")
    parser.add_argument("--config", required=True, type=Path, help="Saved formal config.json")
    parser.add_argument("--output", required=True, type=Path, help="Output profile JSON")
    parser.add_argument(
        "--profile-results",
        required=True,
        type=Path,
        help="Isolated results root used by the one-batch profiling run",
    )
    parser.add_argument("--seed", type=int, default=20260722)
    cli = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    src_root = project_root / "src"
    if not src_root.is_dir():
        raise FileNotFoundError(f"Missing baseline source directory: {src_root}")
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    config_path = cli.config if cli.config.is_absolute() else project_root / cli.config
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    module = importlib.import_module(cli.module)
    native_parser = _capture_parser(module, config)
    native_action_dests = {action.dest for action in native_parser._actions}
    required_overrides = {
        "num_runs": 1,
        "train_epochs": 1,
        "batch_size": 1,
        "debug": 1,
        "debug_train_steps": 1,
        "debug_eval_steps": 1,
        "results": str(cli.profile_results),
        "random_seed": int(cli.seed),
    }
    missing = sorted(set(required_overrides) - native_action_dests)
    if missing:
        raise ValueError(f"Native runner does not expose required profile options: {missing}")
    overrides = dict(required_overrides)
    if "model_id" in native_action_dests:
        overrides["model_id"] = f"efficiency_{config.get('model', cli.module.split('.')[0])}"
    native_argv = _native_argv(native_parser, config, overrides)

    records: dict[str, Any] = {
        "status": "running",
        "module": cli.module,
        "formal_config": str(config_path),
        "native_argv": native_argv,
        "protocol": {
            "profiling_training": "one native training batch at batch size one",
            "formal_training_time": "extracted from existing formal training logs",
            "inference": "one warmed native test forward pass at batch size one",
            "batch_size": 1,
            "weights": "architecture-level measurement; timings do not depend on learned checkpoint values",
        },
        "train_batch_calls": [],
        "evaluation_calls": [],
        "evaluation_forward_calls": [],
    }

    original_build_model = module.build_model
    original_train_epoch = module.train_epoch
    original_eval_model = module.eval_model

    def instrumented_build_model(*args: Any, **kwargs: Any) -> torch.nn.Module:
        model = original_build_model(*args, **kwargs)
        records["trainable_parameters"] = int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        )
        records["total_parameters"] = int(sum(parameter.numel() for parameter in model.parameters()))
        records["average_active_parameters"] = records["total_parameters"]
        records["active_parameter_protocol"] = (
            "all model weights are active for each forward pass; the model has no conditional experts"
        )
        original_forward = model.forward

        def instrumented_forward(*forward_args: Any, **forward_kwargs: Any) -> Any:
            if not records.get("_measure_evaluation_forward", False):
                return original_forward(*forward_args, **forward_kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            result = original_forward(*forward_args, **forward_kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            records["evaluation_forward_calls"].append(
                {"seconds": float(time.perf_counter() - started)}
            )
            return result

        model.forward = instrumented_forward
        return model

    def instrumented_train_epoch(*args: Any, **kwargs: Any) -> Any:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        result = original_train_epoch(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        records["train_batch_calls"].append(
            {"seconds": float(time.perf_counter() - started), "cuda_memory": _cuda_snapshot()}
        )
        return result

    def instrumented_eval_model(*args: Any, **kwargs: Any) -> Any:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        records["_measure_evaluation_forward"] = True
        try:
            result = original_eval_model(*args, **kwargs)
        finally:
            records.pop("_measure_evaluation_forward", None)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        records["evaluation_calls"].append({"seconds": float(time.perf_counter() - started)})
        return result

    module.build_model = instrumented_build_model
    module.train_epoch = instrumented_train_epoch
    module.eval_model = instrumented_eval_model
    sys.argv = [str(config_path)] + native_argv
    cli.profile_results.mkdir(parents=True, exist_ok=True)
    module.main()

    records["status"] = "complete"
    if len(records["train_batch_calls"]) != 1:
        raise RuntimeError(
            "Expected exactly one instrumented training batch, got "
            f"{len(records['train_batch_calls'])}"
        )
    records["train_batch_seconds"] = records["train_batch_calls"][0]["seconds"]
    records["training_cuda_memory"] = records["train_batch_calls"][0]["cuda_memory"]
    if not records["evaluation_calls"]:
        raise RuntimeError("Native baseline runner did not execute any evaluation call")
    if not records["evaluation_forward_calls"]:
        raise RuntimeError("Native baseline runner did not execute an evaluation forward pass")
    records["test_forward_seconds"] = records["evaluation_forward_calls"][-1]["seconds"]
    records["gpu"] = (
        torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
    )
    output_path = cli.output if cli.output.is_absolute() else project_root / cli.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    print(f"EFFICIENCY_PROFILE_JSON={output_path}", flush=True)


if __name__ == "__main__":
    main()
