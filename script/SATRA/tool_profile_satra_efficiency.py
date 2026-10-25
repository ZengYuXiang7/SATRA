#!/usr/bin/env python3
"""Profile PATRA-Net's supervised path without editing the model code.

The run uses the saved formal NASDAQ configuration, restricts the runner to one
native supervised training batch, and measures one warmed native test forward
pass at batch size one. Formal training time is extracted from the existing
training logs. MMR pretraining is retained in the formal-total calculation
because it was run once for each formal seed, but it is not part of inference.
"""

from __future__ import annotations

import argparse
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
        option = next((item for item in action.option_strings if item.startswith("--")), None)
        if option is None:
            continue
        if isinstance(action, argparse._StoreTrueAction):
            if bool(value):
                argv.append(option)
            continue
        if isinstance(action, argparse._StoreFalseAction):
            if not bool(value):
                argv.append(option)
            continue
        argv.extend([option, _format_value(action, value)])
    return argv


def _cuda_snapshot() -> dict[str, float] | None:
    if not torch.cuda.is_available():
        return None
    return {
        "peak_allocated_gib": float(torch.cuda.max_memory_allocated() / (1024**3)),
        "peak_reserved_gib": float(torch.cuda.max_memory_reserved() / (1024**3)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile PATRA-Net efficiency")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile-results", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260722)
    cli = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    src_root = project_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    config_path = cli.config if cli.config.is_absolute() else project_root / cli.config
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    from stock import main as stock_main

    native_parser = stock_main.build_parser()
    actions = {action.dest for action in native_parser._actions}
    overrides = {
        "num_runs": 1,
        "train_epochs": 1,
        "batch_size": 1,
        "debug": 1,
        "debug_train_steps": 1,
        "debug_eval_steps": 1,
        "results": str(cli.profile_results),
        "checkpoints": str(cli.profile_results / "checkpoints"),
        "random_seed": int(cli.seed),
        "model_id": "efficiency_patra",
    }
    missing = sorted(set(overrides) - actions)
    if missing:
        raise ValueError(f"PATRA parser does not expose required profile options: {missing}")
    native_argv = _native_argv(native_parser, config, overrides)

    records: dict[str, Any] = {
        "status": "running",
        "formal_config": str(config_path),
        "native_argv": native_argv,
        "protocol": {
            "profiling_training": "one native supervised training batch at batch size one",
            "formal_training_time": "extracted from existing formal training logs",
            "inference": "one warmed native test forward pass at batch size one",
            "mmr": "included in Total Train from the corresponding formal seed logs",
            "batch_size": 1,
            "profiling_runtime": "uncompiled debug microbenchmark, matching generic forecaster rows",
        },
    }
    original_experiment = stock_main.Exp_Main

    class InstrumentedExperiment(original_experiment):
        def __init__(self, args: argparse.Namespace) -> None:
            super().__init__(args)
            self._profile_training_active = False
            self._profile_test_active = False
            self._active_parameter_sum = 0.0
            self._active_parameter_count = 0
            records["test_forward_calls"] = []
            records["trainable_parameters"] = int(
                sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
            )
            records["total_parameters"] = int(sum(parameter.numel() for parameter in self.model.parameters()))
            model = self._checkpoint_model()
            moe = model.spe.adaptive_bottleneck
            self._expert_parameter_counts = torch.tensor(
                [
                    sum(parameter.numel() for parameter in expert.parameters())
                    for expert in moe.bottlenecks
                ],
                device=self.device,
                dtype=torch.float64,
            )
            records["fixed_parameters"] = int(
                records["total_parameters"] - int(self._expert_parameter_counts.sum().item())
            )
            records["expert_parameter_counts"] = [
                int(value) for value in self._expert_parameter_counts.detach().cpu().tolist()
            ]
            records["active_parameter_protocol"] = (
                "fixed model weights plus the Top-k selected ACE experts, averaged over stock windows "
                "during the test evaluation"
            )

        def _record_active_parameters(self) -> None:
            moe = self._checkpoint_model().spe.adaptive_bottleneck
            selection = moe._last_topk_selection
            if selection is None:
                raise RuntimeError("ACE routing selection was unavailable after the forward pass")
            active_expert_parameters = selection.to(dtype=torch.float64) @ self._expert_parameter_counts
            self._active_parameter_sum += float(active_expert_parameters.sum().item())
            self._active_parameter_count += int(active_expert_parameters.numel())

        def _predict_batch(self, tensor_batch: dict[str, torch.Tensor]) -> torch.Tensor:
            if self._profile_test_active and torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter() if self._profile_test_active else None
            prediction = super()._predict_batch(tensor_batch)
            if self._profile_test_active:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                records["test_forward_calls"].append(
                    {"seconds": float(time.perf_counter() - started)}
                )
                self._record_active_parameters()
            return prediction

        def vali(self, *args: Any, **kwargs: Any) -> Any:
            if self._profile_training_active:
                if "training_cuda_memory" not in records:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    records["training_cuda_memory"] = _cuda_snapshot()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                started = time.perf_counter()
                result = super().vali(*args, **kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                records["validation_seconds"] = float(time.perf_counter() - started)
                return result
            return super().vali(*args, **kwargs)

        def train(self, setting: str) -> Any:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            self._profile_training_active = True
            try:
                result = super().train(setting)
            finally:
                self._profile_training_active = False
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            full_cycle_seconds = float(time.perf_counter() - started)
            records["train_cycle_seconds"] = full_cycle_seconds
            records["train_batch_seconds"] = full_cycle_seconds - float(
                records.get("validation_seconds", 0.0)
            )
            if "training_cuda_memory" not in records:
                records["training_cuda_memory"] = _cuda_snapshot()
            return result

        def test(self, setting: str, test: int = 0) -> Any:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            self._active_parameter_sum = 0.0
            self._active_parameter_count = 0
            self._profile_test_active = True
            try:
                result = super().test(setting, test=test)
            finally:
                self._profile_test_active = False
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            records["test_evaluation_seconds"] = float(time.perf_counter() - started)
            if self._active_parameter_count <= 0:
                raise RuntimeError("No ACE routing selections were recorded during test evaluation")
            if not records["test_forward_calls"]:
                raise RuntimeError("No PATRA-Net test forward pass was recorded")
            records["test_forward_seconds"] = records["test_forward_calls"][-1]["seconds"]
            records["average_active_parameters"] = float(
                records["fixed_parameters"]
                + self._active_parameter_sum / self._active_parameter_count
            )
            return result

    stock_main.Exp_Main = InstrumentedExperiment
    sys.argv = [str(config_path)] + native_argv
    cli.profile_results.mkdir(parents=True, exist_ok=True)
    stock_main.main()

    records["status"] = "complete"
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
