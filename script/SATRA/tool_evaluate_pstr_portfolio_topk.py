#!/usr/bin/env python3
"""Re-evaluate saved PSTR-Net checkpoints for multiple portfolio Top-K values."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from satra.exp.exp_main import Exp_Main
from satra.main import build_parser, prepare_runtime_args, set_random_seed
from satra.utils.metrics import _portfolio_return_stats, evaluate
from satra.utils.stock_loss import stockmixer_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--topks", type=int, nargs="+", required=True)
    parser.add_argument("--num-runs", type=int, default=5)
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_eval_args(config, run_id, seed):
    parser = build_parser()
    args = parser.parse_args(
        [
            "--is_training",
            "0",
            "--model_id",
            "portfolio_topk_eval",
            "--model",
            config["model"],
            "--data",
            config["data"],
        ]
    )
    for key, value in config.items():
        if hasattr(args, key) and value is not None:
            setattr(args, key, value)

    args.run_id = int(run_id)
    args.random_seed = int(seed)
    args.obs_logger = None
    args.use_gpu = 1
    args.use_multi_gpu = False
    args.gpu = 0
    args.torch_compile = 0
    args.num_workers = 0
    args.pin_memory = 1
    return prepare_runtime_args(args)


def configure_deterministic_inference():
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def collect_test_predictions(exp):
    start_index = int(exp.data.test_index)
    end_index = int(exp.data.trade_dates)
    date_count = end_index - start_index
    prediction = np.zeros((exp.data.stock_num, date_count), dtype=np.float32)
    ground_truth = np.zeros_like(prediction)
    mask = np.zeros_like(prediction)
    eval_start_offset = start_index - exp.args.seq_len - exp.args.steps + 1
    observed_columns = np.zeros(date_count, dtype=np.int64)

    exp.model.eval()
    with torch.no_grad():
        for batch in exp.data.eval_loader(start_index, end_index, drop_last=False):
            tensor_batch = exp._to_tensor_batch(batch)
            mask_batch = tensor_batch["mask"]
            price_batch = tensor_batch["price"]
            target_batch = tensor_batch["y_endo"]
            raw_gt_batch = tensor_batch["raw_gt"]
            with exp._autocast_context():
                model_output = exp._predict_batch(tensor_batch)
            _, _, _, return_ratio = stockmixer_loss(
                model_output.float(),
                target_batch,
                price_batch,
                mask_batch,
                exp.data.stock_num,
                exp.args.alpha,
                exp.args.prediction_target,
            )
            columns = exp._offsets_to_numpy(tensor_batch["offset"]) - eval_start_offset
            if np.any(columns < 0) or np.any(columns >= date_count):
                raise IndexError(f"Evaluation column outside [0,{date_count}): {columns.tolist()}")
            prediction[:, columns] = return_ratio.squeeze(-1).detach().cpu().numpy().T
            ground_truth[:, columns] = raw_gt_batch.squeeze(-1).detach().cpu().numpy().T
            mask[:, columns] = mask_batch.squeeze(-1).detach().cpu().numpy().T
            observed_columns[columns] += 1

    if not np.all(observed_columns == 1):
        raise RuntimeError(
            "Test prediction coverage must contain every date exactly once; "
            f"coverage counts={np.unique(observed_columns, return_counts=True)}"
        )
    return prediction, ground_truth, mask


def evaluate_topks(prediction, ground_truth, mask, topks):
    valid_mask = (mask >= 0.5) & np.isfinite(prediction) & np.isfinite(ground_truth)
    results = {}
    for topk in topks:
        daily_returns = []
        for date_index in range(prediction.shape[1]):
            valid = valid_mask[:, date_index]
            valid_count = int(np.sum(valid))
            if valid_count < topk:
                raise ValueError(
                    f"Top-K={topk} requires {topk} valid stocks, but date column "
                    f"{date_index} contains {valid_count}"
                )
            pred_day = prediction[valid, date_index]
            gt_day = ground_truth[valid, date_index]
            selected_order = np.argsort(pred_day, kind="mergesort")[-topk:][::-1]
            daily_returns.append(float(np.mean(gt_day[selected_order])))

        irr, annualized_return, sharpe = _portfolio_return_stats(daily_returns)
        if not np.isfinite(irr) or not np.isfinite(sharpe):
            raise FloatingPointError(
                f"Non-finite portfolio metric for Top-K={topk}: IRR={irr}, SR={sharpe}"
            )
        results[str(topk)] = {
            f"IRR@{topk}": float(irr),
            f"SR@{topk}": float(sharpe),
            f"AR@{topk}": float(annualized_return),
            "test_days": len(daily_returns),
        }
    return results


def verify_top10(recomputed, native_evaluation, recorded, run_id):
    checks = {
        "contract_source": "satra.utils.metrics.evaluate on the same deterministic predictions",
        "historical_reference_note": (
            "Historical values used the original non-deterministic CUDA attention backend and are "
            "recorded for reference, not used as an equality assertion."
        ),
    }
    for name in ("IRR@10", "SR@10"):
        actual = float(recomputed["10"][name])
        native = float(native_evaluation[name])
        historical = float(recorded[name])
        contract_difference = actual - native
        if not np.isclose(actual, native, rtol=1e-6, atol=1e-8):
            raise AssertionError(
                f"Run {run_id + 1} {name} contract mismatch: multi_topk={actual}, "
                f"native_evaluate={native}, difference={contract_difference}"
            )
        checks[name] = {
            "multi_topk": actual,
            "native_evaluate": native,
            "contract_difference": contract_difference,
            "historical_recorded": historical,
            "historical_difference": actual - historical,
        }
    return checks


def aggregate_runs(run_results, topks):
    aggregate = {}
    for topk in topks:
        key = str(topk)
        aggregate[key] = {}
        for metric in (f"IRR@{topk}", f"SR@{topk}"):
            values = np.asarray(
                [run["metrics"][key][metric] for run in run_results], dtype=np.float64
            )
            aggregate[key][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "values": values.tolist(),
            }
    return aggregate


def main():
    cli_args = parse_args()
    topks = sorted(set(cli_args.topks))
    if topks != cli_args.topks:
        raise ValueError("--topks must be unique and strictly increasing")
    if any(topk <= 0 for topk in topks):
        raise ValueError("Every Top-K value must be positive")
    if 10 not in topks:
        raise ValueError("--topks must include 10 to verify the existing metric contract")

    experiment_dir = cli_args.experiment_dir.resolve()
    config_path = experiment_dir / "config.json"
    summary_path = experiment_dir / "summary_all_runs.json"
    config = load_json(config_path)
    summary = load_json(summary_path)
    if int(summary["num_runs"]) != cli_args.num_runs:
        raise ValueError(
            f"Expected {cli_args.num_runs} runs, found {summary['num_runs']} in {summary_path}"
        )
    if len(summary["runs"]) != cli_args.num_runs:
        raise ValueError(
            f"Expected {cli_args.num_runs} run records, found {len(summary['runs'])}"
        )

    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    configure_deterministic_inference()
    run_results = []
    stock_num = None
    test_days = None
    for run_record in summary["runs"]:
        run_id = int(run_record["run_id"])
        seed = int(run_record["seed"])
        checkpoint_path = experiment_dir / "checkpoints" / f"best_model_run{run_id}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        print(
            f"[{config['data']}] checkpoint {run_id + 1}/{cli_args.num_runs} "
            f"seed={seed} path={checkpoint_path}",
            flush=True,
        )
        run_started = time.time()
        set_random_seed(seed)
        model_args = build_eval_args(config, run_id, seed)
        exp = Exp_Main(model_args)
        state_dict = torch.load(checkpoint_path, map_location=exp.device, weights_only=True)
        exp._checkpoint_model().load_state_dict(state_dict, strict=True)
        prediction, ground_truth, mask = collect_test_predictions(exp)
        metrics = evaluate_topks(prediction, ground_truth, mask, topks)
        native_evaluation = evaluate(
            prediction,
            ground_truth,
            mask,
            stock_num=exp.data.stock_num,
        )
        top10_check = verify_top10(
            metrics,
            native_evaluation,
            run_record["test_metrics"],
            run_id,
        )
        elapsed = time.time() - run_started
        stock_num = int(exp.data.stock_num)
        test_days = int(prediction.shape[1])
        run_results.append(
            {
                "run_id": run_id,
                "seed": seed,
                "best_epoch": int(run_record["best_epoch"]),
                "checkpoint": str(checkpoint_path),
                "metrics": metrics,
                "top10_contract_check": top10_check,
                "inference_seconds": elapsed,
            }
        )
        print(
            f"[{config['data']}] completed checkpoint {run_id + 1}/{cli_args.num_runs} "
            f"IRR@10={metrics['10']['IRR@10']:.6f} "
            f"SR@10={metrics['10']['SR@10']:.6f} elapsed={elapsed:.1f}s",
            flush=True,
        )
        del exp, state_dict, prediction, ground_truth, mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "started_at": started_at,
        "source_experiment_dir": str(experiment_dir),
        "source_config": str(config_path),
        "source_summary": str(summary_path),
        "dataset": config["data"],
        "model": config["model"],
        "variant": "full PSTR-Net with MMR",
        "topks": topks,
        "num_runs": cli_args.num_runs,
        "stock_num": stock_num,
        "test_days": test_days,
        "metric_definition": {
            "IRR@K": "Cumulative return of the daily equal-weight portfolio formed from the K highest predictions.",
            "SR@K": "Annualized Sharpe ratio of the same daily portfolio using sqrt(252).",
        },
        "inference_protocol": {
            "precision": config["precision"],
            "deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cuda_attention_backend": "math",
            "historical_top10_values": "reference only because the original CUDA attention backend was non-deterministic",
        },
        "runs": run_results,
        "aggregate": aggregate_runs(run_results, topks),
    }
    cli_args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with cli_args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"Saved {cli_args.output_json.resolve()}", flush=True)


if __name__ == "__main__":
    main()
