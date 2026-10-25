"""Command-line entry point for supervised SATRA experiments.

The runner creates independent seeded runs, delegates one run at a time to
``Exp_Main``, and writes structured per-run plus aggregate reports. Validation
RankIC is the only checkpoint-selection signal; test evaluation is optional
and occurs only after the best validation checkpoint has been fixed."""

import argparse
import copy
import random
import secrets
import sys
from datetime import datetime

import numpy as np
import torch

from satra.exp.exp_main import Exp_Main
from satra.observability.exp_logger import ExpLogger
from satra.observability.exp_paths import build_exp_paths, run_paths
from satra.observability.exp_report import (
    append_project_run_md,
    format_aggregate_report,
    save_config,
    save_model_detail,
    save_run_md,
    save_summary_all_runs,
    update_config_runtime,
)
from satra.utils.metrics import PORTFOLIO_SIZE


def build_parser():
    parser = argparse.ArgumentParser(description="Stock forecasting model")

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Base seed. When omitted, one random base seed is generated and recorded.",
    )
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument(
        "--evaluate_test",
        type=int,
        default=1,
        choices=[0, 1],
        help=(
            "Whether to evaluate the test split after selecting the best checkpoint "
            "on validation data. Hyperparameter searches must set this to 0."
        ),
    )
    parser.add_argument(
        "--portfolio_size",
        type=int,
        default=PORTFOLIO_SIZE,
        choices=[PORTFOLIO_SIZE],
        help=f"Fixed paper portfolio protocol: select the top N={PORTFOLIO_SIZE} stocks each day.",
    )
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--debug_num_runs", type=int, default=1)
    parser.add_argument("--print_freq", type=int, default=1)
    parser.add_argument("--debug_train_steps", type=int, default=8)
    parser.add_argument("--debug_eval_steps", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=int, default=1)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", type=int, default=0)
    parser.add_argument("--train_drop_last", type=int, default=1)
    parser.add_argument("--valid_drop_last", type=int, default=0)
    parser.add_argument("--is_training", type=int, required=True, default=1)
    parser.add_argument("--model_id", type=str, required=True, default="model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="model",
        choices=[
            "model_a_shared_temporal_encoder",
            "model_a_homogeneous_experts",
            "model_a_spe",
            "model_b_spe_only",
            "model_b_spe_semantic_tower",
            "model_b_spe_prior_guided_tower",
            "model_b_spe_drt",
            "model_b_spe_drt_learnable_fusion",
            "model_b_spe_drt_feature_aggregation",
            "model_c_no_prior_constraint",
            "model_c_industry_relation_prior",
            "model_c_euclidean_distance_prior",
            "model_c_cosine_distance_prior",
            "model_c_dtw_based_prior",
            "model_c_learned_adjacency",
            "model_c_adaptive_graph",
        ],
    )

    parser.add_argument("--data", type=str, required=True, choices=["NASDAQ", "NYSE", "SP500"])
    parser.add_argument("--root_path", type=str, default="./data/processed")
    parser.add_argument("--checkpoints", type=str, default="./results/checkpoints")
    parser.add_argument("--results", type=str, default="./results/main")

    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--features", type=int, default=5)
    parser.add_argument(
        "--input_features",
        type=str,
        default="all",
        help="Feature subset used by the model: all, raw_ohlcv, or comma-separated feature names.",
    )
    parser.add_argument(
        "--use_grouped_inputs",
        type=int,
        default=0,
        help="Legacy data-loader switch. PSTR-Net always consumes the dataset's complete inputs tensor.",
    )
    parser.add_argument("--stockmixer_ma_windows", type=str, default="5,10,20,30")
    parser.add_argument("--market_num", type=int, default=20)
    parser.add_argument("--scale_factor", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--output_scale", type=float, default=0.2)
    parser.add_argument("--output_activation", type=str, default="tanh", choices=["tanh", "identity"])
    parser.add_argument(
        "--target_mode",
        type=str,
        default="raw_return",
        choices=["raw_return"],
        help="Training label uses gt_data raw returns.",
    )
    parser.add_argument(
        "--prediction_target",
        type=str,
        default="return",
        choices=["price", "return"],
        help="Model output contract: price is converted to return in loss; return is used directly.",
    )
    parser.add_argument("--dlinear_hidden", type=int, default=64)
    parser.add_argument("--factor_dropout", type=float, default=0.1)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--factor_num", type=int, default=20)
    parser.add_argument("--dlinear_stock_hidden", type=int, default=20)
    parser.add_argument("--dlinear_dropout", type=float, default=0.0)
    parser.add_argument("--close_feature_index", type=int, default=-1)
    parser.add_argument("--ret_feature_index", type=int, default=-1)
    parser.add_argument("--dlinear_residual_scale", type=float, default=0.1)
    parser.add_argument("--stock_prior", type=str, default="dtw", choices=["none", "dtw"])
    parser.add_argument("--stock_prior_path", type=str, default="")
    parser.add_argument("--stock_prior_cache_dir", type=str, default="")
    parser.add_argument("--stock_prior_feature", type=str, default="close")
    parser.add_argument("--stock_prior_topk", type=int, default=16)
    parser.add_argument("--stock_prior_candidate_k", type=int, default=48)
    parser.add_argument("--stock_prior_downsample", type=int, default=64)
    parser.add_argument("--stock_prior_band", type=int, default=6)
    parser.add_argument("--stock_prior_scale", type=float, default=1.0)
    parser.add_argument("--stock_prior_force", type=int, default=0)
    parser.add_argument(
        "--adaptive_graph_topk",
        type=int,
        default=32,
        help=(
            "Per-stock neighbor count selected by the state-adaptive learned graph. "
            "It is independent of every fixed relation prior."
        ),
    )
    parser.add_argument("--moe_top_k", type=int, default=3, help="State-conditioned expert routing top-k.")
    parser.add_argument(
        "--moe_load_balance_weight",
        type=float,
        default=0.0,
        help="Coefficient of the stage-2 MoE importance-times-load balance loss; zero disables it.",
    )
    parser.add_argument("--d_model", type=int, default=64, help="PSTR-Net intermediate feature dimension.")
    parser.add_argument("--drt_load_balance_weight", type=float, default=0.001, help="Two-expert aggregation importance CV-squared weight in both training stages.")
    parser.add_argument("--pstr_depth", type=int, default=10, help="Number of residual temporal CNN blocks minus one.")
    parser.add_argument("--pstr_kernel_size", type=int, default=3)
    parser.add_argument("--pstr_num_heads", type=int, default=4)
    parser.add_argument("--pstr_tower_layers", type=int, default=1)
    parser.add_argument("--pstr_dropout", type=float, default=0.1)
    parser.add_argument(
        "--semantic_support",
        type=str,
        default="global",
        choices=["global", "dtw_prior"],
        help=(
            "Visible relation support for the semantic DRT tower. "
            "dtw_prior reuses the fixed DTW mask in the controlled "
            "prior-support restriction experiment."
        ),
    )
    parser.add_argument(
        "--pstr_relation_path",
        type=str,
        default="",
        help="Exact binary relation .npz used by PSTR prior-guided variants.",
    )
    parser.add_argument("--mae_mask_ratio", type=float, default=0.3)
    parser.add_argument("--pretrained_encoder_path", type=str, default="")
    parser.add_argument("--pretrained_backbone_path", type=str, default="")

    parser.add_argument("--valid_index", type=int, default=756)
    parser.add_argument("--test_index", type=int, default=1008)
    parser.add_argument("--data_start_index", type=int, default=0)

    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--min_epochs", type=int, default=0)
    parser.add_argument("--min_updates", type=int, default=0)
    parser.add_argument("--early_stop_metric", type=str, default="RankIC", choices=["RankIC"])
    parser.add_argument("--early_stop_mode", type=str, default="max", choices=["max"])
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine_hard_restarts",
        choices=[
            "plateau",
            "step",
            "cosine",
            "cosine_hard_restarts",
            "linear",
            "none",
            "type1",
            "type2",
            "type3",
            "constant",
            "3",
            "4",
            "5",
            "6",
        ],
    )
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--num_cycles", type=int, default=1)
    parser.add_argument("--min_ratio", type=float, default=1e-2)
    parser.add_argument("--lr_step_size", type=int, default=10)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    parser.add_argument("--use_gpu", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0,1")
    parser.add_argument("--precision", type=str, default="bf16_amp_tf32", choices=["fp32", "tf32", "bf16_amp_tf32"])
    parser.add_argument("--torch_compile", type=int, default=1)
    parser.add_argument("--torch_compile_mode", type=str, default="reduce-overhead")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    return parser


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_runtime_args(args):
    args.portfolio_size = int(args.portfolio_size)
    if args.portfolio_size != PORTFOLIO_SIZE:
        raise ValueError(
            f"portfolio_size must equal the paper protocol N={PORTFOLIO_SIZE}, "
            f"got {args.portfolio_size}"
        )
    args.pin_memory_requested = bool(args.pin_memory)
    if bool(args.use_gpu):
        args.device_type = "cuda"
        args.use_gpu = True
    else:
        args.device_type = "cpu"
        args.use_gpu = False
        args.use_multi_gpu = False
    args.debug = bool(args.debug)
    if args.debug:
        args.train_epochs = min(args.train_epochs, 1)
    args.patience_requested = int(args.patience)
    args.patience = int(args.patience)
    args.batch_size = int(args.batch_size)
    args.num_workers = int(args.num_workers)
    args.pin_memory = bool(args.pin_memory) and args.device_type == "cuda"
    args.persistent_workers = bool(args.persistent_workers)
    args.train_drop_last = bool(args.train_drop_last)
    args.valid_drop_last = bool(args.valid_drop_last)
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        args.device_ids = [int(device) for device in args.devices.split(",")]
        args.gpu = args.device_ids[0]
    return args


def build_setting(args):
    return "{}_{}_{}_sl{}_run{}_seed{}".format(
        args.model_id, args.model, args.data, args.seq_len, args.run_id, args.random_seed
    )


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def aggregate_runs(run_summaries):
    metric_values = {}
    for summary in run_summaries:
        for key, value in summary["test_metrics"].items():
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                metric_values.setdefault(key, []).append(float(value))

    aggregate = {}
    for key, values in metric_values.items():
        values_array = np.asarray(values, dtype=np.float64)
        aggregate[key] = {
            "mean": float(np.mean(values_array)),
            "std": float(np.std(values_array, ddof=1)) if values_array.size > 1 else 0.0,
            "values": values,
        }
    return aggregate


def format_metric_values(metrics):
    parts = []
    for name, value in metrics.items():
        if isinstance(value, dict):
            if "value" in value:
                value = value["value"]
            elif "values" in value:
                values = value["values"]
                joined = ",".join("{:.6f}".format(float(item)) for item in values)
                parts.append("{}=[{}]".format(name, joined))
                continue
            elif "mean" in value:
                value = value["mean"]
        if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
            parts.append("{}={:.6f}".format(name, float(value)))
        else:
            parts.append("{}={}".format(name, value))
    return ", ".join(parts)


def run_once(base_args, run_id, paths):
    args = copy.deepcopy(base_args)
    args.run_id = run_id
    args.total_runs = base_args.num_runs
    args.completed_run_summaries = base_args.completed_run_summaries
    args.random_seed = base_args.random_seed + run_id
    set_random_seed(args.random_seed)
    prepare_runtime_args(args)
    args.obs_paths = paths
    args.obs_run_paths = run_paths(paths, run_id)
    logger = ExpLogger(args.obs_run_paths["detail_log"])
    args.obs_logger = logger

    logger.log("========== Run {}/{} ==========".format(run_id + 1, base_args.num_runs))
    logger.log("Experiment Config")
    logger.log(
        "model_id={} model={} data={} seed={} run_id={} num_runs={} debug={} "
        "seq_len={} batch_size={} lr={} epochs={} patience={} "
        "min_epochs={} min_updates={} monitor={} precision={} compile={} mode={} max_grad_norm={} "
        "moe_load_balance_weight={}".format(
            args.model_id,
            args.model,
            args.data,
            args.random_seed,
            args.run_id,
            base_args.num_runs,
            args.debug,
            args.seq_len,
            args.batch_size,
            args.learning_rate,
            args.train_epochs,
            args.patience,
            args.min_epochs,
            args.min_updates,
            args.early_stop_metric,
            args.precision,
            bool(args.torch_compile),
            args.torch_compile_mode,
            args.max_grad_norm,
            args.moe_load_balance_weight,
        )
    )

    setting = build_setting(args)

    exp = Exp_Main(args)
    logger.log("==================== Training Start ====================")
    logger.log(f"setting={setting}")
    train_summary = exp.train(setting)
    if bool(args.evaluate_test):
        logger.log("==================== Test Start ====================")
        logger.log(f"setting={setting}")
        test_perf = exp.test(setting)
    else:
        logger.log("==================== Test Skipped ====================")
        logger.log("reason=validation-only hyperparameter selection")
        test_perf = {}

    valid_value = train_summary["best_valid"][args.early_stop_metric]
    if isinstance(valid_value, (int, float, np.integer, np.floating)) and np.isfinite(valid_value):
        valid_value = "{:.6f}".format(float(valid_value))
    test_text = format_metric_values(test_perf)
    if test_text:
        round_line = "Round={} BestEpoch={} Valid{}={} | {} time={:.1f}s".format(
            run_id + 1,
            train_summary["best_epoch"],
            args.early_stop_metric,
            valid_value,
            test_text,
            train_summary["train_seconds"],
        )
    else:
        round_line = "Round={} BestEpoch={} Valid{}={} | test=skipped time={:.1f}s".format(
            run_id + 1,
            train_summary["best_epoch"],
            args.early_stop_metric,
            valid_value,
            train_summary["train_seconds"],
        )
    # 记录验证最优模型的门值，便于和原相加聚合比较。
    if args.model == "model_b_spe_drt_learnable_fusion":
        checkpoint_model = exp._checkpoint_model()
        fusion_alpha = float(torch.sigmoid(checkpoint_model.drt_fusion_logit).detach().cpu())
        logger.log("DRT fusion: alpha_semantic={:.6f} alpha_prior={:.6f}".format(fusion_alpha, 1.0 - fusion_alpha))
    # 在完整测试集上报告特征门的均值和输入间变化幅度。
    if args.model == "model_b_spe_drt_feature_aggregation" and args.evaluate_test:
        logger.log(f"DRT test aggregation: semantic_gate_mean={exp.drt_eval_gate_mean:.6f} semantic_gate_std={exp.drt_eval_gate_std:.6f}")
    logger.log(round_line, highlight=True)
    logger.close()

    run_summary = {
        "run_id": run_id,
        "seed": args.random_seed,
        "setting": setting,
        "best_epoch": train_summary["best_epoch"],
        "best_valid": train_summary["best_valid"],
        "test_metrics": test_perf,
        "test_evaluated": bool(args.evaluate_test),
        "round_summary_line": round_line,
        "round_finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_seconds": train_summary["train_seconds"],
        "runtime": {
            "runtime_protocol": train_summary["runtime_info"]["runtime_protocol"],
        },
        "runtime_info": train_summary["runtime_info"],
        "early_stop": train_summary["early_stop"],
        "params": train_summary["params"],
        "model_text": train_summary["model_text"],
        "artifacts": {
            "detail_log": str(args.obs_run_paths["detail_log"].relative_to(paths["exp_dir"])),
            "history": str(args.obs_run_paths["history"].relative_to(paths["exp_dir"])),
            "curve": str(args.obs_run_paths["curve"].relative_to(paths["exp_dir"])),
            "best_checkpoint": str(args.obs_run_paths["best_ckpt"].relative_to(paths["exp_dir"])),
            "last_checkpoint": str(args.obs_run_paths["last_ckpt"].relative_to(paths["exp_dir"])),
        },
    }
    if args.model == "model_b_spe_drt_learnable_fusion":
        run_summary["drt_fusion_alpha"] = fusion_alpha
    if args.model == "model_b_spe_drt_feature_aggregation" and args.evaluate_test:
        run_summary["drt_test_gate_mean"] = exp.drt_eval_gate_mean
        run_summary["drt_test_gate_std"] = exp.drt_eval_gate_std
    return run_summary


def main():
    args = build_parser().parse_args()
    if args.random_seed is None:
        args.random_seed = secrets.randbelow(1_000_000_000)
    args.completed_run_summaries = []
    args.exp_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    if args.debug:
        args.num_runs = args.debug_num_runs
        args.torch_compile = int(args.torch_compile)

    prepare_runtime_args(args)
    paths = build_exp_paths(args)
    save_config(args, paths)
    run_summaries = []
    total_runs = args.num_runs
    for run_id in range(total_runs):
        args.completed_run_summaries = list(run_summaries)
        run_summaries.append(run_once(args, run_id, paths))

    update_config_runtime(
        paths,
        run_summaries[0]["runtime_info"],
        run_summaries[0]["early_stop"],
    )
    aggregate = aggregate_runs(run_summaries)
    payload = {
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_runs": args.num_runs,
        "base_seed": args.random_seed,
        "runs": run_summaries,
        "aggregate_test": aggregate,
    }
    save_summary_all_runs(paths, payload)
    report = format_aggregate_report(args, paths, run_summaries)
    save_run_md(paths, report)
    append_project_run_md(paths, report)
    first = run_summaries[0]
    save_model_detail(
        args,
        paths,
        first["model_text"],
        first["params"],
        first["runtime_info"],
        run_summaries,
    )
    sys.stdout.write(report + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
