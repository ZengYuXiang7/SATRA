"""Two-stage SATRA training with masked market reconstruction.

Stage 1 reconstructs randomly masked stock-time tokens using the full SATRA
backbone. Stage 2 transfers only the input projection, Adaptive Temporal Pattern
Encoder, and Dual Relation Tower into supervised return prediction. The
reconstruction decoder is never used by the downstream model.

Each run owns its pretraining checkpoint, so weights cannot cross seeds. The
supervised stage follows the project-wide validation RankIC checkpoint rule."""

import copy
import json
import secrets
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import optim

from satra.b_spe_drt import PSTRNet_MMR
from satra.data_provider.stock_data import StockMarketData
from satra.main import (
    aggregate_runs,
    build_parser,
    format_metric_values,
    prepare_runtime_args,
    run_once,
    set_random_seed,
)
from satra.observability.exp_logger import ExpLogger
from satra.observability.exp_paths import build_exp_paths
from satra.observability.exp_report import (
    append_project_run_md,
    format_aggregate_report,
    save_config,
    save_model_detail,
    save_run_md,
    save_summary_all_runs,
    update_config_runtime,
)
from satra.utils.tools import select_scheduler


def build_mmr_parser():
    parser = build_parser()
    parser.add_argument("--pretrain_epochs", type=int, default=20)
    parser.add_argument("--pretrain_learning_rate", type=float, default=0.001)
    parser.add_argument(
        "--pretrain_lr_scheduler",
        type=str,
        default="cosine_hard_restarts",
        choices=["step", "cosine", "cosine_hard_restarts", "linear", "none"],
    )
    return parser


def run_pretraining(base_args, run_id, paths):
    args = copy.deepcopy(base_args)
    args.run_id = run_id
    args.total_runs = base_args.num_runs
    args.random_seed = base_args.random_seed + run_id
    args.pretrained_backbone_path = ""
    args.train_epochs = 1 if args.debug else args.pretrain_epochs
    args.learning_rate = args.pretrain_learning_rate
    args.lr_scheduler = args.pretrain_lr_scheduler
    prepare_runtime_args(args)
    set_random_seed(args.random_seed)

    pretrain_dir = paths["exp_dir"] / "pretrain"
    checkpoint_dir = pretrain_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = pretrain_dir / f"run{run_id}.log"
    history_path = pretrain_dir / f"history_run{run_id}.json"
    checkpoint_path = checkpoint_dir / f"backbone_run{run_id}.pt"
    logger = ExpLogger(log_path)

    data = StockMarketData(args)
    args.stock_num = data.stock_num
    args.features = data.feature_num
    model = PSTRNet_MMR(args).float()
    device = torch.device(f"cuda:{args.gpu}" if args.use_gpu else "cpu")
    model = model.to(device)
    optimizer = (
        optim.AdamW(model.parameters(), lr=args.learning_rate, fused=True)
        if device.type == "cuda"
        else optim.AdamW(model.parameters(), lr=args.learning_rate)
    )
    scheduler = select_scheduler(optimizer, args, nbatches=len(data.train_loader(0)))
    scheduler_steps_per_batch = args.lr_scheduler == "cosine_hard_restarts"

    logger.log(
        "MMR pretraining run={}/{} data={} seed={} epochs={} mask_mode=random_token mask_ratio={:.1f}".format(
            run_id + 1,
            args.total_runs,
            args.data,
            args.random_seed,
            args.train_epochs,
            args.mae_mask_ratio,
        )
    )
    history = []
    best_loss = float("inf")
    best_epoch = 0
    started_at = time.time()

    for epoch in range(args.train_epochs):
        model.train()
        np.random.shuffle(data.train_offsets)
        epoch_started_at = time.time()
        loss_sum = 0.0
        drt_balance_sum = 0.0
        token_count = 0
        mask_ratio_sum = 0.0

        for batch in data.train_loader(epoch):
            inputs = torch.as_tensor(batch["inputs"], dtype=torch.float32, device=device)
            token_mask = model.sample_token_mask(inputs, args.mae_mask_ratio)
            optimizer.zero_grad()
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda" and args.precision == "bf16_amp_tf32"
                else nullcontext()
            )
            with autocast:
                reconstruction_loss = model.reconstruction_loss(inputs, token_mask)
            loss = reconstruction_loss.float()
            # 预训练阶段同样约束两塔平均权重，避免门在初始化阶段集中到单塔。
            if args.model == "model_b_spe_drt_feature_aggregation":
                drt_balance_loss = model.backbone.drt_load_balance_loss()
                loss = loss + args.drt_load_balance_weight * drt_balance_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            if scheduler is not None and scheduler_steps_per_batch:
                scheduler.step()
            batch_tokens = int(inputs.shape[0] * inputs.shape[1] * inputs.shape[2])
            loss_sum += float(reconstruction_loss.detach().item()) * batch_tokens
            if args.model == "model_b_spe_drt_feature_aggregation":
                drt_balance_sum += float(drt_balance_loss.detach().item()) * batch_tokens
            token_count += batch_tokens
            mask_ratio_sum += float(token_mask.float().mean().item())

        if scheduler is not None and not scheduler_steps_per_batch:
            scheduler.step()
        epoch_loss = loss_sum / token_count
        epoch_mask_ratio = mask_ratio_sum / len(data.train_loader(epoch))
        improved = epoch_loss < best_loss
        if improved:
            best_loss = epoch_loss
            best_epoch = epoch + 1
            torch.save(
                {
                    "stage": "masked_market_reconstruction",
                    "data": args.data,
                    "seed": args.random_seed,
                    "seq_len": args.seq_len,
                    "feature_dim": args.features,
                    "mask_mode": "random_token",
                    "mask_ratio": args.mae_mask_ratio,
                    "backbone_state": model.backbone_state(),
                },
                checkpoint_path,
            )
        epoch_seconds = time.time() - epoch_started_at
        history.append(
            {
                "epoch": epoch + 1,
                "reconstruction_mse": epoch_loss,
                "actual_mask_ratio": epoch_mask_ratio,
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": epoch_seconds,
                "saved": improved,
            }
        )
        # 检查点仍按重构误差选取，均衡项作为独立辅助目标记录。
        if args.model == "model_b_spe_drt_feature_aggregation":
            epoch_drt_balance_loss = drt_balance_sum / token_count
            weighted_drt_balance_loss = args.drt_load_balance_weight * epoch_drt_balance_loss
            history[-1]["drt_load_balance_loss"] = epoch_drt_balance_loss
            history[-1]["weighted_drt_load_balance_loss"] = weighted_drt_balance_loss
            logger.log(f"DRT aggregation MMR: balance_loss={epoch_drt_balance_loss:.6f} weighted_balance_loss={weighted_drt_balance_loss:.8f}")
        logger.log(
            "MMR Epoch {}/{} run={}/{} reconstruction_mse={:.6f} mask_ratio={:.6f} "
            "best={:.6f}@{} lr={:.2e} saved={}".format(
                epoch + 1,
                args.train_epochs,
                run_id + 1,
                args.total_runs,
                epoch_loss,
                epoch_mask_ratio,
                best_loss,
                best_epoch,
                optimizer.param_groups[0]["lr"],
                "yes" if improved else "no",
            )
        )

    summary = {
        "stage": "masked_market_reconstruction",
        "seed": args.random_seed,
        "mask_mode": "random_token",
        "mask_ratio": args.mae_mask_ratio,
        "best_epoch": best_epoch,
        "best_reconstruction_mse": best_loss,
        "train_seconds": time.time() - started_at,
        "checkpoint": str(checkpoint_path.relative_to(paths["exp_dir"])),
        "log": str(log_path.relative_to(paths["exp_dir"])),
        "history": str(history_path.relative_to(paths["exp_dir"])),
    }
    with history_path.open("w", encoding="utf-8") as file:
        json.dump({"epochs": history, "summary": summary}, file, indent=2)
    logger.log(
        "MMR complete run={} best_epoch={} reconstruction_mse={:.6f} checkpoint={}".format(
            run_id + 1,
            best_epoch,
            best_loss,
            checkpoint_path,
        ),
        highlight=True,
    )
    logger.close()
    return checkpoint_path, summary


def main():
    args = build_mmr_parser().parse_args()
    if args.random_seed is None:
        args.random_seed = secrets.randbelow(1_000_000_000)
    args.exp_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    if args.debug:
        args.num_runs = args.debug_num_runs
    prepare_runtime_args(args)
    paths = build_exp_paths(args)
    save_config(args, paths)

    run_summaries = []
    pretraining_summaries = []
    for run_id in range(args.num_runs):
        checkpoint_path, pretraining_summary = run_pretraining(args, run_id, paths)
        finetuning_args = copy.deepcopy(args)
        finetuning_args.pretrained_backbone_path = str(checkpoint_path)
        finetuning_args.completed_run_summaries = list(run_summaries)
        run_summary = run_once(finetuning_args, run_id, paths)
        run_summary["mmr_pretraining"] = pretraining_summary
        run_summaries.append(run_summary)
        pretraining_summaries.append(pretraining_summary)

    update_config_runtime(paths, run_summaries[0]["runtime_info"], run_summaries[0]["early_stop"])
    payload = {
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": "SATRA with MMR",
        "num_runs": args.num_runs,
        "base_seed": args.random_seed,
        "pretraining": pretraining_summaries,
        "runs": run_summaries,
        "aggregate_test": aggregate_runs(run_summaries),
    }
    save_summary_all_runs(paths, payload)
    report = format_aggregate_report(args, paths, run_summaries)
    report += "\n\n## MMR Pretraining\n"
    for index, summary in enumerate(pretraining_summaries, start=1):
        report += (
            "- run {}: best_epoch={}, reconstruction_mse={:.6f}, checkpoint={}\n".format(
                index,
                summary["best_epoch"],
                summary["best_reconstruction_mse"],
                summary["checkpoint"],
            )
        )
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
    print(report)


if __name__ == "__main__":
    main()