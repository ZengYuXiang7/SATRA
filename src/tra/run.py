import argparse
import copy
import random
import time

import numpy as np
import torch

from metrics import MONITOR_METRIC, MONITOR_MODE, evaluate
from rsr.losses import return_rank_loss
from rsr.run import (
    autocast_context,
    build_optimizer,
    is_improved,
    make_runtime_protocol,
    prepare_model,
    resolve_monitor_mode,
    setup_device,
    to_device,
)
from rsr.utils import aggregate_runs, make_output_dir, write_json, write_run_md

from .data import TRAData, make_data_args
from .model import Model as TRAModel


DEFAULT_PORTFOLIO_SIZES = (5, 10, 15, 20, 25)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_portfolio_sizes(raw_value):
    if not isinstance(raw_value, str):
        raise TypeError(f"portfolio_sizes must be a comma-separated string, got {type(raw_value).__name__}")
    raw_sizes = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not raw_sizes:
        raise ValueError("portfolio_sizes must contain at least one positive integer")
    try:
        sizes = [int(item) for item in raw_sizes]
    except ValueError as exc:
        raise ValueError(f"invalid portfolio_sizes={raw_value!r}; expected comma-separated positive integers") from exc
    if any(size <= 0 for size in sizes):
        raise ValueError(f"portfolio_sizes must be positive, got {sizes}")
    if len(set(sizes)) != len(sizes):
        raise ValueError(f"portfolio_sizes must not contain duplicates, got {sizes}")
    if 10 not in sizes:
        raise ValueError("portfolio_sizes must include 10 to preserve the standard IRR@10 and SR@10 metrics")
    return tuple(sorted(sizes))


def topk_metric_names(portfolio_sizes):
    return tuple(f"{metric}@{size}" for metric in ("IRR", "SR") for size in portfolio_sizes)


def build_topk_payload(args, run_summaries, aggregate):
    metric_names = topk_metric_names(args.portfolio_sizes)
    run_rows = []
    for row in run_summaries:
        missing = [name for name in metric_names if name not in row["test_metrics"]]
        if missing:
            raise RuntimeError(f"TRA final test did not record required Top-K metrics: {missing}")
        run_rows.append(
            {
                "run_id": row["run_id"],
                "seed": row["seed"],
                "best_epoch": row["best_epoch"],
                "best_valid_RankIC": row["best_valid"].get("RankIC"),
                "topk_metrics": {name: row["test_metrics"][name] for name in metric_names},
            }
        )
    missing_aggregate = [name for name in metric_names if name not in aggregate]
    if missing_aggregate:
        raise RuntimeError(f"TRA aggregate did not record required Top-K metrics: {missing_aggregate}")
    return {
        "model": "tra",
        "data": args.data,
        "monitor_selection": {"metric": args.monitor_metric, "mode": args.monitor_mode},
        "portfolio_sizes": list(args.portfolio_sizes),
        "measurement": "Computed during each selected checkpoint's single final test pass while TRA temporal error memory remained in process; no checkpoint replay was used.",
        "runs": run_rows,
        "aggregate": {name: aggregate[name] for name in metric_names},
    }


def parse_args():
    parser = argparse.ArgumentParser(description="TRA unified-data runner")
    parser.add_argument("--data", required=True, choices=["NASDAQ", "NYSE", "SP500"])
    parser.add_argument("--model_id", default="tra")
    parser.add_argument("--root_path", default="data/processed")
    parser.add_argument("--relation_root", default="data/processed/relations")
    parser.add_argument("--relation_name", default="auto")
    parser.add_argument("--results", default="results")
    parser.add_argument("--input_features", default="stock_all")
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--debug_train_steps", type=int, default=2)
    parser.add_argument("--debug_eval_steps", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=int, default=1)
    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--ot_weight", type=float, default=0.1)
    parser.add_argument("--sinkhorn_epsilon", type=float, default=0.1)
    parser.add_argument("--sinkhorn_iters", type=int, default=20)
    parser.add_argument("--memory_decay", type=float, default=0.9)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_epochs", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--monitor_metric", default=MONITOR_METRIC, choices=[MONITOR_METRIC])
    parser.add_argument("--monitor_mode", default=MONITOR_MODE, choices=[MONITOR_MODE])
    parser.add_argument("--precision", default="bf16_amp_tf32", choices=["bf16_amp_tf32", "fp32"])
    parser.add_argument("--torch_compile", type=int, default=1)
    parser.add_argument("--compile_mode", default="reduce-overhead")
    parser.add_argument("--fused_optimizer", type=int, default=1)
    parser.add_argument("--use_gpu", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--num_predictors", type=int, default=5)
    parser.add_argument("--router_hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--hard_routing", type=int, default=0)
    parser.add_argument(
        "--portfolio_sizes",
        default=",".join(str(size) for size in DEFAULT_PORTFOLIO_SIZES),
        help="Comma-separated Top-K portfolio sizes measured during the final test pass.",
    )
    return parser.parse_args()


def build_model(args, data):
    return TRAModel(
        input_dim=int(data.feature_num),
        num_stocks=int(data.stock_num),
        hidden_size=int(args.hidden_size),
        num_layers=int(args.num_layers),
        num_predictors=int(args.num_predictors),
        router_hidden=int(args.router_hidden),
        dropout=float(args.dropout),
        tau=float(args.temperature),
    )


def sinkhorn_assignment(cost: torch.Tensor, epsilon: float = 0.1, iters: int = 20) -> torch.Tensor:
    """Balanced soft assignment over predictors for TRA's OT regularizer."""
    if cost.numel() == 0:
        return cost
    cost = cost.float()
    n, k = cost.shape
    kernel = torch.exp(-(cost - cost.min(dim=1, keepdim=True).values) / max(float(epsilon), 1e-6))
    kernel = kernel.clamp_min(1e-12)
    row_target = torch.ones(n, device=cost.device, dtype=cost.dtype)
    col_target = torch.full((k,), float(n) / float(k), device=cost.device, dtype=cost.dtype)
    transport = kernel
    for _ in range(int(iters)):
        transport = transport * (row_target / transport.sum(dim=1).clamp_min(1e-12)).unsqueeze(1)
        transport = transport * (col_target / transport.sum(dim=0).clamp_min(1e-12)).unsqueeze(0)
    transport = transport / transport.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return transport


def tra_losses(output, target, mask, stock_num, args):
    pred = output["prediction"].float()
    base_loss, reg_loss, rank_loss = return_rank_loss(
        pred,
        target.float(),
        mask.float(),
        stock_num,
        alpha=args.alpha,
    )
    per_predictor = (output["all_predictions"].float() - target.float()).pow(2)
    valid = mask.squeeze(-1).bool()
    cost = per_predictor[valid]
    route = output["route_prob"].float()[valid]
    if cost.numel() == 0:
        ot_loss = pred.new_zeros(())
    else:
        assignment = sinkhorn_assignment(cost.detach(), args.sinkhorn_epsilon, args.sinkhorn_iters)
        ot_loss = -(assignment.detach() * torch.log(route.clamp_min(1e-8))).sum(dim=-1).mean()
    return base_loss + float(args.ot_weight) * ot_loss, reg_loss, rank_loss, ot_loss, per_predictor.detach()


def attach_temporal_errors(batch, memory, device):
    offsets = torch.as_tensor(batch["offset"], dtype=torch.long, device=device)
    batch["temporal_errors"] = memory[offsets]
    return offsets


def update_memory(memory, offsets, per_predictor_loss, args):
    old = memory[offsets]
    memory[offsets] = float(args.memory_decay) * old + (1.0 - float(args.memory_decay)) * per_predictor_loss


def train_epoch(model, loader, optimizer, device, args, stock_num, runtime_protocol, memory):
    model.train()
    rows = []
    for batch in loader:
        batch = to_device(batch, device)
        offsets = attach_temporal_errors(batch, memory, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, runtime_protocol):
            output = model(batch)
            loss, reg_loss, rank_loss, ot_loss, per_predictor_loss = tra_losses(
                output,
                batch["y_endo"],
                batch["mask"],
                stock_num,
                args,
            )
        loss = loss.float()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
        grad_norm_value = float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm)
        optimizer.step()
        update_memory(memory, offsets, per_predictor_loss, args)
        rows.append(
            {
                "loss": float(loss.detach().cpu()),
                "reg_loss": float(reg_loss.detach().cpu()),
                "rank_loss": float(rank_loss.detach().cpu()),
                "ot_loss": float(ot_loss.detach().cpu()),
                "grad_norm": grad_norm_value,
                "clip_applied": float(grad_norm_value > float(args.max_grad_norm)),
            }
        )
    summary = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    summary["max_grad_norm_seen"] = float(np.max([row["grad_norm"] for row in rows]))
    runtime_protocol["grad_clip"]["clip_ratio"] = summary["clip_applied"]
    runtime_protocol["grad_clip"]["max_grad_norm_seen"] = max(
        float(runtime_protocol["grad_clip"]["max_grad_norm_seen"]),
        summary["max_grad_norm_seen"],
    )
    return summary


@torch.no_grad()
def eval_model(model, loader, device, args, stock_num, runtime_protocol, memory, portfolio_sizes=None):
    model.eval()
    preds = []
    labels = []
    masks = []
    losses = []
    for batch in loader:
        batch = to_device(batch, device)
        attach_temporal_errors(batch, memory, device)
        with autocast_context(device, runtime_protocol):
            output = model(batch)
            pred = output["prediction"].float()
        loss, _, _ = return_rank_loss(
            pred,
            batch["y_endo"].float(),
            batch["mask"].float(),
            stock_num,
            alpha=0.0,
        )
        losses.append(float(loss.detach().cpu()))
        preds.append(pred.detach().cpu().numpy())
        labels.append(batch["y_endo"].detach().cpu().numpy())
        masks.append(batch["mask"].detach().cpu().numpy())
    metrics = evaluate(
        np.concatenate(preds, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(masks, axis=0),
        stock_num=stock_num,
        portfolio_sizes=portfolio_sizes,
    )
    metrics["loss"] = float(np.mean(losses)) if losses else np.nan
    return metrics


def run_once(args, data, out_dir, run_id, device):
    seed = int(args.random_seed) + int(run_id)
    set_seed(seed)
    runtime_protocol = copy.deepcopy(args.runtime_protocol)
    raw_model = build_model(args, data).to(device)
    model = prepare_model(raw_model, args, device, runtime_protocol)
    optimizer = build_optimizer(raw_model, args, device, runtime_protocol)
    memory = torch.zeros(
        int(data.trade_dates),
        int(data.stock_num),
        int(args.num_predictors),
        device=device,
        dtype=torch.float32,
    )
    train_loader = data.train_loader(epoch=run_id)
    valid_loader = data.eval_loader(data.valid_index, data.test_index, drop_last=True)
    test_loader = data.eval_loader(data.test_index, data.trade_dates, drop_last=False)
    runtime_protocol["drop_last"]["valid_effective"] = bool(getattr(valid_loader, "drop_last", False))
    runtime_protocol["drop_last"]["test_effective"] = bool(getattr(test_loader, "drop_last", False))

    monitor_mode = resolve_monitor_mode(args.monitor_metric, args.monitor_mode)
    best_score = -float("inf")
    best_valid_metrics = None
    best_epoch = 0
    patience_count = 0
    early_stopped = False
    history = []
    best_path = out_dir / "checkpoints" / f"best_model_run{run_id}.pt"
    started = time.time()
    epochs = 1 if args.debug else int(args.train_epochs)

    print(
        f"[tra] run={run_id + 1}/{args.num_runs} runtime "
        f"precision={runtime_protocol['precision']['effective']} "
        f"compile={runtime_protocol['compile']['effective']} mode={runtime_protocol['compile']['mode']} "
        f"fused_adamw={runtime_protocol['optimizer']['fused_effective']} "
        f"drop_last=train:{runtime_protocol['drop_last']['train_effective']} "
        f"valid:{runtime_protocol['drop_last']['valid_effective']} "
        f"test:{runtime_protocol['drop_last']['test_effective']}",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device, args, data.stock_num, runtime_protocol, memory)
        valid_metrics = eval_model(model, valid_loader, device, args, data.stock_num, runtime_protocol, memory)
        monitor_value = float(valid_metrics[args.monitor_metric])
        improved = is_improved(monitor_value, best_score, monitor_mode, best_valid_metrics is not None)
        if improved:
            best_score = monitor_value
            best_valid_metrics = dict(valid_metrics)
            best_epoch = epoch
            patience_count = 0
            torch.save(raw_model.state_dict(), best_path)
        else:
            patience_count += 1
        row = {"epoch": epoch, "train": train_metrics, "valid": valid_metrics, "saved": improved}
        history.append(row)
        print(
            f"[tra] run={run_id + 1}/{args.num_runs} epoch={epoch:03d}/{epochs} "
            f"loss={train_metrics['loss']:.6f} reg={train_metrics['reg_loss']:.6f} "
            f"rank={train_metrics['rank_loss']:.6f} ot={train_metrics['ot_loss']:.6f} "
            f"valid_{args.monitor_metric}={monitor_value:.6f} best={best_score:.6f} "
            f"mode={monitor_mode} best_ep={best_epoch} saved={'yes' if improved else 'no'}",
            flush=True,
        )
        if epoch >= int(args.min_epochs) and patience_count >= int(args.patience):
            early_stopped = True
            break

    raw_model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = eval_model(
        model,
        test_loader,
        device,
        args,
        data.stock_num,
        runtime_protocol,
        memory,
        portfolio_sizes=args.portfolio_sizes,
    )
    summary = {
        "run_id": run_id,
        "seed": seed,
        "best_epoch": best_epoch,
        "monitor_metric": args.monitor_metric,
        "monitor_mode": monitor_mode,
        "best_valid": best_valid_metrics or {args.monitor_metric: best_score},
        "test_metrics": test_metrics,
        "runtime_protocol": runtime_protocol,
        "early_stop": {
            "requested": True,
            "effective": True,
            "patience": int(args.patience),
            "triggered": early_stopped,
            "counter": int(patience_count),
        },
        "train_seconds": float(time.time() - started),
        "checkpoint": str(best_path),
    }
    write_json(
        out_dir / f"history_run{run_id}.json",
        {"epochs": history, "summary": summary, "runtime_protocol": runtime_protocol},
    )
    return summary


def main():
    args = parse_args()
    args.portfolio_sizes = parse_portfolio_sizes(args.portfolio_sizes)
    args.model = "tra"
    device = setup_device(args)
    args.runtime_protocol = make_runtime_protocol(args, device)
    data_args = make_data_args(
        args.data,
        root_path=args.root_path,
        relation_root=args.relation_root,
        relation_name=args.relation_name,
        seq_len=args.seq_len,
        steps=args.steps,
        input_features=args.input_features,
        batch_size=args.batch_size,
        random_seed=args.random_seed,
        debug=bool(args.debug),
        debug_train_steps=args.debug_train_steps,
        debug_eval_steps=args.debug_eval_steps,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        train_drop_last=True,
        valid_drop_last=True,
    )
    data = TRAData(data_args)
    args.features = data.feature_num
    args.stock_num = data.stock_num
    args.market_num = int(getattr(data.args, "market_num", data.stock_num))

    out_dir = make_output_dir(args.results, args.data, args.model_id)
    write_json(
        out_dir / "config.json",
        {
            **vars(args),
            "device": str(device),
            "feature_names": data.feature_names,
            "input_transform_metadata": {
                **data.input_transform_metadata,
                "tra_memory": "temporal prediction-error memory indexed by trading-window offset",
                "graph_usage": "none; relation file is only loaded through the shared RSRData validation path",
            },
        },
    )
    print(f"[tra] output_dir={out_dir}", flush=True)
    print(f"[tra] device={device} root_path={args.root_path}", flush=True)
    print(
        f"[tra] data_contract inputs=[B,{data.stock_num},{args.seq_len},{data.feature_num}] "
        f"target=[B,{data.stock_num},1] predictors={args.num_predictors}",
        flush=True,
    )
    print(f"[tra] feature_names={data.feature_names}", flush=True)

    run_count = 1 if args.debug else int(args.num_runs)
    args.num_runs = run_count
    run_summaries = [run_once(args, data, out_dir, run_id, device) for run_id in range(run_count)]
    aggregate = aggregate_runs(run_summaries)
    topk_payload = build_topk_payload(args, run_summaries, aggregate)
    topk_path = out_dir / "portfolio_topk_sensitivity.json"
    write_json(topk_path, topk_payload)
    payload = {
        "model": "tra",
        "data": args.data,
        "root_path": args.root_path,
        "input_features": args.input_features,
        "feature_names": data.feature_names,
        "portfolio_sizes": list(args.portfolio_sizes),
        "portfolio_topk_file": str(topk_path),
        "runs": run_summaries,
        "aggregate": aggregate,
    }
    write_json(out_dir / "summary_all_runs.json", payload)
    write_run_md(
        out_dir / "run.md",
        f"TRA {args.data}",
        args,
        run_summaries,
        aggregate,
        extra_config_lines=[
            f"- source_pdf: `baseline_references/2021_KDD_TRA_Learning_Multiple_Stock_Trading_Patterns_with_Temporal_Routing_Adaptor_and_Optimal_Transport.pdf`",
            f"- input_features: `{args.input_features}`",
            f"- features: `{', '.join(data.feature_names)}`",
            f"- num_predictors: `{args.num_predictors}`",
            f"- router_input: `latent representation + temporal prediction-error memory`",
            f"- ot_regularizer: `Sinkhorn balanced assignment, weight={args.ot_weight}`",
            f"- loss: `mse_return + {args.alpha} * pairwise_rank_loss + {args.ot_weight} * ot_router_loss`",
            f"- monitor_metric: `{args.monitor_metric}`",
            f"- monitor_mode: `{args.monitor_mode}`",
            f"- portfolio_sizes: `{', '.join(str(size) for size in args.portfolio_sizes)}`",
            "- portfolio_measurement: `all requested IRR@K and SR@K values are collected in the selected checkpoint's single final test pass; no checkpoint replay is used.`",
            f"- portfolio_topk_file: `{topk_path}`",
        ],
    )
    print(f"[tra] summary={out_dir / 'summary_all_runs.json'}", flush=True)
    print(f"[tra] run_md={out_dir / 'run.md'}", flush=True)


if __name__ == "__main__":
    main()
