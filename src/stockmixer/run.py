import argparse
import copy
from contextlib import nullcontext
import random
import time

import numpy as np
import torch

from .data import StockMixerData, make_data_args
from .losses import price_return_rank_loss
from metrics import MONITOR_METRIC, MONITOR_MODE, evaluate
from .model import Model as StockMixer
from .utils import aggregate_runs, make_output_dir, write_json, write_run_md


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_run_seeds(args, run_count):
    if args.seeds is None:
        return [int(args.random_seed) + run_id for run_id in range(run_count)]

    seeds = [int(seed) for seed in args.seeds]
    if len(seeds) != int(run_count):
        raise ValueError(
            f"--seeds requires exactly {run_count} values for num_runs={run_count}; got {len(seeds)}"
        )
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain distinct values")
    if any(seed < 0 or seed >= 2**63 for seed in seeds):
        raise ValueError("--seeds values must be in [0, 2**63)")
    return seeds


def parse_args():
    parser = argparse.ArgumentParser(description="StockMixer paper-specific runner")
    parser.add_argument("--data", required=True, choices=["NASDAQ", "NYSE", "SP500"])
    parser.add_argument("--model_id", default="stockmixer")
    parser.add_argument("--root_path", default="data/processed")
    parser.add_argument("--results", default="results")
    parser.add_argument("--input_features", default="stockmixer_ma")
    parser.add_argument("--stockmixer_ma_windows", default="5,10,20,30")
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit distinct seeds; requires exactly --num_runs values.",
    )
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
    parser.add_argument("--stock_hidden", type=int, default=None)
    parser.add_argument("--scale_factor", type=int, default=3)
    return parser.parse_args()


def setup_device(args):
    if bool(args.use_gpu) and torch.cuda.is_available():
        device = torch.device(f"cuda:{int(args.gpu)}")
    elif bool(args.use_gpu) and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    args.pin_memory = bool(args.pin_memory) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    return device


def make_runtime_protocol(args, device):
    requested_precision = str(args.precision)
    bf16_supported = bool(device.type == "cuda" and torch.cuda.is_bf16_supported())
    precision_effective = requested_precision if requested_precision == "bf16_amp_tf32" and bf16_supported else "fp32"
    precision_reason = None
    if requested_precision == "bf16_amp_tf32" and not bf16_supported:
        precision_reason = f"{device.type}_no_bf16_amp"

    compile_requested = bool(args.torch_compile)
    if not compile_requested:
        compile_reason = "not_requested"
    elif bool(args.debug):
        compile_reason = "debug"
    elif device.type != "cuda":
        compile_reason = f"{device.type}_no_compile"
    else:
        compile_reason = "pending"

    fused_requested = bool(args.fused_optimizer)
    optimizer_reason = "pending" if fused_requested and device.type == "cuda" else None
    if fused_requested and device.type != "cuda":
        optimizer_reason = f"{device.type}_no_fused_optimizer"

    return {
        "precision": {
            "requested": requested_precision,
            "effective": precision_effective,
            "reason": precision_reason,
            "tf32": bool(device.type == "cuda"),
        },
        "compile": {
            "requested": compile_requested,
            "effective": False,
            "mode": str(args.compile_mode),
            "reason": compile_reason,
        },
        "optimizer": {
            "name": "AdamW",
            "fused_requested": fused_requested,
            "fused_effective": False,
            "fallback_reason": optimizer_reason,
        },
        "grad_clip": {
            "max_grad_norm": float(args.max_grad_norm),
            "clip_ratio": 0.0,
            "max_grad_norm_seen": 0.0,
        },
        "drop_last": {
            "train_requested": True,
            "train_effective": True,
            "valid_requested": True,
            "valid_effective": True,
            "test_requested": False,
            "test_effective": False,
            "semantics": "train/valid drop_last=True for stable compile batches; test drop_last=False keeps all final samples",
        },
    }


def autocast_context(device, runtime_protocol):
    enabled = bool(
        device.type == "cuda"
        and runtime_protocol.get("precision", {}).get("effective") == "bf16_amp_tf32"
    )
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def prepare_model(raw_model, args, device, runtime_protocol):
    compile_state = runtime_protocol["compile"]
    if not compile_state["requested"]:
        compile_state["reason"] = "not_requested"
        return raw_model
    if bool(args.debug):
        compile_state["reason"] = "debug"
        return raw_model
    if device.type != "cuda":
        compile_state["reason"] = f"{device.type}_no_compile"
        return raw_model
    try:
        compiled = torch.compile(raw_model, mode=str(args.compile_mode))
    except Exception as exc:
        compile_state["effective"] = False
        compile_state["reason"] = f"compile_failed:{type(exc).__name__}:{str(exc)[:160]}"
        return raw_model
    compile_state["effective"] = True
    compile_state["reason"] = None
    return compiled


def build_optimizer(model, args, device, runtime_protocol):
    optimizer_state = runtime_protocol["optimizer"]
    use_fused = bool(optimizer_state["fused_requested"] and device.type == "cuda")
    kwargs = {
        "lr": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
    }
    if use_fused:
        try:
            optimizer = torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
        except (TypeError, RuntimeError) as exc:
            optimizer_state["fused_effective"] = False
            optimizer_state["fallback_reason"] = f"fused_failed:{type(exc).__name__}:{str(exc)[:160]}"
            return torch.optim.AdamW(model.parameters(), **kwargs)
        optimizer_state["fused_effective"] = True
        optimizer_state["fallback_reason"] = None
        return optimizer

    if optimizer_state["fused_requested"] and device.type != "cuda":
        optimizer_state["fallback_reason"] = f"{device.type}_no_fused_optimizer"
    elif not optimizer_state["fused_requested"]:
        optimizer_state["fallback_reason"] = "not_requested"
    return torch.optim.AdamW(model.parameters(), **kwargs)


def to_device(batch, device):
    result = {}
    for key, value in batch.items():
        if key == "offset":
            result[key] = value
        else:
            result[key] = torch.as_tensor(value, dtype=torch.float32).to(device, non_blocking=True)
    return result


def build_model(args, data):
    stock_hidden = args.stock_hidden
    if stock_hidden is None:
        stock_hidden = int(getattr(data.args, "market_num", data.stock_num))
    args.stock_hidden = int(stock_hidden)
    return StockMixer(
        stocks=int(data.stock_num),
        time_steps=int(args.seq_len),
        channels=int(data.feature_num),
        market=int(args.stock_hidden),
        scale=int(args.scale_factor),
    )


def resolve_monitor_mode(metric, mode):
    if metric != MONITOR_METRIC or mode != MONITOR_MODE:
        raise ValueError("Checkpoint selection is fixed to validation RankIC/max.")
    return MONITOR_MODE


def is_improved(value, best_score, mode, has_best):
    if not np.isfinite(value):
        return False
    if not has_best:
        return True
    if mode != MONITOR_MODE:
        raise ValueError("Checkpoint selection is fixed to validation RankIC/max.")
    return value > best_score


def train_epoch(model, loader, optimizer, device, args, stock_num, runtime_protocol):
    model.train()
    rows = []
    for batch in loader:
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, runtime_protocol):
            pred = model(batch["inputs"])
        pred = pred.float()
        loss, reg_loss, rank_loss, _ = price_return_rank_loss(
            pred,
            batch["price"].float(),
            batch["y_endo"].float(),
            batch["mask"].float(),
            stock_num,
            alpha=args.alpha,
        )
        loss = loss.float()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
        grad_norm_value = float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm)
        optimizer.step()
        rows.append(
            {
                "loss": float(loss.detach().cpu()),
                "reg_loss": float(reg_loss.detach().cpu()),
                "rank_loss": float(rank_loss.detach().cpu()),
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
def eval_model(model, loader, device, args, stock_num, runtime_protocol):
    model.eval()
    preds = []
    labels = []
    masks = []
    losses = []
    for batch in loader:
        batch = to_device(batch, device)
        with autocast_context(device, runtime_protocol):
            pred = model(batch["inputs"])
        pred = pred.float()
        loss, _, _, return_pred = price_return_rank_loss(
            pred,
            batch["price"].float(),
            batch["y_endo"].float(),
            batch["mask"].float(),
            stock_num,
            alpha=0.0,
        )
        losses.append(float(loss.detach().cpu()))
        preds.append(return_pred.detach().cpu().numpy())
        labels.append(batch["y_endo"].detach().cpu().numpy())
        masks.append(batch["mask"].detach().cpu().numpy())
    metrics = evaluate(
        np.concatenate(preds, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(masks, axis=0),
        stock_num=stock_num,
    )
    metrics["loss"] = float(np.mean(losses)) if losses else np.nan
    return metrics


def run_once(args, data, out_dir, run_id, device, seed):
    seed = int(seed)
    set_seed(seed)
    runtime_protocol = copy.deepcopy(args.runtime_protocol)
    raw_model = build_model(args, data).to(device)
    model = prepare_model(raw_model, args, device, runtime_protocol)
    optimizer = build_optimizer(raw_model, args, device, runtime_protocol)
    train_loader = data.train_loader(seed=seed)
    valid_loader = data.eval_loader(data.valid_index, data.test_index, drop_last=True)
    test_loader = data.eval_loader(data.test_index, data.trade_dates, drop_last=False)
    runtime_protocol["drop_last"]["valid_effective"] = bool(getattr(valid_loader, "drop_last", False))
    runtime_protocol["drop_last"]["test_effective"] = bool(getattr(test_loader, "drop_last", False))

    monitor_mode = resolve_monitor_mode(args.monitor_metric, args.monitor_mode)
    best_score = -float("inf")
    best_valid_metrics = None
    best_epoch = 0
    patience_count = 0
    history = []
    best_path = out_dir / "checkpoints" / f"best_model_run{run_id}.pt"
    started = time.time()
    epochs = 1 if args.debug else int(args.train_epochs)
    early_stopped = False

    print(
        f"[stockmixer] run={run_id + 1}/{args.num_runs} seed={seed} runtime "
        f"precision={runtime_protocol['precision']['effective']} "
        f"compile={runtime_protocol['compile']['effective']} mode={runtime_protocol['compile']['mode']} "
        f"fused_adamw={runtime_protocol['optimizer']['fused_effective']} "
        f"drop_last=train:{runtime_protocol['drop_last']['train_effective']} "
        f"valid:{runtime_protocol['drop_last']['valid_effective']} "
        f"test:{runtime_protocol['drop_last']['test_effective']}",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device, args, data.stock_num, runtime_protocol)
        valid_metrics = eval_model(model, valid_loader, device, args, data.stock_num, runtime_protocol)
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
            f"[stockmixer] run={run_id + 1}/{args.num_runs} epoch={epoch:03d}/{epochs} "
            f"loss={train_metrics['loss']:.6f} valid_{args.monitor_metric}={monitor_value:.6f} "
            f"best={best_score:.6f} mode={monitor_mode} best_ep={best_epoch} saved={'yes' if improved else 'no'}",
            flush=True,
        )
        if epoch >= int(args.min_epochs) and patience_count >= int(args.patience):
            early_stopped = True
            break

    raw_model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = eval_model(model, test_loader, device, args, data.stock_num, runtime_protocol)
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
    if args.input_features != "stockmixer_ma":
        raise ValueError("StockMixer runner uses paper input_features=stockmixer_ma derived from raw price_data")
    args.model = "stockmixer"
    device = setup_device(args)
    args.runtime_protocol = make_runtime_protocol(args, device)
    data_args = make_data_args(
        args.data,
        root_path=args.root_path,
        seq_len=args.seq_len,
        steps=args.steps,
        batch_size=args.batch_size,
        random_seed=args.random_seed,
        debug=bool(args.debug),
        debug_train_steps=args.debug_train_steps,
        debug_eval_steps=args.debug_eval_steps,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        train_drop_last=True,
        valid_drop_last=True,
        input_features=args.input_features,
        stockmixer_ma_windows=args.stockmixer_ma_windows,
        require_feature_groups=False,
    )
    data = StockMixerData(data_args)
    args.features = data.feature_num
    args.stock_num = data.stock_num
    args.market_num = int(getattr(data.args, "market_num", data.stock_num))
    if args.stock_hidden is None:
        args.stock_hidden = args.market_num

    run_count = 1 if args.debug else int(args.num_runs)
    args.num_runs = run_count
    args.run_seeds = resolve_run_seeds(args, run_count)

    out_dir = make_output_dir(args.results, args.data, args.model_id)
    write_json(
        out_dir / "config.json",
        {
            **vars(args),
            "device": str(device),
            "feature_names": data.feature_names,
            "runtime_protocol": args.runtime_protocol,
        },
    )
    print(f"[stockmixer] output_dir={out_dir}", flush=True)
    print(f"[stockmixer] device={device} root_path={args.root_path}", flush=True)
    print(
        f"[stockmixer] data_contract inputs=[B,{data.stock_num},{args.seq_len},{data.feature_num}] "
        f"input_features={args.input_features} feature_names={data.feature_names}",
        flush=True,
    )
    print(f"[stockmixer] runtime_protocol_requested={args.runtime_protocol}", flush=True)

    run_summaries = [
        run_once(args, data, out_dir, run_id, device, seed)
        for run_id, seed in enumerate(args.run_seeds)
    ]
    aggregate = aggregate_runs(run_summaries)
    payload = {
        "model": "stockmixer",
        "data": args.data,
        "root_path": args.root_path,
        "input_features": args.input_features,
        "feature_names": data.feature_names,
        "runtime_protocol_requested": args.runtime_protocol,
        "runs": run_summaries,
        "aggregate": aggregate,
    }
    write_json(out_dir / "summary_all_runs.json", payload)
    write_run_md(
        out_dir / "run.md",
        f"StockMixer {args.data}",
        args,
        run_summaries,
        aggregate,
        extra_config_lines=[
            f"- input_features: `{args.input_features}`",
            f"- feature_names: `{', '.join(data.feature_names)}`",
            "- StockMixer inputs are derived at load time from `price_data.pkl`: `MA5, MA10, MA20, MA30, Close`",
            f"- loss: `price_to_return_mse + {args.alpha} * pairwise_rank_loss`",
            f"- monitor_metric: `{args.monitor_metric}`",
            f"- monitor_mode: `{args.monitor_mode}`",
            f"- runtime: `precision={args.precision}, torch_compile={bool(args.torch_compile)}, compile_mode={args.compile_mode}, max_grad_norm={args.max_grad_norm}`",
            "- drop_last: `train=True, valid=True, test=False`",
        ],
    )
    print(f"[stockmixer] summary={out_dir / 'summary_all_runs.json'}", flush=True)
    print(f"[stockmixer] run_md={out_dir / 'run.md'}", flush=True)


if __name__ == "__main__":
    main()
