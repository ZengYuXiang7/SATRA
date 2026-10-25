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

from .data import FactorGCLData, make_data_args
from .model import Model as FactorGCLModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="FactorGCL unified-data runner")
    parser.add_argument("--data", required=True, choices=["NASDAQ", "NYSE", "SP500"])
    parser.add_argument("--model_id", default="factorgcl")
    parser.add_argument("--root_path", default="data/processed")
    parser.add_argument("--relation_root", default="data/processed/relations")
    parser.add_argument("--relation_name", default="auto")
    parser.add_argument("--prior_beta_name", default="factor_prior_beta")
    parser.add_argument("--results", default="results")
    parser.add_argument("--input_features", default="stock_all")
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--debug_train_steps", type=int, default=2)
    parser.add_argument("--debug_eval_steps", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--future_seq_len", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=int, default=1)
    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--use_contrastive", type=int, default=1)
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
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--num_hidden_factors", type=int, default=32)
    parser.add_argument("--num_gru_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--projection_dim", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.1)
    return parser.parse_args()


def build_model(args, data):
    projection_dim = None if int(args.projection_dim) <= 0 else int(args.projection_dim)
    return FactorGCLModel(
        input_dim=int(data.feature_num),
        num_stocks=int(data.stock_num),
        prior_beta=data.prior_beta,
        hidden_dim=int(args.hidden_dim),
        num_hidden_factors=int(args.num_hidden_factors),
        num_horizons=1,
        num_gru_layers=int(args.num_gru_layers),
        dropout=float(args.dropout),
        projection_dim=projection_dim,
        temperature=float(args.temperature),
    )


def train_epoch(model, loader, optimizer, device, args, stock_num, runtime_protocol):
    model.train()
    rows = []
    for batch in loader:
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, runtime_protocol):
            output = model(batch)
            pred = output["prediction"]
            base_loss, reg_loss, rank_loss = return_rank_loss(
                pred,
                batch["y_endo"].float(),
                batch["mask"].float(),
                stock_num,
                alpha=args.alpha,
            )
            contrastive_loss = output.get("contrastive_loss", pred.new_zeros(()))
            loss = base_loss + float(args.gamma) * contrastive_loss
        loss = loss.float()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
        grad_norm_value = float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm)
        optimizer.step()
        rows.append(
            {
                "loss": float(loss.detach().cpu()),
                "base_loss": float(base_loss.detach().cpu()),
                "reg_loss": float(reg_loss.detach().cpu()),
                "rank_loss": float(rank_loss.detach().cpu()),
                "contrastive_loss": float(contrastive_loss.detach().cpu()),
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
        f"[factorgcl] run={run_id + 1}/{args.num_runs} runtime "
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
            f"[factorgcl] run={run_id + 1}/{args.num_runs} epoch={epoch:03d}/{epochs} "
            f"loss={train_metrics['loss']:.6f} base={train_metrics['base_loss']:.6f} "
            f"cl={train_metrics['contrastive_loss']:.6f} valid_{args.monitor_metric}={monitor_value:.6f} "
            f"best={best_score:.6f} mode={monitor_mode} best_ep={best_epoch} "
            f"saved={'yes' if improved else 'no'}",
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
    args.model = "factorgcl"
    device = setup_device(args)
    args.runtime_protocol = make_runtime_protocol(args, device)
    data_args = make_data_args(
        args.data,
        root_path=args.root_path,
        relation_root=args.relation_root,
        relation_name=args.relation_name,
        prior_beta_name=args.prior_beta_name,
        seq_len=args.seq_len,
        future_seq_len=args.future_seq_len,
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
        use_contrastive=bool(args.use_contrastive),
    )
    data = FactorGCLData(data_args)
    args.features = data.feature_num
    args.stock_num = data.stock_num
    args.relation_name_resolved = data.relation_name
    args.prior_beta_path = str(data.prior_beta_path)
    args.prior_beta_shape = list(data.prior_beta.shape)
    args.prior_factor_count = data.prior_factor_count
    args.market_num = int(getattr(data.args, "market_num", data.stock_num))

    out_dir = make_output_dir(args.results, args.data, args.model_id)
    write_json(
        out_dir / "config.json",
        {
            **vars(args),
            "device": str(device),
            "feature_names": data.feature_names,
            "input_transform_metadata": data.input_transform_metadata,
        },
    )
    print(f"[factorgcl] output_dir={out_dir}", flush=True)
    print(f"[factorgcl] device={device} root_path={args.root_path}", flush=True)
    print(
        f"[factorgcl] data_contract inputs=[B,{data.stock_num},{args.seq_len},{data.feature_num}] "
        f"target=[B,{data.stock_num},1] prior_beta={list(data.prior_beta.shape)} "
        f"future_inputs={'on' if data.use_contrastive else 'off'}",
        flush=True,
    )
    print(f"[factorgcl] feature_names={data.feature_names}", flush=True)
    print(
        f"[factorgcl] prior_beta_name={data.prior_beta_name} prior_beta_path={data.prior_beta_path}",
        flush=True,
    )

    run_count = 1 if args.debug else int(args.num_runs)
    args.num_runs = run_count
    run_summaries = [run_once(args, data, out_dir, run_id, device) for run_id in range(run_count)]
    aggregate = aggregate_runs(run_summaries)
    payload = {
        "model": "factorgcl",
        "data": args.data,
        "root_path": args.root_path,
        "input_features": args.input_features,
        "prior_beta_name": data.prior_beta_name,
        "prior_beta_path": str(data.prior_beta_path),
        "feature_names": data.feature_names,
        "runs": run_summaries,
        "aggregate": aggregate,
    }
    write_json(out_dir / "summary_all_runs.json", payload)
    write_run_md(
        out_dir / "run.md",
        f"FactorGCL {args.data}",
        args,
        run_summaries,
        aggregate,
        extra_config_lines=[
            f"- source_pdf: `baseline_references/2025_AAAI_FactorGCL_Hypergraph_Based_Factor_Model_with_Temporal_Residual_Contrastive_Learning.pdf`",
            f"- input_features: `{args.input_features}`",
            f"- features: `{', '.join(data.feature_names)}`",
            f"- prior_beta_path: `{data.prior_beta_path}`",
            f"- prior_beta_shape: `{list(data.prior_beta.shape)}`",
            f"- future_seq_len: `{data.future_seq_len}`",
            f"- use_contrastive: `{data.use_contrastive}`",
            f"- hidden_dim: `{args.hidden_dim}`",
            f"- num_hidden_factors: `{args.num_hidden_factors}`",
            f"- num_gru_layers: `{args.num_gru_layers}`",
            f"- loss: `mse_return + {args.alpha} * pairwise_rank_loss + {args.gamma} * temporal_residual_contrastive_loss`",
            f"- monitor_metric: `{args.monitor_metric}`",
            f"- monitor_mode: `{args.monitor_mode}`",
        ],
    )
    print(f"[factorgcl] summary={out_dir / 'summary_all_runs.json'}", flush=True)
    print(f"[factorgcl] run_md={out_dir / 'run.md'}", flush=True)


if __name__ == "__main__":
    main()
