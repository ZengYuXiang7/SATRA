"""Supervised SATRA experiment lifecycle and evaluation control.

This module owns data preparation, model construction, optimization,
validation, checkpointing, and final test evaluation. Formal checkpoints are
selected only by maximum validation RankIC. Test metrics are computed after
selection and do not influence training or model choice."""

import json
import os
import time
import warnings
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from satra.data_provider.stock_data import StockMarketData
from satra.data_provider.stock_prior import build_dtw_stock_prior
from satra.exp.exp_basic import Exp_Basic
import satra.b_spe_drt as stock_model_b_spe_drt
import satra.ablation.a_shared_temporal_encoder as stock_model_a_shared_temporal_encoder
import satra.ablation.a_homogeneous_experts as stock_model_a_homogeneous_experts
import satra.ablation.a_spe as stock_model_a_spe
import satra.ablation.b_spe_only as stock_model_b_spe_only
import satra.ablation.b_spe_semantic_tower as stock_model_b_spe_semantic_tower
import satra.ablation.b_spe_prior_guided_tower as stock_model_b_spe_prior_guided_tower
import satra.ablation.c_no_prior_constraint as stock_model_c_no_prior_constraint
import satra.ablation.c_industry_relation_prior as stock_model_c_industry_relation_prior
import satra.ablation.c_euclidean_distance_prior as stock_model_c_euclidean_distance_prior
import satra.ablation.c_cosine_distance_prior as stock_model_c_cosine_distance_prior
import satra.ablation.c_dtw_based_prior as stock_model_c_dtw_based_prior
import satra.ablation.c_learned_adjacency as stock_model_c_learned_adjacency
import satra.ablation.c_adaptive_graph as stock_model_c_adaptive_graph
from satra.utils.metrics import (
    AR_KEY,
    IRR_KEY,
    PORTFOLIO_SIZE,
    PRECISION_KEY,
    SR_KEY,
    evaluate,
)
from satra.utils.stock_loss import stockmixer_loss
from satra.utils.tools import EarlyStoppingByMetric, adjust_learning_rate, ensure_dir, select_scheduler
from satra.observability.exp_report import save_history, save_training_curve
from satra.observability.exp_runtime import build_runtime_protocol, get_runtime_info, model_param_count

warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy.*")


PSTR_MODELS = {
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
}
class Exp_Main(Exp_Basic):
    def __init__(self, args):
        self.logger = args.obs_logger
        self.history = []
        self.data = StockMarketData(args)
        args.stock_num = self.data.stock_num
        args.trade_dates = self.data.trade_dates
        args.features = self.data.feature_num
        args.feature_group_dims = {
            name: len(indices) for name, indices in self.data.feature_group_indices.items()
        }
        if self.data.feature_names:
            if "Close" in self.data.feature_names:
                args.close_feature_index = self.data.feature_names.index("Close")
            elif "close" in self.data.feature_names:
                args.close_feature_index = self.data.feature_names.index("close")
            if "ret" in self.data.feature_names:
                args.ret_feature_index = self.data.feature_names.index("ret")
        args.valid_index = self.data.valid_index
        args.test_index = self.data.test_index
        self._prepare_stock_prior(args)
        super(Exp_Main, self).__init__(args)
        self._setup_runtime_acceleration()
        self.metric_display_names = {
            "mae": "MAE↓",
            "mape": "MAPE↓",
            "RankIC": "RankIC↑",
            "RankICIR": "RankICIR↑",
            IRR_KEY: f"IRR@{PORTFOLIO_SIZE}↑",
            AR_KEY: f"AR@{PORTFOLIO_SIZE}↑",
            SR_KEY: f"SR@{PORTFOLIO_SIZE}↑",
            "KT": "KT↑",
            PRECISION_KEY: f"P@{PORTFOLIO_SIZE}↑",
            "SR": "SR↑",
            "IC": "IC↑",
            "ICIR": "ICIR↑",
        }
        self.primary_metric_order = ["mae", "RankIC", "RankICIR", IRR_KEY, SR_KEY]
        self.auxiliary_metric_order = [AR_KEY, "KT", "mape", PRECISION_KEY, "IC", "ICIR"]
        self.metric_display_order = self.primary_metric_order + self.auxiliary_metric_order
        self.model = self._build_model().to(self.device)
        self.model = self._maybe_compile_model(self.model)
        self._print_runtime_acceleration()

    def _prepare_stock_prior(self, args):
        if args.model in PSTR_MODELS:
            return
        if str(args.stock_prior) != "dtw":
            args.stock_prior_path = ""
            return
        cache_dir = args.stock_prior_cache_dir
        args.stock_prior_path = build_dtw_stock_prior(
            self.data,
            cache_dir,
            topk=args.stock_prior_topk,
            candidate_k=args.stock_prior_candidate_k,
            downsample=args.stock_prior_downsample,
            band=args.stock_prior_band,
            feature=args.stock_prior_feature,
            force=bool(args.stock_prior_force),
        )
        self._log(f"Stock prior: dtw path={args.stock_prior_path}")

    def _setup_runtime_acceleration(self):
        self.precision_requested = self.args.precision
        self.precision_effective = "fp32"
        self.amp_enabled = False
        self.tf32_enabled = False
        self.compile_requested = bool(self.args.torch_compile)
        self.compile_effective = False
        self.compile_reason = "not_requested" if not self.compile_requested else "pending"
        self.optimizer_fused_effective = None
        self.optimizer_fused_disabled_reason = None

        if self.device.type == "cuda":
            if self.args.precision in {"tf32", "bf16_amp_tf32"}:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.set_float32_matmul_precision("high")
                self.tf32_enabled = True
                self.precision_effective = "tf32"
            if self.args.precision == "bf16_amp_tf32":
                self.amp_enabled = True
                self.precision_effective = "bf16_amp_tf32"
        elif self.args.precision != "fp32":
            self.precision_effective = f"fp32(requested={self.args.precision})"

    def _maybe_compile_model(self, model):
        if not self.compile_requested:
            return model
        if self.args.debug:
            self.compile_reason = "debug"
            return model
        if self.device.type != "cuda":
            self.compile_reason = f"unsupported_device={self.device.type}"
            return model
        compiled = torch.compile(model, mode=self.args.torch_compile_mode)
        self.compile_effective = True
        self.compile_reason = "enabled"
        return compiled

    def _print_runtime_acceleration(self):
        self._log(
            "Runtime acceleration: precision requested={}, effective={}, amp={}, tf32={}, "
            "compile requested={}, effective={}, mode={}, reason={}, max_grad_norm={}".format(
                self.precision_requested,
                self.precision_effective,
                self.amp_enabled,
                self.tf32_enabled,
                self.compile_requested,
                self.compile_effective,
                self.args.torch_compile_mode,
                self.compile_reason,
                self.args.max_grad_norm,
            )
        )

    def _log(self, message=""):
        if self.logger is not None:
            self.logger.log(message)

    def _autocast_context(self):
        if self.amp_enabled:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _checkpoint_model(self):
        model = self.model
        while hasattr(model, "module"):
            model = model.module
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        return model

    def _build_model(self):
        model_dict = {
            "model_a_shared_temporal_encoder": stock_model_a_shared_temporal_encoder,
            "model_a_homogeneous_experts": stock_model_a_homogeneous_experts,
            "model_a_spe": stock_model_a_spe,
            "model_b_spe_only": stock_model_b_spe_only,
            "model_b_spe_semantic_tower": stock_model_b_spe_semantic_tower,
            "model_b_spe_prior_guided_tower": stock_model_b_spe_prior_guided_tower,
            "model_b_spe_drt": stock_model_b_spe_drt,
            "model_b_spe_drt_learnable_fusion": stock_model_b_spe_drt,
            "model_b_spe_drt_feature_aggregation": stock_model_b_spe_drt,
            "model_c_no_prior_constraint": stock_model_c_no_prior_constraint,
            "model_c_industry_relation_prior": stock_model_c_industry_relation_prior,
            "model_c_euclidean_distance_prior": stock_model_c_euclidean_distance_prior,
            "model_c_cosine_distance_prior": stock_model_c_cosine_distance_prior,
            "model_c_dtw_based_prior": stock_model_c_dtw_based_prior,
            "model_c_learned_adjacency": stock_model_c_learned_adjacency,
            "model_c_adaptive_graph": stock_model_c_adaptive_graph,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _select_optimizer(self):
        if self.device.type == "cuda":
            self.optimizer_fused_effective = True
            self.optimizer_fused_disabled_reason = None
            return optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, fused=True)
        self.optimizer_fused_effective = False
        self.optimizer_fused_disabled_reason = f"unsupported_device={self.device.type}"
        return optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)

    def _to_tensor_batch(self, batch):
        if isinstance(batch, dict):
            tensors = {}
            for key, item in batch.items():
                if key == "offset":
                    tensors[key] = item
                    continue
                tensor = torch.as_tensor(item, dtype=torch.float32)
                tensors[key] = tensor.to(self.device, non_blocking=True)
            return tensors
        return tuple(torch.as_tensor(item, dtype=torch.float32).to(self.device, non_blocking=True) for item in batch)

    def _predict_batch(self, batch):
        return self.model(batch["inputs"])

    def _moe_load_balance_loss(self):
        weight = float(self.args.moe_load_balance_weight)
        if weight == 0.0:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        model = self._checkpoint_model()
        if not hasattr(model, "moe_load_balance_loss"):
            raise AttributeError(
                "moe_load_balance_weight is nonzero, but model "
                f"{self.args.model!r} does not expose moe_load_balance_loss()"
            )
        balance_loss = model.moe_load_balance_loss()
        if balance_loss.ndim != 0:
            raise RuntimeError(
                "MoE load-balance loss must be a scalar, got shape "
                f"{tuple(balance_loss.shape)}"
            )
        if not torch.isfinite(balance_loss):
            raise FloatingPointError("MoE load-balance loss is non-finite")
        return balance_loss

    def _offsets_to_numpy(self, offsets):
        if hasattr(offsets, "detach"):
            return offsets.detach().cpu().numpy().astype(np.int64)
        return np.asarray(offsets, dtype=np.int64)

    def vali(self, start_index, end_index, drop_last=False):
        self.model.eval()
        pred_all = np.zeros((self.data.stock_num, end_index - start_index), dtype=np.float32)
        gt_all = np.zeros((self.data.stock_num, end_index - start_index), dtype=np.float32)
        mask_all = np.zeros((self.data.stock_num, end_index - start_index), dtype=np.float32)
        total_loss = 0.0
        total_reg_loss = 0.0
        total_rank_loss = 0.0
        count = 0
        eval_start_offset = start_index - self.args.seq_len - self.args.steps + 1
        # 评估时汇总实际门值，区分随输入变化的权重与近似常数门。
        if self.args.model == "model_b_spe_drt_feature_aggregation":
            gate_sum = torch.zeros((), dtype=torch.float64, device=self.device)
            gate_square_sum = torch.zeros((), dtype=torch.float64, device=self.device)
            gate_count = 0

        with torch.no_grad():
            for batch in self.data.eval_loader(start_index, end_index, drop_last=drop_last):
                tensor_batch = self._to_tensor_batch(batch)
                mask_batch = tensor_batch["mask"]
                price_batch = tensor_batch["price"]
                target_batch = tensor_batch["y_endo"]
                raw_gt_batch = tensor_batch["raw_gt"]
                batch_windows = int(target_batch.shape[0])
                with self._autocast_context():
                    prediction = self._predict_batch(tensor_batch)
                if self.args.model == "model_b_spe_drt_feature_aggregation":
                    semantic_gate = self._checkpoint_model().drt_gate_probabilities[..., 0].double()
                    gate_sum += semantic_gate.sum()
                    gate_square_sum += semantic_gate.square().sum()
                    gate_count += semantic_gate.numel()
                loss, reg_loss, rank_loss, return_ratio = stockmixer_loss(
                    prediction.float(),
                    target_batch,
                    price_batch,
                    mask_batch,
                    self.data.stock_num,
                    self.args.alpha,
                    self.args.prediction_target,
                )
                cols = self._offsets_to_numpy(tensor_batch["offset"]) - eval_start_offset
                eval_return = return_ratio
                return_np = eval_return.squeeze(-1).detach().cpu().numpy()
                gt_np = raw_gt_batch.squeeze(-1).detach().cpu().numpy()
                mask_np = mask_batch.squeeze(-1).detach().cpu().numpy()
                pred_all[:, cols] = return_np.T
                gt_all[:, cols] = gt_np.T
                mask_all[:, cols] = mask_np.T
                total_loss += loss.item() * batch_windows
                total_reg_loss += reg_loss.item() * batch_windows
                total_rank_loss += rank_loss.item() * batch_windows
                count += batch_windows

        # 方差下界仅消除浮点求矩的舍入误差。
        if self.args.model == "model_b_spe_drt_feature_aggregation":
            gate_mean = gate_sum / gate_count
            gate_variance = gate_square_sum / gate_count - gate_mean.square()
            self.drt_eval_gate_mean = gate_mean.item()
            self.drt_eval_gate_std = gate_variance.clamp_min(0.0).sqrt().item()

        metrics = evaluate(
            pred_all,
            gt_all,
            mask_all,
            stock_num=self.data.stock_num,
            portfolio_size=self.args.portfolio_size,
        )
        self.model.train()
        return total_loss / count, total_reg_loss / count, total_rank_loss / count, metrics, count

    def _log_start_blocks(self):
        runtime_info = get_runtime_info(
            self.args,
            self.precision_effective,
            self.amp_enabled,
            self.tf32_enabled,
            self.compile_effective,
            self.compile_reason,
            self.optimizer_fused_effective,
            self.optimizer_fused_disabled_reason,
        )
        params = model_param_count(self._checkpoint_model())
        self._log("==================== Runtime Env ====================")
        for key, value in runtime_info.items():
            self._log(f"- {key}: {value}")
        self._log("")
        self._log("==================== Model Info ====================")
        self._log(f"- model: {self.args.model}")
        self._log(f"- trainable_params_m: {params['trainable_m']:.3f}")
        self._log(
            "- moe_load_balance_weight: {} (supervised stage only)".format(
                self.args.moe_load_balance_weight
            )
        )
        self._log("")
        self._log("==================== Dataset Info ====================")
        self._log(
            "- data: {}, stocks={}, dates={}, features={}, valid_index={}, test_index={}, "
            "train_windows={}, batch_size={}, train_shuffle=True, train_drop_last={}, valid_drop_last={}, test_drop_last=False".format(
                self.args.data,
                self.data.stock_num,
                self.data.trade_dates,
                self.data.feature_num,
                self.data.valid_index,
                self.data.test_index,
                len(self.data.train_offsets),
                self.args.batch_size,
                bool(self.args.train_drop_last),
                bool(self.args.valid_drop_last),
            )
        )
        self._log(f"- feature_set: {self.data.feature_config['feature_set']}")
        self._log(f"- use_ohlcv_features: {self.data.feature_config['use_ohlcv_features']}")
        self._log(f"- use_market_features: {self.data.feature_config['use_market_features']}")
        self._log(f"- ma_windows: {self.data.feature_config['ma_windows']}")
        self._log(f"- return_momentum_windows: {self.data.feature_config['return_momentum_windows']}")
        self._log(f"- market_momentum_window: {self.data.feature_config['market_momentum_window']}")
        self._log(f"- market_drawdown_window: {self.data.feature_config['market_drawdown_window']}")
        self._log(f"- volume_surprise_window: {self.data.feature_config['volume_surprise_window']}")
        self._log(f"- normalization_scope: {self.data.feature_config['normalization_scope']}")
        self._log(f"- feature_names: {self.data.feature_names}")
        self._log(f"- feature_groups: {self.data.feature_config['feature_groups']}")
        self._log(f"- feature_group_dims: {self.args.feature_group_dims}")
        self._log(f"- target_mode: {self.args.target_mode}")
        self._log(f"- prediction_target: {self.args.prediction_target}")
        self._log("")
        return runtime_info, params

    def _log_progress_block(
        self,
        epoch,
        train_loss,
        train_supervised_loss,
        train_moe_load_balance_loss,
        train_weighted_moe_load_balance_loss,
        val_loss,
        val_perf,
        best_epoch,
        best_valid_perf,
        lr,
        epoch_seconds,
        valid_seconds,
        train_steps,
        valid_steps,
        clip_ratio,
        max_grad_norm_seen,
        patience_counter,
        saved,
    ):
        best_value = None if best_valid_perf is None else best_valid_perf[self.args.early_stop_metric]
        current = val_perf[self.args.early_stop_metric]
        mem = "N/A"
        if self.device.type == "cuda":
            max_alloc_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
            total_gb = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
            mem = "{:.2f}/{:.1f}GB".format(max_alloc_gb, total_gb)
        train_speed = train_steps / max(epoch_seconds, 1e-9)
        valid_speed = valid_steps / max(valid_seconds, 1e-9)
        saved_text = "yes" if saved else "no"
        valid_metrics = " ".join(
            f"{name}={value}" for name, value in self._progress_metric_cells(val_perf)
        )
        best_text = "none" if best_value is None else self._fmt(best_value, 6)
        line = (
            "Epoch {epoch:03d}/{total_epochs} run={run_index}/{total_runs} "
            "train_loss={train_loss} sup_loss={supervised_loss} moe_lb={moe_load_balance_loss} "
            "moe_lb_weighted={weighted_moe_load_balance_loss} valid_loss={valid_loss} "
            "{monitor_name}={current} "
            "best={best} best_ep={best_epoch} patience={patience_counter}/{patience} "
            "lr={lr} grad={grad} clip={clip} train_speed={train_speed}/s "
            "valid_speed={valid_speed}/s mem={mem} prec={precision} saved={saved} "
            "valid_metrics=[{valid_metrics}]"
        ).format(
            epoch=epoch,
            total_epochs=self.args.train_epochs,
            run_index=int(self.args.run_id) + 1,
            total_runs=int(self.args.total_runs),
            train_loss=self._fmt(train_loss, 6),
            supervised_loss=self._fmt(train_supervised_loss, 6),
            moe_load_balance_loss=self._fmt(train_moe_load_balance_loss, 6),
            weighted_moe_load_balance_loss=self._fmt(train_weighted_moe_load_balance_loss, 6),
            valid_loss=self._fmt(val_loss, 6),
            monitor_name=self._metric_label(self.args.early_stop_metric),
            current=self._fmt(current, 6),
            best=best_text,
            best_epoch=best_epoch if best_epoch else "none",
            patience_counter=patience_counter,
            patience=self.args.patience,
            lr=f"{float(lr):.2e}",
            grad=self._fmt(max_grad_norm_seen, 4),
            clip=self._fmt(clip_ratio, 4),
            train_speed=self._fmt(train_speed, 1),
            valid_speed=self._fmt(valid_speed, 1),
            mem=mem,
            precision=self.precision_effective,
            saved=saved_text,
            valid_metrics=valid_metrics,
        )
        self._log(line)

    def _metric_label(self, name):
        return self.metric_display_names[name]

    def _progress_metric_cells(self, metrics):
        return self._metric_cells(metrics, self.primary_metric_order)

    def _metric_cells(self, metrics, order=None):
        cells = []
        if order is None:
            ordered = [name for name in self.metric_display_order if name in metrics]
            ordered.extend(name for name in metrics.keys() if name not in ordered and name != "mae")
        else:
            ordered = [name for name in order if name in metrics]
        for name in ordered:
            display_name = self._metric_label(name)
            cells.append((display_name, self._fmt(metrics[name], 6)))
        limit = len(order) if order is not None else len(self.primary_metric_order)
        return cells[:limit]

    def train(self, setting):
        path = os.path.join(self.args.checkpoints, setting)
        ensure_dir(path)
        ensure_dir(self.args.results)

        optimizer = self._select_optimizer()
        nbatches = len(self.data.train_loader(0))
        scheduler = select_scheduler(optimizer, self.args, nbatches=nbatches)
        scheduler_steps_per_batch = self.args.lr_scheduler == "cosine_hard_restarts"
        early_stopping = EarlyStoppingByMetric(
            patience=self.args.patience,
            metric_name=self.args.early_stop_metric,
            mode=self.args.early_stop_mode,
            verbose=False,
            delta=float(self.args.min_delta),
            logger=self.logger,
        )

        best_valid_perf = None
        best_test_perf = None
        best_epoch = 0
        total_updates = 0

        train_started = time.time()
        runtime_info, params = self._log_start_blocks()
        model_text = str(self._checkpoint_model())

        for epoch in range(self.args.train_epochs):
            epoch_time = time.time()
            self.model.train()
            np.random.shuffle(self.data.train_offsets)

            train_loss = 0.0
            train_supervised_loss = 0.0
            train_moe_load_balance_loss = 0.0
            train_weighted_moe_load_balance_loss = 0.0
            train_drt_load_balance_loss = 0.0
            train_reg_loss = 0.0
            train_rank_loss = 0.0
            train_steps = 0
            train_batches = 0
            clip_count = 0
            max_grad_norm_seen = 0.0

            for batch in self.data.train_loader(epoch):
                tensor_batch = self._to_tensor_batch(batch)
                mask_batch = tensor_batch["mask"]
                price_batch = tensor_batch["price"]
                target_batch = tensor_batch["y_endo"]
                batch_windows = int(target_batch.shape[0])
                optimizer.zero_grad()
                with self._autocast_context():
                    prediction = self._predict_batch(tensor_batch)
                supervised_loss, reg_loss, rank_loss, _ = stockmixer_loss(
                    prediction.float(),
                    target_batch,
                    price_batch,
                    mask_batch,
                    self.data.stock_num,
                    self.args.alpha,
                    self.args.prediction_target,
                )
                moe_load_balance_loss = self._moe_load_balance_loss()
                weighted_moe_load_balance_loss = (
                    float(self.args.moe_load_balance_weight) * moe_load_balance_loss
                )
                loss = supervised_loss + weighted_moe_load_balance_loss
                # 新聚合门单独均衡两塔，不改变 ACE 的原有正则权重。
                if self.args.model == "model_b_spe_drt_feature_aggregation":
                    drt_load_balance_loss = self._checkpoint_model().drt_load_balance_loss()
                    loss = loss + self.args.drt_load_balance_weight * drt_load_balance_loss
                    train_drt_load_balance_loss += drt_load_balance_loss.item() * batch_windows
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.max_grad_norm, error_if_nonfinite=True
                )
                grad_norm_value = float(grad_norm.detach().cpu().item() if hasattr(grad_norm, "detach") else grad_norm)
                max_grad_norm_seen = max(max_grad_norm_seen, grad_norm_value)
                if grad_norm_value > self.args.max_grad_norm:
                    clip_count += 1
                optimizer.step()
                if scheduler is not None and scheduler_steps_per_batch:
                    scheduler.step()
                total_updates += 1
                train_loss += loss.item() * batch_windows
                train_supervised_loss += supervised_loss.item() * batch_windows
                train_moe_load_balance_loss += moe_load_balance_loss.item() * batch_windows
                train_weighted_moe_load_balance_loss += (
                    weighted_moe_load_balance_loss.item() * batch_windows
                )
                train_reg_loss += reg_loss.item() * batch_windows
                train_rank_loss += rank_loss.item() * batch_windows
                train_steps += batch_windows
                train_batches += 1

            train_seconds = time.time() - epoch_time
            train_loss /= train_steps
            train_supervised_loss /= train_steps
            train_moe_load_balance_loss /= train_steps
            train_weighted_moe_load_balance_loss /= train_steps
            train_reg_loss /= train_steps
            train_rank_loss /= train_steps

            valid_start = time.time()
            val_loss, val_reg_loss, val_rank_loss, val_perf, valid_steps = self.vali(
                self.data.valid_index, self.data.test_index, drop_last=self.args.valid_drop_last
            )
            valid_seconds = time.time() - valid_start
            monitor_value = val_perf[self.args.early_stop_metric]
            if scheduler is not None and not scheduler_steps_per_batch:
                if self.args.lr_scheduler == "plateau":
                    scheduler.step(monitor_value)
                else:
                    scheduler.step()
            elif scheduler is None:
                adjust_learning_rate(optimizer, epoch + 1, self.args)

            improved = early_stopping(monitor_value)
            min_epoch_ready = (epoch + 1) >= int(self.args.min_epochs)
            min_update_ready = total_updates >= int(self.args.min_updates)
            stop_allowed = min_epoch_ready and min_update_ready
            if improved:
                best_epoch = epoch + 1
                best_valid_perf = val_perf
                best_test_perf = None
                self._save_metrics(setting, best_epoch, best_valid_perf, best_test_perf)
                torch.save(
                    self._checkpoint_model().state_dict(),
                    self.args.obs_run_paths["best_ckpt"],
                )
            torch.save(
                self._checkpoint_model().state_dict(),
                self.args.obs_run_paths["last_ckpt"],
            )
            if early_stopping.early_stop and not stop_allowed:
                early_stopping.early_stop = False
                early_stopping.counter = min(early_stopping.counter, max(self.args.patience - 1, 0))
            epoch_seconds = time.time() - epoch_time
            clip_ratio = clip_count / max(train_batches, 1)
            epoch_runtime_protocol = build_runtime_protocol(
                self.args,
                self.precision_effective,
                self.amp_enabled,
                self.tf32_enabled,
                self.compile_effective,
                self.compile_reason,
                fused_effective=self.optimizer_fused_effective,
                fused_disabled_reason=self.optimizer_fused_disabled_reason,
                clip_ratio=clip_ratio,
                max_grad_norm_seen=max_grad_norm_seen,
            )
            self.history.append(
                {
                    "epoch": epoch + 1,
                    "train": {
                        "loss": train_loss,
                        "supervised_loss": train_supervised_loss,
                        "moe_load_balance_loss": train_moe_load_balance_loss,
                        "weighted_moe_load_balance_loss": train_weighted_moe_load_balance_loss,
                        "reg_loss": train_reg_loss,
                        "rank_loss": train_rank_loss,
                    },
                    "valid": {"loss": val_loss, "reg_loss": val_reg_loss, "rank_loss": val_rank_loss, **val_perf},
                    "lr": optimizer.param_groups[0]["lr"],
                    "seconds": epoch_seconds,
                    "clip_ratio": clip_ratio,
                    "max_grad_norm_seen": max_grad_norm_seen,
                    "total_updates": total_updates,
                    "min_epochs": int(self.args.min_epochs),
                    "min_updates": int(self.args.min_updates),
                    "runtime_protocol": epoch_runtime_protocol,
                    "best_epoch": best_epoch,
                }
            )
            # 保存两专家均衡项的实际数值，便于核对其优化贡献。
            if self.args.model == "model_b_spe_drt_feature_aggregation":
                train_drt_load_balance_loss /= train_steps
                weighted_drt_load_balance_loss = self.args.drt_load_balance_weight * train_drt_load_balance_loss
                self.history[-1]["train"]["drt_load_balance_loss"] = train_drt_load_balance_loss
                self.history[-1]["train"]["weighted_drt_load_balance_loss"] = weighted_drt_load_balance_loss
                self._log(f"DRT aggregation: balance_loss={train_drt_load_balance_loss:.6f} weighted_balance_loss={weighted_drt_load_balance_loss:.8f}")
            self._log_progress_block(
                epoch + 1,
                train_loss,
                train_supervised_loss,
                train_moe_load_balance_loss,
                train_weighted_moe_load_balance_loss,
                val_loss,
                val_perf,
                best_epoch,
                best_valid_perf,
                optimizer.param_groups[0]["lr"],
                train_seconds,
                valid_seconds,
                train_steps,
                valid_steps,
                clip_ratio,
                max_grad_norm_seen,
                early_stopping.counter,
                improved,
            )
            if early_stopping.early_stop:
                self._log(
                    "[EarlyStop] epoch={} counter={} patience={} monitor={} best_epoch={}".format(
                        epoch + 1,
                        early_stopping.counter,
                        self.args.patience,
                        self.args.early_stop_metric,
                        best_epoch,
                    )
                )
                break

        stopped_epoch = epoch + 1 if early_stopping.early_stop else None
        early_stop_summary = {
            "requested": True,
            "effective": True,
            "patience_requested": self.args.patience_requested,
            "patience_effective": int(self.args.patience),
            "min_delta": float(self.args.min_delta),
            "monitor_metric": self.args.early_stop_metric,
            "monitor_source": "valid",
            "monitor_direction": self.args.early_stop_mode,
            "best": None if best_valid_perf is None else best_valid_perf[self.args.early_stop_metric],
            "best_epoch": best_epoch,
            "counter": int(early_stopping.counter),
            "stopped_epoch": stopped_epoch,
            "stop_reason": "patience_exhausted" if stopped_epoch is not None else None,
        }
        best_model_path = (
            self.args.obs_run_paths["best_ckpt"]
            if os.path.exists(self.args.obs_run_paths["best_ckpt"])
            else self.args.obs_run_paths["last_ckpt"]
        )
        self._checkpoint_model().load_state_dict(self._load_checkpoint(best_model_path))
        save_history(self.history, self.args.obs_run_paths["history"], early_stop_summary)
        save_training_curve(self.history, self.args.obs_run_paths["curve"])
        self._log("==================== Training Summary ====================")
        self._log(
            "best_epoch={} monitor={} best_valid={}".format(
                best_epoch,
                self.args.early_stop_metric,
                "N/A" if best_valid_perf is None else self._fmt(best_valid_perf[self.args.early_stop_metric], 6),
            )
        )
        self._log("")
        return {
            "best_epoch": best_epoch,
            "best_valid": best_valid_perf,
            "test_at_best_valid": best_test_perf,
            "train_seconds": time.time() - train_started,
            "runtime_info": runtime_info,
            "early_stop": early_stop_summary,
            "params": params,
            "model_text": model_text,
        }

    def test(self, setting, test=0):
        if test:
            checkpoint_path = os.path.join(self.args.checkpoints, setting, "checkpoint.pth")
            self._log(f"loading model: {checkpoint_path}")
            self._checkpoint_model().load_state_dict(self._load_checkpoint(checkpoint_path))

        test_perf = self._evaluate_test_only()
        self._log("========== Test Results (Run{}) ==========".format(int(self.args.run_id) + 1))
        self._log("source: best checkpoint selected by valid {}".format(self.args.early_stop_metric))
        self._log("Final test performance: {}".format(self._format_metrics(test_perf)))
        self._log("")
        return test_perf

    def _evaluate_test_only(self):
        _, _, _, test_perf, _ = self.vali(self.data.test_index, self.data.trade_dates, drop_last=False)
        return test_perf

    def _load_checkpoint(self, checkpoint_path):
        return torch.load(checkpoint_path, map_location=self.device, weights_only=True)

    def _fmt(self, value, digits=6):
        return f"{float(value):.{digits}f}"

    def _format_metrics(self, metrics):
        return (
            f"MAE↓:{{}}, RankIC↑:{{}}, RankICIR↑:{{}}, "
            f"IRR@{PORTFOLIO_SIZE}↑:{{}}, SR@{PORTFOLIO_SIZE}↑:{{}}, "
            f"AR@{PORTFOLIO_SIZE}↑:{{}}, KT↑:{{}}, MAPE↓:{{}}, "
            f"P@{PORTFOLIO_SIZE}↑:{{}}, IC↑:{{}}, ICIR↑:{{}}"
        ).format(
            self._fmt(metrics["mae"]),
            self._fmt(metrics["RankIC"]),
            self._fmt(metrics["RankICIR"]),
            self._fmt(metrics[IRR_KEY]),
            self._fmt(metrics[SR_KEY]),
            self._fmt(metrics[AR_KEY]),
            self._fmt(metrics["KT"]),
            self._fmt(metrics["mape"]),
            self._fmt(metrics[PRECISION_KEY]),
            self._fmt(metrics["IC"]),
            self._fmt(metrics["ICIR"]),
        )

    def _json_safe(self, value):
        if isinstance(value, dict):
            return {key: self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return self._json_safe(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _save_metrics(self, setting, best_epoch, valid_perf, test_perf):
        result_path = os.path.join(self.args.obs_paths["native_dir"], f"{setting}.json")
        saved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        payload = {
            "saved_at": saved_at,
            "run_id": self.args.run_id,
            "seed": self.args.random_seed,
            "best_epoch": best_epoch,
            "early_stop_metric": self.args.early_stop_metric,
            "precision_requested": self.precision_requested,
            "precision_effective": self.precision_effective,
            "amp_effective": self.amp_enabled,
            "tf32_effective": self.tf32_enabled,
            "torch_compile_requested": self.compile_requested,
            "torch_compile_effective": self.compile_effective,
            "torch_compile_mode": self.args.torch_compile_mode,
            "torch_compile_reason": self.compile_reason,
            "max_grad_norm": self.args.max_grad_norm,
            "best_valid": valid_perf,
            "test_at_best_valid": test_perf,
        }
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(self._json_safe(payload), f, indent=2)

    def _append_result_txt(self, setting, best_epoch, best_valid_perf, best_test_perf):
        result_path = os.path.join(self.args.obs_paths["native_dir"], "result.txt")
        saved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "=" * 80,
            f"time: {saved_at}",
            f"setting: {setting}",
            f"run_id: {self.args.run_id}",
            f"seed: {self.args.random_seed}",
            f"market: {self.args.data}",
            f"best_epoch_by_valid_{self.args.early_stop_metric}: {best_epoch}",
            "config: seq_len={}, steps={}, market_num={}, scale_factor={}, alpha={}, output_scale={}, learning_rate={}, "
            "valid_index={}, test_index={}, data_start_index={}, early_stop_metric={}, patience={}, precision={}, "
            "torch_compile={}, torch_compile_mode={}, max_grad_norm={}".format(
                self.args.seq_len,
                self.args.steps,
                self.args.market_num,
                self.args.scale_factor,
                self._fmt(self.args.alpha),
                self._fmt(self.args.output_scale),
                self._fmt(self.args.learning_rate),
                self.args.valid_index,
                self.args.test_index,
                self.args.data_start_index,
                self.args.early_stop_metric,
                self.args.patience,
                self.precision_effective,
                self.compile_effective,
                self.args.torch_compile_mode,
                self._fmt(self.args.max_grad_norm),
            ),
            f"best_valid_metrics: {self._format_metrics(best_valid_perf)}",
            "test_metrics_at_best_valid_epoch: final_test_only",
            "",
        ]
        with open(result_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))
