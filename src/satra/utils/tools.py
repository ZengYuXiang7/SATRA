import math
import os
from functools import partial

import numpy as np
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _cosine_hard_restarts_warmup_lr_lambda(
    current_step,
    *,
    num_warmup_steps,
    num_training_steps,
    num_cycles,
    min_ratio,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    if progress >= 1.0:
        return min_ratio
    return min_ratio + (1.0 - min_ratio) * 0.5 * (
        1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))
    )


def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    num_cycles=1,
    last_epoch=-1,
    min_ratio=1e-2,
):
    lr_lambda = partial(
        _cosine_hard_restarts_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_ratio=min_ratio,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def select_scheduler(optimizer, args, nbatches=None):
    if args.lr_scheduler == "plateau":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=args.early_stop_mode,
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
        )
    if args.lr_scheduler == "step":
        return lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_factor
        )
    if args.lr_scheduler == "cosine":
        return lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.train_epochs, eta_min=args.min_lr
        )
    if args.lr_scheduler == "linear":
        total_epochs = int(args.train_epochs)
        return LambdaLR(
            optimizer,
            lambda completed_epochs: max(
                0.0,
                1.0 - float(completed_epochs) / float(total_epochs),
            ),
        )
    if args.lr_scheduler == "cosine_hard_restarts":
        if nbatches is None or int(nbatches) <= 0:
            raise ValueError("nbatches must be positive for cosine_hard_restarts scheduler")
        num_training_steps = int(nbatches) * int(args.train_epochs)
        num_warmup_steps = float(args.warmup_ratio) * num_training_steps
        return get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=int(args.num_cycles),
            min_ratio=float(args.min_ratio),
        )
    return None


def adjust_learning_rate(optimizer, epoch, args):
    lr_adjust = {}
    base_lr = args.learning_rate
    policy = args.lr_scheduler

    if policy == "type1":
        lr_adjust = {epoch: base_lr * (0.5 ** ((epoch - 1) // 1))}
    elif policy == "type2":
        lr_adjust = {
            2: 5e-5,
            4: 1e-5,
            6: 5e-6,
            8: 1e-6,
            10: 5e-7,
            15: 1e-7,
            20: 5e-8,
        }
    elif policy == "type3":
        lr_adjust = {
            epoch: base_lr if epoch < 3 else base_lr * (0.8 ** ((epoch - 3) // 1))
        }
    elif policy == "constant":
        lr_adjust = {epoch: base_lr}
    elif policy == "3":
        lr_adjust = {epoch: base_lr if epoch < 10 else base_lr * 0.1}
    elif policy == "4":
        lr_adjust = {epoch: base_lr if epoch < 15 else base_lr * 0.1}
    elif policy == "5":
        lr_adjust = {epoch: base_lr if epoch < 25 else base_lr * 0.1}
    elif policy == "6":
        lr_adjust = {epoch: base_lr if epoch < 5 else base_lr * 0.1}

    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


class EarlyStoppingByMetric:
    """Track the single formal checkpoint criterion: maximum validation RankIC."""

    def __init__(
        self,
        patience=5,
        metric_name="RankIC",
        mode="max",
        verbose=False,
        delta=0.0,
        logger=None,
    ):
        if metric_name != "RankIC" or mode != "max":
            raise ValueError(
                "SATRA checkpoints must be selected by maximum validation RankIC"
            )
        self.patience = patience
        self.metric_name = metric_name
        self.mode = mode
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.logger = logger

    def _log(self, message):
        if self.logger is not None:
            self.logger.log(message)

    def __call__(self, value):
        if not np.isfinite(value):
            raise FloatingPointError(
                f"Validation {self.metric_name} is non-finite: {value!r}"
            )
        score = value
        if self.best_score is None:
            self.best_score = score
            if self.verbose:
                self._log(f"Validation {self.metric_name} improved to {value:.6f}.")
            return True
        if score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

        self.best_score = score
        if self.verbose:
            self._log(f"Validation {self.metric_name} improved to {value:.6f}.")
        self.counter = 0
        return True
