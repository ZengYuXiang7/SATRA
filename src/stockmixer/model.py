"""
StockMixer: A Simple Yet Strong MLP-Based Architecture for Stock Price Forecasting.

Source type: official-source-derived.

This file keeps the architecture and optional ranking loss in one PyTorch file.
Training scripts, data loaders, and evaluator code are intentionally omitted.

Expected input:
    inputs: Tensor [num_stocks, time_steps, channels]
    or Tensor [batch, num_stocks, time_steps, channels]

Output:
    Tensor [num_stocks, 1]
    or Tensor [batch, num_stocks, 1]
"""

import torch
from torch import nn
import torch.nn.functional as F


def stockmixer_loss(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    base_price: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Official regression + pairwise ranking loss used by StockMixer."""
    device = prediction.device
    batch_size = prediction.shape[0]
    all_one = torch.ones(batch_size, 1, dtype=torch.float32, device=device)
    return_ratio = torch.div(prediction - base_price, base_price.clamp_min(1e-12))

    reg_loss = F.mse_loss(return_ratio * mask, ground_truth * mask)
    pre_pw_dif = return_ratio @ all_one.t() - all_one @ return_ratio.t()
    gt_pw_dif = all_one @ ground_truth.t() - ground_truth @ all_one.t()
    mask_pw = mask @ mask.t()
    rank_loss = torch.mean(F.relu(pre_pw_dif * gt_pw_dif * mask_pw))
    loss = reg_loss + alpha * rank_loss
    return loss, reg_loss, rank_loss, return_ratio


class MixerBlock(nn.Module):
    """Two-layer MLP mixer used for channel mixing."""

    def __init__(self, mlp_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout
        self.dense_1 = nn.Linear(mlp_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dense_2 = nn.Linear(hidden_dim, mlp_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_1(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training) if self.dropout else x
        x = self.dense_2(x)
        x = F.dropout(x, p=self.dropout, training=self.training) if self.dropout else x
        return x


class TriU(nn.Module):
    """Upper-triangular causal temporal mixer from the public StockMixer code."""

    def __init__(self, time_step: int):
        super().__init__()
        self.time_step = time_step
        self.tri_u = nn.ModuleList([nn.Linear(i + 1, 1) for i in range(time_step)])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: [num_stocks, channels, time_step]
        x = self.tri_u[0](inputs[:, :, 0].unsqueeze(-1))
        for i in range(1, self.time_step):
            x = torch.cat([x, self.tri_u[i](inputs[:, :, : i + 1])], dim=-1)
        return x


class Mixer2dTriU(nn.Module):
    """Indicator mixer + temporal mixer block."""

    def __init__(self, time_steps: int, channels: int):
        super().__init__()
        self.norm_time = nn.LayerNorm([time_steps, channels])
        self.norm_channel = nn.LayerNorm([time_steps, channels])
        self.time_mixer = TriU(time_steps)
        self.channel_mixer = MixerBlock(channels, channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.norm_time(inputs)
        x = x.permute(0, 2, 1)
        x = self.time_mixer(x)
        x = x.permute(0, 2, 1)

        x = self.norm_channel(x + inputs)
        y = self.channel_mixer(x)
        return x + y


class MultTime2dMixer(nn.Module):
    """Multi-scale temporal mixer used before stock mixing."""

    def __init__(self, time_step: int, channel: int, scale_dim: int):
        super().__init__()
        self.mix_layer = Mixer2dTriU(time_step, channel)
        self.scale_mix_layer = Mixer2dTriU(scale_dim, channel)

    def forward(self, inputs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = self.scale_mix_layer(y)
        x = self.mix_layer(inputs)
        return torch.cat([inputs, x, y], dim=1)


class NoGraphMixer(nn.Module):
    """Cross-stock MLP mixer; StockMixer does not use predefined graph knowledge."""

    def __init__(self, stocks: int, hidden_dim: int = 20):
        super().__init__()
        self.dense1 = nn.Linear(stocks, hidden_dim)
        self.activation = nn.Hardswish()
        self.dense2 = nn.Linear(hidden_dim, stocks)
        self.layer_norm_stock = nn.LayerNorm(stocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: [num_stocks, time_features] or [batch, num_stocks, time_features]
        if inputs.dim() == 2:
            x = inputs.permute(1, 0)
            x = self.layer_norm_stock(x)
            x = self.dense1(x)
            x = self.activation(x)
            x = self.dense2(x)
            return x.permute(1, 0)
        if inputs.dim() != 3:
            raise ValueError(
                "NoGraphMixer expects [num_stocks,time_features] or "
                f"[batch,num_stocks,time_features], got {tuple(inputs.shape)}"
            )
        x = inputs.permute(0, 2, 1)
        x = self.layer_norm_stock(x)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        return x.permute(0, 2, 1)


class StockMixer(nn.Module):
    """StockMixer core model."""

    def __init__(
        self,
        stocks: int,
        time_steps: int,
        channels: int,
        market: int = 20,
        scale: int = 3,
        scale_dim: int | None = None,
    ):
        super().__init__()
        if scale_dim is None:
            scale_dim = time_steps // 2
        if scale_dim < 1:
            raise ValueError("scale_dim must be positive.")

        self.time_steps = time_steps
        # Kept for paper/script parity. The public model receives scale_factor
        # from train.py; the core architecture uses the stride-2 convolution
        # length as the multi-scale branch size.
        self.scale = scale
        self.scale_dim = scale_dim
        self.mixer = MultTime2dMixer(time_steps, channels, scale_dim=scale_dim)
        self.channel_fc = nn.Linear(channels, 1)
        self.time_fc = nn.Linear(time_steps * 2 + scale_dim, 1)
        self.conv = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=2, stride=2)
        self.stock_mixer = NoGraphMixer(stocks, market)
        self.time_fc_stock = nn.Linear(time_steps * 2 + scale_dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        is_batched = inputs.dim() == 4
        if inputs.dim() not in (3, 4):
            raise ValueError(
                "StockMixer expects [num_stocks,time_steps,channels] or "
                f"[batch,num_stocks,time_steps,channels], got {tuple(inputs.shape)}"
            )
        time_dim = 2 if is_batched else 1
        if inputs.size(time_dim) != self.time_steps:
            raise ValueError(f"expected time_steps={self.time_steps}, got {inputs.size(time_dim)}.")

        if is_batched:
            batch_size, stock_num, time_steps, channels = inputs.shape
            flat_inputs = inputs.reshape(batch_size * stock_num, time_steps, channels)
        else:
            batch_size = None
            stock_num = inputs.shape[0]
            flat_inputs = inputs

        x = flat_inputs.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)

        if x.size(1) != self.scale_dim:
            raise ValueError(
                f"conv scale length {x.size(1)} does not match scale_dim={self.scale_dim}; "
                "use an even time_steps value or pass scale_dim explicitly."
            )

        y = self.mixer(flat_inputs, x)
        y = self.channel_fc(y).squeeze(-1)
        if is_batched:
            y = y.reshape(batch_size, stock_num, -1)
        z = self.stock_mixer(y)
        return self.time_fc(y) + self.time_fc_stock(z)


Model = StockMixer
