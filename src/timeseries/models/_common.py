import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def check_input(x, seq_len, d):
    if x.dim() != 3:
        raise ValueError(f"expected [bs, seq_len, d], got {tuple(x.shape)}")
    if x.size(1) != seq_len:
        raise ValueError(f"expected seq_len={seq_len}, got {x.size(1)}")
    if x.size(2) != d:
        raise ValueError(f"expected d={d}, got {x.size(2)}")


class RevIN(nn.Module):
    def __init__(self, d, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, 1, d))
            self.bias = nn.Parameter(torch.zeros(1, 1, d))
        self._mean = None
        self._std = None

    def norm(self, x):
        self._mean = x.mean(dim=1, keepdim=True).detach()
        self._std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps).detach()
        x = (x - self._mean) / self._std
        if self.affine:
            x = x * self.weight + self.bias
        return x

    def denorm(self, x):
        if self.affine:
            x = (x - self.bias) / self.weight.clamp_min(self.eps)
        return x * self._std + self._mean


class MovingAverageDecomp(nn.Module):
    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size = int(kernel_size)

    def forward(self, x):
        pad = (self.kernel_size - 1) // 2
        front = x[:, :1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        padded = torch.cat([front, x, end], dim=1).transpose(1, 2)
        trend = F.avg_pool1d(padded, kernel_size=self.kernel_size, stride=1).transpose(1, 2)
        trend = trend[:, : x.size(1), :]
        return x - trend, trend


class TimeHead(nn.Module):
    def __init__(self, seq_len, pred_len):
        super().__init__()
        self.proj = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        return self.proj(x.transpose(1, 2)).transpose(1, 2)


class ChannelMLP(nn.Module):
    def __init__(self, d, hidden, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d),
        )

    def forward(self, x):
        return self.net(x)


class TokenMLP(nn.Module):
    def __init__(self, length, hidden, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(length)
        self.net = nn.Sequential(
            nn.Linear(length, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, length),
        )

    def forward(self, x):
        y = x.transpose(1, 2)
        y = self.net(self.norm(y))
        return y.transpose(1, 2)


class MixerBlock(nn.Module):
    def __init__(self, length, d, hidden=128, dropout=0.1):
        super().__init__()
        self.token_mlp = TokenMLP(length, hidden, dropout)
        self.channel_mlp = ChannelMLP(d, hidden, dropout)

    def forward(self, x):
        x = x + self.token_mlp(x)
        return x + self.channel_mlp(x)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, length, d_model):
        super().__init__()
        pe = torch.zeros(length, d_model)
        pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class SimpleTransformer(nn.Module):
    def __init__(self, d_model=128, n_heads=4, layers=2, d_ff=256, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, x):
        return self.encoder(x)


class PatchEncoder(nn.Module):
    def __init__(self, seq_len, patch_len=8, stride=4, d_model=128, n_heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.patch_num = max(1, math.floor((seq_len - self.patch_len) / self.stride) + 1)
        self.patch_proj = nn.Linear(self.patch_len, d_model)
        self.pos = SinusoidalPositionalEncoding(self.patch_num, d_model)
        self.encoder = SimpleTransformer(d_model, n_heads, layers, d_model * 4, dropout)

    def forward(self, x):
        b, t, d = x.shape
        if t < self.patch_len:
            x = F.pad(x.transpose(1, 2), (self.patch_len - t, 0), mode="replicate").transpose(1, 2)
        z = x.transpose(1, 2).unfold(-1, self.patch_len, self.stride)
        z = z[:, :, : self.patch_num, :]
        z = z.reshape(b * d, self.patch_num, self.patch_len)
        z = self.patch_proj(z)
        z = self.encoder(self.pos(z))
        return z.reshape(b, d, self.patch_num, -1)


class PatchFlattenHead(nn.Module):
    def __init__(self, patch_num, d_model, pred_len):
        super().__init__()
        self.head = nn.Linear(patch_num * d_model, pred_len)

    def forward(self, z):
        b, d, _, _ = z.shape
        y = self.head(z.reshape(b, d, -1))
        return y.transpose(1, 2)
