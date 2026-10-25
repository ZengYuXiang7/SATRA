"""
HyperStockGAT / Hyperbolic Graph Learning for Algorithmic Trading.

This module ports the useful parts of the official HyperStockGAT code to this
project's unified stock-window contract:

    batch["inputs"]: Tensor [batch, num_stocks, lookback, num_features]
    batch["mask"]  : Tensor [batch, num_stocks, 1]
    output         : Tensor [batch, num_stocks, 1], raw-return score

The official implementation builds a binary stock graph from RSR relation files,
maps per-time-step stock features to the Poincare ball, applies hyperbolic graph
aggregation, then collapses the temporal dimension. This version keeps that
meaning but removes official hard-coding for NASDAQ's 1026 stocks, 5 features,
and a local graph.npy file.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def artanh(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(min=-1 + 1e-7, max=1 - 1e-7)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class PoincareBall:
    """Poincare ball operations used by the official HyperStockGAT HGCN layer."""

    min_norm = 1e-15

    @staticmethod
    def _eps(dtype: torch.dtype) -> float:
        return 4e-3 if dtype == torch.float32 else 1e-5

    def proj(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1.0 - self._eps(x.dtype)) / torch.sqrt(c)
        projected = x / norm * maxnorm
        return torch.where(norm > maxnorm, projected, x)

    def proj_tan0(self, u: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return u

    def expmap0(self, u: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        sqrt_c = torch.sqrt(c)
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), self.min_norm)
        gamma = torch.tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma

    def logmap0(self, p: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        sqrt_c = torch.sqrt(c)
        p_norm = torch.clamp_min(p.norm(dim=-1, p=2, keepdim=True), self.min_norm)
        return artanh(sqrt_c * p_norm) * p / (sqrt_c * p_norm)

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, dim: int = -1) -> torch.Tensor:
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        xy = (x * y).sum(dim=dim, keepdim=True)
        numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denominator = 1 + 2 * c * xy + c.pow(2) * x2 * y2
        return numerator / denominator.clamp_min(self.min_norm)

    def mobius_matvec(self, weight: torch.Tensor, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        sqrt_c = torch.sqrt(c)
        x_norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        mx = x @ weight.transpose(-1, -2)
        mx_norm = torch.clamp_min(mx.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        res = torch.tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
        return torch.where((mx.abs().sum(dim=-1, keepdim=True) == 0), torch.zeros_like(res), res)


class HypLinear(nn.Module):
    """Mobius linear layer from the official HGCN implementation."""

    def __init__(self, manifold: PoincareBall, in_features: int, out_features: int, c: torch.Tensor, dropout: float, bias: bool):
        super().__init__()
        self.manifold = manifold
        self.c = c
        self.dropout = float(dropout)
        self.use_bias = bool(bias)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight, gain=math.sqrt(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.c.to(device=x.device, dtype=x.dtype)
        weight = F.dropout(self.weight, self.dropout, training=self.training)
        out = self.manifold.proj(self.manifold.mobius_matvec(weight, x, c), c)
        if self.use_bias:
            hyp_bias = self.manifold.expmap0(self.manifold.proj_tan0(self.bias.view(1, -1), c), c)
            out = self.manifold.proj(self.manifold.mobius_add(out, hyp_bias, c=c), c)
        return out


class SparseHypAgg(nn.Module):
    """
    Hyperbolic neighborhood aggregation.

    Official HyperStockGAT uses DenseAtt over adj.to_dense(). This sparse version
    keeps the same attention idea while avoiding an [N,N,2D] tensor for NYSE.
    """

    def __init__(
        self,
        manifold: PoincareBall,
        c: torch.Tensor,
        in_features: int,
        relation_mask: np.ndarray,
        dropout: float,
        use_attention: bool,
        attention_alpha: float,
    ):
        super().__init__()
        self.manifold = manifold
        self.c = c
        self.dropout = float(dropout)
        self.use_attention = bool(use_attention)
        mask = np.asarray(relation_mask, dtype=bool).copy()
        np.fill_diagonal(mask, True)
        row, col = np.nonzero(mask)
        self.register_buffer("edge_row", torch.as_tensor(row, dtype=torch.long), persistent=True)
        self.register_buffer("edge_col", torch.as_tensor(col, dtype=torch.long), persistent=True)
        if self.use_attention:
            self.att_left = nn.Linear(in_features, 1, bias=False)
            self.att_right = nn.Linear(in_features, 1, bias=True)
            self.leaky_relu = nn.LeakyReLU(float(attention_alpha))

    def _edge_softmax(self, row: torch.Tensor, score: torch.Tensor, num_nodes: int) -> torch.Tensor:
        score_fp32 = score.float()
        max_per_row = torch.full(
            (num_nodes,),
            torch.finfo(score_fp32.dtype).min,
            device=score.device,
            dtype=score_fp32.dtype,
        )
        max_per_row.scatter_reduce_(0, row, score_fp32, reduce="amax", include_self=True)
        exp_score = torch.exp(score_fp32 - max_per_row[row])
        denom = torch.zeros((num_nodes,), device=score.device, dtype=score_fp32.dtype)
        denom.index_add_(0, row, exp_score)
        return (exp_score / denom[row].clamp_min(1e-12)).to(dtype=score.dtype)

    def forward(self, x: torch.Tensor, stock_mask: torch.Tensor | None = None) -> torch.Tensor:
        c = self.c.to(device=x.device, dtype=x.dtype)
        x_tangent = self.manifold.logmap0(x, c=c)
        n = x_tangent.shape[0]
        row = self.edge_row.to(device=x_tangent.device)
        col = self.edge_col.to(device=x_tangent.device)
        if stock_mask is not None:
            valid = stock_mask.to(device=x_tangent.device).bool()
            keep = valid[row] & valid[col]
            row = row[keep]
            col = col[keep]
        if row.numel() == 0:
            support = torch.zeros_like(x_tangent)
            return self.manifold.proj(self.manifold.expmap0(support, c=c), c=c)

        if self.use_attention:
            score = self.leaky_relu(self.att_left(x_tangent)[row] + self.att_right(x_tangent)[col]).squeeze(-1)
            weight = self._edge_softmax(row, score, n)
            weight = F.dropout(weight, self.dropout, training=self.training)
        else:
            degree = torch.zeros((n,), device=x_tangent.device, dtype=x_tangent.dtype)
            degree.index_add_(0, row, torch.ones_like(row, dtype=x_tangent.dtype))
            weight = degree[row].clamp_min(1.0).reciprocal()

        support = torch.zeros_like(x_tangent)
        support.index_add_(0, row, weight.unsqueeze(-1) * x_tangent[col])
        if stock_mask is not None:
            support = support * stock_mask.to(device=x_tangent.device, dtype=support.dtype).unsqueeze(-1)
        return self.manifold.proj(self.manifold.expmap0(support, c=c), c=c)


class HypAct(nn.Module):
    """Apply an activation in tangent space and map back to the Poincare ball."""

    def __init__(self, manifold: PoincareBall, c: torch.Tensor, activation):
        super().__init__()
        self.manifold = manifold
        self.c = c
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.c.to(device=x.device, dtype=x.dtype)
        tangent = self.activation(self.manifold.logmap0(x, c=c))
        return self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(tangent, c=c), c=c), c=c)


class HyperbolicGraphAttentionLayer(nn.Module):
    """Official HyperbolicGraphConvolution with sparse graph attention aggregation."""

    def __init__(
        self,
        manifold: PoincareBall,
        in_features: int,
        out_features: int,
        c: torch.Tensor,
        relation_mask: np.ndarray,
        dropout: float,
        activation,
        bias: bool,
        use_attention: bool,
        attention_alpha: float,
    ):
        super().__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c, dropout, bias)
        self.agg = SparseHypAgg(manifold, c, out_features, relation_mask, dropout, use_attention, attention_alpha)
        self.hyp_act = HypAct(manifold, c, activation)

    def forward(self, x: torch.Tensor, stock_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.linear(x)
        h = self.agg(h, stock_mask=stock_mask)
        return self.hyp_act(h)


class TemporalAttentionLayer(nn.Module):
    """Temporal attention block adapted from the official HGCN encoder."""

    def __init__(self, in_channels: int, num_stocks: int, seq_len: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_stocks = int(num_stocks)
        self.seq_len = int(seq_len)
        self.U1 = nn.Parameter(torch.empty(num_stocks))
        self.U2 = nn.Parameter(torch.empty(in_channels, num_stocks))
        self.U3 = nn.Parameter(torch.empty(in_channels))
        self.be = nn.Parameter(torch.zeros(1, seq_len, seq_len))
        self.Ve = nn.Parameter(torch.empty(seq_len, seq_len))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.U1, -0.1, 0.1)
        nn.init.xavier_uniform_(self.U2)
        nn.init.uniform_(self.U3, -0.1, 0.1)
        nn.init.xavier_uniform_(self.Ve)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, F, L]
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        rhs = torch.matmul(self.U3, x)
        product = torch.matmul(lhs, rhs)
        energy = torch.matmul(self.Ve, torch.sigmoid(product + self.be))
        return F.softmax(energy, dim=1)


class HyperStockGAT(nn.Module):
    """Unified-data HyperStockGAT baseline."""

    def __init__(
        self,
        input_dim: int,
        num_stocks: int,
        seq_len: int,
        relation_mask: np.ndarray,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.4,
        curvature: float = 1.0,
        use_attention: bool = True,
        attention_alpha: float = 0.2,
        bias: bool = True,
        output_activation: str = "leaky_relu",
    ):
        super().__init__()
        if output_activation not in {"leaky_relu", "identity"}:
            raise ValueError("output_activation must be leaky_relu or identity")
        if int(num_layers) < 1:
            raise ValueError("num_layers must be >= 1")
        self.input_dim = int(input_dim)
        self.num_stocks = int(num_stocks)
        self.seq_len = int(seq_len)
        self.hidden_size = int(hidden_size)
        self.output_activation = output_activation
        self.manifold = PoincareBall()
        self.register_buffer("curvature", torch.tensor([float(curvature)], dtype=torch.float32), persistent=True)

        self.input_temporal_attention = TemporalAttentionLayer(input_dim, num_stocks, seq_len)
        self.input_time_conv = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1)

        dims = [input_dim] + [hidden_size] * int(num_layers)
        layers = []
        for idx in range(len(dims) - 1):
            layers.append(
                HyperbolicGraphAttentionLayer(
                    manifold=self.manifold,
                    in_features=dims[idx],
                    out_features=dims[idx + 1],
                    c=self.curvature,
                    relation_mask=relation_mask,
                    dropout=dropout,
                    activation=F.relu,
                    bias=bias,
                    use_attention=use_attention,
                    attention_alpha=attention_alpha,
                )
            )
        self.graph_layers = nn.ModuleList(layers)
        self.output_temporal_attention = TemporalAttentionLayer(hidden_size, num_stocks, seq_len)
        self.temporal_readout = nn.Conv2d(seq_len, 1, kernel_size=(1, 3), padding=(0, 1))
        self.prediction = nn.Linear(hidden_size, 1)

    def _prepare_input_temporal(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, L, F]
        b, n, l, f = x.shape
        x_nfl = x.permute(0, 1, 3, 2)
        temporal_att = self.input_temporal_attention(x_nfl)
        attended = torch.matmul(x_nfl.reshape(b, n * f, l), temporal_att).reshape(b, n, f, l)
        conv = self.input_time_conv(attended.reshape(b * n, f, l)).reshape(b, n, f, l)
        return conv.permute(0, 1, 3, 2)

    def _graph_encode_one(self, x_t: torch.Tensor, stock_mask: torch.Tensor | None) -> torch.Tensor:
        c = self.curvature.to(dtype=x_t.dtype, device=x_t.device)
        h = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x_t, c), c=c), c=c)
        for layer in self.graph_layers:
            h = layer(h, stock_mask=stock_mask)
        return self.manifold.logmap0(h, c=c)

    def _forward_tensor(self, x: torch.Tensor, stock_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
            if stock_mask is not None and stock_mask.dim() == 1:
                stock_mask = stock_mask.unsqueeze(0)
        if x.dim() != 4:
            raise ValueError(f"HyperStockGAT expects [B,N,L,F] or [N,L,F], got {tuple(x.shape)}")
        b, n, l, f = x.shape
        if n != self.num_stocks:
            raise ValueError(f"stock dimension mismatch: got {n}, expected {self.num_stocks}")
        if l != self.seq_len:
            raise ValueError(f"seq_len mismatch: got {l}, expected {self.seq_len}")
        if f != self.input_dim:
            raise ValueError(f"feature dimension mismatch: got {f}, expected {self.input_dim}")
        if stock_mask is not None:
            if stock_mask.dim() == 3:
                stock_mask = stock_mask.squeeze(-1)
            if stock_mask.shape != (b, n):
                raise ValueError(f"stock_mask must be [B,N], got {tuple(stock_mask.shape)}")
            stock_mask = stock_mask.bool()

        x = self._prepare_input_temporal(x)
        outputs = []
        for batch_idx in range(b):
            valid = stock_mask[batch_idx] if stock_mask is not None else None
            per_time = []
            for time_idx in range(l):
                per_time.append(self._graph_encode_one(x[batch_idx, :, time_idx, :], valid))
            outputs.append(torch.stack(per_time, dim=1))
        h_seq = torch.stack(outputs, dim=0)  # [B, N, L, D]

        h_nfl = h_seq.permute(0, 1, 3, 2)
        temporal_att = self.output_temporal_attention(h_nfl)
        attended = torch.matmul(h_nfl.reshape(b, n * self.hidden_size, l), temporal_att)
        attended = attended.reshape(b, n, self.hidden_size, l).permute(0, 2, 1, 3)
        readout = self.temporal_readout(attended.permute(0, 3, 2, 1)).squeeze(1)
        pred = self.prediction(readout)
        if self.output_activation == "leaky_relu":
            pred = F.leaky_relu(pred, negative_slope=0.2)
        if stock_mask is not None:
            pred = pred * stock_mask.to(dtype=pred.dtype, device=pred.device).unsqueeze(-1)
        return pred

    def forward(self, batch, stock_mask: torch.Tensor | None = None) -> torch.Tensor:
        if isinstance(batch, dict):
            stock_mask = batch.get("mask", stock_mask)
            x = batch["inputs"]
        else:
            x = batch
        return self._forward_tensor(x, stock_mask=stock_mask)


Model = HyperStockGAT
