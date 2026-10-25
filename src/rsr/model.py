"""
Temporal Relational Ranking for Stock Prediction.

Source type: official-source-derived.

This PyTorch implementation keeps the RSR relation module from the official
TensorFlow code, but replaces the paper's precomputed sequential embedding
with an in-model LSTM over this project's unified [B,N,L,F] windows.

Expected input:
    batch["inputs"]: Tensor [batch, num_stocks, lookback, num_features]
    batch["mask"]  : Tensor [batch, num_stocks, 1], optional stock validity mask

Output:
    Tensor [batch, num_stocks, 1], raw-return score.
"""

import torch
from torch import nn
import torch.nn.functional as F


class SparseRelationScorer(nn.Module):
    """Trainable dense-score equivalent of tf.layers.dense(relation, 1)."""

    def __init__(
        self,
        num_stocks: int,
        num_relation_types: int,
        relation_row,
        relation_col,
        relation_type,
        relation_mask,
    ):
        super().__init__()
        self.num_stocks = int(num_stocks)
        self.num_relation_types = int(num_relation_types)
        self.relation_type_weight = nn.Parameter(torch.empty(num_relation_types))
        self.relation_bias = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.relation_type_weight.view(1, -1))

        self.register_buffer("relation_row", torch.as_tensor(relation_row, dtype=torch.long), persistent=True)
        self.register_buffer("relation_col", torch.as_tensor(relation_col, dtype=torch.long), persistent=True)
        self.register_buffer("relation_type", torch.as_tensor(relation_type, dtype=torch.long), persistent=True)
        self.register_buffer("relation_mask", torch.as_tensor(relation_mask, dtype=torch.bool), persistent=True)

    def forward(self) -> torch.Tensor:
        n = self.num_stocks
        flat = self.relation_type_weight.new_zeros(n * n)
        flat_index = self.relation_row * n + self.relation_col
        flat.index_add_(0, flat_index, self.relation_type_weight[self.relation_type])
        score = flat.view(n, n) + self.relation_bias
        return F.leaky_relu(score, negative_slope=0.2)


class RelationRankLSTM(nn.Module):
    """LSTM encoder plus the RSR relation-aware ranking head."""

    def __init__(
        self,
        input_dim: int,
        num_stocks: int,
        relation_channels: int,
        relation_row,
        relation_col,
        relation_type,
        relation_mask,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        flat_relation: bool = False,
        inner_product: bool = False,
        softmax_axis: str = "source",
        output_activation: str = "leaky_relu",
    ):
        super().__init__()
        if softmax_axis not in {"source", "target"}:
            raise ValueError("softmax_axis must be source or target")
        if output_activation not in {"leaky_relu", "identity"}:
            raise ValueError("output_activation must be leaky_relu or identity")
        self.num_stocks = int(num_stocks)
        self.hidden_size = int(hidden_size)
        self.flat_relation = bool(flat_relation)
        self.inner_product = bool(inner_product)
        self.softmax_axis = softmax_axis
        self.output_activation = output_activation

        self.encoder = nn.LSTM(
            input_size=int(input_dim),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
        )
        self.relation_scorer = SparseRelationScorer(
            num_stocks=num_stocks,
            num_relation_types=relation_channels,
            relation_row=relation_row,
            relation_col=relation_col,
            relation_type=relation_type,
            relation_mask=relation_mask,
        )
        self.head_weight = nn.Linear(hidden_size, 1)
        self.tail_weight = nn.Linear(hidden_size, 1)
        if self.flat_relation:
            self.relation_fusion = nn.Linear(hidden_size * 2, hidden_size)
            prediction_dim = hidden_size
        else:
            self.relation_fusion = None
            prediction_dim = hidden_size * 2
        self.prediction = nn.Linear(prediction_dim, 1)

    def _masked_softmax(self, scores: torch.Tensor, stock_mask: torch.Tensor | None) -> torch.Tensor:
        relation_mask = self.relation_scorer.relation_mask.to(device=scores.device)
        scores = scores.masked_fill(~relation_mask.unsqueeze(0), torch.finfo(scores.dtype).min)
        if stock_mask is not None:
            valid = stock_mask.to(device=scores.device).bool()
            if self.softmax_axis == "source":
                scores = scores.masked_fill(~valid.unsqueeze(2), torch.finfo(scores.dtype).min)
            else:
                scores = scores.masked_fill(~valid.unsqueeze(1), torch.finfo(scores.dtype).min)
        dim = 1 if self.softmax_axis == "source" else 2
        weights = torch.softmax(scores, dim=dim)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        if stock_mask is not None:
            valid_float = stock_mask.to(device=scores.device, dtype=weights.dtype)
            weights = weights * valid_float.unsqueeze(1)
        return weights

    def _forward_tensor(self, x: torch.Tensor, stock_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
            if stock_mask is not None and stock_mask.dim() == 1:
                stock_mask = stock_mask.unsqueeze(0)
        if x.dim() != 4:
            raise ValueError(f"RSR expects [B,N,L,F] or [N,L,F], got {tuple(x.shape)}")
        b, n, l, f = x.shape
        if n != self.num_stocks:
            raise ValueError(f"stock dimension mismatch: got {n}, expected {self.num_stocks}")
        if stock_mask is not None:
            if stock_mask.dim() == 3:
                stock_mask = stock_mask.squeeze(-1)
            if stock_mask.shape != (b, n):
                raise ValueError(f"stock_mask must be [B,N], got {tuple(stock_mask.shape)}")

        z, _ = self.encoder(x.reshape(b * n, l, f))
        feature = z[:, -1, :].reshape(b, n, self.hidden_size)
        if stock_mask is not None:
            feature = feature * stock_mask.to(dtype=feature.dtype, device=feature.device).unsqueeze(-1)

        relation_weight = self.relation_scorer().to(dtype=feature.dtype, device=feature.device)
        if self.inner_product:
            inner = torch.matmul(feature, feature.transpose(1, 2))
            score = inner * relation_weight.unsqueeze(0)
        else:
            head = F.leaky_relu(self.head_weight(feature), negative_slope=0.2)
            tail = F.leaky_relu(self.tail_weight(feature), negative_slope=0.2)
            score = head + tail.transpose(1, 2) + relation_weight.unsqueeze(0)

        relation_attention = self._masked_softmax(score, stock_mask)
        propagated = torch.matmul(relation_attention, feature)
        combined = torch.cat([feature, propagated], dim=-1)
        if self.relation_fusion is not None:
            combined = F.leaky_relu(self.relation_fusion(combined), negative_slope=0.2)
        pred = self.prediction(combined)
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


Model = RelationRankLSTM
