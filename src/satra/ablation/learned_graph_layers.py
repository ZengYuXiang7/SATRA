"""End-to-end graph operators used by learned-relation ablations.

The operators in this module deliberately do not read ``pstr_relation_path``
or any externally constructed relation file.  They therefore test whether a
relation structure learned from the supervised objective can replace a fixed
historical DTW neighborhood.
"""

import math

import torch
import torch.nn as nn


class StaticLearnedAdjacency(nn.Module):
    """One dense, directed adjacency matrix shared by every sample and time step.

    Each row is normalized with a softmax.  The parameters are initialized
    close to uniform rather than from a handcrafted relation prior, so every
    edge is learned end to end from the prediction objective.
    """

    def __init__(self, stock_num):
        super().__init__()
        self.stock_num = int(stock_num)
        if self.stock_num <= 1:
            raise ValueError(f"stock_num must exceed one, got {self.stock_num}")
        self.adjacency_logits = nn.Parameter(torch.empty(self.stock_num, self.stock_num))
        nn.init.normal_(self.adjacency_logits, mean=0.0, std=0.02)

    def forward(self, stock_num, dtype):
        if int(stock_num) != self.stock_num:
            raise ValueError(
                "Learned adjacency stock count does not match the configured graph: "
                f"observed={stock_num}, configured={self.stock_num}"
            )
        return torch.softmax(self.adjacency_logits.float(), dim=-1).to(dtype=dtype)


class StaticGraphMessageLayer(nn.Module):
    """Residual graph message passing for a supplied static adjacency matrix."""

    def __init__(self, d_model, dropout):
        super().__init__()
        d_model = int(d_model)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)
        self.output_projection = nn.Linear(d_model, d_model, bias=False)
        self.message_dropout = nn.Dropout(float(dropout))
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(float(dropout)),
        )

    def forward(self, stock_tokens, adjacency):
        if stock_tokens.ndim != 3:
            raise ValueError(
                "Static graph tokens must have shape [batch_time, stocks, features], got "
                f"{tuple(stock_tokens.shape)}"
            )
        stock_num = stock_tokens.shape[1]
        if adjacency.shape != (stock_num, stock_num):
            raise ValueError(
                "Static learned adjacency shape does not match tokens: "
                f"adjacency={tuple(adjacency.shape)}, stocks={stock_num}"
            )
        normalized = self.norm1(stock_tokens)
        values = self.value_projection(normalized)
        messages = torch.einsum("ij,bjd->bid", adjacency.to(dtype=values.dtype), values)
        stock_tokens = stock_tokens + self.message_dropout(self.output_projection(messages))
        return stock_tokens + self.feed_forward(self.norm2(stock_tokens))


class AdaptiveSparseGraphLayer(nn.Module):
    """Build a per-sample sparse graph from current token representations.

    This is intentionally a single-head top-k graph propagator rather than a
    second copy of the multi-head global semantic Transformer.  The learned
    edge support is recomputed for every stock-time panel and contains no
    fixed DTW, industry, cosine, or Euclidean relation input.
    """

    def __init__(self, d_model, top_k, dropout):
        super().__init__()
        d_model = int(d_model)
        self.top_k = int(top_k)
        if self.top_k <= 0:
            raise ValueError(f"adaptive graph top_k must be positive, got {self.top_k}")
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)
        self.output_projection = nn.Linear(d_model, d_model, bias=False)
        self.self_loop_bias = nn.Parameter(torch.zeros(()))
        self.message_dropout = nn.Dropout(float(dropout))
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(float(dropout)),
        )

    def forward(self, stock_tokens):
        if stock_tokens.ndim != 3:
            raise ValueError(
                "Adaptive graph tokens must have shape [batch_time, stocks, features], got "
                f"{tuple(stock_tokens.shape)}"
            )
        batch_time, stock_num, d_model = stock_tokens.shape
        if stock_num <= 1:
            raise ValueError(f"Adaptive graph requires at least two stocks, got {stock_num}")
        top_k = min(self.top_k, stock_num)
        normalized = self.norm1(stock_tokens)
        query = self.query_projection(normalized).float()
        key = self.key_projection(normalized).float()
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(d_model)
        diagonal = torch.eye(stock_num, dtype=scores.dtype, device=scores.device).unsqueeze(0)
        scores = scores + self.self_loop_bias.float() * diagonal
        top_scores, top_indices = scores.topk(k=top_k, dim=-1)
        edge_weights = torch.softmax(top_scores, dim=-1)

        values = self.value_projection(normalized)
        source_values = values.unsqueeze(1).expand(batch_time, stock_num, stock_num, d_model)
        selected_values = torch.gather(
            source_values,
            dim=2,
            index=top_indices.unsqueeze(-1).expand(batch_time, stock_num, top_k, d_model),
        )
        messages = (edge_weights.to(dtype=values.dtype).unsqueeze(-1) * selected_values).sum(dim=2)
        stock_tokens = stock_tokens + self.message_dropout(self.output_projection(messages))
        return stock_tokens + self.feed_forward(self.norm2(stock_tokens))
