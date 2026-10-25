"""
Enhancer: A Distribution-Aware Framework with Temporal-Relational
Meta-Learning for Stock Prediction.

Source type: PDF-derived.
The checked public appendix repository contains an appendix PDF
material only, not model source code.

This file implements the paper's framework-level modules:
    Temporal Meta-Learner (expert distribution + RPPs attention)
    Relational Meta-Learner (matrix-polynomial approximation + decoupled attention)
    SGC-style integration and prediction heads

Core model expected input:
    x: Tensor [num_stocks, lookback, num_features]

Output:
    dict with prediction, invariant-only prediction, intervened prediction, and
    the regularization terms needed for the paper's optimization idea.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


def stock_rank_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pairwise stock ranking loss for the paper's stock recommendation target."""
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    if mask is not None:
        keep = mask.reshape(-1).bool()
        prediction = prediction[keep]
        target = target[keep]
    if prediction.numel() <= 1:
        return prediction.new_zeros(())

    pred_diff = prediction.unsqueeze(1) - prediction.unsqueeze(0)
    target_diff = target.unsqueeze(1) - target.unsqueeze(0)
    pair_sign = target_diff.sign()
    valid_pair = pair_sign.ne(0)
    if not valid_pair.any():
        return prediction.new_zeros(())
    return F.softplus(-pred_diff[valid_pair] * pair_sign[valid_pair]).mean()


class ExpertDistribution(nn.Module):
    """Instance normalization plus attention transition, Eqs. (2)-(3)."""

    def __init__(self, feature_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(feature_dim))
        self.beta = nn.Parameter(torch.zeros(feature_dim))
        self.q = nn.Linear(feature_dim, feature_dim)
        self.k = nn.Linear(feature_dim, feature_dim)
        self.v = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + 1e-6)
        x_hat = x_hat * self.gamma.view(1, 1, -1) + self.beta.view(1, 1, -1)

        q = self.q(x_hat)
        k = self.k(x_hat)
        v = self.v(x_hat)
        attn = torch.softmax(q.matmul(k.transpose(1, 2)) / math.sqrt(x_hat.size(-1)), dim=-1)
        return self.dropout(attn).matmul(v)


class RPPsAttention(nn.Module):
    """Reactive Point Process attention from Eqs. (4)-(6)."""

    def __init__(self, feature_dim: int, dropout: float = 0.1):
        super().__init__()
        self.context = nn.Linear(feature_dim, feature_dim, bias=False)
        self.excitation = nn.Linear(1, 1)
        self.inhibition = nn.Linear(1, 1)
        self.decay_log = nn.Parameter(torch.zeros(()))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, T, D]
        n, t, d = x.shape
        h = self.context(x)
        raw = h.matmul(h.transpose(1, 2)) / math.sqrt(d)
        causal_mask = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device))
        raw = raw.masked_fill(~causal_mask.view(1, t, t), float("-inf"))
        attn = torch.softmax(raw, dim=-1)

        point_score = attn.unsqueeze(-1)
        exc = torch.relu(self.excitation(point_score)).squeeze(-1)
        inh = torch.relu(-self.inhibition(point_score)).squeeze(-1)

        idx = torch.arange(t, device=x.device)
        delta = (idx.view(1, -1) - idx.view(-1, 1)).clamp_min(0).to(dtype=x.dtype)
        decay = torch.exp(-F.softplus(self.decay_log) * delta).view(1, t, t)

        rpp = (1.0 + exc - inh).clamp_min(0.0) * decay
        rpp = rpp.masked_fill(~causal_mask.view(1, t, t), 0.0)
        rpp = rpp / rpp.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return self.dropout(rpp).matmul(x)


class TemporalMetaLearner(nn.Module):
    """TML: distribution information plus temporal point information, Eq. (7)."""

    def __init__(self, feature_dim: int, dropout: float = 0.1):
        super().__init__()
        self.expert = ExpertDistribution(feature_dim, dropout)
        self.rpps = RPPsAttention(feature_dim, dropout)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.expert(x) + self.rpps(x))


class GraphPropagation(nn.Module):
    """Small SGC-style propagation block used by RML and integration."""

    def __init__(self, feature_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(feature_dim, out_dim)

    @staticmethod
    def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
        adj = torch.relu(adj)
        adj = adj + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
        degree = adj.sum(dim=-1).clamp_min(1e-6)
        return degree.rsqrt().unsqueeze(-1) * adj * degree.rsqrt().unsqueeze(0)

    def forward(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        norm_adj = self.normalize_adj(adj)
        return norm_adj.matmul(self.linear(x))


class RelationalMetaLearner(nn.Module):
    """RML approximation and decoupled attention, Eqs. (8)-(13)."""

    def __init__(
        self,
        num_stocks: int,
        feature_dim: int,
        relation_dim: int = 30,
        num_contexts: int = 30,
        polynomial_order: int = 3,
        topk: int | None = None,
    ):
        super().__init__()
        self.num_stocks = num_stocks
        self.num_contexts = num_contexts
        self.polynomial_order = polynomial_order
        self.topk = topk if topk is not None else max(1, math.ceil(math.log(num_stocks)))

        self.e_a = nn.Parameter(torch.randn(num_stocks, relation_dim) * relation_dim**-0.5)
        self.context_embedding = nn.Parameter(torch.randn(num_contexts, relation_dim) * relation_dim**-0.5)
        self.context_to_coeff = nn.Linear(relation_dim, polynomial_order + 1)

        self.context_gru = nn.GRU(feature_dim + num_stocks, num_contexts, batch_first=True)
        self.stock_relation_embedding = nn.Parameter(torch.randn(num_stocks, relation_dim) * relation_dim**-0.5)
        self.q = nn.Linear(relation_dim, relation_dim)
        self.k = nn.Linear(relation_dim, relation_dim)
        self.v = nn.Linear(relation_dim, relation_dim)
        self.feature_selector = nn.Parameter(torch.randn(num_stocks, relation_dim) * relation_dim**-0.5)

        self.inv_graph = GraphPropagation(relation_dim, relation_dim)
        self.var_graph = GraphPropagation(relation_dim, relation_dim)

    def base_adjacency(self) -> torch.Tensor:
        return torch.relu(self.e_a.matmul(self.e_a.t()))

    def polynomial_coefficients(self, context_prob: torch.Tensor | None = None) -> torch.Tensor:
        coeff = torch.softmax(self.context_to_coeff(self.context_embedding), dim=-1)
        if context_prob is None:
            return coeff
        return context_prob.matmul(coeff)

    def polynomial_adjacency(self, context_prob: torch.Tensor) -> torch.Tensor:
        # Probabilistically mix context coefficients before applying Eq. (2).
        # This avoids materializing [P,N,N] dense graphs for large markets.
        base = self.base_adjacency()
        powers = [torch.eye(self.num_stocks, device=base.device, dtype=base.dtype)]
        powers.append(base)
        for _ in range(2, self.polynomial_order + 1):
            powers.append(powers[-1].matmul(base))

        coeff = self.polynomial_coefficients(context_prob)
        adj = 0.0
        for k, basis in enumerate(powers):
            adj = adj + coeff[k] * basis
        return adj

    def context_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        # The paper matches contexts over time; here we use GRU over market state and
        # covariance summary to estimate the last-step context distribution.
        n, t, d = x.shape
        market_state = x.mean(dim=0)  # [T, D]
        last_centered = x[:, -1, :] - x[:, -1, :].mean(dim=0, keepdim=True)
        cov_proxy = last_centered.matmul(last_centered.t()) / max(d, 1)
        cov_summary = cov_proxy.mean(dim=0).view(1, -1).expand(t, -1)
        seq = torch.cat([market_state, cov_summary], dim=-1).unsqueeze(0)
        output, _ = self.context_gru(seq)
        return torch.softmax(output[:, -1, :], dim=-1).squeeze(0)

    def sparsify(self, adj: torch.Tensor) -> torch.Tensor:
        if self.topk >= adj.size(-1):
            return adj
        values, indices = torch.topk(adj, k=self.topk, dim=-1)
        sparse = torch.zeros_like(adj)
        sparse.scatter_(dim=-1, index=indices, src=values)
        return sparse

    def decouple(self, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.stock_relation_embedding
        q = self.q(emb)
        k = self.k(emb)
        v = self.v(emb)
        score = q.matmul(k.t()) / math.sqrt(q.size(-1))

        inv_attn = torch.softmax(score, dim=-1)
        var_attn = torch.softmax(-score, dim=-1)
        selector = torch.softmax(self.feature_selector.matmul(self.feature_selector.t()), dim=-1)
        value_inv = v * selector.matmul(v).sigmoid()

        p_inv = self.inv_graph(inv_attn * adj, value_inv).matmul(value_inv.t())
        p_var = self.var_graph(var_attn * adj, v).matmul(v.t())
        return self.sparsify(torch.relu(p_inv)), self.sparsify(torch.relu(p_var))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        prob = self.context_probabilities(x)
        adj = self.polynomial_adjacency(prob)
        p_inv, p_var = self.decouple(adj)
        return {
            "adjacency": adj,
            "context_prob": prob,
            "invariant_relation": p_inv,
            "variant_relation": p_var,
        }


class Enhancer(nn.Module):
    """Enhancer framework with a lightweight downstream predictor."""

    def __init__(
        self,
        num_stocks: int,
        feature_dim: int,
        hidden_dim: int = 30,
        relation_dim: int = 30,
        num_contexts: int = 30,
        polynomial_order: int = 3,
        sgc_steps: int = 2,
        dropout: float = 0.1,
        relation_topk: int | None = None,
    ):
        super().__init__()
        self.sgc_steps = sgc_steps
        self.temporal_meta_learner = TemporalMetaLearner(feature_dim, dropout)
        self.temporal_proj = nn.Linear(feature_dim, hidden_dim)
        self.relational_meta_learner = RelationalMetaLearner(
            num_stocks=num_stocks,
            feature_dim=feature_dim,
            relation_dim=relation_dim,
            num_contexts=num_contexts,
            polynomial_order=polynomial_order,
            topk=relation_topk,
        )
        self.inv_proj = nn.Linear(hidden_dim, hidden_dim)
        self.var_proj = nn.Linear(hidden_dim, hidden_dim)
        self.predictor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _norm_adj(adj: torch.Tensor) -> torch.Tensor:
        adj = torch.relu(adj)
        adj = adj + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
        degree = adj.sum(dim=-1).clamp_min(1e-6)
        return degree.rsqrt().unsqueeze(-1) * adj * degree.rsqrt().unsqueeze(0)

    def integrate(self, temporal: torch.Tensor, p_inv: torch.Tensor, p_var: torch.Tensor | None = None) -> torch.Tensor:
        # Eq. (16): SGC-style repeated propagation over invariant and optional variant relations.
        last = self.temporal_proj(temporal[:, -1, :])
        inv_adj = self._norm_adj(p_inv)
        h_inv = last
        out = 0.0
        for _ in range(self.sgc_steps):
            h_inv = inv_adj.matmul(h_inv)
            out = out + self.inv_proj(h_inv)

        if p_var is not None:
            var_adj = self._norm_adj(p_var)
            h_var = last
            for _ in range(self.sgc_steps):
                h_var = var_adj.matmul(h_var)
                out = out + self.var_proj(h_var)
        return out

    def intervention_loss(
        self,
        pred_full: torch.Tensor,
        pred_inv: torch.Tensor,
        pred_intervened: torch.Tensor,
    ) -> torch.Tensor:
        # Implements the paper's goal of stable outputs under shuffled variant relations.
        return (pred_full - pred_inv).abs().mean() + (pred_full - pred_intervened).abs().mean()

    def noisy_consistency_loss(
        self,
        x: torch.Tensor,
        base_prediction: torch.Tensor,
        noise_scale: float = 1e-3,
    ) -> torch.Tensor:
        """Noise consistency regularizer matching the paper's robust optimization idea."""
        if noise_scale <= 0:
            return base_prediction.new_zeros(())
        noisy_x = x + torch.randn_like(x) * noise_scale
        noisy_prediction = self.forward(noisy_x)["prediction"]
        return F.mse_loss(noisy_prediction, base_prediction.detach())

    def training_objective(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        mse_weight: float = 1.0,
        rank_weight: float = 1.0,
        reg1_weight: float = 12.0,
        reg2_weight: float = 1.0,
        noise_scale: float = 1e-3,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Combine prediction, rank, intervention, and noise-consistency losses."""
        outputs = self.forward(x)
        prediction = outputs["prediction"]
        target = target.reshape_as(prediction)

        if mask is None:
            mse = F.mse_loss(prediction, target)
        else:
            mask = mask.reshape_as(prediction).to(dtype=prediction.dtype)
            mse = ((prediction - target).pow(2) * mask).sum() / mask.sum().clamp_min(1.0)

        rank = stock_rank_loss(prediction, target, mask)
        reg1 = outputs["intervention_loss"]
        reg2 = self.noisy_consistency_loss(x, prediction, noise_scale)
        loss = mse_weight * mse + rank_weight * rank + reg1_weight * reg1 + reg2_weight * reg2
        return {
            "loss": loss,
            "mse_loss": mse,
            "rank_loss": rank,
            "reg1_intervention_loss": reg1,
            "reg2_noisy_consistency_loss": reg2,
            "outputs": outputs,
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError("Enhancer expects x with shape [num_stocks, lookback, features].")

        temporal = self.temporal_meta_learner(x)
        relational = self.relational_meta_learner(x)
        p_inv = relational["invariant_relation"]
        p_var = relational["variant_relation"]

        z_full = self.integrate(temporal, p_inv, p_var)
        z_inv = self.integrate(temporal, p_inv, None)

        p_var_shuffled = torch.roll(p_var, shifts=(1, 1), dims=(0, 1))
        z_intervened = self.integrate(temporal, p_inv, p_var_shuffled)

        pred_full = self.predictor(z_full)
        pred_inv = self.predictor(z_inv)
        pred_intervened = self.predictor(z_intervened)
        reg = self.intervention_loss(pred_full, pred_inv, pred_intervened)

        return {
            "prediction": pred_full.squeeze(-1),
            "invariant_prediction": pred_inv.squeeze(-1),
            "intervened_prediction": pred_intervened.squeeze(-1),
            "intervention_loss": reg,
            "temporal_representation": temporal,
            "full_representation": z_full,
            "invariant_representation": z_inv,
            **relational,
        }


class EnhancerStockModel(nn.Module):
    """Project wrapper that accepts both [N,L,F] and [B,N,L,F]."""

    def __init__(
        self,
        num_stocks: int,
        feature_dim: int,
        hidden_dim: int = 30,
        relation_dim: int = 30,
        num_contexts: int = 30,
        polynomial_order: int = 3,
        sgc_steps: int = 2,
        dropout: float = 0.1,
        relation_topk: int | None = None,
    ):
        super().__init__()
        self.num_stocks = int(num_stocks)
        self.feature_dim = int(feature_dim)
        self.core = Enhancer(
            num_stocks=num_stocks,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            relation_dim=relation_dim,
            num_contexts=num_contexts,
            polynomial_order=polynomial_order,
            sgc_steps=sgc_steps,
            dropout=dropout,
            relation_topk=relation_topk,
        )

    def _check_single(self, x: torch.Tensor) -> None:
        if x.dim() != 3:
            raise ValueError(f"Enhancer expects [N,L,F] or [B,N,L,F], got {tuple(x.shape)}")
        if x.size(0) != self.num_stocks:
            raise ValueError(f"stock dimension mismatch: got {x.size(0)}, expected {self.num_stocks}")
        if x.size(-1) != self.feature_dim:
            raise ValueError(f"feature dimension mismatch: got {x.size(-1)}, expected {self.feature_dim}")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        if x.dim() == 3:
            self._check_single(x)
            return self.core(x)
        if x.dim() != 4:
            raise ValueError(f"Enhancer expects [N,L,F] or [B,N,L,F], got {tuple(x.shape)}")
        outputs = []
        predictions = []
        invariant_predictions = []
        intervened_predictions = []
        intervention_losses = []
        for index in range(x.size(0)):
            self._check_single(x[index])
            out = self.core(x[index])
            outputs.append(out)
            predictions.append(out["prediction"].reshape(-1, 1))
            invariant_predictions.append(out["invariant_prediction"].reshape(-1, 1))
            intervened_predictions.append(out["intervened_prediction"].reshape(-1, 1))
            intervention_losses.append(out["intervention_loss"].float())
        return {
            "prediction": torch.stack(predictions, dim=0),
            "invariant_prediction": torch.stack(invariant_predictions, dim=0),
            "intervened_prediction": torch.stack(intervened_predictions, dim=0),
            "intervention_loss": torch.stack(intervention_losses).mean(),
            "outputs": outputs,
        }


CoreModel = Enhancer
Model = EnhancerStockModel
