"""
FactorGCL: A Hypergraph-Based Factor Model with Temporal Residual
Contrastive Learning for Stock Return Prediction.

Source type: PDF-derived.
The searched public PGCL repository is not this financial FactorGCL model.

This implementation follows the paper's module decomposition:
    feature extractor -> prior beta HyperGCN -> hidden beta HyperGCN
    -> individual alpha -> multi-horizon prediction

Core model expected input:
    x: Tensor [num_stocks, lookback, num_features]
    prior_beta: Tensor [num_stocks, num_prior_factors]
        Soft or binary factor exposures. It is used as the hypergraph incidence
        matrix for the prior beta module.
    future_x: optional Tensor [num_stocks, future_lookback, num_features]
        Used only to compute temporal residual contrastive learning.

Output:
    dict with prediction and intermediate embeddings.
"""

import torch
from torch import nn
import torch.nn.functional as F


def _safe_degree(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return values.clamp_min(eps)


class HypergraphConvolution(nn.Module):
    """Dense soft-incidence HyperGCN layer from Eq. (1)."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, node_features: torch.Tensor, incidence: torch.Tensor) -> torch.Tensor:
        # node_features: [N, H], incidence: [N, E]
        if incidence.dim() != 2:
            raise ValueError("incidence must have shape [num_stocks, num_hyperedges].")
        if node_features.size(0) != incidence.size(0):
            raise ValueError("node_features and incidence must share the stock dimension.")

        h = incidence.to(dtype=node_features.dtype, device=node_features.device)
        transformed = self.linear(node_features)

        node_degree = _safe_degree(h.sum(dim=1))  # Dn
        edge_degree = _safe_degree(h.sum(dim=0))  # De

        left = transformed * node_degree.rsqrt().unsqueeze(-1)
        edge_messages = h.transpose(0, 1).matmul(left)
        edge_messages = edge_messages / edge_degree.unsqueeze(-1)
        node_messages = h.matmul(edge_messages)
        node_messages = node_messages * node_degree.rsqrt().unsqueeze(-1)
        return self.activation(self.dropout(node_messages))


class StockFeatureExtractor(nn.Module):
    """GRU + batch normalization feature extractor phi_feat."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, T, D]
        n, t, d = x.shape
        z = self.input_norm(x.reshape(n * t, d)).reshape(n, t, d)
        out, _ = self.gru(z)
        return out[:, -1, :]


class ProjectionHead(nn.Module):
    """Two-layer MLP p(.) used by the InfoNCE loss in Eq. (11)."""

    def __init__(self, hidden_dim: int, projection_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FactorGCL(nn.Module):
    """Cascading residual hypergraph architecture from Fig. 3 and Eqs. (5)-(8)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_hidden_factors: int = 32,
        num_horizons: int = 4,
        num_gru_layers: int = 2,
        dropout: float = 0.1,
        projection_dim: int | None = None,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.num_hidden_factors = num_hidden_factors
        self.num_horizons = num_horizons
        self.temperature = temperature

        self.feature_extractor = StockFeatureExtractor(input_dim, hidden_dim, num_gru_layers, dropout)
        # The paper uses phi_feat for historical windows and phi'_feat for
        # future windows in temporal residual contrastive learning.
        self.future_feature_extractor = StockFeatureExtractor(input_dim, hidden_dim, num_gru_layers, dropout)
        self.prior_beta_module = HypergraphConvolution(hidden_dim, hidden_dim, dropout)
        self.hidden_beta_module = HypergraphConvolution(hidden_dim, hidden_dim, dropout)

        # Hidden factor prototypes c^(j), used to generate soft hyperedges beta_h.
        self.hidden_factor_prototypes = nn.Parameter(torch.randn(num_hidden_factors, hidden_dim) * hidden_dim**-0.5)

        self.alpha_module = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
        )
        self.prior_heads = nn.ModuleList([nn.Linear(hidden_dim, 1, bias=False) for _ in range(num_horizons)])
        self.hidden_heads = nn.ModuleList([nn.Linear(hidden_dim, 1, bias=False) for _ in range(num_horizons)])
        self.alpha_heads = nn.ModuleList([nn.Linear(hidden_dim, 1, bias=False) for _ in range(num_horizons)])
        self.output_bias = nn.Parameter(torch.zeros(num_horizons))

        if projection_dim is None:
            projection_dim = hidden_dim
        self.projection_head = ProjectionHead(hidden_dim, projection_dim)

    def encode_components(
        self,
        x: torch.Tensor,
        prior_beta: torch.Tensor,
        hidden_beta: torch.Tensor | None = None,
        feature_extractor: StockFeatureExtractor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return es, ep, eh, e_alpha, and beta_h."""
        if feature_extractor is None:
            feature_extractor = self.feature_extractor
        stock_features = feature_extractor(x)  # es

        # Eq. (5): prior beta module over the prior factor hypergraph.
        prior_embeddings = self.prior_beta_module(stock_features, prior_beta)
        residual_after_prior = stock_features - prior_embeddings

        # Hidden soft hyperedges beta_h = sigmoid(er c^T).
        if hidden_beta is None:
            hidden_beta = torch.sigmoid(residual_after_prior.matmul(self.hidden_factor_prototypes.t()))

        # Eq. (6): hidden beta module over generated hidden factor hypergraph.
        hidden_embeddings = self.hidden_beta_module(residual_after_prior, hidden_beta)
        residual_after_factors = stock_features - prior_embeddings - hidden_embeddings

        # Eq. (7): individual alpha module.
        alpha_embeddings = self.alpha_module(residual_after_factors)
        return {
            "stock_features": stock_features,
            "prior_embeddings": prior_embeddings,
            "hidden_embeddings": hidden_embeddings,
            "alpha_embeddings": alpha_embeddings,
            "hidden_beta": hidden_beta,
        }

    def predict_from_components(
        self,
        prior_embeddings: torch.Tensor,
        hidden_embeddings: torch.Tensor,
        alpha_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        # Eq. (8): separate linear maps for prior beta, hidden beta, and alpha.
        preds = []
        for i in range(self.num_horizons):
            y_i = (
                self.prior_heads[i](prior_embeddings)
                + self.hidden_heads[i](hidden_embeddings)
                + self.alpha_heads[i](alpha_embeddings)
                + self.output_bias[i]
            )
            preds.append(y_i.squeeze(-1))
        return torch.stack(preds, dim=-1)  # [N, L]

    def temporal_residual_contrastive_loss(
        self,
        alpha_past: torch.Tensor,
        alpha_future: torch.Tensor,
    ) -> torch.Tensor:
        """InfoNCE over same-stock past/future alpha embeddings, Eq. (11)."""
        z1 = F.normalize(self.projection_head(alpha_past), dim=-1)
        z2 = F.normalize(self.projection_head(alpha_future), dim=-1)
        logits = z1.matmul(z2.t()) / self.temperature
        labels = torch.arange(alpha_past.size(0), device=alpha_past.device)
        return F.cross_entropy(logits, labels)

    def objective(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        contrastive_loss: torch.Tensor | None = None,
        gamma: float = 0.1,
    ) -> torch.Tensor:
        """Eq. (12)-(13): MSE over horizons plus optional contrastive loss."""
        mse = F.mse_loss(prediction, target)
        return mse if contrastive_loss is None else mse + gamma * contrastive_loss

    def forward(
        self,
        x: torch.Tensor,
        prior_beta: torch.Tensor,
        future_x: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        components = self.encode_components(x, prior_beta)
        prediction = self.predict_from_components(
            components["prior_embeddings"],
            components["hidden_embeddings"],
            components["alpha_embeddings"],
        )
        out = {"prediction": prediction, **components}

        if future_x is not None:
            # Eq. (9)-(10): future alpha uses shared prior/hidden modules and the
            # hidden hyperedges mined from historical data.
            future_components = self.encode_components(
                future_x,
                prior_beta,
                hidden_beta=components["hidden_beta"].detach(),
                feature_extractor=self.future_feature_extractor,
            )
            out["future_alpha_embeddings"] = future_components["alpha_embeddings"]
            out["contrastive_loss"] = self.temporal_residual_contrastive_loss(
                components["alpha_embeddings"],
                future_components["alpha_embeddings"],
            )

        return out


class FactorGCLStockModel(nn.Module):
    """Project wrapper for the unified [B,N,L,F] stock baseline contract."""

    def __init__(
        self,
        input_dim: int,
        num_stocks: int,
        prior_beta,
        hidden_dim: int = 32,
        num_hidden_factors: int = 32,
        num_horizons: int = 1,
        num_gru_layers: int = 2,
        dropout: float = 0.1,
        projection_dim: int | None = None,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_stocks = int(num_stocks)
        beta = torch.as_tensor(prior_beta, dtype=torch.float32)
        if beta.dim() != 2:
            raise ValueError(f"prior_beta must be [N,K], got {tuple(beta.shape)}")
        if beta.size(0) != self.num_stocks:
            raise ValueError(
                f"prior_beta stock dimension mismatch: got {beta.size(0)}, expected {self.num_stocks}"
            )
        self.register_buffer("prior_beta", beta, persistent=True)
        self.core = FactorGCL(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_hidden_factors=num_hidden_factors,
            num_horizons=num_horizons,
            num_gru_layers=num_gru_layers,
            dropout=dropout,
            projection_dim=projection_dim,
            temperature=temperature,
        )

    def _ensure_input(self, x: torch.Tensor, name: str) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() != 4:
            raise ValueError(f"{name} must be [B,N,L,F] or [N,L,F], got {tuple(x.shape)}")
        if x.size(1) != self.num_stocks:
            raise ValueError(f"{name} stock dimension mismatch: got {x.size(1)}, expected {self.num_stocks}")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"{name} feature dimension mismatch: got {x.size(-1)}, expected {self.input_dim}")
        return x

    def _ensure_mask(self, mask: torch.Tensor | None, batch_size: int) -> torch.Tensor | None:
        if mask is None:
            return None
        if mask.dim() == 2:
            if mask.shape == (self.num_stocks, 1) and batch_size == 1:
                mask = mask.unsqueeze(0)
            else:
                mask = mask.unsqueeze(-1)
        if mask.dim() != 3:
            raise ValueError(f"mask must be [B,N,1] or [B,N], got {tuple(mask.shape)}")
        if mask.shape[:2] != (batch_size, self.num_stocks):
            raise ValueError(
                f"mask shape {tuple(mask.shape)} does not match [B,N]=[{batch_size},{self.num_stocks}]"
            )
        return mask

    def forward(
        self,
        batch,
        future_x: torch.Tensor | None = None,
        stock_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if isinstance(batch, dict):
            x = batch["inputs"]
            future_x = batch.get("future_inputs", future_x)
            stock_mask = batch.get("mask", stock_mask)
        else:
            x = batch

        x = self._ensure_input(x, "inputs")
        future = None if future_x is None else self._ensure_input(future_x, "future_inputs")
        if future is not None and future.size(0) != x.size(0):
            raise ValueError(
                f"future_inputs batch dimension {future.size(0)} does not match inputs batch {x.size(0)}"
            )
        stock_mask = self._ensure_mask(stock_mask, x.size(0))

        outputs = []
        predictions = []
        contrastive_losses = []
        for index in range(x.size(0)):
            out = self.core(
                x[index],
                self.prior_beta,
                future_x=None if future is None else future[index],
            )
            pred = out["prediction"]
            predictions.append(pred)
            if "contrastive_loss" in out:
                contrastive_losses.append(out["contrastive_loss"].float())
            outputs.append(out)

        prediction = torch.stack(predictions, dim=0)
        if stock_mask is not None:
            prediction = prediction * stock_mask.to(device=prediction.device, dtype=prediction.dtype)
        result = {"prediction": prediction, "outputs": outputs}
        if contrastive_losses:
            result["contrastive_loss"] = torch.stack(contrastive_losses).mean()
        return result


CoreModel = FactorGCL
Model = FactorGCLStockModel
