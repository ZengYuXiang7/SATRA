"""Reference-causal Temporal Routing Adaptor for the project stock contract.

The project batches stock windows as ``inputs [B, N, L, F]``.  TRA itself
operates independently on each stock window, while its router receives the
causal predictor-error history ``temporal_errors [B, N, H, K]``.  ``H`` only
contains losses from target dates preceding the current target date.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


class AttentionLSTMBackbone(nn.Module):
    """Attention LSTM used by the reference TRA implementation."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        use_attn: bool = True,
        dropout: float = 0.0,
        input_drop: float = 0.0,
        noise_level: float = 0.0,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.use_attn = bool(use_attn)
        self.noise_level = float(noise_level)

        self.input_drop = nn.Dropout(float(input_drop))
        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=float(dropout),
        )
        if self.use_attn:
            self.W = nn.Linear(self.hidden_size, self.hidden_size)
            self.u = nn.Linear(self.hidden_size, 1, bias=False)
            self.output_size = self.hidden_size * 2
        else:
            self.output_size = self.hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_drop(x)
        if self.training and self.noise_level > 0.0:
            x = x + torch.randn_like(x) * self.noise_level

        rnn_out, _ = self.rnn(x)
        last_out = rnn_out[:, -1]
        if not self.use_attn:
            return last_out

        latent = self.W(rnn_out).tanh()
        scores = self.u(latent).softmax(dim=1)
        att_out = (rnn_out * scores).sum(dim=1)
        return torch.cat([last_out, att_out], dim=1)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from the reference Transformer option."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=float(dropout))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[: x.size(0)])


class TransformerBackbone(nn.Module):
    """Transformer backbone available in the reference TRA code."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.0,
        input_drop: float = 0.0,
        noise_level: float = 0.0,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.noise_level = float(noise_level)

        self.input_drop = nn.Dropout(float(input_drop))
        self.input_proj = nn.Linear(self.input_size, self.hidden_size)
        self.pe = PositionalEncoding(self.input_size, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_size * 4,
            dropout=float(dropout),
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.num_layers)
        self.output_size = self.hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_drop(x)
        if self.training and self.noise_level > 0.0:
            x = x + torch.randn_like(x) * self.noise_level

        x = x.permute(1, 0, 2).contiguous()
        x = self.pe(x)
        return self.encoder(self.input_proj(x))[-1]


class TRA(nn.Module):
    """Temporal Routing Adaptor with causal temporal prediction-error input."""

    def __init__(
        self,
        input_dim: int,
        num_stocks: int,
        model_type: str = "LSTM",
        hidden_size: int = 64,
        num_layers: int = 2,
        use_attn: bool = True,
        num_heads: int = 2,
        dropout: float = 0.0,
        input_drop: float = 0.0,
        noise_level: float = 0.0,
        num_predictors: int = 5,
        router_hidden: int = 8,
        tau: float = 1.0,
        src_info: str = "LR_TPE",
    ):
        super().__init__()
        if int(num_predictors) < 1:
            raise ValueError("num_predictors must be >= 1")
        if float(tau) <= 0.0:
            raise ValueError("tau must be > 0")

        self.input_dim = int(input_dim)
        self.num_stocks = int(num_stocks)
        self.num_predictors = int(num_predictors)
        self.tau = float(tau)
        self.src_info = str(src_info)
        source_tokens = {token for token in self.src_info.split("_") if token}
        if source_tokens - {"LR", "TPE"}:
            raise ValueError("src_info may only combine LR and TPE, for example LR_TPE")

        backbone_kwargs = {
            "input_size": self.input_dim,
            "hidden_size": int(hidden_size),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
            "input_drop": float(input_drop),
            "noise_level": float(noise_level),
        }
        if model_type == "LSTM":
            self.backbone = AttentionLSTMBackbone(use_attn=bool(use_attn), **backbone_kwargs)
        elif model_type == "Transformer":
            self.backbone = TransformerBackbone(num_heads=int(num_heads), **backbone_kwargs)
        else:
            raise ValueError("model_type must be LSTM or Transformer")

        self.model_type = model_type
        self.predictors = nn.Linear(self.backbone.output_size, self.num_predictors)
        if self.num_predictors > 1:
            self.router = nn.LSTM(
                input_size=self.num_predictors,
                hidden_size=int(router_hidden),
                num_layers=1,
                batch_first=True,
            )
            self.fc = nn.Linear(int(router_hidden) + self.backbone.output_size, self.num_predictors)

    def _validate_inputs(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x = batch["inputs"]
        if x.dim() != 4:
            raise ValueError(f"TRA inputs must be [B,N,L,F], got {tuple(x.shape)}")
        if x.size(1) != self.num_stocks:
            raise ValueError(f"stock dimension mismatch: got {x.size(1)}, expected {self.num_stocks}")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"feature dimension mismatch: got {x.size(-1)}, expected {self.input_dim}")

        temporal_errors = batch["temporal_errors"]
        if temporal_errors.dim() != 4:
            raise ValueError(f"temporal_errors must be [B,N,H,K], got {tuple(temporal_errors.shape)}")
        if temporal_errors.shape[0] != x.shape[0] or temporal_errors.shape[1] != x.shape[1]:
            raise ValueError("temporal_errors batch and stock dimensions must match inputs")
        if temporal_errors.shape[-1] != self.num_predictors:
            raise ValueError(
                f"temporal_errors predictor dimension mismatch: got {temporal_errors.shape[-1]}, "
                f"expected {self.num_predictors}"
            )
        if temporal_errors.shape[2] < 1:
            raise ValueError("temporal_errors history length must be positive")
        return x, temporal_errors

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x, temporal_errors = self._validate_inputs(batch)
        batch_size, stock_num, sequence_length, feature_num = x.shape
        hidden = self.backbone(x.reshape(batch_size * stock_num, sequence_length, feature_num))
        all_predictions = self.predictors(hidden).reshape(batch_size, stock_num, self.num_predictors)

        if self.num_predictors == 1:
            return {
                "prediction": all_predictions,
                "all_predictions": all_predictions,
                "route_logits": None,
                "route_prob": None,
                "latent": hidden.reshape(batch_size, stock_num, -1),
            }

        history_length = temporal_errors.shape[2]
        router_out, _ = self.router(temporal_errors.reshape(batch_size * stock_num, history_length, self.num_predictors))
        latent_for_router = hidden if "LR" in self.src_info else torch.randn_like(hidden)
        error_for_router = router_out[:, -1] if "TPE" in self.src_info else torch.randn_like(router_out[:, -1])
        route_logits = self.fc(torch.cat([error_for_router, latent_for_router], dim=-1))
        route_prob = F.gumbel_softmax(route_logits, tau=self.tau, hard=False, dim=-1)
        route_prob = route_prob.reshape(batch_size, stock_num, self.num_predictors)

        if self.training:
            prediction = (all_predictions * route_prob).sum(dim=-1, keepdim=True)
        else:
            selected = route_prob.argmax(dim=-1, keepdim=True)
            prediction = all_predictions.gather(dim=-1, index=selected)

        return {
            "prediction": prediction,
            "all_predictions": all_predictions,
            "route_logits": route_logits.reshape(batch_size, stock_num, self.num_predictors),
            "route_prob": route_prob,
            "latent": hidden.reshape(batch_size, stock_num, -1),
        }


Model = TRA
