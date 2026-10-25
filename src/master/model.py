"""
MASTER: Market-Guided Stock Transformer for Stock Price Forecasting.

Source type: official-source-derived.

This file keeps only the core neural modules from the public implementation.
It removes the Qlib/SequenceModel wrapper so the model can be reviewed as a
single PyTorch file.

Expected input:
    x: Tensor [num_stocks, lookback, stock_features + market_features]
       or [batch, num_stocks, lookback, stock_features + market_features].
       The first `d_feat` features are stock features. Features in
       [gate_input_start_index, gate_input_end_index) are market status
       features copied to each stock by the caller.

Output:
    Tensor [num_stocks] or [batch, num_stocks]
"""

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal temporal position encoding used before temporal attention."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(0):
            raise ValueError(f"sequence length {x.size(1)} exceeds max_len {self.pe.size(0)}")
        return x + self.pe[: x.size(1), :]


class SAttention(nn.Module):
    """Inter-stock self-attention over the cross-sectional stock dimension."""

    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(d_model / nhead)

        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(nhead)])

        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, stock_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [num_stocks, lookback, d_model]
        if stock_mask is not None:
            stock_mask = stock_mask.to(device=x.device).reshape(-1).bool()
            if stock_mask.numel() != x.size(0):
                raise ValueError(
                    f"stock_mask length {stock_mask.numel()} does not match stock count {x.size(0)}"
                )
            if not torch.compiler.is_compiling() and not bool(stock_mask.any()):
                stock_mask = None

        x = self.norm1(x)
        q = self.qtrans(x).transpose(0, 1)  # [lookback, num_stocks, d_model]
        k = self.ktrans(x).transpose(0, 1)
        v = self.vtrans(x).transpose(0, 1)

        dim = int(self.d_model / self.nhead)
        outputs = []
        for i in range(self.nhead):
            start = i * dim
            end = None if i == self.nhead - 1 else (i + 1) * dim
            qh = q[:, :, start:end]
            kh = k[:, :, start:end]
            vh = v[:, :, start:end]
            scores = torch.matmul(qh, kh.transpose(1, 2)) / self.temperature
            if stock_mask is not None:
                scores = scores.masked_fill(~stock_mask.view(1, 1, -1), torch.finfo(scores.dtype).min)
            attn = torch.softmax(scores, dim=-1)
            attn = self.attn_dropout[i](attn)
            head_output = torch.matmul(attn, vh).transpose(0, 1)
            if stock_mask is not None:
                head_output = head_output * stock_mask.view(-1, 1, 1).to(dtype=head_output.dtype)
            outputs.append(head_output)

        att_output = torch.cat(outputs, dim=-1)
        xt = x + att_output
        xt = self.norm2(xt)
        return xt + self.ffn(xt)


class TAttention(nn.Module):
    """Intra-stock temporal self-attention along each stock's lookback window."""

    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(nhead)])

        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [num_stocks, lookback, d_model]
        x = self.norm1(x)
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)

        dim = int(self.d_model / self.nhead)
        outputs = []
        for i in range(self.nhead):
            start = i * dim
            end = None if i == self.nhead - 1 else (i + 1) * dim
            qh = q[:, :, start:end]
            kh = k[:, :, start:end]
            vh = v[:, :, start:end]
            attn = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)), dim=-1)
            attn = self.attn_dropout[i](attn)
            outputs.append(torch.matmul(attn, vh))

        att_output = torch.cat(outputs, dim=-1)
        xt = x + att_output
        xt = self.norm2(xt)
        return xt + self.ffn(xt)


class Gate(nn.Module):
    """Market-guided feature gate. The softmax weights are scaled by feature count."""

    def __init__(self, d_input: int, d_output: int, beta: float = 1.0):
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.d_output = d_output
        self.temperature = beta

    def forward(self, gate_input: torch.Tensor) -> torch.Tensor:
        output = self.trans(gate_input)
        return self.d_output * torch.softmax(output / self.temperature, dim=-1)


class TemporalAttention(nn.Module):
    """Aggregate the temporal sequence into one stock representation."""

    def __init__(self, d_model: int):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [num_stocks, lookback, d_model]
        h = self.trans(z)
        query = h[:, -1, :].unsqueeze(-1)
        lam = torch.matmul(h, query).squeeze(-1)
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        return torch.matmul(lam, z).squeeze(1)


class MASTER(nn.Module):
    """Market-guided stock transformer from the MASTER paper."""

    def __init__(
        self,
        d_feat: int,
        d_model: int = 256,
        t_nhead: int = 4,
        s_nhead: int = 2,
        t_dropout_rate: float = 0.5,
        s_dropout_rate: float = 0.5,
        gate_input_start_index: int = 158,
        gate_input_end_index: int = 221,
        beta: float = 5.0,
        max_len: int = 512,
    ):
        super().__init__()
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        d_gate_input = gate_input_end_index - gate_input_start_index
        if gate_input_start_index != d_feat:
            raise ValueError(
                "MASTER requires gate_input_start_index == d_feat so the stock feature slice "
                f"matches the input projection; got d_feat={d_feat}, "
                f"gate_input_start_index={gate_input_start_index}."
            )
        if d_gate_input <= 0:
            raise ValueError(
                "MASTER requires at least one market status feature; "
                f"got gate_input_start_index={gate_input_start_index}, "
                f"gate_input_end_index={gate_input_end_index}."
            )

        self.feature_gate = Gate(d_gate_input, d_feat, beta=beta)
        self.input_projection = nn.Linear(d_feat, d_model)
        self.position = PositionalEncoding(d_model, max_len=max_len)
        self.temporal_encoder = TAttention(d_model=d_model, nhead=t_nhead, dropout=t_dropout_rate)
        self.stock_encoder = SAttention(d_model=d_model, nhead=s_nhead, dropout=s_dropout_rate)
        self.temporal_aggregator = TemporalAttention(d_model=d_model)
        self.prediction = nn.Linear(d_model, 1)

    def _forward_single(self, x: torch.Tensor, stock_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("MASTER expects x with shape [num_stocks, lookback, features].")
        if x.size(-1) < self.gate_input_end_index:
            raise ValueError(
                f"MASTER input feature dim {x.size(-1)} is smaller than gate_input_end_index "
                f"{self.gate_input_end_index}."
            )

        src = x[:, :, : self.gate_input_start_index]
        gate_input = x[:, -1, self.gate_input_start_index : self.gate_input_end_index]
        src = src * self.feature_gate(gate_input).unsqueeze(1)
        z = self.input_projection(src)
        z = self.position(z)
        z = self.temporal_encoder(z)
        z = self.stock_encoder(z, stock_mask=stock_mask)
        z = self.temporal_aggregator(z)
        return self.prediction(z).squeeze(-1)

    def forward(self, x: torch.Tensor, stock_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() == 3:
            if stock_mask is not None and stock_mask.dim() == 2:
                if stock_mask.size(0) != 1:
                    raise ValueError("single-date MASTER input received a batched stock_mask")
                stock_mask = stock_mask.squeeze(0)
            return self._forward_single(x, stock_mask=stock_mask)

        if x.dim() == 4:
            masks = None
            if stock_mask is not None:
                if stock_mask.dim() == 1:
                    masks = stock_mask.unsqueeze(0).expand(x.size(0), -1)
                elif stock_mask.dim() == 2:
                    if stock_mask.size(0) != x.size(0):
                        raise ValueError(
                            f"stock_mask batch {stock_mask.size(0)} does not match input batch {x.size(0)}"
                        )
                    masks = stock_mask
                else:
                    raise ValueError(f"stock_mask must be [stocks] or [batch,stocks], got {tuple(stock_mask.shape)}")
            outputs = []
            for index in range(x.size(0)):
                mask = None if masks is None else masks[index]
                outputs.append(self._forward_single(x[index], stock_mask=mask))
            return torch.stack(outputs, dim=0)

        raise ValueError("MASTER expects x with shape [num_stocks, lookback, features] or [batch,num_stocks,lookback,features].")


Model = MASTER
