"""
CI-STHPAN: Pre-trained Attention Network for Stock Selection with
Channel-Independent Spatio-Temporal Hypergraph.

Source type: official-source-derived with dependency reduction.

The public implementation spreads the model over PatchTST, PatchTST_backbone,
patch layers, RevIN, and torch_geometric HypergraphConv. This standalone file
keeps the core idea in one file:
    channel-independent patch embedding -> temporal Transformer encoder
    -> optional spatio-temporal hypergraph attention -> prediction head

Expected input:
    x: Tensor [num_stocks, channels, context_window]
       `num_stocks` is treated as the cross-sectional node dimension when a
       hypergraph is supplied.
    hyperedge_index: optional Tensor [2, num_memberships] or list of such tensors
       Row 0 stores stock/node ids, row 1 stores hyperedge ids.

Output:
    By default, Tensor [num_stocks, target_window] for stock-selection
    regression. Set head_type="prediction" to obtain
    Tensor [num_stocks, target_window, channels] PatchTST-style forecasts.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


class RevIN(nn.Module):
    """Reversible instance normalization used by PatchTST-style models."""

    def __init__(self, num_features: int, affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_features, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_features, 1))

    def norm(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(self.eps)
        z = (x - mean) / std
        if self.affine:
            z = z * self.weight + self.bias
        return z, mean, std

    def denorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = (x - self.bias) / self.weight.clamp_min(self.eps)
        return x * std + mean


class PureHypergraphConv(nn.Module):
    """Pure PyTorch hypergraph propagation with soft attention over memberships."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.proj = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.edge_proj = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.attn = nn.Parameter(torch.empty(heads, out_dim * 2))
        self.dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.attn)

    @staticmethod
    def incidence_from_index(
        hyperedge_index: torch.Tensor,
        num_nodes: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if hyperedge_index is None:
            return torch.eye(num_nodes, dtype=dtype, device=device)
        if hyperedge_index.dim() != 2 or hyperedge_index.size(0) != 2:
            raise ValueError("hyperedge_index must have shape [2, num_memberships].")
        node_idx = hyperedge_index[0].long().to(device)
        edge_idx = hyperedge_index[1].long().to(device)
        num_edges = int(edge_idx.max().item()) + 1 if edge_idx.numel() else num_nodes
        h = torch.zeros(num_nodes, num_edges, dtype=dtype, device=device)
        h[node_idx, edge_idx] = 1.0
        return h

    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor | None = None) -> torch.Tensor:
        # x: [num_stocks, in_dim]
        num_nodes = x.size(0)
        h = self.incidence_from_index(hyperedge_index, num_nodes, x.dtype, x.device)
        edge_degree = h.sum(dim=0).clamp_min(1.0)

        node_feat = self.proj(x).view(num_nodes, self.heads, self.out_dim)
        edge_input = h.t().matmul(x) / edge_degree.unsqueeze(-1)
        edge_feat = self.edge_proj(edge_input).view(h.size(1), self.heads, self.out_dim)

        node_ids, edge_ids = h.nonzero(as_tuple=True)
        if node_ids.numel() == 0:
            return x.new_zeros(num_nodes, self.out_dim)

        pair = torch.cat([node_feat[node_ids], edge_feat[edge_ids]], dim=-1)
        score = F.leaky_relu((pair * self.attn.unsqueeze(0)).sum(dim=-1), negative_slope=0.2)
        score_exp = torch.exp(score - score.max(dim=0, keepdim=True).values)

        denom = x.new_zeros(num_nodes, self.heads)
        denom.index_add_(0, node_ids, score_exp)
        alpha = score_exp / denom[node_ids].clamp_min(1e-6)
        alpha = self.dropout(alpha)

        messages = alpha.unsqueeze(-1) * edge_feat[edge_ids]
        out = x.new_zeros(num_nodes, self.heads, self.out_dim)
        out.index_add_(0, node_ids, messages)
        return out.mean(dim=1) + self.bias


class HGATBlock(nn.Module):
    """Two-layer hypergraph attention block from the CI-STHPAN HGAT branch."""

    def __init__(self, patch_num: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.patch_num = patch_num
        self.d_model = d_model
        flat_dim = patch_num * d_model
        self.linear1 = nn.Linear(flat_dim, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.hconv1 = PureHypergraphConv(d_model, d_model, heads=4, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.hconv2 = PureHypergraphConv(d_model, d_model, heads=1, dropout=dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.linear2 = nn.Linear(d_model, flat_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor | None = None) -> torch.Tensor:
        # x: [num_stocks, patch_num, d_model]
        src = x
        z = F.leaky_relu(self.linear1(x.reshape(x.size(0), -1)), negative_slope=0.2)
        z = self.norm1(self.dropout(z))

        h = self.hconv1(z, hyperedge_index)
        z = self.norm2(z + self.dropout(h))
        h = self.hconv2(z, hyperedge_index)
        z = self.norm3(z + self.dropout(h))

        z = F.leaky_relu(self.linear2(z), negative_slope=0.2)
        z = z.view(x.size(0), self.patch_num, self.d_model)
        return self.out_norm(src + self.dropout(z))


class PatchEncoder(nn.Module):
    """Channel-independent PatchTST encoder."""

    def __init__(
        self,
        channels: int,
        patch_len: int,
        patch_num: int,
        d_model: int = 128,
        n_heads: int = 16,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
        use_channel_independence: bool = True,
        use_hypergraph: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.patch_len = patch_len
        self.patch_num = patch_num
        self.d_model = d_model
        self.use_channel_independence = use_channel_independence
        self.use_hypergraph = use_hypergraph

        if use_channel_independence:
            self.patch_proj = nn.Linear(patch_len, d_model)
        else:
            self.patch_proj = nn.Linear(channels * patch_len, d_model)
            self.channel_restore = nn.Linear(d_model, channels * d_model)

        self.positional = nn.Parameter(torch.zeros(1, patch_num, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)

        if use_hypergraph:
            self.hgat = HGATBlock(patch_num, d_model, dropout)

    def _edge_for_channel(
        self,
        hyperedge_index: torch.Tensor | list[torch.Tensor] | None,
        channel_idx: int,
    ) -> torch.Tensor | None:
        if hyperedge_index is None:
            return None
        if isinstance(hyperedge_index, list):
            return hyperedge_index[channel_idx]
        return hyperedge_index

    def forward(
        self,
        patches: torch.Tensor,
        hyperedge_index: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        # patches: [N, C, patch_num, patch_len]
        n, c, p, l = patches.shape
        if self.use_channel_independence:
            z = self.patch_proj(patches)  # [N, C, P, D]
            z = z.reshape(n * c, p, self.d_model)
            z = self.encoder(self.dropout(z + self.positional))
            z = z.view(n, c, p, self.d_model)

            if self.use_hypergraph:
                outs = []
                for i in range(c):
                    outs.append(self.hgat(z[:, i, :, :], self._edge_for_channel(hyperedge_index, i)))
                z = torch.stack(outs, dim=1)
            return z  # [N, C, P, D]

        mixed = patches.permute(0, 2, 1, 3).reshape(n, p, c * l)
        z = self.patch_proj(mixed)
        z = self.encoder(self.dropout(z + self.positional))
        if self.use_hypergraph:
            z = self.hgat(z, self._edge_for_channel(hyperedge_index, 0))
        z = self.channel_restore(z).view(n, p, c, self.d_model)
        return z.permute(0, 2, 1, 3)


class CISTHPAN(nn.Module):
    """Standalone CI-STHPAN core model."""

    def __init__(
        self,
        channels: int,
        context_window: int,
        target_window: int = 1,
        patch_len: int = 12,
        stride: int = 12,
        d_model: int = 128,
        n_heads: int = 16,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
        use_revin: bool = True,
        use_channel_independence: bool = True,
        use_hypergraph: bool = True,
        head_type: str = "regression",
    ):
        super().__init__()
        if context_window < patch_len:
            raise ValueError("context_window must be >= patch_len.")
        patch_num = int((context_window - patch_len) / stride + 1)
        if patch_num < 1:
            raise ValueError("patch_num must be positive.")

        self.channels = channels
        self.context_window = context_window
        self.target_window = target_window
        self.patch_len = patch_len
        self.stride = stride
        self.patch_num = patch_num
        if head_type not in {"regression", "prediction", "pretrain"}:
            raise ValueError("head_type must be one of: regression, prediction, pretrain.")
        self.head_type = head_type
        self.use_revin = use_revin
        self.revin = RevIN(channels) if use_revin else None

        self.encoder = PatchEncoder(
            channels=channels,
            patch_len=patch_len,
            patch_num=patch_num,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            use_channel_independence=use_channel_independence,
            use_hypergraph=use_hypergraph,
        )
        self.prediction_head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Dropout(dropout),
            nn.Linear(patch_num * d_model, target_window),
        )
        self.regression_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(dropout),
            nn.Linear(channels * patch_num * d_model, target_window),
        )
        self.pretrain_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, patch_len),
        )

    def make_patches(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # [N, C, patch_num, patch_len]
        if patches.size(2) != self.patch_num:
            raise ValueError(f"expected patch_num={self.patch_num}, got {patches.size(2)}.")
        return patches

    def forward(
        self,
        x: torch.Tensor,
        hyperedge_index: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("CISTHPAN expects x with shape [num_stocks, channels, context_window].")
        if x.size(1) != self.channels or x.size(2) != self.context_window:
            raise ValueError(
                f"expected [N, {self.channels}, {self.context_window}], got {tuple(x.shape)}."
            )

        mean = std = None
        if self.revin is not None:
            x, mean, std = self.revin.norm(x)

        patches = self.make_patches(x)
        z = self.encoder(patches, hyperedge_index)  # [N, C, P, D]

        if self.head_type == "pretrain":
            return self.pretrain_head(z)  # [N, C, patch_num, patch_len]

        if self.head_type == "prediction":
            y = self.prediction_head(z).permute(0, 2, 1)  # [N, target_window, C]
            if self.revin is not None and mean is not None and std is not None:
                # RevIN denormalization is only valid for channel forecasts.
                y = self.revin.denorm(y.permute(0, 2, 1), mean, std).permute(0, 2, 1)
            return y

        # Paper stock-selection setting: predict a scalar return/rank signal per stock.
        y = self.regression_head(z)
        if self.target_window == 1:
            return y.squeeze(-1)
        return y


class CISTHPANStockModel(nn.Module):
    """Project wrapper for the unified [B,N,L,F] stock baseline contract."""

    def __init__(
        self,
        input_dim: int,
        num_stocks: int,
        seq_len: int,
        hyperedge_index,
        target_window: int = 1,
        patch_len: int = 12,
        stride: int = 12,
        d_model: int = 128,
        n_heads: int = 16,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
        use_revin: bool = True,
        use_channel_independence: bool = True,
        use_hypergraph: bool = True,
        share_hypergraph_across_channels: bool = True,
        head_type: str = "regression",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_stocks = int(num_stocks)
        self.seq_len = int(seq_len)
        self.target_window = int(target_window)
        self.share_hypergraph_across_channels = bool(share_hypergraph_across_channels)
        edge = torch.as_tensor(hyperedge_index, dtype=torch.long)
        if edge.dim() != 2 or edge.size(0) != 2:
            raise ValueError(f"hyperedge_index must be [2,E], got {tuple(edge.shape)}")
        self.register_buffer("hyperedge_index", edge, persistent=True)
        self.core = CISTHPAN(
            channels=input_dim,
            context_window=seq_len,
            target_window=target_window,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            use_revin=use_revin,
            use_channel_independence=use_channel_independence,
            use_hypergraph=use_hypergraph,
            head_type=head_type,
        )

    def _edge_arg(self):
        if self.share_hypergraph_across_channels:
            return self.hyperedge_index
        return [self.hyperedge_index for _ in range(self.input_dim)]

    def _forward_tensor(self, x: torch.Tensor, stock_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
            if stock_mask is not None and stock_mask.dim() == 1:
                stock_mask = stock_mask.unsqueeze(0)
        if x.dim() != 4:
            raise ValueError(f"CI-STHPAN expects [B,N,L,F] or [N,L,F], got {tuple(x.shape)}")
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

        edge = self._edge_arg()
        outputs = []
        for idx in range(b):
            # Core model follows the paper notation [|S|, M, L].
            outputs.append(self.core(x[idx].permute(0, 2, 1), edge))
        pred = torch.stack(outputs, dim=0)
        if pred.dim() == 2:
            pred = pred.unsqueeze(-1)
        if stock_mask is not None:
            pred = pred * stock_mask.to(device=pred.device, dtype=pred.dtype).unsqueeze(-1)
        return pred

    def forward(self, batch, stock_mask: torch.Tensor | None = None) -> torch.Tensor:
        if isinstance(batch, dict):
            stock_mask = batch.get("mask", stock_mask)
            x = batch["inputs"]
        else:
            x = batch
        return self._forward_tensor(x, stock_mask=stock_mask)


CoreModel = CISTHPAN
Model = CISTHPANStockModel
