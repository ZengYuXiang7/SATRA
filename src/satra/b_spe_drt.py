"""Core SATRA architecture and masked-reconstruction pretraining model.

The supervised path maps a market window ``[B, N, T, F]`` to one return score
per stock:

``feature projection -> Adaptive Temporal Pattern Encoder -> Dual Relation Tower
-> prediction head``.

The Adaptive Temporal Pattern Encoder combines temporal convolutions with sparse
top-k routing over heterogeneous bottleneck experts. The Dual Relation Tower
then processes each date independently across stocks. Its semantic branch has
global attention, while its prior-guided branch can attend only to neighbors
in a fixed relation matrix. An input-conditioned two-way softmax gate fuses their outputs before prediction.

Some public class and argument names retain the earlier ``PSTRNet`` prefix for
checkpoint and launcher compatibility. They implement the SATRA model
described above."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


MOE_EXPANSION_RATIO = 4
HETEROGENEOUS_EXPERT_RATIOS = (
    1.0 / 16.0,
    1.0 / 8.0,
    1.0 / 4.0,
    1.0 / 2.0,
    3.0 / 4.0,
    1.0,
)


class SamePadConv(nn.Module):
    """Length-preserving one-dimensional temporal convolution."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        receptive_field = (int(kernel_size) - 1) * int(dilation) + 1
        self.remove = 1 if receptive_field % 2 == 0 else 0
        self.conv = nn.Conv1d(
            int(in_channels),
            int(out_channels),
            int(kernel_size),
            padding=receptive_field // 2,
            dilation=int(dilation),
            groups=int(groups),
        )

    def forward(self, x):
        out = self.conv(x)
        if self.remove:
            out = out[:, :, : -self.remove]
        return out


class ConvBlock(nn.Module):
    """Two temporal convolutions with a residual connection."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = (
            nn.Conv1d(int(in_channels), int(out_channels), 1)
            if int(in_channels) != int(out_channels) or final
            else None
        )

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = self.conv1(F.gelu(x))
        x = self.conv2(F.gelu(x))
        return x + residual


class DilatedConvEncoder(nn.Module):
    """Serial residual CNN operating on the original history axis ``T``."""

    def __init__(self, in_channels, channels, kernel_size=3):
        super().__init__()
        channels = tuple(int(channel) for channel in channels)
        self.net = nn.Sequential(
            *[
                ConvBlock(
                    channels[index - 1] if index else int(in_channels),
                    channels[index],
                    kernel_size=kernel_size,
                    dilation=1,
                    final=index == len(channels) - 1,
                )
                for index in range(len(channels))
            ]
        )

    def forward(self, x):
        return self.net(x)


class BottleNeck(nn.Module):
    """One heterogeneous MoE expert for one stock's full history."""

    def __init__(self, seq_len, model_dim, bottleneck_dim, dropout=0.1):
        super().__init__()
        self.seq_len = int(seq_len)
        self.model_dim = int(model_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.net = nn.Sequential(
            nn.Linear(self.model_dim, self.bottleneck_dim),
            nn.GELU(),
            nn.Linear(self.bottleneck_dim, self.bottleneck_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.bottleneck_dim, self.model_dim),
        )

    def forward(self, representation):
        residual = representation
        update = self.net(
            representation.reshape(representation.shape[0], self.seq_len, self.model_dim)
        )
        return residual + update.reshape(
            representation.shape[0], self.seq_len * self.model_dim
        )


class SparseDispatcher:
    """Dispatch flattened stock histories to the selected experts."""

    def __init__(self, num_experts, gates):
        self.gates = gates
        self.num_experts = int(num_experts)
        nonzero = torch.nonzero(gates, as_tuple=False)
        routed = nonzero[torch.argsort(nonzero[:, 1])]
        self.batch_index = routed[:, 0]
        self.expert_index = routed[:, 1]
        self.part_sizes = (gates > 0).sum(dim=0).tolist()
        self.nonzero_gates = gates[self.batch_index, self.expert_index]

    def dispatch(self, inputs):
        return torch.split(inputs[self.batch_index], self.part_sizes, dim=0)

    def combine(self, expert_outputs):
        stitched = torch.cat(expert_outputs, dim=0)
        weighted = stitched * self.nonzero_gates.unsqueeze(-1).to(dtype=stitched.dtype)
        combined = stitched.new_zeros((self.gates.shape[0], stitched.shape[1]))
        return combined.index_add(0, self.batch_index, weighted)


class AdaptiveBottleNeck(nn.Module):
    """Noisy top-k heterogeneous MoE with an optional load-balance diagnostic."""

    def __init__(self, seq_len, model_dim, noisy_gating=True, top_k=3, dropout=0.1):
        super().__init__()
        self.seq_len = int(seq_len)
        self.model_dim = int(model_dim)
        self.bottleneck_dims = tuple(
            max(1, int(round(self.model_dim * ratio)))
            for ratio in HETEROGENEOUS_EXPERT_RATIOS
        )
        self.num_bottlenecks = len(self.bottleneck_dims)
        self.top_k = int(top_k)
        self.noisy_gating = bool(noisy_gating)
        self.bottlenecks = nn.ModuleList(
            [
                BottleNeck(self.seq_len, self.model_dim, bottleneck_dim, dropout=dropout)
                for bottleneck_dim in self.bottleneck_dims
            ]
        )
        router_width = self.seq_len * self.model_dim
        self.w_gate = nn.Parameter(torch.zeros(router_width, self.num_bottlenecks))
        self.w_noise = nn.Parameter(torch.zeros(router_width, self.num_bottlenecks))
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)
        self._last_router_probabilities = None
        self._last_topk_selection = None

    def noisy_top_k_gating(self, router_input):
        clean_logits = router_input @ self.w_gate
        # Keep a full, noise-free router distribution for the auxiliary
        # load-balance loss.  It is intentionally not truncated to Top-K so
        # every expert logit can receive a learning signal.
        router_probabilities = torch.softmax(clean_logits.float(), dim=1)
        if self.noisy_gating and self.training:
            noise_stddev = self.softplus(router_input @ self.w_noise) + 1e-2
            logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
        else:
            logits = clean_logits
        top_logits, top_indices = logits.topk(self.top_k, dim=1)
        # AMP may keep router logits in FP32 while the selected values are
        # BF16.  ``scatter`` requires both tensors to share a dtype.
        gates = self.softmax(top_logits.float()).to(dtype=logits.dtype)
        sparse_gates = torch.zeros_like(logits).scatter(1, top_indices, gates)
        self._last_router_probabilities = router_probabilities
        self._last_topk_selection = (
            sparse_gates.detach().ne(0).to(dtype=router_probabilities.dtype)
        )
        return sparse_gates

    def load_balance_loss(self):
        """Return the Switch-style importance-times-load balance objective.

        The Top-K selection is discrete, so it is treated as a fixed routing
        load measurement.  Gradients flow through the full router probability
        distribution, including experts absent from the selected Top-K set.
        """

        if self._last_router_probabilities is None or self._last_topk_selection is None:
            raise RuntimeError("MoE load-balance loss requested before a routing forward pass")
        importance = self._last_router_probabilities.mean(dim=0)
        load = self._last_topk_selection.mean(dim=0) / float(self.top_k)
        return self.num_bottlenecks * torch.sum(importance * load)

    def forward(self, router_input, representation):
        gates = self.noisy_top_k_gating(router_input)
        dispatcher = SparseDispatcher(self.num_bottlenecks, gates)
        expert_inputs = dispatcher.dispatch(representation)
        expert_outputs = [
            expert(expert_inputs[index])
            for index, expert in enumerate(self.bottlenecks)
        ]
        return dispatcher.combine(expert_outputs)


class SPELayer(nn.Module):
    """Temporal CNN and heterogeneous MoE applied before stock relations.

    ``[B, N, T, d_model]`` is reshaped to ``[B*N, d_model, T]`` for the CNN,
    so convolution learns the history axis.  MoE then routes one complete
    ``T x (4*d_model)`` representation for each stock window.
    """

    def __init__(self, d_model, seq_len, depth=10, kernel_size=3, top_k=3, dropout=0.1):
        super().__init__()
        self.d_model = int(d_model)
        self.seq_len = int(seq_len)
        self.encoder = DilatedConvEncoder(
            self.d_model,
            [self.d_model] * (int(depth) + 1),
            kernel_size=kernel_size,
        )
        self.dropout = nn.Dropout(float(dropout))
        self.moe_dim = self.d_model * MOE_EXPANSION_RATIO
        self.moe_input_projection = nn.Linear(self.d_model, self.moe_dim)

        # ABLATION (a) Temporal specialization design:
        # Replace this complete SPE layer with CNN-only or homogeneous-expert
        # variants.  The full model always uses heterogeneous
        # Top-3 state-conditioned routing over the six hard-coded widths.
        self.adaptive_bottleneck = AdaptiveBottleNeck(
            seq_len=self.seq_len,
            model_dim=self.moe_dim,
            top_k=top_k,
            dropout=dropout,
        )
        self.moe_output_projection = nn.Linear(self.moe_dim, self.d_model)

    def forward(self, inputs):
        batch_size, stock_num, seq_len, d_model = inputs.shape
        representation = inputs.reshape(batch_size * stock_num, seq_len, d_model)
        representation = self.encoder(representation.transpose(1, 2)).transpose(1, 2)
        representation = self.dropout(representation)
        representation = self.moe_input_projection(representation)
        flattened = representation.reshape(batch_size * stock_num, -1)
        flattened = self.adaptive_bottleneck(flattened, flattened)
        representation = self.moe_output_projection(
            flattened.reshape(batch_size * stock_num, seq_len, self.moe_dim)
        )
        return representation.reshape(batch_size, stock_num, seq_len, d_model)

    def moe_load_balance_loss(self):
        return self.adaptive_bottleneck.load_balance_loss()


class CrossStockTransformerLayer(nn.Module):
    """One pre-normalized Transformer layer over the stock axis ``N``."""

    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.RMSNorm(int(d_model))
        self.norm2 = nn.RMSNorm(int(d_model))
        self.self_attention = nn.MultiheadAttention(
            int(d_model),
            int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(float(dropout))
        self.feed_forward = nn.Sequential(
            nn.Linear(int(d_model), int(d_model) * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(d_model) * 2, int(d_model)),
            nn.Dropout(float(dropout)),
        )

    def forward(self, stock_tokens, attention_mask):
        normalized = self.norm1(stock_tokens)
        attention_output, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            need_weights=False,
        )
        stock_tokens = stock_tokens + self.attention_dropout(attention_output)
        return stock_tokens + self.feed_forward(self.norm2(stock_tokens))

    def forward_diagnostics(self, stock_tokens, attention_mask, return_attention_weights):
        """Return the layer output and relation-specific attention update.

        This path is inference-only.  It keeps the production ``forward``
        unchanged while exposing the tensor immediately before the attention
        residual addition.  Per-head attention weights are materialized only
        when explicitly requested by the diagnostic runner.
        """
        if self.training:
            raise RuntimeError("forward_diagnostics requires eval mode")
        normalized = self.norm1(stock_tokens)
        attention_output, attention_weights = self.self_attention(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            need_weights=bool(return_attention_weights),
            average_attn_weights=False,
        )
        attention_update = self.attention_dropout(attention_output)
        post_attention = stock_tokens + attention_update
        output = post_attention + self.feed_forward(self.norm2(post_attention))
        if return_attention_weights and attention_weights is None:
            raise RuntimeError("MultiheadAttention did not return requested attention weights")
        return output, attention_update, attention_weights

    def attention_topk_indices(self, stock_tokens, top_k, query_chunk_size=128):
        """Return exact per-head attention Top-k indices without an ``N x N`` output.

        Softmax preserves logit order in evaluation mode, so Top-k indices can
        be obtained directly from scaled query-key logits.  Query chunking caps
        diagnostic memory while matching the layer's semantic attention ranks.
        """
        if self.training:
            raise RuntimeError("attention_topk_indices requires eval mode")
        if stock_tokens.ndim != 3:
            raise ValueError(
                f"Expected stock tokens [B, N, D], got {tuple(stock_tokens.shape)}"
            )
        batch_size, stock_num, embed_dim = stock_tokens.shape
        top_k = int(top_k)
        query_chunk_size = int(query_chunk_size)
        if top_k <= 0 or top_k >= stock_num:
            raise ValueError(f"Invalid attention Top-k={top_k} for stock count {stock_num}")
        if query_chunk_size <= 0:
            raise ValueError(f"query_chunk_size must be positive, got {query_chunk_size}")
        attention = self.self_attention
        if attention.in_proj_weight is None:
            raise RuntimeError("Expected packed MultiheadAttention projection weights")
        if embed_dim != attention.embed_dim:
            raise ValueError(
                f"Token width {embed_dim} does not match attention width {attention.embed_dim}"
            )
        if embed_dim % attention.num_heads != 0:
            raise ValueError(
                f"Attention width {embed_dim} is not divisible by {attention.num_heads} heads"
            )

        normalized = self.norm1(stock_tokens)
        projected = F.linear(
            normalized,
            attention.in_proj_weight,
            attention.in_proj_bias,
        )
        query, key, _ = projected.chunk(3, dim=-1)
        head_dim = embed_dim // attention.num_heads
        query = query.reshape(
            batch_size, stock_num, attention.num_heads, head_dim
        ).transpose(1, 2)
        key = key.reshape(
            batch_size, stock_num, attention.num_heads, head_dim
        ).transpose(1, 2)
        key_transposed = key.transpose(-2, -1)
        scale = head_dim ** -0.5
        chunks = []
        for query_start in range(0, stock_num, query_chunk_size):
            query_end = min(query_start + query_chunk_size, stock_num)
            logits = torch.matmul(
                query[:, :, query_start:query_end],
                key_transposed,
            ) * scale
            local_queries = torch.arange(
                query_end - query_start,
                device=stock_tokens.device,
            )
            global_queries = torch.arange(
                query_start,
                query_end,
                device=stock_tokens.device,
            )
            logits[:, :, local_queries, global_queries] = -torch.inf
            chunks.append(logits.topk(k=top_k, dim=-1).indices)
        return torch.cat(chunks, dim=2)


def build_relation_attention_mask(relation_weight_matrix):
    """Convert directed binary relation support to a boolean attention mask.

    A matrix value of one means the row stock may attend to the column stock.
    The diagonal is always enabled for self-attention.  PyTorch boolean masks
    use ``True`` for forbidden attention pairs, so the returned matrix is the
    complement of the allowed DTW support.
    """

    # ABLATION (d) Relation prior construction:
    # This is the single prior boundary.  Supply a DTW, industry, or Euclidean
    # binary adjacency here.  "No prior constraint" is deliberately semantic
    # tower only and therefore does not instantiate this path in its forward.
    matrix = torch.as_tensor(relation_weight_matrix)
    stock_num = matrix.shape[0]
    allowed = matrix.to(dtype=torch.bool)
    allowed = allowed | torch.eye(stock_num, dtype=torch.bool, device=matrix.device)
    return ~allowed


class PredictionHead(nn.Module):
    """Map fused per-stock history features to a one-step scalar target."""

    OUTPUT_ACTIVATIONS = ("identity", "tanh")

    def __init__(self, d_model, seq_len, output_activation="identity"):
        super().__init__()
        self.output_activation = str(output_activation)
        self.feature_to_target = nn.Linear(int(d_model), 1)
        self.history_to_future = nn.Linear(int(seq_len), 1)

    def forward(self, representation):
        # Keep the user-defined order: d_model -> 1, then T -> 1.
        target_history = self.feature_to_target(representation)
        prediction = self.history_to_future(target_history.transpose(-1, -2)).transpose(-1, -2)
        if self.output_activation == "tanh":
            return torch.tanh(prediction)
        return prediction


class PSTRNet_SPE_DRT(nn.Module):
    """PSTR-Net with parallel semantic and DTW-prior cross-stock Transformers."""

    def __init__(
        self,
        enc_in,
        d_model,
        seq_len,
        dtw_weight_matrix,
        depth=10,
        kernel_size=3,
        moe_top_k=3,
        num_heads=4,
        tower_layers=1,
        dropout=0.1,
        semantic_support="global",
    ):
        super().__init__()
        self.semantic_support = str(semantic_support)
        self.learnable_fusion = False
        self.feature_aggregation = False
        if self.semantic_support not in {"global", "dtw_prior"}:
            raise ValueError(
                "semantic_support must be either 'global' or 'dtw_prior', got "
                f"{self.semantic_support!r}"
            )
        self.input_projection = nn.Linear(int(enc_in), int(d_model))
        self.spe = SPELayer(
            d_model=d_model,
            seq_len=seq_len,
            depth=depth,
            kernel_size=kernel_size,
            top_k=moe_top_k,
            dropout=dropout,
        )

        # ABLATION (b) Cross-stock relation design (DRT, Dual Relation Tower):
        # Disable both towers for "No relation learning"; retain only the
        # named tower for a single-tower variant; retain both for full PSTR-Net.
        # HYPERPARAMETER SEARCH (3): pstr_tower_layers changes both towers
        # together.  The two towers never share parameters.
        self.semantic_stock_transformer = nn.ModuleList(
            [
                CrossStockTransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(int(tower_layers))
            ]
        )
        self.prior_stock_transformer = nn.ModuleList(
            [
                CrossStockTransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(int(tower_layers))
            ]
        )
        self.register_buffer(
            "dtw_attention_mask",
            build_relation_attention_mask(dtw_weight_matrix),
            persistent=False,
        )

        # ABLATION (c) Auxiliary pretraining:
        # MMR pretraining is explicitly deferred.  This forward is only the
        # supervised prediction path and has no reconstruction branch or loss.
        self.prediction_head = PredictionHead(d_model=d_model, seq_len=seq_len)

    def route_relation_experts(self, stock_tokens):
        # 从进入两塔前的 ATPE 表征生成每个股票时间位置的两专家权重。
        gate_logits = self.drt_router(stock_tokens)
        gate_probabilities = torch.softmax(gate_logits.float(), dim=-1)
        return gate_probabilities

    def fuse_relation_states(self, semantic_state, prior_state):
        # 训练前向和机制诊断使用同一双塔聚合。
        if self.feature_aggregation:
            gate_weights = self.drt_gate_probabilities.to(dtype=semantic_state.dtype)
            fused = gate_weights[..., :1] * semantic_state + gate_weights[..., 1:] * prior_state
        elif self.learnable_fusion:
            fusion_alpha = torch.sigmoid(self.drt_fusion_logit)
            fused = fusion_alpha * semantic_state + (1.0 - fusion_alpha) * prior_state
        else:
            fused = semantic_state + prior_state
        return fused

    def _cross_stock_relation(self, representation):
        batch_size, stock_num, seq_len, d_model = representation.shape
        stock_tokens = representation.permute(0, 2, 1, 3).reshape(
            batch_size * seq_len,
            stock_num,
            d_model,
        )
        # 路由与两个关系专家读取同一 ATPE 输出，在双塔计算前确定权重。
        if self.feature_aggregation:
            self.drt_gate_probabilities = self.route_relation_experts(stock_tokens)

        # Controlled DRT support restriction: the semantic tower remains an
        # independent Transformer in both settings.  Only its visible support
        # changes; ``dtw_prior`` reuses the exact fixed mask used by the
        # prior-guided tower.
        semantic_attention_mask = (
            None
            if self.semantic_support == "global"
            else self.dtw_attention_mask
        )

        # Transformer 1: global semantic cross-stock attention by default.
        semantic_state = stock_tokens
        for transformer in self.semantic_stock_transformer:
            semantic_state = transformer(
                semantic_state,
                attention_mask=semantic_attention_mask,
            )

        # ABLATION (b), prior-guided tower only: retain this state and omit semantic_state.
        # Transformer 2: attention pairs with relation support 0 are masked out.
        prior_state = stock_tokens
        for transformer in self.prior_stock_transformer:
            prior_state = transformer(
                prior_state,
                attention_mask=self.dtw_attention_mask,
            )

        # 使用该变体的双塔权重生成融合表示。
        fused = self.fuse_relation_states(semantic_state, prior_state)
        return fused.reshape(batch_size, seq_len, stock_num, d_model).permute(0, 2, 1, 3)

    @torch.inference_mode()
    def drt_diagnostics(self, representation):
        """Measure DRT support, update, and prediction-impact diagnostics.

        ``representation`` is the SPE output with shape ``[B, N, T, D]``.
        The method batches the ``(batch, time)`` stock sets exactly as the
        production relation path does. Semantic attention ranks are still
        computed in query chunks, so the diagnostic never saves or returns a
        full per-head ``N x N`` attention matrix.
        """
        if self.training:
            raise RuntimeError("drt_diagnostics requires eval mode")
        if self.semantic_support != "global":
            raise RuntimeError(
                "drt_diagnostics is defined for the global-semantic DRT only; "
                "the prior-constrained variant is evaluated through the controlled "
                "performance comparison."
            )
        if representation.ndim != 4:
            raise ValueError(
                f"Expected SPE representation [B, N, T, D], got {tuple(representation.shape)}"
            )
        if len(self.semantic_stock_transformer) != len(self.prior_stock_transformer):
            raise RuntimeError("Semantic and prior towers must have the same number of layers")
        if len(self.semantic_stock_transformer) == 0:
            raise RuntimeError("DRT diagnostics require at least one layer per tower")

        batch_size, stock_num, seq_len, d_model = representation.shape
        if self.dtw_attention_mask.shape != (stock_num, stock_num):
            raise ValueError(
                "DTW attention mask shape does not match representation stock count: "
                f"mask={tuple(self.dtw_attention_mask.shape)}, stocks={stock_num}"
            )
        stock_tokens = representation.permute(0, 2, 1, 3).reshape(
            batch_size * seq_len,
            stock_num,
            d_model,
        )
        # 诊断沿用共同输入得到的权重，干预塔输出时不重新改变路由。
        if self.feature_aggregation:
            self.drt_gate_probabilities = self.route_relation_experts(stock_tokens)

        prior_support = ~self.dtw_attention_mask
        prior_support = prior_support.clone()
        prior_support.fill_diagonal_(False)
        support_counts = prior_support.sum(dim=1)
        unique_support_counts = torch.unique(support_counts)
        if unique_support_counts.numel() != 1:
            raise ValueError(
                "DRT support overlap requires a uniform prior degree; got "
                f"{unique_support_counts.detach().cpu().tolist()}"
            )
        prior_topk = int(unique_support_counts.item())
        if prior_topk <= 0 or prior_topk >= stock_num:
            raise ValueError(
                f"Invalid prior degree {prior_topk} for stock count {stock_num}"
            )

        layer_count = len(self.semantic_stock_transformer)
        support_overlap_sum = torch.zeros(
            layer_count,
            dtype=torch.float64,
            device=representation.device,
        )
        support_overlap_count = torch.zeros(
            layer_count,
            dtype=torch.int64,
            device=representation.device,
        )
        relation_dissimilarity_sum = torch.zeros(
            layer_count,
            dtype=torch.float64,
            device=representation.device,
        )
        relation_dissimilarity_count = torch.zeros(
            layer_count,
            dtype=torch.int64,
            device=representation.device,
        )
        query_indices = torch.arange(stock_num, device=representation.device).view(
            1, 1, stock_num, 1
        )
        norm_epsilon = torch.finfo(torch.float32).eps

        semantic_state = stock_tokens
        prior_state = stock_tokens
        for layer_index, (semantic_layer, prior_layer) in enumerate(
            zip(self.semantic_stock_transformer, self.prior_stock_transformer)
        ):
            semantic_top_indices = semantic_layer.attention_topk_indices(
                semantic_state,
                top_k=prior_topk,
            )
            semantic_state, semantic_relation_update, _ = (
                semantic_layer.forward_diagnostics(
                    semantic_state,
                    attention_mask=None,
                    return_attention_weights=False,
                )
            )
            prior_state, prior_relation_update, _ = prior_layer.forward_diagnostics(
                prior_state,
                attention_mask=self.dtw_attention_mask,
                return_attention_weights=False,
            )

            selected_prior_support = prior_support[
                query_indices.expand_as(semantic_top_indices),
                semantic_top_indices,
            ]
            support_overlap_sum[layer_index] += selected_prior_support.double().sum()
            support_overlap_count[layer_index] += selected_prior_support.numel()

            semantic_relation_float = semantic_relation_update.float()
            prior_relation_float = prior_relation_update.float()
            semantic_relation_norm = torch.linalg.vector_norm(
                semantic_relation_float,
                dim=-1,
            )
            prior_relation_norm = torch.linalg.vector_norm(
                prior_relation_float,
                dim=-1,
            )
            active = (semantic_relation_norm > norm_epsilon) & (
                prior_relation_norm > norm_epsilon
            )
            if torch.any(active):
                relation_cosine = F.cosine_similarity(
                    semantic_relation_float,
                    prior_relation_float,
                    dim=-1,
                )
                relation_dissimilarity = (1.0 - relation_cosine) / 2.0
                relation_dissimilarity_sum[layer_index] += relation_dissimilarity[
                    active
                ].double().sum()
                relation_dissimilarity_count[layer_index] += active.sum()

            del semantic_top_indices, selected_prior_support

        if torch.any(support_overlap_count == 0):
            raise RuntimeError("No semantic support-overlap observations were accumulated")
        if torch.any(relation_dissimilarity_count == 0):
            raise RuntimeError("No active relation-update pairs were accumulated")

        semantic_update = semantic_state - stock_tokens
        prior_update = prior_state - stock_tokens

        semantic_update_float = semantic_update.float()
        prior_update_float = prior_update.float()
        semantic_update_norm = torch.linalg.vector_norm(semantic_update_float, dim=-1)
        prior_update_norm = torch.linalg.vector_norm(prior_update_float, dim=-1)
        update_norm_sum = semantic_update_norm + prior_update_norm
        active_tower_updates = update_norm_sum > norm_epsilon
        if not torch.any(active_tower_updates):
            raise RuntimeError("Both DRT tower updates are zero")
        update_magnitude_balance = (
            2.0 * torch.minimum(semantic_update_norm, prior_update_norm)
            / update_norm_sum.clamp_min(norm_epsilon)
        )
        tower_update_cosine = F.cosine_similarity(
            semantic_update_float,
            prior_update_float,
            dim=-1,
        )
        tower_update_dissimilarity = (1.0 - tower_update_cosine) / 2.0

        def restore_shape(state):
            return state.reshape(batch_size, seq_len, stock_num, d_model).permute(
                0, 2, 1, 3
            )

        # 干预后的表示保留同一聚合权重，避免诊断偏离模型前向。
        full_representation = restore_shape(self.fuse_relation_states(semantic_state, prior_state))
        without_semantic_update = restore_shape(self.fuse_relation_states(stock_tokens, prior_state))
        without_prior_update = restore_shape(self.fuse_relation_states(semantic_state, stock_tokens))
        prediction_full = self.prediction_head(full_representation)
        prediction_without_semantic = self.prediction_head(without_semantic_update)
        prediction_without_prior = self.prediction_head(without_prior_update)
        semantic_prediction_impact = (
            prediction_full - prediction_without_semantic
        ).abs().flatten(1).mean(dim=1)
        prior_prediction_impact = (
            prediction_full - prediction_without_prior
        ).abs().flatten(1).mean(dim=1)
        prediction_impact_sum = semantic_prediction_impact + prior_prediction_impact
        if torch.any(prediction_impact_sum <= norm_epsilon):
            raise RuntimeError("At least one sample has zero total DRT prediction impact")
        prediction_impact_balance = (
            2.0 * torch.minimum(semantic_prediction_impact, prior_prediction_impact)
            / prediction_impact_sum
        )

        support_overlap_by_layer = support_overlap_sum / support_overlap_count
        relation_dissimilarity_by_layer = (
            relation_dissimilarity_sum / relation_dissimilarity_count
        )
        random_support_overlap = prior_topk / float(stock_num - 1)
        return {
            "prior_topk": prior_topk,
            "random_support_overlap": random_support_overlap,
            "support_overlap": support_overlap_by_layer.mean(),
            "support_overlap_by_layer": support_overlap_by_layer,
            "support_overlap_enrichment": support_overlap_by_layer.mean()
            / random_support_overlap,
            "relation_update_dissimilarity": relation_dissimilarity_by_layer.mean(),
            "relation_update_dissimilarity_by_layer": relation_dissimilarity_by_layer,
            "tower_update_dissimilarity": tower_update_dissimilarity[
                active_tower_updates
            ].mean(),
            "semantic_update_norm": semantic_update_norm.mean(),
            "prior_update_norm": prior_update_norm.mean(),
            "update_magnitude_balance": update_magnitude_balance[
                active_tower_updates
            ].mean(),
            "semantic_prediction_impact": semantic_prediction_impact,
            "prior_prediction_impact": prior_prediction_impact,
            "prediction_impact_balance": prediction_impact_balance,
            "prediction_full": prediction_full,
        }

    def forward_backbone(self, embedded_tokens):
        representation = self.spe(embedded_tokens)
        return self._cross_stock_relation(representation)

    def moe_load_balance_loss(self):
        return self.spe.moe_load_balance_loss()

    def drt_load_balance_loss(self):
        # 对批次内的两专家平均权重施加 CV² 约束，允许单个输入偏向任一塔。
        importance = self.drt_gate_probabilities.mean(dim=(0, 1))
        balance_loss = 2.0 * (importance - 0.5).square().sum()
        return balance_loss

    def forward(self, inputs):
        embedded_tokens = self.input_projection(inputs)
        representation = self.forward_backbone(embedded_tokens)
        return self.prediction_head(representation)


def load_relation_weight_matrix(path):
    with np.load(Path(path), allow_pickle=False) as archive:
        matrix = np.asarray(archive["weight_matrix"])
    return torch.from_numpy(matrix.astype(np.uint8, copy=True))


def make_empty_relation_weight_matrix(stock_num):
    return torch.zeros(int(stock_num), int(stock_num), dtype=torch.uint8)


class Model(PSTRNet_SPE_DRT):
    """Complete SPE + DRT experiment model."""

    variant_name = "b_spe_drt"
    variant_label = "SPE + DRT (Dual Relation Tower)"

    def __init__(self, args):
        relation = load_relation_weight_matrix(args.pstr_relation_path)
        super().__init__(
            enc_in=int(args.features),
            d_model=int(args.d_model),
            seq_len=int(args.seq_len),
            dtw_weight_matrix=relation,
            depth=int(args.pstr_depth),
            kernel_size=int(args.pstr_kernel_size),
            moe_top_k=int(args.moe_top_k),
            num_heads=int(args.pstr_num_heads),
            tower_layers=int(args.pstr_tower_layers),
            dropout=float(args.pstr_dropout),
            semantic_support=str(args.semantic_support),
        )
        # 独立实验变体新增一个门参数，两个训练阶段共同学习。
        if args.model == "model_b_spe_drt_learnable_fusion":
            self.learnable_fusion = True
            self.drt_fusion_logit = nn.Parameter(torch.zeros(()))
            self.variant_name = "b_spe_drt_learnable_fusion"
            self.variant_label = "SPE + DRT (Learnable Scalar Fusion)"
        # 两专家门从等权初始化，随后随当前 ATPE 嵌入生成聚合权重。
        if args.model == "model_b_spe_drt_feature_aggregation":
            self.feature_aggregation = True
            self.drt_router = nn.Linear(int(args.d_model), 2)
            nn.init.zeros_(self.drt_router.weight)
            nn.init.zeros_(self.drt_router.bias)
            self.variant_name = "b_spe_drt_feature_aggregation"
            self.variant_label = "SPE + DRT (Feature-Adaptive Two-Expert Aggregation)"
        self.prediction_head = PredictionHead(
            d_model=int(args.d_model),
            seq_len=int(args.seq_len),
            output_activation=args.output_activation,
        )
        if args.pretrained_backbone_path:
            self.load_mmr_backbone(args.pretrained_backbone_path)

    def load_mmr_backbone(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = checkpoint["backbone_state"]
        self.input_projection.load_state_dict(state["input_projection"])
        self.spe.load_state_dict(state["spe"])
        self.semantic_stock_transformer.load_state_dict(state["semantic_stock_transformer"])
        self.prior_stock_transformer.load_state_dict(state["prior_stock_transformer"])
        # 微调接续该次 MMR 检查点中的门参数。
        if self.learnable_fusion:
            with torch.no_grad():
                self.drt_fusion_logit.copy_(state["drt_fusion_logit"])
        # 微调继承预训练得到的特征门，继续端到端更新。
        if self.feature_aggregation:
            self.drt_router.load_state_dict(state["drt_router"])


    def forward(self, inputs):
        embedded_tokens = self.input_projection(inputs)
        representation = self.forward_backbone(embedded_tokens)
        prediction = self.prediction_head(representation)
        return prediction.squeeze(-1)


class PSTRNet_MMR(nn.Module):
    """Masked market reconstruction model used only for PSTR-Net pretraining."""

    def __init__(self, args):
        super().__init__()
        self.backbone = Model(args)
        del self.backbone.prediction_head
        self.mask_token = nn.Parameter(torch.empty(1, 1, 1, int(args.d_model)))
        nn.init.normal_(self.mask_token, std=0.02)
        self.reconstruction_decoder = nn.Sequential(
            nn.Linear(int(args.d_model), int(args.d_model)),
            nn.GELU(),
            nn.Linear(int(args.d_model), int(args.features)),
        )

    def sample_token_mask(self, inputs, mask_ratio):
        return torch.rand(inputs.shape[:3], device=inputs.device) < float(mask_ratio)

    def forward(self, inputs, token_mask):
        embedded_tokens = self.backbone.input_projection(inputs)
        masked_tokens = torch.where(
            token_mask.unsqueeze(-1),
            self.mask_token.to(dtype=embedded_tokens.dtype),
            embedded_tokens,
        )
        representation = self.backbone.forward_backbone(masked_tokens)
        return self.reconstruction_decoder(representation)

    def reconstruction_loss(self, inputs, token_mask):
        reconstruction = self.forward(inputs, token_mask)
        masked_error = (reconstruction - inputs).square() * token_mask.unsqueeze(-1)
        return masked_error.sum() / (token_mask.sum() * inputs.shape[-1])

    def backbone_state(self):
        backbone_state = {
            "input_projection": self.backbone.input_projection.state_dict(),
            "spe": self.backbone.spe.state_dict(),
            "semantic_stock_transformer": self.backbone.semantic_stock_transformer.state_dict(),
            "prior_stock_transformer": self.backbone.prior_stock_transformer.state_dict(),
        }
        # 仅门控变体扩展检查点，原模型的序列化结构保持一致。
        if self.backbone.learnable_fusion:
            backbone_state["drt_fusion_logit"] = self.backbone.drt_fusion_logit.detach().clone()
        if self.backbone.feature_aggregation:
            backbone_state["drt_router"] = self.backbone.drt_router.state_dict()
        return backbone_state