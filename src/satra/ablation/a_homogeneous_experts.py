"""Panel (a): temporal CNN with homogeneous state-conditioned Top-3 experts.

The model applies every operation independently to each stock window.  It has
no relation matrix and no cross-stock attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        receptive_field = (int(kernel_size) - 1) * int(dilation) + 1
        self.remove = 1 if receptive_field % 2 == 0 else 0
        self.conv = nn.Conv1d(
            int(in_channels),
            int(out_channels),
            int(kernel_size),
            padding=receptive_field // 2,
            dilation=int(dilation),
        )

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.remove:
            return outputs[:, :, :-self.remove]
        return outputs


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation)
        self.projector = nn.Conv1d(int(in_channels), int(out_channels), 1)
        self.final = bool(final)

    def forward(self, inputs):
        hidden = self.conv2(F.gelu(self.conv1(inputs)))
        outputs = self.projector(inputs) + hidden
        if self.final:
            return outputs
        return F.gelu(outputs)


class DilatedConvEncoder(nn.Module):
    def __init__(self, d_model, depth, kernel_size):
        super().__init__()
        channels = [int(d_model)] * (int(depth) + 1)
        blocks = []
        current = int(d_model)
        for index, out_channels in enumerate(channels):
            blocks.append(
                ConvBlock(
                    current,
                    out_channels,
                    kernel_size,
                    dilation=2**index,
                    final=index == len(channels) - 1,
                )
            )
            current = out_channels
        self.net = nn.Sequential(*blocks)

    def forward(self, inputs):
        return self.net(inputs)


class BottleNeck(nn.Module):
    def __init__(self, seq_len, model_dim, bottleneck_dim, dropout):
        super().__init__()
        width = int(seq_len) * int(model_dim)
        self.width = width
        self.encoder = nn.Sequential(
            nn.Linear(width, int(bottleneck_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.decoder = nn.Sequential(
            nn.Linear(int(bottleneck_dim), width),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

    def forward(self, representation):
        flat = representation.reshape(representation.shape[0], self.width)
        return self.decoder(self.encoder(flat)).reshape_as(representation)


class SparseDispatcher:
    def __init__(self, gates):
        nonzero = torch.nonzero(gates, as_tuple=False)
        routed = nonzero[torch.argsort(nonzero[:, 1])]
        self.batch_index = routed[:, 0]
        self.expert_index = routed[:, 1]
        self.part_sizes = (gates > 0).sum(dim=0).tolist()
        self.nonzero_gates = gates[self.batch_index, self.expert_index]
        self.batch_size = gates.shape[0]

    def dispatch(self, inputs):
        return torch.split(inputs[self.batch_index], self.part_sizes, dim=0)

    def combine(self, expert_outputs):
        stitched = torch.cat(expert_outputs, dim=0)
        weighted = stitched * self.nonzero_gates.unsqueeze(-1).to(dtype=stitched.dtype)
        combined = stitched.new_zeros((self.batch_size, stitched.shape[1]))
        return combined.index_add(0, self.batch_index, weighted)


class HomogeneousMoE(nn.Module):
    """Six same-width experts with noisy state-conditioned Top-3 routing."""

    def __init__(self, seq_len, model_dim, top_k, dropout):
        super().__init__()
        self.top_k = int(top_k)
        self.num_experts = 6
        expert_width = max(1, int(round(int(model_dim) * 43.0 / 96.0)))
        self.experts = nn.ModuleList(
            [BottleNeck(seq_len, model_dim, expert_width, dropout) for _ in range(self.num_experts)]
        )
        router_width = int(seq_len) * int(model_dim)
        self.w_gate = nn.Parameter(torch.zeros(router_width, self.num_experts))
        self.w_noise = nn.Parameter(torch.zeros(router_width, self.num_experts))
        self.softplus = nn.Softplus()

    def forward(self, representation):
        clean_logits = representation @ self.w_gate
        if self.training:
            noise_stddev = self.softplus(representation @ self.w_noise) + 1e-2
            logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
        else:
            logits = clean_logits
        top_logits, top_indices = logits.topk(self.top_k, dim=1)
        gates = torch.softmax(top_logits.float(), dim=1).to(dtype=logits.dtype)
        gates = torch.zeros_like(logits).scatter(1, top_indices, gates)
        dispatcher = SparseDispatcher(gates)
        expert_inputs = dispatcher.dispatch(representation)
        expert_outputs = [expert(expert_inputs[index]) for index, expert in enumerate(self.experts)]
        return dispatcher.combine(expert_outputs)


class HomogeneousTemporalEncoder(nn.Module):
    def __init__(self, d_model, seq_len, depth, kernel_size, top_k, dropout):
        super().__init__()
        self.d_model = int(d_model)
        self.seq_len = int(seq_len)
        self.cnn = DilatedConvEncoder(self.d_model, depth, kernel_size)
        self.dropout = nn.Dropout(float(dropout))
        self.moe_dim = self.d_model * 4
        self.expand = nn.Linear(self.d_model, self.moe_dim)
        self.moe = HomogeneousMoE(self.seq_len, self.moe_dim, top_k, dropout)
        self.reduce = nn.Linear(self.moe_dim, self.d_model)

    def forward(self, inputs):
        batch_size, stock_num, seq_len, d_model = inputs.shape
        temporal_windows = inputs.reshape(batch_size * stock_num, seq_len, d_model)
        temporal_windows = self.cnn(temporal_windows.transpose(1, 2)).transpose(1, 2)
        temporal_windows = self.dropout(temporal_windows)
        temporal_windows = self.expand(temporal_windows)
        routed = self.moe(temporal_windows.reshape(batch_size * stock_num, -1))
        temporal_windows = self.reduce(
            routed.reshape(batch_size * stock_num, seq_len, self.moe_dim)
        )
        return temporal_windows.reshape(batch_size, stock_num, seq_len, d_model)


class PredictionHead(nn.Module):
    def __init__(self, d_model, seq_len, output_activation):
        super().__init__()
        self.output_activation = str(output_activation)
        self.feature_to_target = nn.Linear(int(d_model), 1)
        self.history_to_future = nn.Linear(int(seq_len), 1)

    def forward(self, representation):
        target_history = self.feature_to_target(representation)
        prediction = self.history_to_future(target_history.transpose(-1, -2)).transpose(-1, -2)
        if self.output_activation == "tanh":
            return torch.tanh(prediction)
        return prediction


class PSTRNet_HOMOGENEOUS_EXPERTS(nn.Module):
    """CNN plus homogeneous experts, with no cross-stock relation learning."""

    variant_name = "a_homogeneous_experts"
    variant_label = "Homogeneous experts"

    def __init__(self, args):
        super().__init__()
        self.input_projection = nn.Linear(int(args.features), int(args.d_model))
        self.temporal_encoder = HomogeneousTemporalEncoder(
            d_model=int(args.d_model),
            seq_len=int(args.seq_len),
            depth=int(args.pstr_depth),
            kernel_size=int(args.pstr_kernel_size),
            top_k=int(args.moe_top_k),
            dropout=float(args.pstr_dropout),
        )
        self.prediction_head = PredictionHead(
            d_model=int(args.d_model),
            seq_len=int(args.seq_len),
            output_activation=args.output_activation,
        )

    def forward(self, inputs):
        representation = self.input_projection(inputs)
        representation = self.temporal_encoder(representation)
        return self.prediction_head(representation).squeeze(-1)


class Model(PSTRNet_HOMOGENEOUS_EXPERTS):
    pass
