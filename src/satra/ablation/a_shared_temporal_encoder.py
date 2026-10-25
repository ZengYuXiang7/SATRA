"""Panel (a): CNN-only temporal encoder ablation for PSTR-Net.

``[B, N, T, F]`` -> Linear(F, d_model) -> temporal CNN over ``T``
-> d_model-to-target projection -> history-to-one projection.

This experiment intentionally contains no MoE, relation matrix, stock
Transformer, semantic tower, or prior-guided tower.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamePadConv(nn.Module):
    """Length-preserving one-dimensional convolution on the history axis."""

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

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.remove:
            return outputs[:, :, :-self.remove]
        return outputs


class ConvBlock(nn.Module):
    """Residual temporal CNN block used by the full SPE temporal encoder."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(int(in_channels), int(out_channels), 1)
        self.final = bool(final)

    def forward(self, inputs):
        residual = self.projector(inputs)
        hidden = F.gelu(self.conv1(inputs))
        hidden = self.conv2(hidden)
        outputs = residual + hidden
        if self.final:
            return outputs
        return F.gelu(outputs)


class DilatedConvEncoder(nn.Module):
    """Dilated CNN stack that learns only along the history axis ``T``."""

    def __init__(self, in_channels, channels, kernel_size=3):
        super().__init__()
        blocks = []
        current_channels = int(in_channels)
        for index, out_channels in enumerate(channels):
            blocks.append(
                ConvBlock(
                    current_channels,
                    int(out_channels),
                    kernel_size,
                    dilation=2**index,
                    final=index == len(channels) - 1,
                )
            )
            current_channels = int(out_channels)
        self.net = nn.Sequential(*blocks)

    def forward(self, inputs):
        return self.net(inputs)


class CNNOnlySPELayer(nn.Module):
    """Shared temporal CNN, applied independently to every stock window."""

    def __init__(self, d_model, depth=10, kernel_size=3, dropout=0.1):
        super().__init__()
        self.d_model = int(d_model)
        self.encoder = DilatedConvEncoder(
            self.d_model,
            [self.d_model] * (int(depth) + 1),
            kernel_size=kernel_size,
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, inputs):
        batch_size, stock_num, seq_len, d_model = inputs.shape
        temporal_windows = inputs.reshape(batch_size * stock_num, seq_len, d_model)
        temporal_windows = self.encoder(temporal_windows.transpose(1, 2)).transpose(1, 2)
        temporal_windows = self.dropout(temporal_windows)
        return temporal_windows.reshape(batch_size, stock_num, seq_len, d_model)


class PredictionHead(nn.Module):
    """Map each stock history to one scalar prediction."""

    def __init__(self, d_model, seq_len, output_activation="identity"):
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


class PSTRNet_SHARED_TEMPORAL_ENCODER(nn.Module):
    """Shared temporal encoder only: CNN time-axis learning followed by prediction."""

    variant_name = "a_shared_temporal_encoder"
    variant_label = "Shared temporal encoder"

    def __init__(self, args):
        super().__init__()
        self.input_projection = nn.Linear(int(args.features), int(args.d_model))
        self.temporal_encoder = CNNOnlySPELayer(
            d_model=int(args.d_model),
            depth=int(args.pstr_depth),
            kernel_size=int(args.pstr_kernel_size),
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


class Model(PSTRNet_SHARED_TEMPORAL_ENCODER):
    pass
