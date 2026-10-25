"""Static learned-adjacency relation ablation.

This variant keeps PATRA-Net's input projection, ACE temporal encoder, and
prediction head.  It replaces both DRT towers and every external relation
prior with a dense directed adjacency matrix learned end to end from the
supervised objective.  The same matrix is shared across stock-time panels,
which makes it a learned relation graph rather than a current-state attention
mechanism.
"""

import torch.nn as nn

from satra.ablation.learned_graph_layers import StaticGraphMessageLayer, StaticLearnedAdjacency
from satra.b_spe_drt import PredictionHead, SPELayer


class Model(nn.Module):
    """SPE followed by a learned static relation graph without DTW input."""

    variant_name = "c_learned_adjacency"
    variant_label = "SPE + learned static adjacency"

    def __init__(self, args):
        super().__init__()
        tower_layers = int(args.pstr_tower_layers)
        if tower_layers <= 0:
            raise ValueError(f"pstr_tower_layers must be positive, got {tower_layers}")
        self.stock_num = int(args.stock_num)
        self.input_projection = nn.Linear(int(args.features), int(args.d_model))
        self.spe = SPELayer(
            d_model=int(args.d_model),
            seq_len=int(args.seq_len),
            depth=int(args.pstr_depth),
            kernel_size=int(args.pstr_kernel_size),
            top_k=int(args.moe_top_k),
            dropout=float(args.pstr_dropout),
        )
        # One graph is deliberately shared across propagation layers.  This
        # avoids giving the learned-graph comparator one N-by-N parameter
        # matrix per layer merely because the DRT depth is increased.
        self.learned_adjacency = StaticLearnedAdjacency(self.stock_num)
        self.graph_layers = nn.ModuleList(
            [
                StaticGraphMessageLayer(
                    d_model=int(args.d_model),
                    dropout=float(args.pstr_dropout),
                )
                for _ in range(tower_layers)
            ]
        )
        self.prediction_head = PredictionHead(
            d_model=int(args.d_model),
            seq_len=int(args.seq_len),
            output_activation=args.output_activation,
        )

    def _cross_stock_relation(self, representation):
        if representation.ndim != 4:
            raise ValueError(
                "Expected SPE representation [batch, stocks, time, features], got "
                f"{tuple(representation.shape)}"
            )
        batch_size, stock_num, seq_len, d_model = representation.shape
        if stock_num != self.stock_num:
            raise ValueError(
                "Input stock count does not match the learned adjacency: "
                f"observed={stock_num}, configured={self.stock_num}"
            )
        stock_tokens = representation.permute(0, 2, 1, 3).reshape(
            batch_size * seq_len,
            stock_num,
            d_model,
        )
        adjacency = self.learned_adjacency(stock_num=stock_num, dtype=stock_tokens.dtype)
        for graph_layer in self.graph_layers:
            stock_tokens = graph_layer(stock_tokens, adjacency)
        return stock_tokens.reshape(batch_size, seq_len, stock_num, d_model).permute(0, 2, 1, 3)

    def forward_backbone(self, embedded_tokens):
        return self._cross_stock_relation(self.spe(embedded_tokens))

    def moe_load_balance_loss(self):
        return self.spe.moe_load_balance_loss()

    def forward(self, inputs):
        representation = self.forward_backbone(self.input_projection(inputs))
        return self.prediction_head(representation).squeeze(-1)
