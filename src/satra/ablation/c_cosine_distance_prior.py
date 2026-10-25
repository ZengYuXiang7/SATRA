"""Cosine-prior PSTR-Net relation-prior ablation.

The temporal SPE and dual relation tower are intentionally identical to the
Euclidean-prior ablation. The only experimental variable is the directed
Top-k relation matrix supplied through ``--pstr_relation_path``.
"""

from satra.ablation.c_euclidean_distance_prior import (
    PSTRNet_EUCLIDEAN_DISTANCE_PRIOR,
    PredictionHead,
    load_relation_weight_matrix,
)


class PSTRNet_COSINE_DISTANCE_PRIOR(PSTRNet_EUCLIDEAN_DISTANCE_PRIOR):
    """Complete PSTR-Net using a cosine-similarity historical relation prior."""


class Model(PSTRNet_COSINE_DISTANCE_PRIOR):
    """Directly registered Cosine-distance prior PSTR-Net experiment model."""

    variant_name = "c_cosine_distance_prior"
    variant_label = "Cosine-distance prior"

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
            dropout=float(args.pstr_dropout),
        )
        self.prediction_head = PredictionHead(
            d_model=int(args.d_model),
            seq_len=int(args.seq_len),
            output_activation=args.output_activation,
        )

    def forward(self, inputs):
        representation = self.spe(self.input_projection(inputs))
        representation = self._cross_stock_relation(representation)
        prediction = self.prediction_head(representation)
        return prediction.squeeze(-1)
