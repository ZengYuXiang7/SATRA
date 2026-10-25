# SATRA Ablation Variants

The full SATRA model (`SPE + DRT`) is implemented in
`src/satra/b_spe_drt.py`. Except for
MMR, each Table 2 row has a dedicated Python module. Every variant accepts the
complete input tensor with shape `[B, N, T, F]` and returns predictions with
shape `[B, N, 1]`. Training model identifiers use the same panel prefixes as
the source files.

| Table 2 panel | Variant | Source file | Training model |
|---|---|---|---|
| (a) | Shared temporal encoder | `a_shared_temporal_encoder.py` | `model_a_shared_temporal_encoder` |
| (a) | Homogeneous experts | `a_homogeneous_experts.py` | `model_a_homogeneous_experts` |
| (a) | SPE | `a_spe.py` | `model_a_spe` |
| (b) | SPE only | `b_spe_only.py` | `model_b_spe_only` |
| (b) | SPE + semantic tower | `b_spe_semantic_tower.py` | `model_b_spe_semantic_tower` |
| (b) | SPE + prior guided tower | `b_spe_prior_guided_tower.py` | `model_b_spe_prior_guided_tower` |
| (b) | SPE + DRT | `../b_spe_drt.py` | `model_b_spe_drt` |
| (c) | No prior constraint | `c_no_prior_constraint.py` | `model_c_no_prior_constraint` |
| (c) | Industry relation prior | `c_industry_relation_prior.py` | `model_c_industry_relation_prior` |
| (c) | Euclidean distance prior | `c_euclidean_distance_prior.py` | `model_c_euclidean_distance_prior` |
| (c) | Cosine distance prior | `c_cosine_distance_prior.py` | `model_c_cosine_distance_prior` |
| (c) | DTW prior | `c_dtw_based_prior.py` | `model_c_dtw_based_prior` |

Panel (d) uses the two-stage MMR workflow in `../mmr_two_stage.py`; its
reconstruction model retains the compatibility class name `PSTRNet_MMR` in
`../b_spe_drt.py`.

The data pipeline applies global standardization using training-set statistics.
Model modules do not normalize inputs again. The prediction head maps
`d_model -> 1` and then `T -> 1`; `--output_activation {tanh,identity}`
controls its output activation.
