#!/usr/bin/env bash
set -euo pipefail

PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"

project_root="${PACKAGE_PROJECT_ROOT}"
cd "$project_root"

results_root="results/astra_sp500_table_ablation_1run_20260728"
data_root="data/processed"
dtw_relation="${data_root}/SP500_ohlcv/priors/dtw_binary_imputed_inputstandard_SP500_train915-2042_close_ds64_band6_cand48_top32.npz"
industry_relation="${data_root}/SP500_ohlcv/priors/industry_binary_SP500.npz"
cosine_relation="${data_root}/SP500_ohlcv/priors/cosine_binary_imputed_inputstandard_SP500_train915-2042_close_ds64_top32.npz"
euclidean_relation="${data_root}/SP500_ohlcv/priors/euclidean_binary_imputed_inputstandard_SP500_train915-2042_close_ds64_top32.npz"

for relation_path in \
  "$dtw_relation" \
  "$industry_relation" \
  "$cosine_relation" \
  "$euclidean_relation"; do
  if [[ ! -f "$relation_path" ]]; then
    echo "Missing required relation matrix: $relation_path" >&2
    exit 1
  fi
done

python_entry=(
  env
  PYTHONPATH=src
  python
  -c
  'import sys; sys.path.pop(0); from satra.main import main; main()'
)

common_args=(
  --is_training 1
  --num_runs 1
  --random_seed 1
  --evaluate_test 1
  --portfolio_size 20
  --data SP500
  --market_num 8
  --root_path "$data_root"
  --results "$results_root"
  --checkpoints "${results_root}/checkpoints"
  --seq_len 16
  --steps 1
  --input_features all
  --use_grouped_inputs 0
  --batch_size 1
  --num_workers 0
  --pin_memory 1
  --train_drop_last 1
  --valid_drop_last 1
  --scale_factor 3
  --alpha 0.1
  --output_scale 0.2
  --output_activation tanh
  --target_mode raw_return
  --prediction_target return
  --stock_prior dtw
  --stock_prior_topk 16
  --stock_prior_candidate_k 48
  --stock_prior_downsample 64
  --stock_prior_band 6
  --adaptive_graph_topk 32
  --moe_top_k 3
  --moe_load_balance_weight 0.0
  --d_model 32
  --pstr_depth 10
  --pstr_kernel_size 3
  --pstr_num_heads 4
  --pstr_tower_layers 1
  --pstr_dropout 0.1
  --semantic_support global
  --mae_mask_ratio 0.3
  --pretrained_encoder_path ""
  --pretrained_backbone_path ""
  --valid_index 1127
  --test_index 1288
  --data_start_index 915
  --train_epochs 50
  --learning_rate 0.001
  --patience 20
  --min_delta 0.0
  --min_epochs 0
  --min_updates 0
  --early_stop_metric RankIC
  --early_stop_mode max
  --lr_scheduler cosine
  --warmup_ratio 0.1
  --num_cycles 1
  --min_ratio 0.01
  --min_lr 0.000001
  --use_gpu 1
  --gpu 0
  --precision bf16_amp_tf32
  --torch_compile 0
  --max_grad_norm 1.0
)

printf '[1/10] Homogeneous Capacity\n'
"${python_entry[@]}" "${common_args[@]}" \
  --model_id astra_sp500_1run_homogeneous_capacity \
  --model model_a_homogeneous_experts

printf '[2/10] Adaptive Capacity Encoder\n'
"${python_entry[@]}" "${common_args[@]}" \
  --model_id astra_sp500_1run_ace \
  --model model_a_spe

printf '[3/10] ACE + Learned adjacency\n'
"${python_entry[@]}" "${common_args[@]}" \
  --model_id astra_sp500_1run_learned_adjacency \
  --model model_c_learned_adjacency

printf '[4/10] ACE + Adaptive graph\n'
"${python_entry[@]}" "${common_args[@]}" \
  --model_id astra_sp500_1run_adaptive_graph \
  --model model_c_adaptive_graph

printf '[5/10] ACE + Semantic Tower only\n'
"${python_entry[@]}" "${common_args[@]}" \
  --model_id astra_sp500_1run_semantic_tower_only \
  --model model_b_spe_semantic_tower

printf '[6/10] ACE + Prior Tower only\n'
"${python_entry[@]}" "${common_args[@]}" \
  --model_id astra_sp500_1run_prior_tower_only \
  --model model_b_spe_prior_guided_tower \
  --pstr_relation_path "$dtw_relation"

printf '[7/10] ACE + Semantic Tower + Industry prior\n'
"${python_entry[@]}" "${common_args[@]}" \
  --model_id astra_sp500_1run_industry_prior \
  --model model_c_industry_relation_prior \
  --pstr_relation_path "$industry_relation"

printf '[8/10] ACE + Semantic Tower + Cosine-distance prior\n'
"${python_entry[@]}" "${common_args[@]}" \
  --model_id astra_sp500_1run_cosine_prior \
  --model model_c_cosine_distance_prior \
  --pstr_relation_path "$cosine_relation"

printf '[9/10] ACE + Semantic Tower + Euclidean-distance prior\n'
"${python_entry[@]}" "${common_args[@]}" \
  --model_id astra_sp500_1run_euclidean_prior \
  --model model_c_euclidean_distance_prior \
  --pstr_relation_path "$euclidean_relation"

printf '[10/10] Shared Encoder\n'
"${python_entry[@]}" "${common_args[@]}" \
  --model_id astra_sp500_1run_shared_encoder \
  --model model_a_shared_temporal_encoder
