#!/usr/bin/env bash
set -euo pipefail

PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"

SATRA_PROJECT_ROOT="${PACKAGE_PROJECT_ROOT}"
while [ ! -f "${SATRA_PROJECT_ROOT}/pyproject.toml" ] && [ "${SATRA_PROJECT_ROOT}" != "/" ]; do SATRA_PROJECT_ROOT="$(dirname "${SATRA_PROJECT_ROOT}")"; done
[ -f "${SATRA_PROJECT_ROOT}/pyproject.toml" ] || { echo "Unable to locate the project root." >&2; exit 1; }
cd "${SATRA_PROJECT_ROOT}"

export PYTHONSAFEPATH=1
export PYTHONPATH=src

DATA_ROOT=data/processed

# The only changed factor relative to the formal MMR Full DRT result is
# --semantic_support dtw_prior.  The semantic and prior towers remain separate.
python -m satra.mmr_two_stage \
  --is_training 1 --model_id pstr_drt_prior_constrained_nasdaq_5run \
  --model model_b_spe_drt --data NASDAQ \
  --root_path "${DATA_ROOT}" \
  --results results/pstr_drt_prior_constrained_5run_3markets \
  --checkpoints results/pstr_drt_prior_constrained_5run_3markets/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 \
  --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine_hard_restarts \
  --learning_rate 0.001 --train_epochs 100 --patience 20 --early_stop_metric RankIC --early_stop_mode max \
  --lr_scheduler cosine_hard_restarts --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 64 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 --pstr_tower_layers 1 --pstr_dropout 0.1 \
  --moe_top_k 3 --moe_load_balance_weight 0.0 --semantic_support dtw_prior \
  --pstr_relation_path "${DATA_ROOT}/NASDAQ_ohlcv/priors/dtw_binary_imputed_inputstandard_NASDAQ_train0-871_close_ds64_band6_cand48_top32.npz" \
  --valid_index 871 --test_index 996 --data_start_index 0 --num_runs 5 --random_seed 151567275

python -m satra.mmr_two_stage \
  --is_training 1 --model_id pstr_drt_prior_constrained_nyse_5run \
  --model model_b_spe_drt --data NYSE \
  --root_path "${DATA_ROOT}" \
  --results results/pstr_drt_prior_constrained_5run_3markets \
  --checkpoints results/pstr_drt_prior_constrained_5run_3markets/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 \
  --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine_hard_restarts \
  --learning_rate 0.001 --train_epochs 100 --patience 20 --early_stop_metric RankIC --early_stop_mode max \
  --lr_scheduler cosine_hard_restarts --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 64 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 --pstr_tower_layers 1 --pstr_dropout 0.1 \
  --moe_top_k 3 --moe_load_balance_weight 0.0 --semantic_support dtw_prior \
  --pstr_relation_path "${DATA_ROOT}/NYSE_ohlcv/priors/dtw_binary_imputed_inputstandard_NYSE_train0-871_close_ds64_band6_cand48_top32.npz" \
  --valid_index 871 --test_index 996 --data_start_index 0 --num_runs 5 --random_seed 701943994

python -m satra.mmr_two_stage \
  --is_training 1 --model_id pstr_drt_prior_constrained_sp500_5run \
  --model model_b_spe_drt --data SP500 \
  --root_path "${DATA_ROOT}" \
  --results results/pstr_drt_prior_constrained_5run_3markets \
  --checkpoints results/pstr_drt_prior_constrained_5run_3markets/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 \
  --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine_hard_restarts \
  --learning_rate 0.001 --train_epochs 100 --patience 20 --early_stop_metric RankIC --early_stop_mode max \
  --lr_scheduler cosine_hard_restarts --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 64 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 --pstr_tower_layers 1 --pstr_dropout 0.1 \
  --moe_top_k 3 --moe_load_balance_weight 0.0 --semantic_support dtw_prior \
  --pstr_relation_path "${DATA_ROOT}/SP500_ohlcv/priors/dtw_binary_imputed_inputstandard_SP500_train915-2042_close_ds64_band6_cand48_top32.npz" \
  --valid_index 1127 --test_index 1245 --data_start_index 915 --num_runs 5 --random_seed 761671755
