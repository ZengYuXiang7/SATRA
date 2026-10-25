#!/usr/bin/env bash
set -e

PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"

SATRA_PROJECT_ROOT="${PACKAGE_PROJECT_ROOT}"
while [ ! -f "${SATRA_PROJECT_ROOT}/pyproject.toml" ] && [ "${SATRA_PROJECT_ROOT}" != "/" ]; do SATRA_PROJECT_ROOT="$(dirname "${SATRA_PROJECT_ROOT}")"; done
[ -f "${SATRA_PROJECT_ROOT}/pyproject.toml" ] || { echo "Unable to locate the project root." >&2; exit 1; }
cd "${SATRA_PROJECT_ROOT}"

# Panel (d): complete PSTR-Net, independently pretrain and fine-tune every run.
python -m satra.mmr_two_stage \
  --is_training 1 --model_id pstr_d_pstr_net_with_mmr_nasdaq_5run \
  --model model_b_spe_drt --data NASDAQ \
  --root_path data/processed \
  --results results/pstr_d_pstr_net_with_mmr_5run_3markets \
  --checkpoints results/pstr_d_pstr_net_with_mmr_5run_3markets/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 \
  --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 \
  --learning_rate 0.001 --train_epochs 100 --patience 20 --early_stop_metric RankIC --early_stop_mode max \
  --lr_scheduler cosine_hard_restarts --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 64 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 --pstr_dropout 0.1 --moe_top_k 3 \
  --pstr_relation_path data/processed/NASDAQ_ohlcv/priors/dtw_binary_imputed_inputstandard_NASDAQ_train0-871_close_ds64_band6_cand48_top32.npz \
  --valid_index 871 --test_index 996 --data_start_index 0 --num_runs 5

python -m satra.mmr_two_stage \
  --is_training 1 --model_id pstr_d_pstr_net_with_mmr_nyse_5run \
  --model model_b_spe_drt --data NYSE \
  --root_path data/processed \
  --results results/pstr_d_pstr_net_with_mmr_5run_3markets \
  --checkpoints results/pstr_d_pstr_net_with_mmr_5run_3markets/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 \
  --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 \
  --learning_rate 0.001 --train_epochs 100 --patience 20 --early_stop_metric RankIC --early_stop_mode max \
  --lr_scheduler cosine_hard_restarts --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 64 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 --pstr_dropout 0.1 --moe_top_k 3 \
  --pstr_relation_path data/processed/NYSE_ohlcv/priors/dtw_binary_imputed_inputstandard_NYSE_train0-871_close_ds64_band6_cand48_top32.npz \
  --valid_index 871 --test_index 996 --data_start_index 0 --num_runs 5

python -m satra.mmr_two_stage \
  --is_training 1 --model_id pstr_d_pstr_net_with_mmr_sp500_5run \
  --model model_b_spe_drt --data SP500 \
  --root_path data/processed \
  --results results/pstr_d_pstr_net_with_mmr_5run_3markets \
  --checkpoints results/pstr_d_pstr_net_with_mmr_5run_3markets/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 \
  --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 \
  --learning_rate 0.001 --train_epochs 100 --patience 20 --early_stop_metric RankIC --early_stop_mode max \
  --lr_scheduler cosine_hard_restarts --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 64 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 --pstr_dropout 0.1 --moe_top_k 3 \
  --pstr_relation_path data/processed/SP500_ohlcv/priors/dtw_binary_imputed_inputstandard_SP500_train915-2042_close_ds64_band6_cand48_top32.npz \
  --valid_index 1127 --test_index 1245 --data_start_index 915 --num_runs 5
