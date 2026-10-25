#!/usr/bin/env bash
set -euo pipefail

PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"

SATRA_PROJECT_ROOT="${PACKAGE_PROJECT_ROOT}"
while [ ! -f "${SATRA_PROJECT_ROOT}/pyproject.toml" ] && [ "${SATRA_PROJECT_ROOT}" != "/" ]; do SATRA_PROJECT_ROOT="$(dirname "${SATRA_PROJECT_ROOT}")"; done
[ -f "${SATRA_PROJECT_ROOT}/pyproject.toml" ] || { echo "Unable to locate the project root." >&2; exit 1; }
cd "${SATRA_PROJECT_ROOT}"

export PYTHONSAFEPATH=1
export PATH="${HOME}/.local/bin:${PATH}"

NUM_RUNS=5
RESULTS=results/pstr_a_shared_temporal_encoder_5run_3markets
CHECKPOINTS="${RESULTS}/checkpoints"
DATA_ROOT=data/processed

COMMON_ARGS=(
  --num_runs "${NUM_RUNS}"
  --debug 0
  --is_training 1
  --model model_a_shared_temporal_encoder
  --root_path "${DATA_ROOT}"
  --results "${RESULTS}"
  --checkpoints "${CHECKPOINTS}"
  --input_features all
  --use_grouped_inputs 0
  --seq_len 16
  --batch_size 1
  --num_workers 0
  --use_gpu 1
  --pin_memory 1
  --train_drop_last 1
  --valid_drop_last 1
  --prediction_target return
  --target_mode raw_return
  --output_activation tanh
  --learning_rate 0.001
  --train_epochs 100
  --patience 20
  --early_stop_metric RankIC
  --early_stop_mode max
  --lr_scheduler cosine_hard_restarts
  --precision bf16_amp_tf32
  --torch_compile 0
  --max_grad_norm 1.0
  --d_model 64
  --pstr_depth 10
  --pstr_kernel_size 3
  --pstr_num_heads 4
  --pstr_dropout 0.1
  --moe_top_k 3
  --stock_prior none
)

PYTHONPATH=src python run.py "${COMMON_ARGS[@]}" --model_id pstr_a_shared_temporal_encoder_nasdaq_5run --data NASDAQ --market_num 20 --valid_index 871 --test_index 996 --data_start_index 0
PYTHONPATH=src python run.py "${COMMON_ARGS[@]}" --model_id pstr_a_shared_temporal_encoder_nyse_5run --data NYSE --market_num 25 --valid_index 871 --test_index 996 --data_start_index 0
PYTHONPATH=src python run.py "${COMMON_ARGS[@]}" --model_id pstr_a_shared_temporal_encoder_sp500_5run --data SP500 --market_num 8 --valid_index 1127 --test_index 1288 --data_start_index 915
