#!/usr/bin/env bash
set -euo pipefail
PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"


DATASETS="${DATASETS:-NASDAQ NYSE SP500}"
NUM_RUNS="${NUM_RUNS:-5}"
DEBUG="${DEBUG:-0}"
DEBUG_TRAIN_STEPS="${DEBUG_TRAIN_STEPS:-1}"
DEBUG_EVAL_STEPS="${DEBUG_EVAL_STEPS:-1}"
USE_GPU="${USE_GPU:-1}"
PIN_MEMORY="${PIN_MEMORY:-1}"

for dataset in ${DATASETS}; do
  case "${dataset}" in
    NASDAQ)
      market_num=20
      valid_index=871
      test_index=996
      data_start_index=0
      ;;
    NYSE)
      market_num=25
      valid_index=871
      test_index=996
      data_start_index=0
      ;;
    SP500)
      market_num=8
      valid_index=1127
      test_index=1288
      data_start_index=915
      ;;
    *)
      echo "unsupported dataset=${dataset}" >&2
      exit 1
      ;;
  esac

  lower_dataset="$(printf '%s' "${dataset}" | tr '[:upper:]' '[:lower:]')"
  echo "[pstr-net-supervised-backbone] start dataset=${dataset} runs=${NUM_RUNS} $(date +%Y%m%d_%H%M%S)"
  PYTHONPATH=src python run.py \
    --num_runs "${NUM_RUNS}" \
    --debug "${DEBUG}" \
    --debug_train_steps "${DEBUG_TRAIN_STEPS}" \
    --debug_eval_steps "${DEBUG_EVAL_STEPS}" \
    --is_training 1 \
    --model_id "pstr_net_full_independent_${lower_dataset}_5run" \
    --model model \
    --data "${dataset}" \
    --root_path data/processed \
    --results results/pstr_net_full_independent_5run \
    --input_features all \
    --seq_len 16 \
    --batch_size 8 \
    --num_workers 0 \
    --use_gpu "${USE_GPU}" \
    --pin_memory "${PIN_MEMORY}" \
    --train_drop_last 1 \
    --valid_drop_last 1 \
    --market_num "${market_num}" \
    --valid_index "${valid_index}" \
    --test_index "${test_index}" \
    --data_start_index "${data_start_index}" \
    --prediction_target return \
    --target_mode raw_return \
    --learning_rate 0.001 \
    --train_epochs 100 \
    --patience 20 \
    --early_stop_metric RankIC \
    --early_stop_mode max \
    --lr_scheduler cosine_hard_restarts \
    --precision bf16_amp_tf32 \
    --torch_compile 0 \
    --max_grad_norm 1.0 \
    --moe_top_k 3 \
    --stock_prior dtw \
    --stock_prior_cache_dir "data/processed/${dataset}_ohlcv/priors" \
    --stock_prior_feature close \
    --stock_prior_topk 16 \
    --stock_prior_candidate_k 48 \
    --stock_prior_downsample 64 \
    --stock_prior_band 6 \
    --stock_prior_scale 1.0
  echo "[pstr-net-supervised-backbone] done dataset=${dataset} $(date +%Y%m%d_%H%M%S)"
done
