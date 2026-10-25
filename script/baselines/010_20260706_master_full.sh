#!/usr/bin/env bash
set -euo pipefail
PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"


python -m master.run \
  --data "${DATASET:-NASDAQ}" \
  --root_path "${ROOT_PATH:-data/processed}" \
  --results "${RESULTS:-results}" \
  --input_features "${INPUT_FEATURES:-master_stock_market}" \
  --num_runs "${NUM_RUNS:-3}" \
  --seq_len "${SEQ_LEN:-16}" \
  --steps "${STEPS:-1}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --train_epochs "${TRAIN_EPOCHS:-100}" \
  --learning_rate "${LEARNING_RATE:-1e-5}" \
  --weight_decay "${WEIGHT_DECAY:-0.0}" \
  --d_model "${D_MODEL:-256}" \
  --t_nhead "${T_NHEAD:-4}" \
  --s_nhead "${S_NHEAD:-2}" \
  --gate_beta "${GATE_BETA:-5.0}" \
  --dropout "${DROPOUT:-0.5}" \
  --alpha "${ALPHA:-0.1}" \
  --patience "${PATIENCE:-10}" \
  --max_grad_norm "${MAX_GRAD_NORM:-1.0}" \
  --monitor_metric "${MONITOR_METRIC:-RankIC}" \
  --monitor_mode "${MONITOR_MODE:-max}" \
  --use_gpu "${USE_GPU:-1}" \
  --gpu "${GPU:-0}"
