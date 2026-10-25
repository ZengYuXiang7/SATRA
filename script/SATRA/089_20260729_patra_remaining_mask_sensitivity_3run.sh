#!/usr/bin/env bash
set -euo pipefail

PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"

PROJECT_ROOT="${PACKAGE_PROJECT_ROOT}"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_MODE="${RUN_MODE:-formal}"
NYSE_BATCH_SIZE="${NYSE_BATCH_SIZE:-16}"
SP500_BATCH_SIZE="${SP500_BATCH_SIZE:-16}"
ONLY_CANDIDATE="${ONLY_CANDIDATE:-}"

FORMAL_RESULT_ROOT="results/patra_remaining_mask_sensitivity_3run_20260729"
NYSE_LAYER_SELECTION="results/patra_remaining_hyperparameter_sensitivity_5run_20260728/NYSE/drt_layers/stage_selection.json"
NYSE_MASK_LAYER=4

case "$RUN_MODE" in
  final_test)
    RESULT_ROOT="results/final_test/patra_remaining_mask_sensitivity_3run_20260729"
    LOG_DIR="logs/final_test/patra_remaining_mask_sensitivity_3run_20260729"
    EXPECTED_RUNS=1
    DEBUG_FLAG=0
    PRETRAIN_EPOCHS=1
    TRAIN_EPOCHS=1
    PATIENCE=1
    ;;
  formal)
    RESULT_ROOT="$FORMAL_RESULT_ROOT"
    LOG_DIR="logs/patra_remaining_mask_sensitivity_3run_20260729"
    EXPECTED_RUNS=3
    DEBUG_FLAG=0
    PRETRAIN_EPOCHS=20
    TRAIN_EPOCHS=50
    PATIENCE=20
    ;;
  *)
    echo "RUN_MODE must be final_test or formal; got: $RUN_MODE" >&2
    exit 2
    ;;
esac

if [[ "$NYSE_BATCH_SIZE" != "16" && "$NYSE_BATCH_SIZE" != "8" ]]; then
  echo "NYSE_BATCH_SIZE must be 16 or 8; got: $NYSE_BATCH_SIZE" >&2
  exit 2
fi
if [[ "$SP500_BATCH_SIZE" != "16" && "$SP500_BATCH_SIZE" != "8" ]]; then
  echo "SP500_BATCH_SIZE must be 16 or 8; got: $SP500_BATCH_SIZE" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing executable Python: $PYTHON_BIN" >&2
  exit 1
fi

NYSE_RELATION="data/processed/NYSE_ohlcv/priors/dtw_binary_imputed_inputstandard_NYSE_train0-871_close_ds64_band6_cand48_top32.npz"
SP500_RELATION="data/processed/SP500_ohlcv/priors/dtw_binary_imputed_inputstandard_SP500_train915-2042_close_ds64_band6_cand48_top32.npz"
test -f "$NYSE_RELATION"
test -f "$SP500_RELATION"
test -f "$NYSE_LAYER_SELECTION"

"$PYTHON_BIN" - "$NYSE_LAYER_SELECTION" "$NYSE_MASK_LAYER" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = str(sys.argv[2])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("selection_source") != "validation_only":
    raise RuntimeError(f"Unexpected NYSE layer selection source: {payload.get('selection_source')}")
if str(payload.get("selected_value")) != expected:
    raise RuntimeError(
        f"NYSE mask layer mismatch: expected {expected}, got {payload.get('selected_value')}"
    )
PY

mkdir -p "$RESULT_ROOT" "$LOG_DIR"
MASTER_LOG="${LOG_DIR}/${RUN_MODE}_nyse_batch_${NYSE_BATCH_SIZE}_sp500_batch_${SP500_BATCH_SIZE}.log"
exec > >(tee -a "$MASTER_LOG") 2>&1

candidate_complete() {
  "$PYTHON_BIN" - "$1" "$EXPECTED_RUNS" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_runs = int(sys.argv[2])
summaries = sorted(root.rglob("summary_all_runs.json"))
if len(summaries) != 1:
    raise SystemExit(1)
payload = json.loads(summaries[0].read_text(encoding="utf-8"))
runs = payload.get("runs", [])
required = {"mae", "RankIC", "RankICIR", "IRR@20", "SR@20"}
complete = (
    payload.get("num_runs") == expected_runs
    and len(runs) == expected_runs
    and all(bool(run.get("test_evaluated")) for run in runs)
    and all(required <= set(run.get("test_metrics", {})) for run in runs)
    and all(
        all(
            isinstance(run["test_metrics"][metric], (int, float))
            and math.isfinite(float(run["test_metrics"][metric]))
            for metric in required
        )
        for run in runs
    )
)
raise SystemExit(0 if complete else 1)
PY
}

should_run() {
  [[ -z "$ONLY_CANDIDATE" || "$ONLY_CANDIDATE" == "$1" ]]
}

run_candidate() {
  local candidate="$1"
  local dataset="$2"
  local label="$3"
  local layers="$4"
  local mask_ratio="$5"
  local batch_size="$6"
  local base_seed="$7"

  if ! should_run "$candidate"; then
    return
  fi

  local market_num
  local valid_index
  local test_index
  local data_start_index
  local relation_path
  case "$dataset" in
    NYSE)
      market_num=25
      valid_index=871
      test_index=996
      data_start_index=0
      relation_path="$NYSE_RELATION"
      ;;
    SP500)
      market_num=8
      valid_index=1127
      test_index=1288
      data_start_index=915
      relation_path="$SP500_RELATION"
      ;;
    *)
      echo "Unsupported dataset: $dataset" >&2
      exit 2
      ;;
  esac

  local candidate_root="${RESULT_ROOT}/${dataset}/mmr_mask_ratio/${label}"
  local model_id="${RUN_MODE}_patra_${candidate}_${EXPECTED_RUNS}run"
  if candidate_complete "$candidate_root"; then
    echo "SKIP completed candidate=${candidate} root=${candidate_root}"
    return
  fi

  echo "START candidate=${candidate} dataset=${dataset} d_model=32 layers=${layers} mask=${mask_ratio} batch=${batch_size} runs=${EXPECTED_RUNS}"
  env PYTHONPATH=src "$PYTHON_BIN" \
    -c 'import sys; sys.path.pop(0); from satra.mmr_two_stage import main; main()' \
    --is_training 1 \
    --model_id "$model_id" \
    --model model_b_spe_drt \
    --num_runs "$EXPECTED_RUNS" \
    --random_seed "$base_seed" \
    --evaluate_test 1 \
    --portfolio_size 20 \
    --debug "$DEBUG_FLAG" \
    --debug_num_runs "$EXPECTED_RUNS" \
    --debug_train_steps 8 \
    --debug_eval_steps 8 \
    --data "$dataset" \
    --market_num "$market_num" \
    --root_path data/processed \
    --results "$candidate_root" \
    --checkpoints "${candidate_root}/checkpoints" \
    --input_features all \
    --use_grouped_inputs 0 \
    --seq_len 16 \
    --batch_size "$batch_size" \
    --num_workers 0 \
    --use_gpu 1 \
    --pin_memory 1 \
    --train_drop_last 0 \
    --valid_drop_last 0 \
    --prediction_target return \
    --target_mode raw_return \
    --output_activation tanh \
    --pretrain_epochs "$PRETRAIN_EPOCHS" \
    --pretrain_learning_rate 0.001 \
    --pretrain_lr_scheduler cosine \
    --mae_mask_ratio "$mask_ratio" \
    --learning_rate 0.001 \
    --train_epochs "$TRAIN_EPOCHS" \
    --patience "$PATIENCE" \
    --min_epochs 0 \
    --min_updates 0 \
    --early_stop_metric RankIC \
    --early_stop_mode max \
    --lr_scheduler cosine \
    --precision bf16_amp_tf32 \
    --torch_compile 0 \
    --max_grad_norm 1.0 \
    --d_model 32 \
    --pstr_depth 10 \
    --pstr_kernel_size 3 \
    --pstr_num_heads 4 \
    --pstr_tower_layers "$layers" \
    --pstr_dropout 0.1 \
    --moe_top_k 3 \
    --moe_load_balance_weight 0.0 \
    --pstr_relation_path "$relation_path" \
    --valid_index "$valid_index" \
    --test_index "$test_index" \
    --data_start_index "$data_start_index"

  if ! candidate_complete "$candidate_root"; then
    echo "Candidate finished without a complete N=20 test record: $candidate_root" >&2
    exit 1
  fi
  echo "DONE candidate=${candidate}"
}

verify_all_candidates() {
  local missing=0
  local root
  for root in \
    "$RESULT_ROOT/NYSE/mmr_mask_ratio/mask03" \
    "$RESULT_ROOT/NYSE/mmr_mask_ratio/mask04" \
    "$RESULT_ROOT/SP500/mmr_mask_ratio/mask01" \
    "$RESULT_ROOT/SP500/mmr_mask_ratio/mask02" \
    "$RESULT_ROOT/SP500/mmr_mask_ratio/mask03" \
    "$RESULT_ROOT/SP500/mmr_mask_ratio/mask04"
  do
    if ! candidate_complete "$root"; then
      echo "MISSING or incomplete candidate root: $root" >&2
      missing=1
    fi
  done
  ((missing == 0))
}

echo "PATRA remaining mask-ratio sensitivity"
echo "mode=${RUN_MODE} runs=${EXPECTED_RUNS} only_candidate=${ONLY_CANDIDATE:-all}"
echo "contract=six remaining CSV points; validation-selected checkpoint; test N=20"
echo "NYSE=d32/layers${NYSE_MASK_LAYER}/batch${NYSE_BATCH_SIZE}/mask{0.3,0.4}"
echo "SP500=d32/layers1/batch${SP500_BATCH_SIZE}/mask{0.1,0.2,0.3,0.4}"

run_candidate "nyse_mask03" "NYSE" "mask03" "$NYSE_MASK_LAYER" "0.3" "$NYSE_BATCH_SIZE" "20260721"
run_candidate "nyse_mask04" "NYSE" "mask04" "$NYSE_MASK_LAYER" "0.4" "$NYSE_BATCH_SIZE" "20260721"
run_candidate "sp500_mask01" "SP500" "mask01" "1" "0.1" "$SP500_BATCH_SIZE" "1"
run_candidate "sp500_mask02" "SP500" "mask02" "1" "0.2" "$SP500_BATCH_SIZE" "1"
run_candidate "sp500_mask03" "SP500" "mask03" "1" "0.3" "$SP500_BATCH_SIZE" "1"
run_candidate "sp500_mask04" "SP500" "mask04" "1" "0.4" "$SP500_BATCH_SIZE" "1"

if [[ -z "$ONLY_CANDIDATE" ]]; then
  verify_all_candidates
  echo "ALL COMPLETE mode=${RUN_MODE} root=${RESULT_ROOT}"
fi
