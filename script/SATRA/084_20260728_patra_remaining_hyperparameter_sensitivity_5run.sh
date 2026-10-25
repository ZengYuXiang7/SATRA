#!/usr/bin/env bash
set -euo pipefail

PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"

PROJECT_ROOT="${PACKAGE_PROJECT_ROOT}"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_MODE="${RUN_MODE:-formal}"
NYSE_MASK_BATCH_SIZE="${NYSE_MASK_BATCH_SIZE:-16}"
SP500_MASK_BATCH_SIZE="${SP500_MASK_BATCH_SIZE:-16}"
ONLY_CANDIDATE="${ONLY_CANDIDATE:-}"
NYSE_MASK_LAYER_OVERRIDE="${NYSE_MASK_LAYER_OVERRIDE:-4}"

FORMAL_RESULT_ROOT="results/patra_remaining_hyperparameter_sensitivity_5run_20260728"

case "$RUN_MODE" in
  final_test)
    RESULT_ROOT="results/final_test/patra_remaining_hyperparameter_sensitivity_20260728/nyse_d32_batch_${NYSE_MASK_BATCH_SIZE}_sp500_d32_batch_${SP500_MASK_BATCH_SIZE}"
    LOG_DIR="logs/final_test/patra_remaining_hyperparameter_sensitivity_20260728"
    EXPECTED_RUNS=1
    DEBUG_FLAG=0
    PRETRAIN_EPOCHS=1
    TRAIN_EPOCHS=1
    PATIENCE=1
    ;;
  formal)
    RESULT_ROOT="$FORMAL_RESULT_ROOT"
    LOG_DIR="logs/patra_remaining_hyperparameter_sensitivity_5run_20260728"
    EXPECTED_RUNS=5
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

if [[ "$NYSE_MASK_BATCH_SIZE" != "16" && "$NYSE_MASK_BATCH_SIZE" != "8" && "$NYSE_MASK_BATCH_SIZE" != "1" ]]; then
  echo "NYSE_MASK_BATCH_SIZE must be 16, 8, or the fallback 1; got: $NYSE_MASK_BATCH_SIZE" >&2
  exit 2
fi

if [[ "$SP500_MASK_BATCH_SIZE" != "16" && "$SP500_MASK_BATCH_SIZE" != "8" ]]; then
  echo "SP500_MASK_BATCH_SIZE must be 16 or 8; got: $SP500_MASK_BATCH_SIZE" >&2
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

mkdir -p "$RESULT_ROOT" "$LOG_DIR"
MASTER_LOG="${LOG_DIR}/${RUN_MODE}_nyse_batch_${NYSE_MASK_BATCH_SIZE}_sp500_batch_${SP500_MASK_BATCH_SIZE}.log"
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
required_metrics = {"mae", "RankIC", "RankICIR", "IRR@20", "SR@20"}
complete = (
    payload.get("num_runs") == expected_runs
    and len(runs) == expected_runs
    and all(bool(run.get("test_evaluated")) for run in runs)
    and all(required_metrics <= set(run.get("test_metrics", {})) for run in runs)
    and all(
        all(
            isinstance(run["test_metrics"][metric], (int, float))
            and math.isfinite(float(run["test_metrics"][metric]))
            for metric in required_metrics
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
  local group="$3"
  local label="$4"
  local d_model="$5"
  local tower_layers="$6"
  local mask_ratio="$7"
  local batch_size="$8"
  local base_seed="$9"

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

  local candidate_root="${RESULT_ROOT}/${dataset}/${group}/${label}"
  local model_id="${RUN_MODE}_patra_${candidate}_${EXPECTED_RUNS}run"

  if candidate_complete "$candidate_root"; then
    echo "SKIP completed candidate=${candidate} root=${candidate_root}"
    return
  fi

  echo "START candidate=${candidate} dataset=${dataset} d_model=${d_model} layers=${tower_layers} mask=${mask_ratio} batch=${batch_size} runs=${EXPECTED_RUNS}"
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
    --d_model "$d_model" \
    --pstr_depth 10 \
    --pstr_kernel_size 3 \
    --pstr_num_heads 4 \
    --pstr_tower_layers "$tower_layers" \
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

select_nyse_layer() {
  if [[ "$RUN_MODE" != "formal" ]]; then
    echo "$NYSE_MASK_LAYER_OVERRIDE"
    return
  fi

  "$PYTHON_BIN" - "$FORMAL_RESULT_ROOT" <<'PY'
import json
import os
import statistics
import sys
from pathlib import Path

new_root = Path(sys.argv[1])
roots = {
    "1": new_root / "NYSE" / "drt_layers" / "layers1",
    "2": new_root / "NYSE" / "drt_layers" / "layers2",
    "3": new_root / "NYSE" / "drt_layers" / "layers3",
    "4": new_root / "NYSE" / "drt_layers" / "layers4",
}
rows = []
for value, root in roots.items():
    summaries = sorted(root.rglob("summary_all_runs.json"))
    if len(summaries) != 1:
        raise RuntimeError(f"Expected one completed summary for NYSE layers={value}, found {len(summaries)} below {root}")
    payload = json.loads(summaries[0].read_text(encoding="utf-8"))
    runs = payload.get("runs", [])
    if payload.get("num_runs") != 5 or len(runs) != 5:
        raise RuntimeError(f"NYSE layers={value} is not a complete five-run candidate")
    values = [float(run["best_valid"]["RankIC"]) for run in runs]
    rows.append(
        {
            "candidate": f"layers{value}",
            "value": value,
            "summary": str(summaries[0]),
            "valid_rankic_values": values,
            "valid_rankic_mean": statistics.fmean(values),
            "valid_rankic_std": statistics.stdev(values),
        }
    )

winner = max(rows, key=lambda row: row["valid_rankic_mean"])
payload = {
    "selection_source": "validation_only",
    "selection_metric": "five-run mean of runs[*].best_valid.RankIC",
    "dataset": "NYSE",
    "parameter": "pstr_tower_layers",
    "candidates": rows,
    "selected_candidate": winner["candidate"],
    "selected_value": winner["value"],
    "selected_valid_rankic_mean": winner["valid_rankic_mean"],
    "selected_valid_rankic_std": winner["valid_rankic_std"],
}
path = new_root / "NYSE" / "drt_layers" / "stage_selection.json"
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
print(winner["value"])
PY
}

verify_all_candidates() {
  local missing=0
  local candidate_root
  for candidate_root in \
    "$RESULT_ROOT/NYSE/drt_layers/layers1" \
    "$RESULT_ROOT/NYSE/drt_layers/layers2" \
    "$RESULT_ROOT/NYSE/drt_layers/layers3" \
    "$RESULT_ROOT/NYSE/drt_layers/layers4" \
    "$RESULT_ROOT/NYSE/mmr_mask_ratio/mask01" \
    "$RESULT_ROOT/NYSE/mmr_mask_ratio/mask02" \
    "$RESULT_ROOT/NYSE/mmr_mask_ratio/mask03" \
    "$RESULT_ROOT/NYSE/mmr_mask_ratio/mask04" \
    "$RESULT_ROOT/SP500/mmr_mask_ratio/mask01" \
    "$RESULT_ROOT/SP500/mmr_mask_ratio/mask02" \
    "$RESULT_ROOT/SP500/mmr_mask_ratio/mask03" \
    "$RESULT_ROOT/SP500/mmr_mask_ratio/mask04"
  do
    if ! candidate_complete "$candidate_root"; then
      echo "MISSING or incomplete candidate root: $candidate_root" >&2
      missing=1
    fi
  done
  if ((missing != 0)); then
    return 1
  fi
}

echo "PATRA remaining hyperparameter sensitivity"
echo "mode=${RUN_MODE} nyse_mask_batch_size=${NYSE_MASK_BATCH_SIZE} sp500_mask_batch_size=${SP500_MASK_BATCH_SIZE} only_candidate=${ONLY_CANDIDATE:-all}"
echo "contract=10 missing CSV points plus consistent reruns of NYSE layers 1/2; five formal runs; validation-selected checkpoint; test N=20"
echo "layer_search=NYSE d_model=32 layers 1/2/3/4 at batch_size=1; old d16 batch-4 values are not reused"
echo "mask_search=NYSE d_model=32 ratios 0.1/0.2/0.3/0.4 at batch_size=${NYSE_MASK_BATCH_SIZE}"
echo "mask_search=SP500 ratios 0.1/0.2/0.3/0.4 at batch_size=${SP500_MASK_BATCH_SIZE}; batch 16/8 both validated"

run_candidate "nyse_layer1" "NYSE" "drt_layers" "layers1" "32" "1" "0.3" "1" "20260721"
run_candidate "nyse_layer2" "NYSE" "drt_layers" "layers2" "32" "2" "0.3" "1" "20260721"
run_candidate "nyse_layer3" "NYSE" "drt_layers" "layers3" "32" "3" "0.3" "1" "20260721"
run_candidate "nyse_layer4" "NYSE" "drt_layers" "layers4" "32" "4" "0.3" "1" "20260721"

NYSE_MASK_LAYER="$(select_nyse_layer)"
echo "NYSE mask-ratio fixed layer=${NYSE_MASK_LAYER}; selection source=$([[ "$RUN_MODE" == "formal" ]] && echo validation || echo override)"

run_candidate "nyse_mask01" "NYSE" "mmr_mask_ratio" "mask01" "32" "$NYSE_MASK_LAYER" "0.1" "$NYSE_MASK_BATCH_SIZE" "20260721"
run_candidate "nyse_mask02" "NYSE" "mmr_mask_ratio" "mask02" "32" "$NYSE_MASK_LAYER" "0.2" "$NYSE_MASK_BATCH_SIZE" "20260721"
run_candidate "nyse_mask03" "NYSE" "mmr_mask_ratio" "mask03" "32" "$NYSE_MASK_LAYER" "0.3" "$NYSE_MASK_BATCH_SIZE" "20260721"
run_candidate "nyse_mask04" "NYSE" "mmr_mask_ratio" "mask04" "32" "$NYSE_MASK_LAYER" "0.4" "$NYSE_MASK_BATCH_SIZE" "20260721"

# SP500 mask-ratio stage continues the validation-selected d32/layers1 configuration.
run_candidate "sp500_mask01" "SP500" "mmr_mask_ratio" "mask01" "32" "1" "0.1" "$SP500_MASK_BATCH_SIZE" "1"
run_candidate "sp500_mask02" "SP500" "mmr_mask_ratio" "mask02" "32" "1" "0.2" "$SP500_MASK_BATCH_SIZE" "1"
run_candidate "sp500_mask03" "SP500" "mmr_mask_ratio" "mask03" "32" "1" "0.3" "$SP500_MASK_BATCH_SIZE" "1"
run_candidate "sp500_mask04" "SP500" "mmr_mask_ratio" "mask04" "32" "1" "0.4" "$SP500_MASK_BATCH_SIZE" "1"

if [[ -z "$ONLY_CANDIDATE" ]]; then
  verify_all_candidates
  echo "ALL COMPLETE mode=${RUN_MODE} root=${RESULT_ROOT}"
fi
