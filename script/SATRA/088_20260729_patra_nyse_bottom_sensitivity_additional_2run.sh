#!/usr/bin/env bash
set -euo pipefail

PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"

PROJECT_ROOT="${PACKAGE_PROJECT_ROOT}"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_MODE="formal"
ONLY_CANDIDATE="${ONLY_CANDIDATE:-}"

EXPECTED_SOURCE_MANIFEST="f065edf8e9fd6f34864e6968dce12086de930160870f0eae8dca1ce7200c28ab"
FORMAL_RESULT_ROOT="results/patra_nyse_d32_layers3_bottom_sensitivity_additional_2run_20260729"
FORMAL_LOG_DIR="logs/patra_nyse_d32_layers3_bottom_sensitivity_additional_2run_20260729"

RESULT_ROOT="$FORMAL_RESULT_ROOT"
LOG_DIR="$FORMAL_LOG_DIR"
EXPECTED_RUNS=2
DEBUG_FLAG=0
BASE_SEED=20260722

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing executable Python: $PYTHON_BIN" >&2
  exit 1
fi

PRIOR_DIR="data/processed/NYSE_ohlcv/priors"
FORMAL_TOP32_PRIOR="${PRIOR_DIR}/dtw_binary_imputed_inputstandard_NYSE_train0-871_close_ds64_band6_cand48_top32.npz"
DTW_TOP8_PRIOR="${PRIOR_DIR}/dtw_binary_imputed_inputstandard_NYSE_train0-871_close_ds64_band6_cand96_top8.npz"
DTW_TOP16_PRIOR="${PRIOR_DIR}/dtw_binary_imputed_inputstandard_NYSE_train0-871_close_ds64_band6_cand96_top16.npz"
DTW_TOP32_PRIOR="${PRIOR_DIR}/dtw_binary_imputed_inputstandard_NYSE_train0-871_close_ds64_band6_cand96_top32.npz"
DTW_TOP64_PRIOR="${PRIOR_DIR}/dtw_binary_imputed_inputstandard_NYSE_train0-871_close_ds64_band6_cand96_top64.npz"

"$PYTHON_BIN" - \
  "$EXPECTED_SOURCE_MANIFEST" \
  "$FORMAL_TOP32_PRIOR" \
  "$DTW_TOP8_PRIOR" \
  "$DTW_TOP16_PRIOR" \
  "$DTW_TOP32_PRIOR" \
  "$DTW_TOP64_PRIOR" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys

import numpy as np

expected_manifest = sys.argv[1]
source_root = Path("src/satra")
digest = sha256()
for path in sorted(source_root.rglob("*.py")):
    digest.update(path.as_posix().encode("utf-8"))
    digest.update(b"\0")
    digest.update(path.read_bytes())
    digest.update(b"\0")
actual_manifest = digest.hexdigest()
if actual_manifest != expected_manifest:
    raise RuntimeError(
        f"Runtime source manifest mismatch: expected {expected_manifest}, "
        f"got {actual_manifest}"
    )

expected = (
    (Path(sys.argv[2]), 32, 48),
    (Path(sys.argv[3]), 8, 96),
    (Path(sys.argv[4]), 16, 96),
    (Path(sys.argv[5]), 32, 96),
    (Path(sys.argv[6]), 64, 96),
)
for path, expected_topk, expected_candidate_k in expected:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        matrix = np.asarray(payload["weight_matrix"])
        topk = int(payload["topk"])
        candidate_k = int(payload["candidate_k"])
        train_start = int(payload["train_start"])
        train_end = int(payload["train_end"])
    if matrix.shape != (1737, 1737):
        raise ValueError(f"{path}: expected shape (1737, 1737), got {matrix.shape}")
    if topk != expected_topk or candidate_k != expected_candidate_k:
        raise ValueError(
            f"{path}: expected topk={expected_topk}, "
            f"candidate_k={expected_candidate_k}; got topk={topk}, "
            f"candidate_k={candidate_k}"
        )
    if (train_start, train_end) != (0, 871):
        raise ValueError(
            f"{path}: expected train interval [0, 871), "
            f"got [{train_start}, {train_end})"
        )
    if not np.all(matrix.sum(axis=1) == expected_topk):
        raise ValueError(f"{path}: every row must have {expected_topk} neighbors")
    if np.any(np.diag(matrix) != 0):
        raise ValueError(f"{path}: self edges are forbidden")

print(f"RUNTIME_SOURCE_OK manifest={actual_manifest}")
print("NYSE_PRIORS_OK formal=cand48/top32 controlled=cand96/top{8,16,32,64}")
PY

mkdir -p "$RESULT_ROOT" "$LOG_DIR"
MASTER_LOG="${LOG_DIR}/${RUN_MODE}.log"
exec > >(tee -a "$MASTER_LOG") 2>&1

candidate_complete() {
  "$PYTHON_BIN" - "$1" "$EXPECTED_RUNS" "$BASE_SEED" <<'PY'
import json
import math
from pathlib import Path
import sys

root = Path(sys.argv[1])
expected_runs = int(sys.argv[2])
base_seed = int(sys.argv[3])
required = {"mae", "RankIC", "RankICIR", "IRR@20", "SR@20"}
complete = []

for path in sorted(root.rglob("summary_all_runs.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    runs = payload.get("runs", [])
    seeds = [int(run.get("seed")) for run in runs if run.get("seed") is not None]
    valid = (
        payload.get("num_runs") == expected_runs
        and payload.get("base_seed") == base_seed
        and len(runs) == expected_runs
        and seeds == list(range(base_seed, base_seed + expected_runs))
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
    if valid:
        complete.append(path)

if len(complete) > 1:
    raise RuntimeError(f"Ambiguous complete summaries below {root}: {complete}")
if len(complete) == 1:
    print(f"COMPLETE_RECORD {complete[0]}")
    raise SystemExit(0)
raise SystemExit(1)
PY
}

should_run() {
  [[ -z "$ONLY_CANDIDATE" || "$ONLY_CANDIDATE" == "$1" ]]
}

COMMON_ARGS=(
  --is_training 1
  --model model_b_spe_drt
  --num_runs "$EXPECTED_RUNS"
  --random_seed "$BASE_SEED"
  --evaluate_test 1
  --portfolio_size 20
  --debug "$DEBUG_FLAG"
  --debug_num_runs "$EXPECTED_RUNS"
  --debug_train_steps 8
  --debug_eval_steps 8
  --data NYSE
  --market_num 25
  --root_path data/processed
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
  --pretrain_epochs 20
  --pretrain_learning_rate 0.001
  --pretrain_lr_scheduler cosine
  --mae_mask_ratio 0.3
  --learning_rate 0.001
  --train_epochs 50
  --patience 20
  --min_epochs 0
  --min_updates 0
  --early_stop_metric RankIC
  --early_stop_mode max
  --lr_scheduler cosine
  --precision bf16_amp_tf32
  --torch_compile 0
  --max_grad_norm 1.0
  --d_model 32
  --pstr_depth 10
  --pstr_kernel_size 3
  --pstr_num_heads 4
  --pstr_tower_layers 3
  --pstr_dropout 0.1
  --valid_index 871
  --test_index 996
  --data_start_index 0
)

run_candidate() {
  local candidate="$1"
  local panel="$2"
  local label="$3"
  local top_k="$4"
  local balance_weight="$5"
  local relation_path="$6"
  local candidate_root="${RESULT_ROOT}/${panel}/${label}"
  local model_id="${RUN_MODE}_nyse_d32_l3_${panel}_${label}_seed${BASE_SEED}_${EXPECTED_RUNS}run"

  if ! should_run "$candidate"; then
    return
  fi
  if candidate_complete "$candidate_root"; then
    echo "SKIP completed candidate=${candidate}"
    return
  fi

  echo "START candidate=${candidate} seeds=${BASE_SEED}..$((BASE_SEED + EXPECTED_RUNS - 1))"
  env PYTHONPATH=src PYTHONUNBUFFERED=1 "$PYTHON_BIN" \
    -c 'import sys; sys.path.pop(0); from satra.mmr_two_stage import main; main()' \
    "${COMMON_ARGS[@]}" \
    --model_id "$model_id" \
    --results "$candidate_root" \
    --checkpoints "${candidate_root}/checkpoints" \
    --moe_top_k "$top_k" \
    --moe_load_balance_weight "$balance_weight" \
    --pstr_relation_path "$relation_path"

  if ! candidate_complete "$candidate_root"; then
    echo "Candidate ended without a complete N=20 record: ${candidate_root}" >&2
    exit 1
  fi
  echo "DONE candidate=${candidate}"
}

wait_pair() {
  local left_pid="$1"
  local left_name="$2"
  local right_pid="$3"
  local right_name="$4"
  local status=0

  if ! wait "$left_pid"; then
    echo "FAILED candidate=${left_name}" >&2
    status=1
  fi
  if ! wait "$right_pid"; then
    echo "FAILED candidate=${right_name}" >&2
    status=1
  fi
  return "$status"
}

run_pair() {
  local left_candidate="$1"
  local left_panel="$2"
  local left_label="$3"
  local left_topk="$4"
  local left_balance="$5"
  local left_prior="$6"
  local right_candidate="$7"
  local right_panel="$8"
  local right_label="$9"
  local right_topk="${10}"
  local right_balance="${11}"
  local right_prior="${12}"

  run_candidate \
    "$left_candidate" "$left_panel" "$left_label" \
    "$left_topk" "$left_balance" "$left_prior" &
  local left_pid=$!
  run_candidate \
    "$right_candidate" "$right_panel" "$right_label" \
    "$right_topk" "$right_balance" "$right_prior" &
  local right_pid=$!
  wait_pair "$left_pid" "$left_candidate" "$right_pid" "$right_candidate"
}

verify_candidate() {
  local panel="$1"
  local label="$2"
  candidate_complete "${RESULT_ROOT}/${panel}/${label}"
}

verify_all_formal_candidates() {
  verify_candidate ace_topk k1
  verify_candidate ace_topk k2
  verify_candidate ace_topk k3
  verify_candidate ace_topk k6
  verify_candidate dtw_neighbors k8
  verify_candidate dtw_neighbors k16
  verify_candidate dtw_neighbors k32
  verify_candidate dtw_neighbors k64
  verify_candidate balance_weight lb0
  verify_candidate balance_weight lb1e-4
  verify_candidate balance_weight lb1e-3
  verify_candidate balance_weight lb1e-2
}

echo "PATRA NYSE bottom sensitivity additional runs"
echo "mode=${RUN_MODE} base_seed=${BASE_SEED} runs=${EXPECTED_RUNS} batch_size=1"
echo "fixed=d_model32/layers3/mask0.3; checkpoint=validation RankIC; test=N20"
echo "execution=two concurrent candidates"
echo "source_manifest=${EXPECTED_SOURCE_MANIFEST}"

run_pair \
  ace_k1 ace_topk k1 1 1e-3 "$FORMAL_TOP32_PRIOR" \
  ace_k2 ace_topk k2 2 1e-3 "$FORMAL_TOP32_PRIOR"
run_pair \
  ace_k3 ace_topk k3 3 1e-3 "$FORMAL_TOP32_PRIOR" \
  ace_k6 ace_topk k6 6 1e-3 "$FORMAL_TOP32_PRIOR"
run_pair \
  dtw_k8 dtw_neighbors k8 3 1e-3 "$DTW_TOP8_PRIOR" \
  dtw_k16 dtw_neighbors k16 3 1e-3 "$DTW_TOP16_PRIOR"
run_pair \
  dtw_k32 dtw_neighbors k32 3 1e-3 "$DTW_TOP32_PRIOR" \
  dtw_k64 dtw_neighbors k64 3 1e-3 "$DTW_TOP64_PRIOR"
run_pair \
  lb0 balance_weight lb0 3 0 "$FORMAL_TOP32_PRIOR" \
  lb1e-4 balance_weight lb1e-4 3 1e-4 "$FORMAL_TOP32_PRIOR"
run_pair \
  lb1e-3 balance_weight lb1e-3 3 1e-3 "$FORMAL_TOP32_PRIOR" \
  lb1e-2 balance_weight lb1e-2 3 1e-2 "$FORMAL_TOP32_PRIOR"

if [[ -z "$ONLY_CANDIDATE" ]]; then
  verify_all_formal_candidates
  echo "ALL COMPLETE root=${RESULT_ROOT}"
fi
