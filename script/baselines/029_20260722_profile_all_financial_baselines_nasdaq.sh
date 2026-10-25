#!/usr/bin/env bash
set -euo pipefail
PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"


PYTHON_BIN=${PYTHON_BIN:-python}
DATASET=${DATASET:-NASDAQ}
PROFILE_TAG=${PROFILE_TAG:-20260722}
PROFILE_ROOT=${PROFILE_ROOT:-"results/efficiency_profile_${PROFILE_TAG}/financial"}
PROFILE_RESULTS_ROOT="${PROFILE_ROOT}/native_runs"

methods=(ci_sthpan enhancer factorgcl factorvae hstockgat master rsr stockmixer tra)

for method in "${methods[@]}"; do
  mapfile -t configs < <(find "results_rankic/${DATASET}/${method}" -mindepth 2 -maxdepth 2 -name config.json -type f | sort)
  if [ "${#configs[@]}" -eq 0 ]; then
    echo "Missing formal config for ${method} on ${DATASET}" >&2
    exit 1
  fi
  config="${configs[$((${#configs[@]} - 1))]}"
  echo "=== Profiling ${method} with ${config} ==="
  "$PYTHON_BIN" script/baselines/tool_profile_baseline_efficiency.py \
    --module "${method}.run" \
    --config "${config}" \
    --output "${PROFILE_ROOT}/${method}.json" \
    --profile-results "${PROFILE_RESULTS_ROOT}/${method}" \
    --seed 20260722
done

echo "Financial efficiency profiles: ${PROFILE_ROOT}"
