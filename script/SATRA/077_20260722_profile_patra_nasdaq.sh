#!/usr/bin/env bash
set -euo pipefail
PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"


PYTHON_BIN=${PYTHON_BIN:-python}
PROFILE_TAG=${PROFILE_TAG:-20260722}
PROFILE_ROOT=${PROFILE_ROOT:-"results/efficiency_profile_${PROFILE_TAG}/patra"}
FORMAL_CONFIG=${FORMAL_CONFIG:-"results/pstr_d_pstr_net_with_mmr_5run_3markets/NASDAQ/model_b_spe_drt/20260715_1637/config.json"}

"$PYTHON_BIN" script/SATRA/tool_profile_satra_efficiency.py \
  --config "$FORMAL_CONFIG" \
  --output "${PROFILE_ROOT}/patra_net.json" \
  --profile-results "${PROFILE_ROOT}/native_run" \
  --seed 20260722

echo "PATRA efficiency profile: ${PROFILE_ROOT}/patra_net.json"
