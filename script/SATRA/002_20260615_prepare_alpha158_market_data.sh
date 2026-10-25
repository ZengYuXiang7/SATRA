#!/usr/bin/env bash
set -euo pipefail
PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"


python prepare_data.py preprocess \
  --markets NASDAQ NYSE SP500 \
  --alpha158 \
  --clean-output
