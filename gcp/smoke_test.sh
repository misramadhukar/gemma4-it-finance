#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv/bin/activate"
cd "${REPO_ROOT}"

python train.py \
  --limit 100 \
  --max-length 512 \
  --max-steps 5 \
  --save-steps 5 \
  --logging-steps 1 \
  --overwrite-output-dir \
  --output-dir outputs/smoke \
  "$@"
