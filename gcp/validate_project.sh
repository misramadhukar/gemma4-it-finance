#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

for script in gcp/*.sh; do
  bash -n "${script}"
done

python3 -m py_compile \
  train.py \
  evaluate.py \
  infer.py \
  merge_adapter.py \
  preflight.py \
  dataset_utils.py \
  hub_utils.py \
  sft_utils.py
python3 -m unittest discover -s tests -v
bash tests/test_training_job.sh

echo "Project validation passed."
