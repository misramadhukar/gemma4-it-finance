#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=train_common.sh
source "${SCRIPT_DIR}/train_common.sh"
activate_training_venv

args=(
  evaluate.py
  "--adapter=${OUTPUT_DIR}/final_adapter"
  "--base-model=${MODEL_ID}"
  "--model-revision=auto"
  "--dataset-id=${DATASET_ID}"
  "--dataset-revision=auto"
  "--dataset-split=${DATASET_SPLIT:-train}"
  "--limit=${LIMIT:-5000}"
  "--test-size=${TEST_SIZE:-0.1}"
  "--seed=${SEED:-42}"
  "--max-length=${MAX_LENGTH:-1024}"
  "--max-examples=${EVAL_MAX_EXAMPLES:-200}"
  "--output-dir=${OUTPUT_DIR}/evaluation"
)

if [[ -n "${MIN_SENTIMENT_ACCURACY:-}" ]]; then
  args+=("--min-accuracy=${MIN_SENTIMENT_ACCURACY}")
fi
if [[ -n "${MIN_SENTIMENT_PARSE_RATE:-}" ]]; then
  args+=("--min-parse-rate=${MIN_SENTIMENT_PARSE_RATE}")
fi
if [[ -n "${MIN_ROUGE_L:-}" ]]; then
  args+=("--min-rouge-l=${MIN_ROUGE_L}")
fi

cd "${REPO_ROOT}"
exec python "${args[@]}" "$@"
