#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=train_common.sh
source "${SCRIPT_DIR}/train_common.sh"
activate_training_venv

args=(
  train.py
  "--model-id=${MODEL_ID}"
  "--model-revision=${MODEL_REVISION:-main}"
  "--dataset-id=${DATASET_ID}"
  "--dataset-revision=${DATASET_REVISION:-main}"
  "--dataset-split=${DATASET_SPLIT:-train}"
  "--output-dir=${OUTPUT_DIR}"
  "--limit=${LIMIT:-5000}"
  "--test-size=${TEST_SIZE:-0.1}"
  "--seed=${SEED:-42}"
  "--max-length=${MAX_LENGTH:-1024}"
  "--epochs=${EPOCHS:-3}"
  "--learning-rate=${LEARNING_RATE:-1e-4}"
  "--train-batch-size=${TRAIN_BATCH_SIZE:-1}"
  "--eval-batch-size=${EVAL_BATCH_SIZE:-1}"
  "--gradient-accumulation-steps=${GRADIENT_ACCUMULATION_STEPS:-8}"
  "--lora-r=${LORA_R:-16}"
  "--lora-alpha=${LORA_ALPHA:-32}"
  "--lora-dropout=${LORA_DROPOUT:-0.05}"
  "--lora-targets=${LORA_TARGETS:-all-linear}"
  "--save-steps=${SAVE_STEPS:-100}"
  "--save-total-limit=${SAVE_TOTAL_LIMIT:-3}"
  "--logging-steps=${LOGGING_STEPS:-10}"
  "--dataloader-workers=${DATALOADER_WORKERS:-2}"
)

if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  args+=("--resume-from-checkpoint=${RESUME_FROM_CHECKPOINT}")
fi

if [[ "${OVERWRITE_OUTPUT_DIR:-false}" == "true" ]]; then
  args+=(--overwrite-output-dir)
fi

if [[ "${PUSH_TO_HUB:-false}" == "true" ]]; then
  : "${HUB_MODEL_ID:?Set HUB_MODEL_ID when PUSH_TO_HUB=true}"
  args+=(--push-to-hub "--hub-model-id=${HUB_MODEL_ID}")
fi

cd "${REPO_ROOT}"
exec python "${args[@]}" "$@"
