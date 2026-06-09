#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_CONFIG="${TRAIN_CONFIG:-${SCRIPT_DIR}/train.env}"

if [[ ! -f "${TRAIN_CONFIG}" ]]; then
  echo "Missing ${TRAIN_CONFIG}" >&2
  echo "Create it with: cp gcp/train.env.example gcp/train.env" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${TRAIN_CONFIG}"

: "${MODEL_ID:?Set MODEL_ID in gcp/train.env}"
: "${DATASET_ID:?Set DATASET_ID in gcp/train.env}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR in gcp/train.env}"

if [[ "${OUTPUT_DIR}" == /* || "${OUTPUT_DIR}" == *".."* || ! "${OUTPUT_DIR}" =~ ^[A-Za-z0-9._/-]+$ ]]; then
  echo "OUTPUT_DIR must be a safe relative path below the repository." >&2
  exit 1
fi

for boolean_name in OVERWRITE_OUTPUT_DIR PUSH_TO_HUB SKIP_READINESS_CHECKS; do
  boolean_value="${!boolean_name:-false}"
  if [[ "${boolean_value}" != "true" && "${boolean_value}" != "false" ]]; then
    echo "${boolean_name} must be true or false." >&2
    exit 1
  fi
done

TMUX_SESSION="${TMUX_SESSION:-gemma4}"
if [[ ! "${TMUX_SESSION}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "TMUX_SESSION contains unsupported characters." >&2
  exit 1
fi

activate_training_venv() {
  if [[ ! -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    echo "Missing virtual environment. Run: bash gcp/setup_vm.sh" >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
}
