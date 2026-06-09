#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${GCP_CONFIG:-${SCRIPT_DIR}/config.env}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Missing ${CONFIG_FILE}" >&2
  echo "Create it with: cp gcp/config.env.example gcp/config.env" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${CONFIG_FILE}"

: "${PROJECT_ID:?Set PROJECT_ID in gcp/config.env}"
: "${ZONE:?Set ZONE in gcp/config.env}"
: "${VM_NAME:?Set VM_NAME in gcp/config.env}"
: "${MACHINE_TYPE:?Set MACHINE_TYPE in gcp/config.env}"
: "${BOOT_DISK_SIZE:?Set BOOT_DISK_SIZE in gcp/config.env}"
: "${BOOT_DISK_TYPE:?Set BOOT_DISK_TYPE in gcp/config.env}"
: "${IMAGE_FAMILY:?Set IMAGE_FAMILY in gcp/config.env}"
: "${IMAGE_PROJECT:?Set IMAGE_PROJECT in gcp/config.env}"
: "${PROVISIONING_MODEL:?Set PROVISIONING_MODEL in gcp/config.env}"
: "${REMOTE_DIR:?Set REMOTE_DIR in gcp/config.env}"

TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"
PROVISIONING_MODEL="${PROVISIONING_MODEL^^}"

if [[ "${PROVISIONING_MODEL}" != "STANDARD" && "${PROVISIONING_MODEL}" != "SPOT" ]]; then
  echo "PROVISIONING_MODEL must be STANDARD or SPOT." >&2
  exit 1
fi

if [[ "${PROJECT_ID}" == "your-gcp-project-id" ]]; then
  echo "Replace the PROJECT_ID placeholder in gcp/config.env." >&2
  exit 1
fi

if [[ "${REMOTE_DIR}" == /* || "${REMOTE_DIR}" == *".."* || ! "${REMOTE_DIR}" =~ ^[A-Za-z0-9._/-]+$ ]]; then
  echo "REMOTE_DIR must be a safe relative path below the VM user's home directory." >&2
  exit 1
fi

if [[ ! "${TENSORBOARD_PORT}" =~ ^[0-9]+$ ]]; then
  echo "TENSORBOARD_PORT must be numeric." >&2
  exit 1
fi

require_gcloud() {
  if ! command -v gcloud >/dev/null 2>&1; then
    echo "gcloud is required. Run these scripts from Google Cloud Shell." >&2
    exit 1
  fi
}

gcloud_base() {
  gcloud --project="${PROJECT_ID}" "$@"
}
