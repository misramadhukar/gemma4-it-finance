#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
require_gcloud

timeout_seconds="${READY_TIMEOUT_SECONDS:-1800}"
started_at="$(date +%s)"

echo "Waiting for SSH and the NVIDIA driver on ${VM_NAME}..."
while true; do
  if gcloud_base compute ssh "${VM_NAME}" \
    --zone="${ZONE}" \
    --quiet \
    --command="test -f /var/tmp/gemma4-startup-complete && nvidia-smi" \
    >/dev/null 2>&1; then
    echo "VM is ready."
    gcloud_base compute ssh "${VM_NAME}" \
      --zone="${ZONE}" \
      --quiet \
      --command="nvidia-smi"
    break
  fi

  now="$(date +%s)"
  if (( now - started_at >= timeout_seconds )); then
    echo "Timed out after ${timeout_seconds}s." >&2
    echo "Inspect startup logs with: bash gcp/status.sh" >&2
    exit 1
  fi

  sleep 20
done
