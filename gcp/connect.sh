#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
require_gcloud

ssh_args=()
if [[ "${1:-}" == "--tensorboard" ]]; then
  ssh_args=(-- -L "${TENSORBOARD_PORT}:localhost:${TENSORBOARD_PORT}")
fi

gcloud_base compute ssh "${VM_NAME}" \
  --zone="${ZONE}" \
  "${ssh_args[@]}"
