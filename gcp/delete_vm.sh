#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
require_gcloud

read -r -p "Delete VM ${VM_NAME} and its auto-delete boot disk? Type the VM name: " answer
if [[ "${answer}" != "${VM_NAME}" ]]; then
  echo "Cancelled."
  exit 1
fi

gcloud_base compute instances delete "${VM_NAME}" --zone="${ZONE}" --quiet
