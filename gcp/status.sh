#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
require_gcloud

gcloud_base compute instances describe "${VM_NAME}" \
  --zone="${ZONE}" \
  --format="table(name,status,machineType.basename(),zone.basename(),networkInterfaces[0].accessConfigs[0].natIP)"

echo
echo "Recent serial-port output:"
gcloud_base compute instances get-serial-port-output "${VM_NAME}" \
  --zone="${ZONE}" \
  --port=1 \
  --start=0 | tail -n 120
