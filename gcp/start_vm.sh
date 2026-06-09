#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
require_gcloud

gcloud_base compute instances start "${VM_NAME}" --zone="${ZONE}"
echo "Next: bash gcp/wait_ready.sh"
