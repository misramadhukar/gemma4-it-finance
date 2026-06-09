#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
require_gcloud

destination="${1:-${SCRIPT_DIR}/downloads}"
mkdir -p "${destination}"

gcloud_base compute scp \
  --recurse \
  "${VM_NAME}:~/${REMOTE_DIR}/outputs" \
  "${destination}/" \
  --zone="${ZONE}" \
  --quiet

echo "Downloaded outputs to ${destination}."
