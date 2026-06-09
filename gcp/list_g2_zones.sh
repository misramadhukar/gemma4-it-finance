#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
require_gcloud

gcloud_base compute machine-types list \
  --filter="name=${MACHINE_TYPE}" \
  --format="table(zone.basename():label=ZONE,name:label=MACHINE_TYPE,guestCpus:label=VCPUS,memoryMb:label=MEMORY_MB)"
