#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
require_gcloud

gcloud_base services enable compute.googleapis.com

if gcloud_base compute instances describe "${VM_NAME}" --zone="${ZONE}" >/dev/null 2>&1; then
  echo "VM ${VM_NAME} already exists in ${ZONE}."
  exit 0
fi

create_args=(
  compute instances create "${VM_NAME}"
  "--zone=${ZONE}"
  "--machine-type=${MACHINE_TYPE}"
  "--boot-disk-size=${BOOT_DISK_SIZE}"
  "--boot-disk-type=${BOOT_DISK_TYPE}"
  "--image-family=${IMAGE_FAMILY}"
  "--image-project=${IMAGE_PROJECT}"
  "--metadata-from-file=startup-script=${SCRIPT_DIR}/startup.sh"
  "--metadata=enable-oslogin=TRUE"
  "--labels=workload=gemma4-finetune"
  "--no-service-account"
  "--no-scopes"
  "--no-shielded-secure-boot"
  "--shielded-vtpm"
  "--shielded-integrity-monitoring"
)

if [[ "${PROVISIONING_MODEL}" == "SPOT" ]]; then
  create_args+=(
    "--provisioning-model=SPOT"
    "--instance-termination-action=STOP"
  )
else
  create_args+=(
    "--provisioning-model=STANDARD"
    "--maintenance-policy=TERMINATE"
    "--restart-on-failure"
  )
fi

gcloud_base "${create_args[@]}"

echo
echo "Created ${VM_NAME}. Driver installation can take several minutes and may reboot it."
echo "Next: bash gcp/wait_ready.sh"
