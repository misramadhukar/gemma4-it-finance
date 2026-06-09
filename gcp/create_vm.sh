#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
require_gcloud

gcloud_base services enable compute.googleapis.com

if [[ "${IMAGE_FAMILY}" == "ubuntu-2204-lts-amd64" ]]; then
  echo "IMAGE_FAMILY=ubuntu-2204-lts-amd64 is invalid." >&2
  echo "Change it in gcp/config.env to IMAGE_FAMILY=ubuntu-2204-lts." >&2
  exit 1
fi

if ! image_name="$(
  gcloud compute images describe-from-family "${IMAGE_FAMILY}" \
    --project="${IMAGE_PROJECT}" \
    --format="value(name)"
)"; then
  echo "Could not resolve image family ${IMAGE_PROJECT}/${IMAGE_FAMILY}." >&2
  echo "Check IMAGE_FAMILY and IMAGE_PROJECT in gcp/config.env." >&2
  exit 1
fi

if [[ -z "${image_name}" ]]; then
  echo "Image family ${IMAGE_PROJECT}/${IMAGE_FAMILY} returned no image." >&2
  exit 1
fi

echo "Using image ${IMAGE_PROJECT}/${image_name}."

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
  "--image=${image_name}"
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
