#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
require_gcloud

region="${ZONE%-*}"
candidate_zones=("${ZONE}")
for suffix in a b c; do
  regional_zone="${region}-${suffix}"
  if [[ "${regional_zone}" != "${ZONE}" ]]; then
    candidate_zones+=("${regional_zone}")
  fi
done
found=0

echo "Checking ${MACHINE_TYPE} in ${region}..."
printf "%-20s %-20s %-8s %-10s\n" "ZONE" "MACHINE_TYPE" "VCPUS" "MEMORY_MB"

for candidate_zone in "${candidate_zones[@]}"; do
  output=""
  for attempt in 1 2 3; do
    if output="$(
      gcloud_base compute machine-types describe "${MACHINE_TYPE}" \
        --zone="${candidate_zone}" \
        --format="value(zone.basename(),name,guestCpus,memoryMb)" \
        2>/dev/null
    )"; then
      read -r returned_zone name vcpus memory_mb <<<"${output}"
      printf "%-20s %-20s %-8s %-10s\n" \
        "${returned_zone}" "${name}" "${vcpus}" "${memory_mb}"
      found=1
      break
    fi

    if (( attempt < 3 )); then
      sleep $((attempt * 3))
    fi
  done
done

if (( found == 0 )); then
  echo "Could not describe ${MACHINE_TYPE} in ${region}." >&2
  echo "Verify billing/API access with:" >&2
  echo "  gcloud services list --enabled --project=${PROJECT_ID} --filter='config.name=compute.googleapis.com'" >&2
  echo "  gcloud compute machine-types describe ${MACHINE_TYPE} --zone=${ZONE} --project=${PROJECT_ID}" >&2
  exit 1
fi
