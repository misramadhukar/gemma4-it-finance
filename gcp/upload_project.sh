#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
require_gcloud

archive="$(mktemp --suffix=.tar.gz)"
remote_archive="/tmp/gemma4-fine-tune.tar.gz"
trap 'rm -f "${archive}"' EXIT

tar -czf "${archive}" \
  --exclude="gcp/config.env" \
  --exclude="gcp/train.env" \
  --exclude="gcp/downloads" \
  -C "${REPO_ROOT}" \
  README.md \
  PREPROD_CHECKLIST.md \
  requirements-cuda.txt \
  requirements.txt \
  train.py \
  infer.py \
  merge_adapter.py \
  evaluate.py \
  preflight.py \
  dataset_utils.py \
  hub_utils.py \
  sft_utils.py \
  sample_article.txt \
  tests \
  gcp

gcloud_base compute scp "${archive}" \
  "${VM_NAME}:${remote_archive}" \
  --zone="${ZONE}" \
  --quiet

gcloud_base compute ssh "${VM_NAME}" \
  --zone="${ZONE}" \
  --quiet \
  --command="mkdir -p ~/${REMOTE_DIR} && tar -xzf ${remote_archive} -C ~/${REMOTE_DIR} && rm -f ${remote_archive}"

echo "Uploaded project to ~/${REMOTE_DIR}."
echo "Next: bash gcp/connect.sh"
