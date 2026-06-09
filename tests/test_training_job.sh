#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
temporary_roots=()

cleanup() {
  local root
  for root in "${temporary_roots[@]}"; do
    if [[ -n "${root}" && -d "${root}" && "${root}" == "${TMPDIR:-/tmp}/"* ]]; then
      rm -rf -- "${root}"
    fi
  done
}
trap cleanup EXIT

for expected_exit_code in 0 7; do
  test_root="$(mktemp -d)"
  temporary_roots+=("${test_root}")
  mkdir -p "${test_root}/gcp"

  cp \
    "${REPO_ROOT}/gcp/train_common.sh" \
    "${REPO_ROOT}/gcp/training_job.sh" \
    "${test_root}/gcp/"

  cat >"${test_root}/gcp/run_training.sh" <<EOF
#!/usr/bin/env bash
exit ${expected_exit_code}
EOF
  chmod +x "${test_root}/gcp/run_training.sh"

  cat >"${test_root}/gcp/train.env" <<'EOF'
MODEL_ID=test-model
DATASET_ID=test-dataset
OUTPUT_DIR=outputs/test
AUTO_SHUTDOWN=false
SHUTDOWN_ON_FAILURE=true
NOTIFICATION_URL=
EOF

  set +e
  TRAIN_CONFIG="${test_root}/gcp/train.env" \
    bash "${test_root}/gcp/training_job.sh"
  actual_exit_code=$?
  set -e

  if (( actual_exit_code != expected_exit_code )); then
    echo "Expected exit ${expected_exit_code}, got ${actual_exit_code}." >&2
    exit 1
  fi

  python3 - \
    "${test_root}/outputs/training_job_status.json" \
    "${expected_exit_code}" <<'PY'
import json
import sys
from pathlib import Path

status_path = Path(sys.argv[1])
expected_exit_code = int(sys.argv[2])
status = json.loads(status_path.read_text(encoding="utf-8"))

assert status["exit_code"] == expected_exit_code
assert status["status"] == (
    "completed" if expected_exit_code == 0 else "failed"
)
assert status["output_dir"] == "outputs/test"
PY
done

test_root="$(mktemp -d)"
temporary_roots+=("${test_root}")
mkdir -p "${test_root}/bin" "${test_root}/gcp"

cp \
  "${REPO_ROOT}/gcp/train_common.sh" \
  "${REPO_ROOT}/gcp/training_job.sh" \
  "${test_root}/gcp/"

cat >"${test_root}/gcp/run_training.sh" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
chmod +x "${test_root}/gcp/run_training.sh"

cat >"${test_root}/bin/curl" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >"${CURL_LOG}"
EOF
cat >"${test_root}/bin/sudo" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >"${SUDO_LOG}"
EOF
chmod +x "${test_root}/bin/curl" "${test_root}/bin/sudo"

cat >"${test_root}/gcp/train.env" <<'EOF'
MODEL_ID=test-model
DATASET_ID=test-dataset
OUTPUT_DIR=outputs/test
AUTO_SHUTDOWN=true
SHUTDOWN_ON_FAILURE=true
AUTO_SHUTDOWN_DELAY_MINUTES=2
NOTIFICATION_KIND=ntfy
NOTIFICATION_URL=https://ntfy.sh/test-topic
EOF

PATH="${test_root}/bin:${PATH}" \
  CURL_LOG="${test_root}/curl.log" \
  SUDO_LOG="${test_root}/sudo.log" \
  TRAIN_CONFIG="${test_root}/gcp/train.env" \
  bash "${test_root}/gcp/training_job.sh"

grep -q "https://ntfy.sh/test-topic" "${test_root}/curl.log"
grep -q "shutdown -h +2" "${test_root}/sudo.log"

echo "Training wrapper success/failure tests passed."
