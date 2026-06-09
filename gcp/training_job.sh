#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=train_common.sh
source "${SCRIPT_DIR}/train_common.sh"

status_file="${REPO_ROOT}/outputs/training_job_status.json"
started_at="$(date --utc +%Y-%m-%dT%H:%M:%SZ)"
host_name="$(hostname)"

send_notification() {
  local message="$1"
  local payload=""

  if [[ -z "${NOTIFICATION_URL:-}" ]]; then
    return 0
  fi

  case "${NOTIFICATION_KIND}" in
    ntfy)
      curl --fail --silent --show-error --max-time 20 \
        -H "Title: Gemma fine-tuning" \
        -H "Priority: default" \
        --data-binary "${message}" \
        "${NOTIFICATION_URL}" >/dev/null
      ;;
    slack)
      payload="$(python3 -c 'import json,sys; print(json.dumps({"text": sys.argv[1]}))' "${message}")"
      curl --fail --silent --show-error --max-time 20 \
        -H "Content-Type: application/json" \
        --data-binary "${payload}" \
        "${NOTIFICATION_URL}" >/dev/null
      ;;
    discord)
      payload="$(python3 -c 'import json,sys; print(json.dumps({"content": sys.argv[1]}))' "${message}")"
      curl --fail --silent --show-error --max-time 20 \
        -H "Content-Type: application/json" \
        --data-binary "${payload}" \
        "${NOTIFICATION_URL}" >/dev/null
      ;;
    generic)
      payload="$(python3 -c 'import json,sys; print(json.dumps({"message": sys.argv[1]}))' "${message}")"
      curl --fail --silent --show-error --max-time 20 \
        -H "Content-Type: application/json" \
        --data-binary "${payload}" \
        "${NOTIFICATION_URL}" >/dev/null
      ;;
  esac
}

write_status() {
  local exit_code="$1"
  local status="$2"
  local finished_at="$3"

  mkdir -p "$(dirname "${status_file}")"
  STATUS_FILE="${status_file}" \
    JOB_STATUS="${status}" \
    JOB_EXIT_CODE="${exit_code}" \
    JOB_STARTED_AT="${started_at}" \
    JOB_FINISHED_AT="${finished_at}" \
    JOB_HOST="${host_name}" \
    JOB_OUTPUT_DIR="${OUTPUT_DIR}" \
    python3 - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["STATUS_FILE"])
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(
    json.dumps(
        {
            "status": os.environ["JOB_STATUS"],
            "exit_code": int(os.environ["JOB_EXIT_CODE"]),
            "started_at": os.environ["JOB_STARTED_AT"],
            "finished_at": os.environ["JOB_FINISHED_AT"],
            "host": os.environ["JOB_HOST"],
            "output_dir": os.environ["JOB_OUTPUT_DIR"],
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
temporary.replace(path)
PY
}

finish_job() {
  local exit_code="$1"
  local status="failed"
  local finished_at
  local message
  local should_shutdown="false"

  trap - EXIT INT TERM

  if (( exit_code == 0 )); then
    status="completed"
  fi
  finished_at="$(date --utc +%Y-%m-%dT%H:%M:%SZ)"

  if ! write_status "${exit_code}" "${status}" "${finished_at}"; then
    echo "Warning: could not write ${status_file}." >&2
  fi

  message="Gemma training ${status} on ${host_name}. Exit code: ${exit_code}. Output: ${OUTPUT_DIR}. Finished: ${finished_at}."
  if ! send_notification "${message}"; then
    echo "Warning: training notification could not be delivered." >&2
  fi

  if [[ "${AUTO_SHUTDOWN}" == "true" ]]; then
    if [[ "${status}" == "completed" || "${SHUTDOWN_ON_FAILURE}" == "true" ]]; then
      should_shutdown="true"
    fi
  fi

  if [[ "${should_shutdown}" == "true" ]]; then
    echo "Scheduling VM shutdown in ${AUTO_SHUTDOWN_DELAY_MINUTES} minute(s)."
    if ! sudo shutdown -h "+${AUTO_SHUTDOWN_DELAY_MINUTES}" \
      "Gemma training ${status}; automatic cost-control shutdown."; then
      echo "Warning: automatic VM shutdown could not be scheduled." >&2
    fi
  fi

  exit "${exit_code}"
}

trap 'finish_job $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Training job started at ${started_at} on ${host_name}."
bash "${SCRIPT_DIR}/run_training.sh" "$@"
