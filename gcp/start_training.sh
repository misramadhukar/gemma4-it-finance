#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=train_common.sh
source "${SCRIPT_DIR}/train_common.sh"
session="${TMUX_SESSION}"
log_dir="${REPO_ROOT}/outputs/logs"
log_file="${log_dir}/training-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "${log_dir}"

if [[ "${SKIP_READINESS_CHECKS:-false}" != "true" ]]; then
  if [[ ! -f "${REPO_ROOT}/preflight_report.json" ]]; then
    echo "Missing preflight_report.json. Run: python preflight.py" >&2
    exit 1
  fi
  smoke_manifest="${REPO_ROOT}/outputs/smoke/run_manifest.json"
  if [[ ! -f "${smoke_manifest}" ]]; then
    echo "Missing smoke-test manifest. Run: bash gcp/smoke_test.sh" >&2
    exit 1
  fi
  smoke_status="$(
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("status", ""))' \
      "${smoke_manifest}"
  )"
  if [[ "${smoke_status}" != "completed" ]]; then
    echo "Smoke test status is ${smoke_status:-missing}, not completed." >&2
    exit 1
  fi
fi

if tmux has-session -t "${session}" 2>/dev/null; then
  echo "tmux session ${session} is already running."
  echo "Attach with: tmux attach -t ${session}"
  exit 1
fi

printf -v command 'cd %q && exec bash %q >> %q 2>&1' \
  "${REPO_ROOT}" \
  "${SCRIPT_DIR}/run_training.sh" \
  "${log_file}"

tmux new-session -d -s "${session}" "${command}"

echo "Training started in tmux session ${session}."
echo "Log: ${log_file}"
echo "Attach: tmux attach -t ${session}"
