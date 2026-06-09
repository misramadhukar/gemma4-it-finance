#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PORT="${TENSORBOARD_PORT:-6006}"
SESSION="${TENSORBOARD_SESSION:-tensorboard}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv/bin/activate"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "TensorBoard session ${SESSION} is already running."
  exit 0
fi

printf -v command 'cd %q && exec tensorboard --logdir outputs --host 127.0.0.1 --port %q' \
  "${REPO_ROOT}" \
  "${PORT}"
tmux new-session -d -s "${SESSION}" "${command}"

echo "TensorBoard started on VM localhost:${PORT}."
echo "From Cloud Shell/local terminal run: bash gcp/connect.sh --tensorboard"
