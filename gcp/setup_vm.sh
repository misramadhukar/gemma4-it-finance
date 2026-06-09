#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable. Wait for the startup script or reboot the VM." >&2
  exit 1
fi

nvidia-smi

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  python3-pip \
  python3-venv \
  tmux

python3 -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${REPO_ROOT}/requirements-cuda.txt"
python -m pip install -r "${REPO_ROOT}/requirements.txt"
python -m pip check
hf --help >/dev/null

python - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"torch CUDA runtime={torch.version.cuda}")
print(f"CUDA available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access the GPU.")
print(f"GPU={torch.cuda.get_device_name(0)}")
print(f"VRAM={torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
print(f"bf16 supported={torch.cuda.is_bf16_supported()}")
PY

python -m pip freeze > "${REPO_ROOT}/environment-lock.txt"

if [[ ! -f "${SCRIPT_DIR}/train.env" ]]; then
  cp "${SCRIPT_DIR}/train.env.example" "${SCRIPT_DIR}/train.env"
fi

echo
echo "Environment setup complete."
echo "Next:"
echo "  source .venv/bin/activate"
echo "  hf auth login"
echo "  python preflight.py"
