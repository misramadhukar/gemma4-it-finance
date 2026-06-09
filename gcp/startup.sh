#!/usr/bin/env bash
set -euxo pipefail

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  curl \
  git \
  jq \
  python3-pip \
  python3-venv \
  tmux

if ! nvidia-smi >/dev/null 2>&1; then
  mkdir -p /opt/google/cuda-installer
  cd /opt/google/cuda-installer
  curl -fSsL -O \
    https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz
  python3 cuda_installer.pyz install_driver \
    --installation-mode=repo \
    --installation-branch=prod
fi

if nvidia-smi >/dev/null 2>&1; then
  touch /var/tmp/gemma4-startup-complete
else
  echo "NVIDIA driver installation is still in progress; startup will retry after reboot."
fi
