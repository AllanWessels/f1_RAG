#!/usr/bin/env bash
# Quick smoke check that nvidia-container-toolkit is configured and Docker
# can run a CUDA container with GPU passthrough. Exits 0 if working.
#
# If this fails on a machine that has an NVIDIA GPU + driver, install the
# toolkit:
#   sudo apt-get install -y nvidia-container-toolkit
#   sudo nvidia-ctk runtime configure --runtime=docker
#   sudo systemctl restart docker
# (or distro-equivalent — see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)

set -u

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[gpu-check] nvidia-smi not found on host. Install the NVIDIA driver first." >&2
    exit 2
fi

echo "[gpu-check] host driver/GPU:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo
echo "[gpu-check] running 'docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi' ..."

if docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi --query-gpu=name,memory.total --format=csv 2>&1; then
    echo
    echo "[gpu-check] GPU passthrough is working ✓"
    exit 0
fi

echo
echo "[gpu-check] FAILED. The host can see the GPU but Docker cannot pass it through." >&2
echo "             Install nvidia-container-toolkit and re-run." >&2
exit 1
