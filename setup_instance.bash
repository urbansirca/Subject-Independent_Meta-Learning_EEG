#!/bin/bash
set -euo pipefail

# -----------------------------
# Environment Setup Script for Vast.ai (RTX 50xx friendly)
# -----------------------------

REPO_SSH="git@github.com:urbansirca/Subject-Independent_Meta-Learning_EEG.git"
REPO_HTTPS="https://github.com/urbansirca/Subject-Independent_Meta-Learning_EEG.git"
DEST_DIR="/workspace/Subject-Independent_Meta-Learning_EEG"
ENV_NAME="eeg"

TORCH_INDEX_CU128="https://download.pytorch.org/whl/cu128"
TORCH_INDEX_CU124="https://download.pytorch.org/whl/cu124"

# ---- helper: compare versions (returns 0 if $1 >= $2) ----
version_ge () { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]; }

echo "==> Checking NVIDIA driver..."
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || true)"
if [[ -z "${DRIVER:-}" ]]; then
  echo "WARN: nvidia-smi not found or no GPU visible. Proceeding, but CUDA wheels won't work."
  DRIVER="0.0"
else
  echo "Driver version: $DRIVER"
fi

# Minimum driver versions for these CUDA runtimes
MIN_128="570.26"     # CUDA 12.8
MIN_124="550.54.14"  # CUDA 12.4

USE_INDEX="$TORCH_INDEX_CU128"
CUDA_TAG="cu128"
if ! version_ge "$DRIVER" "$MIN_128"; then
  echo "NOTE: Driver $DRIVER is below $MIN_128 -> falling back to CUDA 12.4 wheels."
  if version_ge "$DRIVER" "$MIN_124"; then
    USE_INDEX="$TORCH_INDEX_CU124"
    CUDA_TAG="cu124"
  else
    echo "ERROR: Driver ($DRIVER) is too old for CUDA 12.4 wheels ($MIN_124)."
    echo "Please choose a Vast.ai image/host with driver >= $MIN_124, or upgrade the driver."
    exit 1
  fi
fi

# 1) Clone repo
if [ ! -d "$DEST_DIR" ]; then
  echo "==> Cloning repository into $DEST_DIR ..."
  git clone "$REPO_SSH" "$DEST_DIR" || git clone "$REPO_HTTPS" "$DEST_DIR"
else
  echo "==> Repository already exists at $DEST_DIR. Skipping clone."
fi
cd "$DEST_DIR" || exit 1

# 2) Create / activate conda env
echo "==> Creating conda environment '$ENV_NAME' with Python 3.10 (if not exists)..."
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Env $ENV_NAME already exists."
else
  conda create -n "$ENV_NAME" python=3.10 -y
fi

echo "==> Activating environment..."
conda activate "$ENV_NAME"

# 3) Ensure fresh pip/setuptools/wheel
echo "==> Upgrading pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel

# 4) Install PyTorch first (GPU wheel), then the rest
echo "==> Installing PyTorch/vision/audio for $CUDA_TAG ..."
# remove any preinstalled torch* to avoid conflicts
python - <<'PY'
import sys, subprocess
pkgs = ["torch", "torchvision", "torchaudio"]
for p in pkgs:
    try:
        __import__(p)
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", p], check=False)
    except Exception:
        pass
PY

pip install torch torchvision torchaudio --index-url "$USE_INDEX"

# 5) Install project requirements, excluding any torch pins
REQS_FILE="requirements.txt"
if [ -f "$REQS_FILE" ]; then
  echo "==> Installing other requirements..."
  # filter out torch/torchvision/torchaudio lines (case-insensitive)
  awk 'BEGIN{IGNORECASE=1} !/^[[:space:]]*(torch|torchvision|torchaudio)([[:space:]=><].*)?$/' "$REQS_FILE" > requirements.notorch.txt
  # If the filtered file is empty or only comments, skip; else install
  if grep -q '[^[:space:]#]' requirements.notorch.txt; then
    pip install -r requirements.notorch.txt
  else
    echo "No non-torch requirements to install."
  fi
else
  echo "requirements.txt not found; skipping extras."
fi

# 6) Make sure the data folder exists
mkdir -p "$DEST_DIR/data"

# 7) Quick verification
echo "==> Verifying CUDA availability in PyTorch..."
python - <<'PY'
import torch, sys
print("Torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
if torch.cuda.is_available():
    print("CUDA is available:", torch.cuda.is_available())
    print("Device:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
    x = torch.randn(512,512, device='cuda')
    print("Tensor device:", x.device)
else:
    print("ERROR: CUDA not available. Check driver/index-url and instance image.", file=sys.stderr)
    sys.exit(1)
PY

echo "--------------------------------------------"
echo "Environment setup complete!"
echo "Activate with: conda activate $ENV_NAME"
echo "Repository: $DEST_DIR"
echo "Data folder: $DEST_DIR/data/"
echo "PyTorch wheel used: $CUDA_TAG ($USE_INDEX)"
echo "--------------------------------------------"
