#!/bin/bash

# -----------------------------
# Environment Setup Script for Vast.ai
# -----------------------------

# 1. Clone the repository into /workspace
REPO_SSH="git@github.com:urbansirca/Subject-Independent_Meta-Learning_EEG.git"
REPO_HTTPS="https://github.com/urbansirca/Subject-Independent_Meta-Learning_EEG.git"
DEST_DIR="/workspace/Subject-Independent_Meta-Learning_EEG"

if [ ! -d "$DEST_DIR" ]; then
    echo "Cloning repository into /workspace..."
    git clone $REPO_SSH $DEST_DIR || git clone $REPO_HTTPS $DEST_DIR
else
    echo "Repository already exists at $DEST_DIR. Skipping clone."
fi

cd $DEST_DIR || exit 1

# 2. Create conda environment
ENV_NAME="eeg"
echo "Creating conda environment '$ENV_NAME' with Python 3.10..."
conda create -n $ENV_NAME python=3.10 -y

# 3. Activate environment
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# 4. Ensure pip is installed
conda install pip -y

# 5. Install requirements with PyTorch CUDA 12.1
echo "Installing requirements..."
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# 6. Make sure the data folder exists
mkdir -p /workspace/Subject-Independent_Meta-Learning_EEG/data

echo "--------------------------------------------"
echo "Environment setup complete!"
echo "Activate the environment with: conda activate $ENV_NAME"
echo "Repository is at: $DEST_DIR"
echo "Data folder is ready at: $DEST_DIR/data/"
echo "--------------------------------------------"