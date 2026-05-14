#!/bin/bash
set -e  # Tells bash to stop immediately if any command fails

echo "Starting environment setup..."

ENV_NAME="${1:-rtpg}"
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "Wiping old environment if it exists to ensure a clean slate..."
conda env remove --name "$ENV_NAME" -y || true 

echo "Creating new environment..."
conda env create -f environment.yml --name "$ENV_NAME" -y

conda activate "$ENV_NAME"
pip install -r requirements.txt
conda deactivate

echo "Setup complete!"
echo "To start working, just run: conda activate $ENV_NAME"