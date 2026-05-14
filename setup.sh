#!/bin/bash

echo "Starting environment setup..."

ENV_NAME="rtpg"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env create -f environment.yml -y
conda activate "$ENV_NAME"
grep -v "file://" requirements.txt > requirements.txt
pip install -r requirements.txt
conda deactivate

echo "Setup complete!"
echo "To start working, just run: conda activate $ENV_NAME"