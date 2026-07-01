#!/bin/bash
set -e

CORES="0-20"  # comma-separated or range: "0,1,2,3" or "0-47"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
SCRIPT="$ROOT/examples/learn_test_grpo.py"

echo "Running learn_test_grpo.py on cores: $CORES"
HF_HUB_OFFLINE=1 taskset -c "$CORES" "$PYTHON" "$SCRIPT"
