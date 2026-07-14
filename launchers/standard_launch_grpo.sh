#!/bin/bash
set -e

CORES="0-7"
NUM_THREADS=8           # torch intraop threads (experiment.num_threads)

# parametri più comuni
TASK=gsm8k              # simple_sums | gsm8k
DTYPE=float32           # float32 | bfloat16
N_SAMPLES=200
BATCH_SIZE=16           # per_device_train_batch_size
NUM_GENERATIONS=4
LR=1.0e-7
BETA=0.0                # peso penalità KL
EPSILON=0.2             # clipping del ratio
MAX_STEPS=-1            # -1 = nessun limite
MAX_COMPLETION_LENGTH=128
TEMPERATURE=1.0
SEED=42

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Running run_grpo.py on cores: $CORES ($NUM_THREADS threads, dtype=$DTYPE)"
HF_HUB_OFFLINE=1 taskset -c "$CORES" "$ROOT/.venv/bin/python" "$ROOT/run/run_grpo.py" \
    experiment.task_name=$TASK \
    experiment.num_threads=$NUM_THREADS \
    experiment.dtype=$DTYPE \
    experiment.seed=$SEED \
    experiment.n_samples=$N_SAMPLES \
    experiment.per_device_train_batch_size=$BATCH_SIZE \
    experiment.num_generations=$NUM_GENERATIONS \
    experiment.learning_rate=$LR \
    experiment.beta=$BETA \
    experiment.epsilon=$EPSILON \
    experiment.max_steps=$MAX_STEPS \
    experiment.max_completion_length=$MAX_COMPLETION_LENGTH \
    experiment.temperature=$TEMPERATURE \
    "$@"
