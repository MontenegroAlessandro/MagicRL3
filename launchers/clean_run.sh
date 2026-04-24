#!/usr/bin/env bash
set -euo pipefail


taskset -c 16 python3 examples/learn_ppo_test.py -m \
  --config-name ppo_hopper \
  \
  experiment.dir_name="/work/fis1/RT-DeepRL/outputs" \
  wandb.entity="alessandro-montenegro-polimi" \
  wandb.project="temp" \
  wandb.tags="[]" \
  experiment.env_name="Hopper-v5" \
  \
  experiment.norm_reward=False \
  experiment.total_timesteps=1000000 \
  \
  experiment.n_minibatch=8 \
  \
  experiment.window_size=8 \
  experiment.weight_type=bh \
  \
  experiment.seed=0 \

