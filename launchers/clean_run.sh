#!/usr/bin/env bash
set -euo pipefail


taskset -c 36 python3 examples/learn_ppo_test.py -m \
  --config-name ppo_reacher \
  \
  experiment.dir_name="/work/fis1/RT-DeepRL/outputs" \
  wandb.entity="alessandro-montenegro-polimi" \
  wandb.project="temp" \
  wandb.tags="[]" \
  experiment.env_name="Reacher-v5" \
  \
  experiment.norm_reward=False \
  experiment.total_timesteps=1000000 \
  \
  experiment.window_size=1 \
  experiment.weight_type=naive \
  \
  experiment.on_policy_critic=false \
  \
  experiment.seed=0,1,2,3,4,5,6,7,8,9 \

