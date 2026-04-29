# script per eseguire n_runs = 3*2*10 = 60


set -euo pipefail


# to change
WANDB_PROJECT="temp"
ENV_NAME="Hopper-v5"
CONFIG_NAME="ppo_hopper"
N_EPOCHS=10
N_MINIBATCH=32
N_STEPS=2048
NORM_REW="False"
TOT_TIMESTEPS="300000"
CPU_SET="1"


# fixed
WANDB_ENTITY="alessandro-montenegro-polimi"
OUTPUT_DIR="/work/fis1/RT-DeepRL/outputs"
TAGS='["K_fixed","B_fixed","wppo_1"]'
SEEDS="0,1,2,3,4,5"


for w in 2 4 8; do

  w_N_MINIBATCH=$((N_MINIBATCH * w))

  taskset -c "${CPU_SET}" python3 examples/learn_ppo_test.py -m \
    --config-name "${CONFIG_NAME}" \
    \
    experiment.env_name="${ENV_NAME}" \
    experiment.dir_name="${OUTPUT_DIR}" \
    wandb.entity="${WANDB_ENTITY}" \
    wandb.project="${WANDB_PROJECT}" \
    wandb.tags="${TAGS}" \
    \
    experiment.norm_reward="${NORM_REW}" \
    experiment.total_timesteps="${TOT_TIMESTEPS}" \
    \
    experiment.n_minibatch="${w_N_MINIBATCH}" \
    \
    experiment.window_size="${w}" \
    experiment.weight_type=bh,naive \
    \
    experiment.seed="${SEEDS}"

done

