# script per eseguire n_runs = 3*2*10 = 60


set -euo pipefail


# to change
WANDB_PROJECT="daje-rt-ppo-swimmer"
ENV_NAME="Swimmer-v5"
CONFIG_NAME="ppo_swimmer"
N_EPOCHS=10
N_MINIBATCH=16
N_STEPS=1024
CPU_SET="22"
NORM_REW="False"
TOT_TIMESTEPS="1000000"


# fixed
WANDB_ENTITY="alessandro-montenegro-polimi"
OUTPUT_DIR="/work/fis1/RT-DeepRL/outputs"
TAGS='["K_fixed","nb_fixed","nu_fixed","wppo_3"]'
SEEDS="0,1,2,3,4,5,6,7,8,9"


for w in 2 4 8; do

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
    experiment.window_size="${w}" \
    experiment.weight_type=bh,naive \
    \
    experiment.seed="${SEEDS}"

done

