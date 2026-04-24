# script per eseguire n_runs = 4*10 = 40


set -euo pipefail


# to change
WANDB_PROJECT="daje-rt-ppo-hopper"
ENV_NAME="Hopper-v5"
CONFIG_NAME="ppo_hopper"
N_EPOCHS=10
N_MINIBATCH=32
N_STEPS=2048
CPU_SET=""
NORM_REW="False"
TOT_TIMESTEPS="1000000"


# fixed
WANDB_ENTITY="alessandro-montenegro-polimi"
OUTPUT_DIR="/work/fis1/RT-DeepRL/outputs"
TAGS='["K_fixed","nb_fixed","B_fixed","ns_fixed","nu_fixed","baseline"]'
SEEDS="0,1,2,3,4,5,6,7,8,9"


for w in 1 2 4 8; do

  EPOCHS=$((w * N_EPOCHS))

  taskset -c "${CPU_SET}" python3 examples/learn_ppo_test.py -m \
    --config-name "${CONFIG_NAME}" \
    \
    experiment.env_name="${ENV_NAME}" \
    experiment.dir_name="${OUTPUT_DIR}" \
    wandb.entity="${WANDB_ENTITY}" \
    wandb.project="${WANDB_PROJECT}" \
    wandb.tags="${TAGS}" \
    experiment.window_size=1 \
    \
    experiment.norm_reward="${NORM_REW}" \
    experiment.total_timesteps="${TOT_TIMESTEPS}" \
    \
    experiment.n_epochs="${EPOCHS}" \
    \
    experiment.seed="${SEEDS}"

done

