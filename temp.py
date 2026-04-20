import os
from pathlib import Path

import gymnasium as gym
import torch as th
import torch.nn as nn
import wandb

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from wandb.integration.sb3 import WandbCallback

# ============================================================
# PPO training script rewritten with the parameters from image
# ============================================================

# =========================
# USER PATHS / RUN CONFIG
# =========================
OUTPUT_DIR = "/work/fis1/RT-DeepRL/outputs"   # <-- cambia questo
WANDB_PROJECT = "debug-ppo-tuned"
WANDB_ENTITY = None                       # es: "tuo-username" oppure None
WANDB_RUN_NAME = "test_01"
WANDB_TAGS = []

# =========================
# General config
# =========================
SEED = 0
ENV_ID = "Hopper-v5"
TOTAL_TIMESTEPS = int(1_000_000)

EVAL_FREQ = 50_000
EVAL_EPISODES = 50
N_EVAL_ENVS = 10
NUM_THREADS = 2
VERBOSE = 0

# =========================
# Parameters from the image
# =========================
N_ENVS = 1

# Batch size N = 2048 in the table
# In SB3 PPO, rollout size = n_envs * n_steps
N_STEPS = 2048

# Minibatches per epoch = 32
# so minibatch size = 2048 / 32 = 64
BATCH_SIZE = 64

# Epochs per update = 10
N_EPOCHS = 10

# General
GAMMA = 0.995
GAE_LAMBDA = 0.97
LEARNING_RATE = 3e-4
CLIP_RANGE = 0.2

# Not specified in the image, keep SB3 defaults unless you want otherwise
ENT_COEF = 0.0
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

# Initial policy std. deviation multiple = 1.0
# For Gaussian policies in SB3: std = exp(log_std_init)
# Therefore std = 1.0 -> log_std_init = 0.0
POLICY_KWARGS = dict(
    activation_fn=nn.Tanh,
    net_arch=dict(pi=[64, 64], vf=[64, 64]),
    log_std_init=0.0,   # approximate match, not exact implementation match
)

# =========================
# Normalization
# Keep configurable as before
# =========================
NORM_OBS = True
NORM_REWARD = False


def make_env(env_id: str, rank: int, seed: int):
    def _init():
        env = gym.make(env_id)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        try:
            env.action_space.seed(seed + rank)
        except Exception:
            pass
        return env
    return _init


def build_train_env():
    env = DummyVecEnv([make_env(ENV_ID, i, SEED) for i in range(N_ENVS)])
    env = VecNormalize(
        env,
        norm_obs=NORM_OBS,
        norm_reward=NORM_REWARD,
        training=True,
    )
    return env


def build_eval_env(train_env: VecNormalize):
    eval_env = DummyVecEnv(
        [make_env(ENV_ID, 10_000 + i, SEED) for i in range(N_EVAL_ENVS)]
    )
    eval_env = VecNormalize(
        eval_env,
        norm_obs=NORM_OBS,
        norm_reward=NORM_REWARD,
        training=False,
    )

    # Share normalization statistics with training env
    eval_env.obs_rms = train_env.obs_rms
    if hasattr(train_env, "ret_rms"):
        eval_env.ret_rms = train_env.ret_rms

    return eval_env


def main():
    # ---------------------------------------------------------
    # Paths
    # ---------------------------------------------------------
    output_dir = Path(OUTPUT_DIR)
    tb_dir = output_dir / "tensorboard"
    checkpoints_dir = output_dir / "checkpoints"
    best_model_dir = output_dir / "best_model"
    model_dir = output_dir / "model"
    vecnorm_dir = output_dir / "vecnormalize"
    eval_dir = output_dir / "eval"

    for d in [output_dir, tb_dir, checkpoints_dir, best_model_dir, model_dir, vecnorm_dir, eval_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Reproducibility / threads
    # ---------------------------------------------------------
    th.set_num_threads(NUM_THREADS)
    set_random_seed(SEED)

    # ---------------------------------------------------------
    # W&B init
    # ---------------------------------------------------------
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=WANDB_RUN_NAME,
        dir=str(output_dir),
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=True,
        tags=WANDB_TAGS,
        config={
            "env_id": ENV_ID,
            "seed": SEED,
            "total_timesteps": TOTAL_TIMESTEPS,
            "eval_freq": EVAL_FREQ,
            "eval_episodes": EVAL_EPISODES,
            "n_eval_envs": N_EVAL_ENVS,
            "num_threads": NUM_THREADS,
            "n_envs": N_ENVS,
            "n_steps": N_STEPS,
            "rollout_batch_size": N_ENVS * N_STEPS,
            "minibatches_per_epoch": 32,
            "batch_size": BATCH_SIZE,
            "n_epochs": N_EPOCHS,
            "gamma": GAMMA,
            "gae_lambda": GAE_LAMBDA,
            "learning_rate": LEARNING_RATE,
            "clip_range": CLIP_RANGE,
            "ent_coef": ENT_COEF,
            "vf_coef": VF_COEF,
            "max_grad_norm": MAX_GRAD_NORM,
            "policy": "MlpPolicy",
            "policy_kwargs": {
                "log_std_init": 0.0,
            },
            "optimizer_note": (
                "SB3 PPO uses a single Adam optimizer and one shared learning rate "
                "for both policy and value networks."
            ),
            "normalize": True,
            "normalize_kwargs": {
                "norm_obs": NORM_OBS,
                "norm_reward": NORM_REWARD,
            },
        },
    )

    # ---------------------------------------------------------
    # Environments
    # ---------------------------------------------------------
    train_env = build_train_env()
    eval_env = build_eval_env(train_env)

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        policy_kwargs=POLICY_KWARGS,
        seed=SEED,
        verbose=VERBOSE,
        tensorboard_log=str(tb_dir),
        device="cpu",
    )

    # ---------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_dir),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        render=False,
    )

    wandb_callback = WandbCallback(
        model_save_path=str(checkpoints_dir),
        model_save_freq=0,
        verbose=VERBOSE,
    )

    callbacks = CallbackList([eval_callback, wandb_callback])

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        log_interval=1,
        progress_bar=True,
    )

    # ---------------------------------------------------------
    # Final saves
    # ---------------------------------------------------------
    model.save(str(model_dir / "ppo_hopper_image_defaults"))
    train_env.save(str(vecnorm_dir / "vecnormalize.pkl"))

    wandb.save(str(model_dir / "ppo_hopper_image_defaults.zip"))
    wandb.save(str(vecnorm_dir / "vecnormalize.pkl"))

    run.finish()

    print(f"Training completed. All local outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()