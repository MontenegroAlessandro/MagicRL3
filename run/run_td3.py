import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.noise import NormalActionNoise
import wandb
from wandb.integration.sb3 import WandbCallback
import hydra
from omegaconf import DictConfig, OmegaConf
import torch.nn as nn

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import envs  # triggers the registration of new envs


@hydra.main(version_base=None, config_path="config", config_name="conf_td3")
def main(cfg: DictConfig):
    exp = cfg.experiment

    base_name = f"TD3 {exp.env_name} baseline"

    conf = OmegaConf.to_container(cfg, resolve=True)
    conf["group"] = base_name
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        config=conf,
        sync_tensorboard=cfg.wandb.sync_tensorboard,
        group=base_name,
        name=f"{base_name} seed={exp.seed}",
        tags=cfg.wandb.tags,
        dir=exp.dir_name,
    )

    # --- Training env ---
    env = make_vec_env(exp.env_name, n_envs=exp.n_envs, seed=exp.seed)
    env = VecNormalize(env, norm_reward=exp.normalize_reward, norm_obs=exp.normalize_obs, gamma=exp.gamma, training=True)

    # --- Evaluation env (shares obs stats, never normalizes rewards) ---
    eval_env = make_vec_env(exp.env_name, n_envs=1, seed=exp.seed + 1000)
    eval_env = VecNormalize(eval_env, norm_reward=False, norm_obs=exp.normalize_obs, gamma=exp.gamma, training=False)
    eval_env.obs_rms = env.obs_rms

    policy_kwargs = OmegaConf.to_container(exp.policy_kwargs, resolve=True) if exp.policy_kwargs is not None else None
    if policy_kwargs is not None and "activation_fn" in policy_kwargs:
        activation_map = {
            "relu":       nn.ReLU,
            "tanh":       nn.Tanh,
            "elu":        nn.ELU,
            "leaky_relu": nn.LeakyReLU,
            "selu":       nn.SELU,
            "gelu":       nn.GELU,
        }
        key = policy_kwargs["activation_fn"].lower()
        if key not in activation_map:
            raise ValueError(f"Unknown activation_fn '{key}'. Choose from: {list(activation_map)}")
        policy_kwargs["activation_fn"] = activation_map[key]

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=exp.action_noise_std * np.ones(n_actions))

    model = TD3(
        policy=exp.policy_type,
        env=env,
        learning_rate=exp.learning_rate,
        buffer_size=exp.buffer_size,
        learning_starts=exp.learning_starts,
        batch_size=exp.batch_size,
        tau=exp.tau,
        gamma=exp.gamma,
        train_freq=exp.train_freq,
        gradient_steps=exp.gradient_steps,
        action_noise=action_noise,
        replay_buffer_class=None,
        replay_buffer_kwargs=None,
        optimize_memory_usage=exp.optimize_memory_usage,
        policy_delay=exp.policy_delay,
        target_policy_noise=exp.target_policy_noise,
        target_noise_clip=exp.target_noise_clip,
        stats_window_size=exp.stats_window_size,
        policy_kwargs=policy_kwargs,
        verbose=exp.verbose,
        tensorboard_log=f"{exp.dir_name}/runs/{run.id}",
        seed=exp.seed,
        device=exp.device,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{exp.dir_name}/models/{run.id}",
        log_path=f"{exp.dir_name}/logs/{run.id}",
        eval_freq=exp.eval_freq,
        n_eval_episodes=exp.n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=0,
    )

    wandb_callback = WandbCallback(
        model_save_path=f"{exp.dir_name}/models/{run.id}",
        verbose=2,
    )

    callbacks = CallbackList([eval_callback, wandb_callback]) if exp.eval_freq is not None else CallbackList([wandb_callback])

    model.learn(
        total_timesteps=int(exp.total_timesteps),
        progress_bar=True,
        callback=callbacks,
    )

    env_name = str(exp.env_name).split("-")[0]
    model.save(f"{exp.dir_name}/{env_name}_TD3_{run.id}")

    if exp.render:
        eval_env_render = gym.make(exp.env_name, render_mode="human")
        obs, _ = eval_env_render.reset()
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = eval_env_render.step(action)
            if terminated or truncated:
                obs, _ = eval_env_render.reset()
        eval_env_render.close()

    wandb.finish()


if __name__ == "__main__":
    main()
