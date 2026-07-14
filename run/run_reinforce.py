import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
import wandb
from wandb.integration.sb3 import WandbCallback
import hydra
from omegaconf import DictConfig, OmegaConf
import torch.nn as nn

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import envs  # triggers the registration of new envs
from algorithms import Reinforce


@hydra.main(version_base=None, config_path="configs", config_name="conf_reinforce")
def main(cfg: DictConfig):
    exp = cfg.experiment

    base_name = (
        f"REINFORCE Ne={exp.n_envs} H={exp.n_steps} "
        f"lr={exp.learning_rate} γ={exp.gamma} "
        f"ent={exp.ent_coef} norm_G={exp.normalize_returns}"
    )

    conf = OmegaConf.to_container(cfg, resolve=True)
    conf["group"] = base_name
    run = wandb.init(
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

    # Parse policy kwargs (activation_fn must be converted from string to class)
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

    model = Reinforce(
        policy=exp.policy_type,
        env=env,
        learning_rate=exp.learning_rate,
        n_steps=exp.n_steps,
        gamma=exp.gamma,
        max_grad_norm=exp.max_grad_norm,
        ent_coef=exp.ent_coef,
        normalize_returns=exp.normalize_returns,
        use_sde=exp.use_sde,
        sde_sample_freq=exp.sde_sample_freq,
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

    if exp.eval_freq is not None:
        callbacks = CallbackList([eval_callback, wandb_callback])
    else:
        callbacks = CallbackList([wandb_callback])

    model.learn(
        total_timesteps=int(exp.total_timesteps),
        progress_bar=True,
        callback=callbacks,
    )

    env_name = str(exp.env_name).split("-")[0]
    model.save(f"{exp.dir_name}/{env_name}_REINFORCE_{run.id}")

    if exp.render:
        eval_env_render = gym.make(exp.env_name, render_mode="human")
        obs, _ = eval_env_render.reset()
        for _ in range(1000):
            obs_input = env.normalize_obs(obs) if exp.normalize_obs else obs
            action, _ = model.predict(obs_input, deterministic=True)
            obs, _, terminated, truncated, _ = eval_env_render.step(action)
            print(f"Obs: {obs}, Action: {action}, Terminated: {terminated}, Truncated: {truncated}")
            if terminated or truncated:
                obs, _ = eval_env_render.reset()
        eval_env_render.close()

    wandb.finish()


if __name__ == "__main__":
    main()
