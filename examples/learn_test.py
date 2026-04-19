import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
import wandb
from wandb.integration.sb3 import WandbCallback
import hydra
from omegaconf import DictConfig, OmegaConf

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from algorithms.rt_ppo import RT_PPO
from buffers.buffers import MultiRolloutBuffer
import torch.nn as nn

@hydra.main(version_base=None, config_path=".", config_name="conf")
def main(cfg: DictConfig):
    exp = cfg.experiment

    # Derive batch_size from n_minibatch so we have direct control over gradient steps per epoch.
    # batch_size is always based on the on-policy data size (n_steps * n_envs * window_size).
    if "batch_size" in exp and exp.batch_size is not None:
        batch_size = exp.batch_size
        n_minibatch_effective = (exp.n_steps * exp.n_envs * exp.window_size) // batch_size
    else:
        batch_size = (exp.n_steps * exp.n_envs * exp.window_size) // exp.n_minibatch
        n_minibatch_effective = exp.n_minibatch

    # logger
    if exp.window_size > 1:
        if not exp.sequential_window_training and not exp.fresh_adv:
            base_name = f"RT-PPO w={exp.window_size} envs={exp.n_envs} steps={exp.n_steps} epochs={exp.n_epochs} on_policy_critic={exp.on_policy_critic} weight_type={exp.weight_type} kl_target={exp.target_kl} n_minibatch={n_minibatch_effective} batch_size={batch_size}"
        elif exp.sequential_window_training and not exp.fresh_adv:
            base_name = f"RT-PPO SEQ w={exp.window_size} envs={exp.n_envs} steps={exp.n_steps} epochs={exp.n_epochs} on_policy_critic={exp.on_policy_critic} weight_type={exp.weight_type} kl_target={exp.target_kl} n_minibatch={n_minibatch_effective} batch_size={batch_size}"
        elif exp.sequential_window_training and exp.fresh_adv:
            base_name = f"RT-PPO FRESH w={exp.window_size} envs={exp.n_envs} steps={exp.n_steps} epochs={exp.n_epochs} on_policy_critic={exp.on_policy_critic} weight_type={exp.weight_type} kl_target={exp.target_kl} n_minibatch={n_minibatch_effective} batch_size={batch_size}"
        else:
            base_name = f"RT-PPO SEQ FRESH w={exp.window_size} envs={exp.n_envs} steps={exp.n_steps} epochs={exp.n_epochs} on_policy_critic={exp.on_policy_critic} weight_type={exp.weight_type} kl_target={exp.target_kl} n_minibatch={n_minibatch_effective} batch_size={batch_size}"
    else:
        base_name = f"PPO envs={exp.n_envs} steps={exp.n_steps} epochs={exp.n_epochs} kl_target={exp.target_kl} n_minibatch={n_minibatch_effective} batch_size={batch_size}"
    conf = OmegaConf.to_container(cfg, resolve=True)
    conf["group"] = base_name
    run = wandb.init(
        project=cfg.wandb.project,
        config=conf,
        sync_tensorboard=cfg.wandb.sync_tensorboard,
        group=base_name,
        name=f"{base_name} seed={exp.seed}",
    )

    # --- Training env ---
    env = make_vec_env(exp.env_name, n_envs=exp.n_envs, seed=exp.seed)
    env = VecNormalize(env, norm_reward=exp.normalize_reward, norm_obs=exp.normalize_obs)

    # parse policy args
    policy_kwargs = OmegaConf.to_container(exp.policy_kwargs, resolve=True) if exp.policy_kwargs is not None else None

    if exp.window_size == 1:
        model = PPO(
            policy=exp.policy_type,
            env=env,
            learning_rate=exp.learning_rate,
            n_steps=exp.n_steps,
            batch_size=batch_size,
            n_epochs=exp.n_epochs,
            gamma=exp.gamma,
            gae_lambda=exp.gae_lambda,
            clip_range=exp.clip_range,
            clip_range_vf=exp.clip_range_vf,
            normalize_advantage=exp.normalize_advantage,
            ent_coef=exp.ent_coef,
            vf_coef=exp.vf_coef,
            max_grad_norm=exp.max_grad_norm,
            use_sde=exp.use_sde,
            sde_sample_freq=exp.sde_sample_freq,
            target_kl=exp.target_kl,
            stats_window_size=exp.stats_window_size,
            policy_kwargs=policy_kwargs,
            verbose=exp.verbose,
            tensorboard_log=f"{exp.dir_name}/runs/{run.id}",
            seed=exp.seed,
            device=exp.device
        )
    else:
        model = RT_PPO(
            on_policy_critic=exp.on_policy_critic,
            rollout_buffer_class=MultiRolloutBuffer,
            rollout_buffer_kwargs=dict(
                window_size=exp.window_size,
                use_bh=(exp.weight_type == "bh"),
            ),
            is_weight_type=exp.weight_type,
            sequential_window_training=exp.sequential_window_training,
            fresh_adv=exp.fresh_adv,
            # Old PPO args
            policy=exp.policy_type,
            env=env,
            learning_rate=exp.learning_rate,
            n_steps=exp.n_steps,
            batch_size=batch_size,
            n_epochs=exp.n_epochs,
            gamma=exp.gamma,
            gae_lambda=exp.gae_lambda,
            clip_range=exp.clip_range,
            clip_range_vf=exp.clip_range_vf,
            normalize_advantage=exp.normalize_advantage,
            ent_coef=exp.ent_coef,
            vf_coef=exp.vf_coef,
            max_grad_norm=exp.max_grad_norm,
            use_sde=exp.use_sde,
            sde_sample_freq=exp.sde_sample_freq,
            target_kl=exp.target_kl,
            stats_window_size=exp.stats_window_size,
            policy_kwargs=policy_kwargs,
            verbose=exp.verbose,
            tensorboard_log=f"{exp.dir_name}/runs/{run.id}",
            seed=exp.seed,
            device=exp.device
        )

    wandb_callback = WandbCallback(
        model_save_path=f"{exp.dir_name}/models/{run.id}",
        verbose=2,
    )

    model.learn(
        total_timesteps=int(exp.total_timesteps),
        progress_bar=True,
        callback=CallbackList([wandb_callback]),
    )
    model.save(f"{exp.dir_name}/ppo_halfcheetah")

    # evaluate
    if exp.render:
        eval_env_render = gym.make(exp.env_name, render_mode="human")
        obs, info = eval_env_render.reset()
        for i in range(1000):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env_render.step(action)
            if terminated or truncated:
                obs, info = eval_env_render.reset()
        eval_env_render.close()

    wandb.finish()


if __name__ == "__main__":
    main()