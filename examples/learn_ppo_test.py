import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import wandb
from wandb.integration.sb3 import WandbCallback
import hydra
from omegaconf import DictConfig, OmegaConf

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from algorithms.rt_ppo import RT_PPO
from buffers.buffers import MultiRolloutBuffer

@hydra.main(version_base=None, config_path="../config/ppo", config_name="")
def main(cfg: DictConfig):
    exp = cfg.experiment

    # Derive batch_size from n_minibatch so we have direct control over gradient steps per epoch.
    # batch_size is always based on the on-policy data size (n_steps * n_envs * window_size).
    batch_size = (exp.n_steps * exp.n_envs * exp.window_size) // exp.n_minibatch
    n_updates = exp.n_epochs * exp.n_minibatch

    # logger
    ppo_info = f"{exp.n_envs}x{exp.n_steps}={exp.n_envs * exp.n_steps} epochs={exp.n_epochs} n_minibatch={exp.n_minibatch} batch_size={batch_size} n_updates={n_updates} kl_target={exp.target_kl}"
    if exp.window_size > 1:
        base_name = f"RT-PPO {ppo_info} w={exp.window_size} opc={exp.on_policy_critic} weight_type={exp.weight_type} kl_target={exp.target_kl}"
    else:
        base_name = f"PPO {ppo_info}"
    
    conf = OmegaConf.to_container(cfg, resolve=True)
    conf["group"] = base_name
    conf["n_updates"] = n_updates
    conf["batch_size"] = batch_size

    run = wandb.init(
        project=cfg.wandb.project,
        config=conf,
        sync_tensorboard=cfg.wandb.sync_tensorboard,
        group=base_name,                       
        name=f"{base_name} seed={exp.seed}",  
        tags=cfg.wandb.tags, 
    )

    # make the env
    env = make_vec_env(exp.env_name, n_envs=exp.n_envs, seed=exp.seed)
    env = VecNormalize(env, norm_reward=True, norm_obs=True)
    
    # parse policy args
    policy_kwargs=OmegaConf.to_container(exp.policy_kwargs, resolve=True) if exp.policy_kwargs is not None else None

    # Derive batch_size from n_minibatch so we have direct control over gradient steps per epoch.
    # batch_size is always based on the on-policy data size (n_steps * n_envs) regardless of window.
    batch_size = (exp.n_steps * exp.n_envs) // exp.n_minibatch
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
    model.learn(
        total_timesteps=int(exp.total_timesteps),
        progress_bar=True,
        callback=WandbCallback(
            model_save_path=f"{exp.dir_name}/models/{run.id}",
            verbose=2,
        ),
    )
    model.save(f"{exp.dir_name}/ppo_halfcheetah")

    # evaluate
    if exp.render:
        eval_env = gym.make(exp.env_name, render_mode="human")
        obs, info = eval_env.reset()
        for i in range(1000):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            if terminated or truncated:
                obs, info = eval_env.reset()
        eval_env.close()
    
    # close the wandb run
    wandb.finish()


if __name__ == "__main__":
    main()