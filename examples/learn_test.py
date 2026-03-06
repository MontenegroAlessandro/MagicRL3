import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
import wandb
from wandb.integration.sb3 import WandbCallback
import hydra
from omegaconf import DictConfig, OmegaConf

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from algorithms.rt_ppo import RT_PPO
from buffers.buffers import MultiRolloutBuffer


@hydra.main(version_base=None, config_path=".", config_name="conf")
def main(cfg: DictConfig):
    exp = cfg.experiment

    # logger
    run = wandb.init(
        project=cfg.wandb.project,
        config=OmegaConf.to_container(cfg, resolve=True),
        sync_tensorboard=cfg.wandb.sync_tensorboard,
    )

    # make the env
    env = make_vec_env(exp.env_name, n_envs=exp.n_envs, seed=exp.seed)

    # learn
    model = RT_PPO(
        on_policy_critic=exp.on_policy_critic,
        rollout_buffer_class=MultiRolloutBuffer,
        rollout_buffer_kwargs=dict(
            window_size=exp.window_size,
        ),
        # Old PPO args
        policy=exp.policy_type,
        env=env,
        learning_rate=exp.learning_rate,
        n_steps=exp.n_steps,
        batch_size=exp.batch_size,
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
        policy_kwargs=exp.policy_kwargs,
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
    eval_env = gym.make(exp.env_name, render_mode="human")
    obs, info = eval_env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        if terminated or truncated:
            obs, info = eval_env.reset()
    eval_env.close()


if __name__ == "__main__":
    main()