import gymnasium as gym
from pathlib import Path
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
from utils.config_utils import resolve_policy_kwargs
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from omegaconf import open_dict
from datetime import datetime


def get_full_config(cfg):
    exp = cfg.experiment

    batch_size = (exp.n_steps * exp.n_envs * exp.window_size) // exp.n_minibatch
    n_updates = exp.n_epochs * exp.n_minibatch
    n_sample_reuse = exp.n_epochs * exp.window_size

    if exp.window_size == 1:
        group_name = "PPO"
    else:
        weight_suffix = "-BH" if exp.weight_type == "bh" else "-N"
        seq_suffix = "-SEQ" if exp.sequential_window_training else ""
        group_name = f"RT-PPO{seq_suffix}{weight_suffix} w={exp.window_size}"

    group_name += (
        f" {exp.n_envs}x{exp.n_steps}={exp.n_envs * exp.n_steps}"
        f" K={exp.n_epochs}"
        f" nb={exp.n_minibatch}"
        f" B={batch_size}"
        f" nu={n_updates}"
        f" ns={n_sample_reuse}"
    )

    with open_dict(cfg):
        cfg.experiment.batch_size = batch_size
        cfg.experiment.n_updates = n_updates
        cfg.experiment.n_sample_reuse = n_sample_reuse
        cfg.group_name = group_name
        cfg.run_name = f"{group_name} seed={exp.seed}"

    return cfg


@hydra.main(version_base=None, config_path="../config/ppo", config_name="")
def main(cfg: DictConfig):
    cfg = get_full_config(cfg)
    exp = cfg.experiment

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(exp.dir_name) / now
    wandb_dir = output_dir / "wandb"
    tb_dir = output_dir / "tensorboard"
    checkpoints_dir = output_dir / "checkpoints"
    model_dir = output_dir / "model"
    eval_dir = output_dir / "eval"

    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        config=OmegaConf.to_container(cfg, resolve=True),
        sync_tensorboard=cfg.wandb.sync_tensorboard,
        group=cfg.group_name,                       
        name=cfg.run_name,  
        tags=cfg.wandb.tags, 
        dir=wandb_dir,
    )

    # make the env
    train_env = make_vec_env(exp.env_name, n_envs=exp.n_envs, seed=exp.seed)
    train_env = VecNormalize(
        train_env, 
        norm_reward=exp.norm_reward, 
        norm_obs=exp.norm_obs, 
        gamma=exp.gamma
    )
    
    eval_env = make_vec_env(exp.env_name, n_envs=exp.n_eval_envs, seed=exp.seed)
    eval_env = VecNormalize(
        eval_env, 
        norm_reward=False, 
        norm_obs=exp.norm_obs, 
        gamma=exp.gamma,
        training = False
    )

    # parse policy args
    policy_kwargs=OmegaConf.to_container(exp.policy_kwargs, resolve=True) if exp.policy_kwargs is not None else None
    policy_kwargs = resolve_policy_kwargs(policy_kwargs)

    # initialize PPO or RT-PPO
    if exp.window_size == 1:
        model = PPO(
            policy=exp.policy_type,
            env=train_env,
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
            policy_kwargs=policy_kwargs,
            verbose=exp.verbose,
            tensorboard_log=tb_dir,
            seed=exp.seed,
            device=exp.device
        )
    else:
        model = RT_PPO(
            on_policy_critic=exp.on_policy_critic,
            is_weight_type=exp.weight_type,
            sequential_window_training=exp.sequential_window_training,
            fresh_adv=exp.fresh_adv,
            rollout_buffer_class=MultiRolloutBuffer,
            rollout_buffer_kwargs=dict(
                window_size=exp.window_size,
                use_bh=(exp.weight_type == "bh"),
            ),
            # Old PPO args
            policy=exp.policy_type,
            env=train_env,
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
            policy_kwargs=policy_kwargs,
            verbose=exp.verbose,
            tensorboard_log=tb_dir,
            seed=exp.seed,
            device=exp.device
        )

    eval_callback = EvalCallback(
        eval_env,
        log_path=eval_dir,
        eval_freq=max(exp.eval_freq // exp.n_envs, 1),
        n_eval_episodes=exp.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    wandb_callback = WandbCallback(
        model_save_path=checkpoints_dir,
        model_save_freq=0,
        verbose=2,
    )

    callabacks = CallbackList([eval_callback, wandb_callback])

    model.learn(
        total_timesteps=int(exp.total_timesteps),
        callback=callabacks,
        progress_bar=True,
        log_interval=10,
    )

    model.save(model_dir)

    wandb.finish()


if __name__ == "__main__":
    main()