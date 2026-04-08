import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import wandb
from wandb.integration.sb3 import WandbCallback
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from algorithms.rt_a2c import RT_A2C
from buffers.buffers import MultiRolloutBuffer

@hydra.main(version_base=None, config_path="../config/a2c/", config_name="")
def main(cfg: DictConfig):
    exp = cfg.experiment
    run_dir = HydraConfig.get().runtime.output_dir

    # logger
    if exp.window_size > 1:
        base_name = f"RT-A2C {exp.n_envs}x{exp.n_steps}={exp.n_steps * exp.n_envs} w={exp.window_size} opc={exp.on_policy_critic} wt={exp.weight_type}  lr={exp.learning_rate} ent={exp.ent_coef}"
    else:
        base_name = f"A2C {exp.n_envs}x{exp.n_steps}={exp.n_steps * exp.n_envs} lr={exp.learning_rate} ent={exp.ent_coef}"
    
    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    wandb_config["base_name"] = base_name

    run = wandb.init(
        project=cfg.wandb.project,
        config=wandb_config,
        sync_tensorboard=cfg.wandb.sync_tensorboard,
        tags=cfg.wandb.tags,   
        dir=f"{run_dir}/wandb",  
        group=base_name,                  
        name=f"{base_name} seed={exp.seed}",   
        reinit="finish_previous",
    )

    # make the env
    env = make_vec_env(exp.env_name, n_envs=exp.n_envs, seed=exp.seed)
    env = VecNormalize(env, norm_reward=True, norm_obs=True)

    # parse policy args
    policy_kwargs=OmegaConf.to_container(exp.policy_kwargs, resolve=True) if exp.policy_kwargs is not None else None

    # learn
    if exp.window_size == 1:
        model = A2C(
            policy=exp.policy_type,
            env=env,
            learning_rate=exp.learning_rate,
            n_steps=exp.n_steps,
            gamma=exp.gamma,
            gae_lambda=exp.gae_lambda,
            ent_coef=exp.ent_coef,
            vf_coef=exp.vf_coef,
            max_grad_norm=exp.max_grad_norm,
            rms_prop_eps=exp.rms_prop_eps,
            use_rms_prop=exp.use_rms_prop,
            use_sde=exp.use_sde,
            sde_sample_freq=exp.sde_sample_freq,
            normalize_advantage=exp.normalize_advantage,
            stats_window_size=exp.stats_window_size,
            tensorboard_log=f"{run_dir}/tb",
            policy_kwargs=policy_kwargs,
            verbose=exp.verbose,
            seed=exp.seed,
            device=exp.device
        )
    else:
        model = RT_A2C(
            on_policy_critic=exp.on_policy_critic,
            rollout_buffer_class=MultiRolloutBuffer,
            rollout_buffer_kwargs=dict(
                window_size=exp.window_size,
                use_bh=(exp.weight_type == "bh"),
            ),
            is_weight_type=exp.weight_type,
            # Old A2C args
            policy=exp.policy_type,
            env=env,
            learning_rate=exp.learning_rate,
            n_steps=exp.n_steps,
            gamma=exp.gamma,
            gae_lambda=exp.gae_lambda,
            ent_coef=exp.ent_coef,
            vf_coef=exp.vf_coef,
            max_grad_norm=exp.max_grad_norm,
            rms_prop_eps=exp.rms_prop_eps,
            use_rms_prop=exp.use_rms_prop,
            use_sde=exp.use_sde,
            sde_sample_freq=exp.sde_sample_freq,
            normalize_advantage=exp.normalize_advantage,
            stats_window_size=exp.stats_window_size,
            tensorboard_log=f"{run_dir}/tb",
            policy_kwargs=policy_kwargs,
            verbose=exp.verbose,
            seed=exp.seed,
            device=exp.device
        )
    model.learn(
        total_timesteps=int(exp.total_timesteps),
        progress_bar=True,
        callback=WandbCallback(
            model_save_path=f"{run_dir}/models/{run.id}",
            verbose=2,
        ),
        log_interval=1,
    )
    model.save(f"{run_dir}/model")

if __name__ == "__main__":
    main()