import warnings

import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
import wandb
from wandb.integration.sb3 import WandbCallback
import hydra
from omegaconf import DictConfig, OmegaConf
import torch.nn as nn
import torch as th

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import envs  # triggers the registration of new envs
from algorithms import PolicyGradient, FDPG
from callbacks.trajectory_eval_callback import TrajectoryEvalCallback
from callbacks.accurate_progress_bar_callback import AccurateProgressBarCallback

ALGOS = ("reinforce", "gpomdp", "fdpg")


def build_run_name(exp) -> str:
    algo = exp.algo
    if algo.name == "reinforce":
        return (
            f"REINFORCE Ne={exp.n_envs} H={exp.n_steps} "
            f"lr={exp.learning_rate} γ={exp.gamma} "
            f"ent={algo.ent_coef} norm_G={algo.normalize_returns}"
        )
    elif algo.name == "gpomdp":
        return (
            f"GPOMDP Ne={exp.n_envs} H={exp.n_steps} "
            f"lr={exp.learning_rate} γ={exp.gamma} "
            f"ent={algo.ent_coef} norm_G={algo.normalize_returns}"
        )
    elif algo.name == "fdpg":
        return (
            f"{algo.mode.capitalize()}-FDPG Ne={exp.n_envs} H={exp.n_steps} "
            f"lr={exp.learning_rate} γ={exp.gamma} σ={algo.sigma} "
            f"{algo.sampling_mode}/{algo.sampling_strategy}"
        )
    else:
        raise ValueError(f"Unknown experiment.algo.name '{algo.name}'. Choose from: {ALGOS}")


def build_model(exp, env, policy_kwargs, tensorboard_log):
    algo = exp.algo
    if algo.name in ["reinforce", "gpomdp"]:
        return PolicyGradient(
            g_estimator=algo.name,
            policy=exp.policy_type,
            env=env,
            learning_rate=exp.learning_rate,
            n_steps=exp.n_steps,
            gamma=exp.gamma,
            max_grad_norm=exp.max_grad_norm,
            ent_coef=algo.ent_coef,
            normalize_returns=algo.normalize_returns,
            use_sde=exp.use_sde,
            sde_sample_freq=exp.sde_sample_freq,
            stats_window_size=exp.stats_window_size,
            policy_kwargs=policy_kwargs,
            verbose=exp.verbose,
            tensorboard_log=tensorboard_log,
            seed=exp.seed,
            device=exp.device,
        )
    elif algo.name == "fdpg":
        return FDPG(
            policy=exp.policy_type,
            env=env,
            learning_rate=exp.learning_rate,
            n_steps=exp.n_steps,
            gamma=exp.gamma,
            sigma=algo.sigma,
            # FDPG requires env.num_envs == batch_size (one reference trajectory per
            # sub-env), so batch_size is derived from the shared n_envs knob rather
            # than configured separately.
            batch_size=exp.n_envs,
            mode=algo.mode,
            sampling_mode=algo.sampling_mode,
            sampling_strategy=algo.sampling_strategy,
            env_id=exp.env_name,
            max_grad_norm=exp.max_grad_norm,
            use_sde=exp.use_sde,
            sde_sample_freq=exp.sde_sample_freq,
            stats_window_size=exp.stats_window_size,
            policy_kwargs=policy_kwargs,
            verbose=exp.verbose,
            tensorboard_log=tensorboard_log,
            seed=exp.seed,
            device=exp.device,
        )
    else:
        raise ValueError(f"Unknown experiment.algo.name '{algo.name}'. Choose from: {ALGOS}")


@hydra.main(version_base=None, config_path=".", config_name="conf")
def main(cfg: DictConfig):

    exp = cfg.experiment

    base_name = build_run_name(exp)

    if exp.algo.name == "fdpg" and exp.mode == "trajectory" and exp.sampling_strategy == "trajectory":
        return

    conf = OmegaConf.to_container(cfg, resolve=True)
    conf["group"] = base_name
    run = wandb.init(
        project=cfg.wandb.project,
        config=conf,
        sync_tensorboard=cfg.wandb.sync_tensorboard,
        group=base_name,
        name=f"{base_name} seed={exp.seed}",
        tags=cfg.wandb.tags,
    )

    # sync_tensorboard bundles every scalar from one logger.dump() call into a single
    # wandb.log(), so train/n_updates (resp. eval/n_updates) is always logged alongside
    # the other train/* (resp. eval/*) metrics from that same call -- define it as their
    # default x-axis so those charts plot against parameter updates out of the box,
    # without needing to hand-edit each panel's x-axis in the wandb UI. The usual
    # timesteps/global_step axis stays available too; this only changes the default.
    wandb.define_metric("train/n_updates")
    wandb.define_metric("train/*", step_metric="train/n_updates")
    wandb.define_metric("eval/n_updates")
    wandb.define_metric("eval/*", step_metric="eval/n_updates")

    # --- Training env ---
    env = make_vec_env(exp.env_name, n_envs=exp.n_envs, seed=exp.seed)

    # --- Evaluation env ---
    eval_env = make_vec_env(exp.env_name, n_envs=exp.n_eval_episodes, seed=exp.seed + 1000)

    # FDPG builds its own perturbed-env pool internally via raw gym.make() (see
    # FDPG._setup_model), bypassing VecNormalize. Wrapping the main env in VecNormalize
    # for FDPG would desync the reference rollout (normalized obs/reward) from the
    # perturbed one (raw obs/reward), corrupting the g-b estimator. So normalization
    # is only applied for algos that don't have that side pool.
    use_vecnormalize = exp.algo.name != "fdpg" and (exp.normalize_obs or exp.normalize_reward)
    if use_vecnormalize:
        env = VecNormalize(env, norm_reward=exp.normalize_reward, norm_obs=exp.normalize_obs, gamma=exp.gamma, training=True)
        # shares obs stats, never normalizes rewards
        eval_env = VecNormalize(eval_env, norm_reward=False, norm_obs=exp.normalize_obs, gamma=exp.gamma, training=False)
        eval_env.obs_rms = env.obs_rms
    elif exp.normalize_obs or exp.normalize_reward:
        warnings.warn(
            f"experiment.normalize_obs/normalize_reward are ignored for algo='{exp.algo.name}': "
            "its perturbed-env pool bypasses VecNormalize, so normalizing here would silently "
            "desync the reference and perturbed rollouts."
        )

    # Parse policy kwargs (activation_fn must be converted from string to class)
    if exp.algo.sigma is not None:
        exp.policy_kwargs.log_std_init = float(th.log(th.tensor(exp.algo.sigma)))
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

    model = build_model(exp, env, policy_kwargs, tensorboard_log=f"{exp.dir_name}/runs/{run.id}")

    # eval_callback = EvalCallback(
    #     eval_env,
    #     best_model_save_path=f"{exp.dir_name}/models/{run.id}",
    #     log_path=f"{exp.dir_name}/logs/{run.id}",
    #     eval_freq=exp.eval_freq,
    #     n_eval_episodes=exp.n_eval_episodes,
    #     deterministic=True,
    #     render=False,
    #     verbose=0,
    # )
    eval_callback = TrajectoryEvalCallback(
        eval_env,
        n_eval_episodes=exp.n_eval_episodes,
        eval_freq=exp.eval_freq,
        deterministic=True,
        verbose=0,
    )

    wandb_callback = WandbCallback(
        model_save_path=f"{exp.dir_name}/models/{run.id}",
        verbose=2,
    )

    # SB3's built-in progress_bar=True assumes one on_step() call == num_envs real
    # timesteps, which FDPG breaks (perturbed-rollout steps bypass on_step()) --
    # use our own bar that tracks num_timesteps directly instead.
    progress_bar_callback = AccurateProgressBarCallback()

    if exp.eval_freq is not None:
        callbacks = CallbackList([eval_callback, wandb_callback, progress_bar_callback])
    else:
        callbacks = CallbackList([wandb_callback, progress_bar_callback])

    model.learn(
        total_timesteps=int(exp.total_timesteps),
        progress_bar=False,
        callback=callbacks,
    )

    env_name = str(exp.env_name).split("-")[0]
    model.save(f"{exp.dir_name}/{env_name}_{exp.algo.name.upper()}_{run.id}")

    if exp.render:
        eval_env_render = gym.make(exp.env_name, render_mode="human")
        obs, _ = eval_env_render.reset()
        for _ in range(1000):
            obs_input = env.normalize_obs(obs) if use_vecnormalize and exp.normalize_obs else obs
            action, _ = model.predict(obs_input, deterministic=True)
            obs, _, terminated, truncated, _ = eval_env_render.step(action)
            print(f"Obs: {obs}, Action: {action}, Terminated: {terminated}, Truncated: {truncated}")
            if terminated or truncated:
                obs, _ = eval_env_render.reset()
        eval_env_render.close()

    wandb.finish()


if __name__ == "__main__":
    main()
