import gymnasium as gym
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
from algorithms import POSER, MyPPO
from algorithms.utils.wandb_logging import WandbMetricsCallback
import envs
from buffers.poser_rollout_buffer import PoserRolloutBuffer
import torch.nn as nn

WANDB_LABEL_MAX_LENGTH = 128


def _wandb_group_and_name(base_name: str, seed) -> tuple[str, str]:
    """Fit base_name within W&B's 128-char GroupName/Name limits, keeping the seed visible.

    Truncates base_name (not the seed suffix) so the run name always ends in "seed=<n>",
    and the group (which drops the suffix) stays a prefix of the run name.
    """
    seed_suffix = f" seed={seed}"
    budget = WANDB_LABEL_MAX_LENGTH - len(seed_suffix)
    group = base_name if len(base_name) <= budget else base_name[:budget]
    return group, group + seed_suffix


ACTIVATION_FN = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
    "selu": nn.SELU,
    "gelu": nn.GELU,
}

def parse_policy_kwargs(policy_kwargs):
    """Config policy_kwargs -> SB3 ones: activation_fn goes from name to nn class."""
    if policy_kwargs is None:
        return None
    kwargs = OmegaConf.to_container(policy_kwargs, resolve=True)
    name = kwargs.get("activation_fn")
    if name is not None:
        if name.lower() not in ACTIVATION_FN:
            raise ValueError(f"Unknown activation_fn '{name}'. Choose from: {list(ACTIVATION_FN)}")
        kwargs["activation_fn"] = ACTIVATION_FN[name.lower()]
    return kwargs

@hydra.main(version_base=None, config_path="config", config_name="conf_poser")
def main(cfg: DictConfig):
    exp = cfg.experiment

    # Derive batch_size from n_minibatches so we have direct control over gradient steps per epoch.
    # batch_size is always based on the on-policy data size (n_steps * n_envs * window_size).
    window_size = exp.window_size or 1
    if exp.batch_size is not None:
        batch_size = exp.batch_size
        n_minibatch_effective = (exp.n_steps * exp.n_envs * window_size) // batch_size
    else:
        batch_size = (exp.n_steps * exp.n_envs * window_size) // exp.n_minibatches
        n_minibatch_effective = exp.n_minibatches

    # logger
    # if window_size > 1:
    if window_size >= 1:
        sampling = exp.batch_sampling
        psr_label = exp.psr_threshold if exp.psr_threshold is not None else "off"
        base_name = (
            f"POSER w={window_size} bs={sampling} wt={exp.weight_type or 'uniform'} "
            f"(Ne,H)=({exp.n_envs},{exp.n_steps}) K={exp.n_epochs} "
            f"(n_b,b_s)=({n_minibatch_effective},{batch_size}) "
            f"psr={psr_label} disc={exp.discard_policy or 'oldest'} "
            f"clip_adapt={exp.clip_range_adaptation or 'none'} "
            f"clip={exp.clip_range}"
        )
    else:
        base_name = f"MyPPO (Ne,H)=({exp.n_envs},{exp.n_steps}) K={exp.n_epochs} (n_b,b_s)=({n_minibatch_effective},{batch_size})"
    base_name += f" norm_r={exp.normalize_reward} gamma={exp.gamma} eps={exp.clip_range}"
    if exp.extra_name is not None:
        base_name += f" {exp.extra_name}"
    if exp.name is not None:
        base_name = exp.name

    group_name, run_name = _wandb_group_and_name(base_name, exp.seed)

    conf = OmegaConf.to_container(cfg, resolve=True)
    conf["group"] = group_name
    # POSER dumps two clocks into TensorBoard. Send its metrics directly to W&B
    # so tensorboard synchronization cannot merge unrelated rows.
    direct_wandb_metrics = window_size > 1 and cfg.wandb.sync_tensorboard
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        config=conf,
        sync_tensorboard=cfg.wandb.sync_tensorboard and not direct_wandb_metrics,
        group=group_name,
        name=run_name,
        tags=cfg.wandb.tags,
        dir=exp.dir_name,
    )

    # --- Training env ---
    env = make_vec_env(exp.env_name, n_envs=exp.n_envs, seed=exp.seed)
    env = VecNormalize(env, norm_reward=exp.normalize_reward, norm_obs=exp.normalize_obs, gamma=exp.gamma, training=True)

    # --- Evaluation env ---
    eval_env = make_vec_env(exp.env_name, n_envs=1, seed=exp.seed + 1000)
    eval_env = VecNormalize(eval_env, norm_reward=False, norm_obs=exp.normalize_obs, gamma=exp.gamma, training=False)
    eval_env.obs_rms = env.obs_rms
    if hasattr(env, "ret_rms"):
        eval_env.ret_rms = env.ret_rms

    # --- PPO common config ---
    PPO_config = dict(
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
        policy_kwargs=parse_policy_kwargs(exp.policy_kwargs),
        verbose=exp.verbose,
        tensorboard_log=f"{exp.dir_name}/runs/{run.id}",
        seed=exp.seed,
        device=exp.device,
    )

    # --- Model selection ---
    # if window_size == 1:
    if window_size < 1:
        model = MyPPO(**PPO_config)
    else:
        model = POSER(
            weight_type=exp.weight_type or "uniform",
            weighted_critic=exp.weighted_critic,
            weight_discard_threshold=exp.weight_discard_threshold,
            psr_threshold=exp.psr_threshold,
            discard_policy=exp.discard_policy or "oldest",
            clip_range_adaptation=exp.clip_range_adaptation or "none",
            rollout_buffer_class=PoserRolloutBuffer,
            rollout_buffer_kwargs=dict(
                window_size=window_size,
                batch_sampling=exp.batch_sampling or "balanced",
            ),
            debug=exp.debug,
            **PPO_config,
        )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{exp.dir_name}/models/{run.id}",
        log_path=f"{exp.dir_name}/logs/{run.id}",
        eval_freq=cfg.experiment.eval_freq,
        n_eval_episodes=cfg.experiment.n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=0
    )

    wandb_callback = WandbCallback(
        model_save_path=f"{exp.dir_name}/models/{run.id}",
        verbose=2,
    )

    callbacks = [wandb_callback, eval_callback]
    if direct_wandb_metrics:
        callbacks.insert(0, WandbMetricsCallback(run))

    model.learn(
        total_timesteps=int(exp.total_timesteps),
        progress_bar=True,
        callback=CallbackList(callbacks),
    )
    env_name = str(exp.env_name).split("-")[0]
    method_name = "POSER" if window_size > 1 else "PPO"
    model.save(f"{exp.dir_name}/{env_name}_{method_name}_{run.id}")

    wandb.finish()


if __name__ == "__main__":
    main()
