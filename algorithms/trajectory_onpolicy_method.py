import sys
import time
import warnings
from typing import Any, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

from buffers import TrajectoryBuffer
from policies import ActorOnlyPolicy

SelfTrajectoryOnPolicyAlgorithm = TypeVar("SelfTrajectoryOnPolicyAlgorithm", bound="TrajectoryOnPolicyAlgorithm")


class TrajectoryOnPolicyAlgorithm(BaseAlgorithm):
    """
    Base class for trajectory-based on-policy algorithms (e.g., REINFORCE, FDPG).

    Unlike step-based on-policy methods (A2C, PPO), each environment collects one
    complete trajectory per iteration. Trajectories may have different lengths.
    n_steps acts as a maximum horizon, not a fixed batch size.

    :param policy: Policy class or string alias
    :param env: Training environment
    :param learning_rate: Learning rate or schedule
    :param n_steps: Maximum trajectory length (hard cap); actual lengths may be shorter
    :param gamma: Discount factor for returns-to-go
    :param max_grad_norm: Gradient clipping threshold
    :param use_sde: Whether to use State Dependent Exploration (gSDE)
    :param sde_sample_freq: Resample gSDE noise every N steps (-1 = once per rollout)
    :param rollout_buffer_class: Buffer class to use; defaults to TrajectoryBuffer
    :param rollout_buffer_kwargs: Extra kwargs forwarded to the buffer constructor
    :param stats_window_size: Number of episodes to average for rollout logging
    :param tensorboard_log: TensorBoard log directory
    :param monitor_wrapper: Whether to auto-wrap envs with Monitor
    :param policy_kwargs: Extra kwargs forwarded to the policy constructor
    :param verbose: Verbosity level (0 = silent, 1 = info, 2 = debug)
    :param seed: Random seed
    :param device: PyTorch device
    :param _init_setup_model: Build networks on construction
    :param supported_action_spaces: Whitelist of allowed action space types
    """

    rollout_buffer: TrajectoryBuffer
    policy: ActorOnlyPolicy

    def __init__(
        self,
        policy: Union[str, type[ActorOnlyPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        max_grad_norm: float,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[TrajectoryBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[tuple[type[spaces.Space], ...]] = None,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer_class = rollout_buffer_class
        self.rollout_buffer_kwargs = rollout_buffer_kwargs or {}

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.rollout_buffer_class is None:
            self.rollout_buffer_class = TrajectoryBuffer

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )

        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)
        self._maybe_recommend_cpu()

    def _maybe_recommend_cpu(self, mlp_class_name: str = "ActorOnlyPolicy") -> None:
        policy_class_name = self.policy_class.__name__
        if self.device != th.device("cpu") and policy_class_name == mlp_class_name:
            warnings.warn(
                f"You are trying to run {self.__class__.__name__} on the GPU, "
                "but it is primarily intended to run on the CPU when not using a CNN policy "
                f"(you are using {policy_class_name} which should be a MlpPolicy). "
                "See https://github.com/DLR-RM/stable-baselines3/issues/1245 for more info. "
                "You can pass `device='cpu'` or `export CUDA_VISIBLE_DEVICES=` to force using the CPU.",
                UserWarning,
            )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: TrajectoryBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect one complete trajectory per environment.

        All environments are reset at the start so every trajectory begins from
        a clean initial state. Collection continues until every environment has
        reached a terminal state or n_rollout_steps (the maximum horizon) is hit.

        num_timesteps is incremented by the number of *active* environments at
        each step, so it exactly equals the total number of transitions stored.

        :param env: The training environment
        :param callback: Callback invoked at each step
        :param rollout_buffer: Buffer to populate
        :param n_rollout_steps: Maximum horizon (safety cap per trajectory)
        :return: False if the callback requested early termination, True otherwise
        """
        self.policy.set_training_mode(False)
        rollout_buffer.reset()

        # Fresh reset: we always want complete trajectories, never mid-episode starts
        self._last_obs = env.reset()  # type: ignore[assignment]
        self._last_episode_starts = np.ones(env.num_envs, dtype=bool)

        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        # active[i] = True while env i is still collecting its trajectory
        active = np.ones(env.num_envs, dtype=bool)
        n_steps = 0

        while active.any() and n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)  # type: ignore[arg-type]
                actions, _ = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            # Count only transitions that will actually land in the buffer
            self.num_timesteps += int(active.sum())

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            # Log episode stats only for envs finishing their *first* (active) trajectory
            newly_done = dones & active
            self._update_info_buffer(infos, newly_done)

            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            active_indices = np.where(active)[0]
            rollout_buffer.add(
                self._last_obs[active_indices],  # type: ignore[arg-type]
                actions[active_indices],
                rewards[active_indices],
                self._last_episode_starts[active_indices],
                env_indices=active_indices,
            )

            # Deactivate envs whose trajectory just ended
            active &= ~dones

            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        # if active.any():
        #     warnings.warn(
        #         f"{active.sum()} environment(s) did not finish their trajectory within "
        #         f"the maximum horizon of {n_rollout_steps} steps. Their returns will be "
        #         "computed on the truncated trajectory (no bootstrapping).",
        #         UserWarning,
        #     )

        rollout_buffer.compute_returns()

        callback.update_locals(locals())
        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """Update policy parameters from the current rollout. Implemented by subclasses."""
        raise NotImplementedError

    def dump_logs(self, iteration: int = 0) -> None:
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)

        if iteration > 0:
            self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        # Log actual mean trajectory length this iteration (not the fixed n_steps cap)
        lengths = self.rollout_buffer._traj_lengths
        if lengths.sum() > 0:
            self.logger.record("rollout/mean_traj_len", float(lengths[lengths > 0].mean()))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))

        self.logger.dump(step=self.num_timesteps)

    def learn(
        self: SelfTrajectoryOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "TrajectoryOnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfTrajectoryOnPolicyAlgorithm:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(
                self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps
            )

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            if log_interval is not None and iteration % log_interval == 0:
                self.dump_logs(iteration)

            self.train()

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        return ["policy", "policy.optimizer"], []
