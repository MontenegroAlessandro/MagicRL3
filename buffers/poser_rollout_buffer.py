"""Rollout-buffer pipeline and storage layout.

H: rollout steps; N_E: parallel environments; n_rollouts: rollouts in the window.

reset:
    self.* = new D_k [H, N_E, ...]
    previous D_k -> self.history as [H, N_E, ...]
    cache = None

add -> compute_returns_and_advantage -> record_behavior_distribution -> recompute_advantages:
    self.* is filled and remains [H, N_E, ...]
    self.history remains [H, N_E, ...]

get:
    current D_k -> cache as [H, N_E, ...]
    self.history remains [H, N_E, ...]
    self.* -> flattened window [n_rollouts * N_E * H, ...] for PPO minibatches

next reset:
    cached D_k -> self.history
    self.* -> new D_k [H, N_E, ...]
    cache = None
"""

from collections import deque
from collections.abc import Generator
from dataclasses import dataclass
from typing import NamedTuple, Optional

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer


BATCH_SAMPLING_MODES = ("random", "balanced")


class PoserRolloutBufferSamples(NamedTuple):
    """The batch of samples used in the update of the policy."""

    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    behavior_mean: th.Tensor
    behavior_std: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    window_id: th.Tensor


@dataclass(frozen=True)
class StoredRollout:
    """One rollout stored in SB3 time-major format [H, N_E, ...]."""

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    episode_starts: np.ndarray
    values: np.ndarray
    log_probs: np.ndarray
    behavior_mean: np.ndarray
    behavior_std: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray


class PoserRolloutBuffer(RolloutBuffer):
    """Store n_rollouts blocks ordered as [D_k, D_(k-1), ...]."""

    def __init__(
        self,
        *args,
        window_size: int = 1,
        batch_sampling: str = "balanced",
        vtrace_rho_clip: float = 1.0,
        vtrace_c_clip: float = 1.0,
        **kwargs,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        if batch_sampling not in BATCH_SAMPLING_MODES:
            raise ValueError(f"batch_sampling must be one of {BATCH_SAMPLING_MODES}")
        if vtrace_c_clip <= 0 or vtrace_rho_clip < vtrace_c_clip:
            raise ValueError("V-trace requires vtrace_rho_clip >= vtrace_c_clip > 0")

        self.window_size = window_size
        self.batch_sampling = batch_sampling
        self.vtrace_rho_clip = vtrace_rho_clip
        self.vtrace_c_clip = vtrace_c_clip
        self.history: deque[StoredRollout] = deque(maxlen=window_size - 1)
        self.window_id = np.empty(0, dtype=np.int64)
        self._current_rollout_cache: Optional[StoredRollout] = None
        super().__init__(*args, **kwargs)

    @property
    def n_rollouts(self) -> int:
        return 1 + len(self.history)

    def reset(self) -> None:
        """Archive the completed rollout, then let SB3 reset its arrays."""

        if getattr(self, "full", False) and self.window_size > 1:
            self.history.appendleft(self._current_rollout())

        super().reset()
        self.window_id = np.empty(0, dtype=np.int64)
        distribution_shape = (
            self.buffer_size,
            self.n_envs,
            self.action_dim,
        )
        self.behavior_mean = np.zeros(
            distribution_shape,
            dtype=np.float32,
        )
        self.behavior_std = np.zeros(
            distribution_shape,
            dtype=np.float32,
        )
        self._current_rollout_cache = None

    def get(self, batch_size: Optional[int] = None) -> Generator[PoserRolloutBufferSamples, None, None]:
        """Yield one full pass using random or rollout-balanced indices."""
        if not self.full:
            raise RuntimeError("The current rollout is not full")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self._prepare_window()

        if self.batch_sampling == "balanced" and self.n_rollouts > 1:
            indices_by_rollout = []
            for rollout in range(self.n_rollouts):
                rollout_indices = np.flatnonzero(self.window_id == rollout)
                indices_by_rollout.append(np.random.permutation(rollout_indices))
            indices = np.stack(indices_by_rollout, axis=1).reshape(-1)
        elif self.batch_sampling == "random" or self.n_rollouts == 1:
            indices = np.random.permutation(len(self.window_id))

        if batch_size is None:
            batch_size = len(indices)

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            if self.batch_sampling == "balanced":
                batch_indices = np.random.permutation(batch_indices)
            yield self._get_samples(batch_indices)

    def get_all(self) -> PoserRolloutBufferSamples:
        """Return every sample from every rollout window."""
        if not self.full:
            raise RuntimeError("The current rollout is not full")

        self._prepare_window()
        indices = np.arange(len(self.window_id))
        return self._get_samples(indices)

    def get_window(self, window_id: int) -> PoserRolloutBufferSamples:
        """Return every sample from one rollout window."""
        if not self.full:
            raise RuntimeError("The current rollout is not full")
        if window_id < 0 or window_id >= self.n_rollouts:
            raise IndexError(f"window_id must be between 0 and {self.n_rollouts - 1}")

        self._prepare_window()
        indices = np.flatnonzero(self.window_id == window_id)
        return self._get_samples(indices)

    def record_behavior_distribution(self, policy) -> None:
        """Store the Gaussian that generated the current rollout."""
        number_of_samples = self.buffer_size * self.n_envs
        observations = self.observations.reshape(
            number_of_samples,
            *self.observations.shape[2:],
        )
        observations = th.as_tensor(
            observations,
            device=policy.device,
        )

        with th.no_grad():
            distribution = policy.get_distribution(observations).distribution
            behavior_mean = distribution.mean.cpu().numpy()
            behavior_std = distribution.stddev.cpu().numpy()

        distribution_shape = (
            self.buffer_size,
            self.n_envs,
            self.action_dim,
        )
        self.behavior_mean[...] = behavior_mean.reshape(distribution_shape)
        self.behavior_std[...] = behavior_std.reshape(distribution_shape)

    def recompute_advantages(self, policy) -> None:
        """Recompute historical V-trace targets with the current policy and critic."""
        if not self.full:
            raise RuntimeError("The current rollout is not full")
        if not self.history:
            return

        rollouts = [self._current_rollout(), *self.history]

        # Current and historical rollouts stay in [n_rollouts, H, N_E, ...].
        window_observations = np.stack([rollout.observations for rollout in rollouts])
        window_actions = np.stack([rollout.actions for rollout in rollouts])
        window_rewards = np.stack([rollout.rewards for rollout in rollouts])
        window_episode_starts = np.stack([rollout.episode_starts for rollout in rollouts])
        window_log_mu = np.stack([rollout.log_probs for rollout in rollouts])

        # Only policy inference temporarily flattens [n_rollouts, H, N_E, ...] to [n_rollouts * H * N_E, ...].
        n_samples = self.n_rollouts * self.buffer_size * self.n_envs
        observations = window_observations.reshape(n_samples, *window_observations.shape[3:])
        actions = window_actions.reshape(n_samples, *window_actions.shape[3:])

        observations = th.as_tensor(observations, device=policy.device)
        actions = th.as_tensor(actions, device=policy.device)
        if isinstance(self.action_space, spaces.Discrete):
            actions = actions.long().flatten()

        was_training = policy.training
        policy.set_training_mode(False)
        try:
            with th.no_grad():
                values, log_pi, _ = policy.evaluate_actions(observations, actions)
        finally:
            policy.set_training_mode(was_training)

        # Reshape the outputs back to [n_rollouts, H, N_E, ...] for V-trace.
        window_shape = (self.n_rollouts, self.buffer_size, self.n_envs)
        window_V = values.cpu().numpy().reshape(window_shape)
        window_log_pi = log_pi.cpu().numpy().reshape(window_shape)

        # Index 0 is D_k; historical quantities remain [n_rollouts - 1, H, N_E].
        V_t = window_V[1:]
        rewards_t = window_rewards[1:]
        episode_starts_t = window_episode_starts[1:]
        log_mu_t = window_log_mu[1:]
        log_pi_t = window_log_pi[1:]

        # D_(k-i+1) provides x_(H+1) for D_(k-i); shape [n_rollouts - 1, N_E].
        bootstrap_V = np.concatenate([window_V[0:1, 0, :], V_t[:-1, 0, :]], axis=0)
        bootstrap_episode_starts = np.concatenate([window_episode_starts[0:1, 0, :], episode_starts_t[:-1, 0, :]], axis=0)

        v_t, A_t = self._compute_vtrace(rewards_t, episode_starts_t, V_t, log_mu_t, log_pi_t, bootstrap_V, bootstrap_episode_starts)

        # V-trace already returns [n_rollouts - 1, H, N_E], the same layout used by history.
        for rollout_index, rollout in enumerate(self.history):
            rollout.returns[...] = v_t[rollout_index]
            rollout.advantages[...] = A_t[rollout_index]

        # get() stores each rollout as [N_E, H] before flattening the two axes.
        if self.generator_ready:
            historical_start = self.buffer_size * self.n_envs
            stored_returns = self.returns[historical_start:]
            stored_advantages = self.advantages[historical_start:]
            flattened_v_t = v_t.swapaxes(1, 2).reshape(stored_returns.shape)
            flattened_A_t = A_t.swapaxes(1, 2).reshape(stored_advantages.shape)
            self.returns[historical_start:] = flattened_v_t
            self.advantages[historical_start:] = flattened_A_t

    def _compute_vtrace(self, rewards_t: np.ndarray, episode_starts_t: np.ndarray, V_t: np.ndarray, log_mu_t: np.ndarray, log_pi_t: np.ndarray, bootstrap_V: np.ndarray, bootstrap_episode_starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply V-trace to arrays shaped [historical rollout, timestep, environment]."""

        # IMPALA notation: mu is the behavior policy, pi is the current target policy, and V_t = V(x_t).
        log_w_t = log_pi_t - log_mu_t

        # rho_t = min(rho_bar, pi(a_t|x_t) / mu(a_t|x_t)).
        log_rho_t = np.minimum(log_w_t, np.log(self.vtrace_rho_clip))
        rho_t = np.exp(log_rho_t)

        # c_t = lambda min(c_bar, pi(a_t|x_t) / mu(a_t|x_t)).
        log_clipped_c_t = np.minimum(log_w_t, np.log(self.vtrace_c_clip))
        clipped_c_t = np.exp(log_clipped_c_t)
        c_t = self.gae_lambda * clipped_c_t

        # Position [:, t, :] contains the quantity at t + 1.
        V_t_plus_1_without_bootstrap = V_t[:, 1:, :]
        bootstrap_V_axis = bootstrap_V[:, None, :]
        V_t_plus_1 = np.concatenate([V_t_plus_1_without_bootstrap, bootstrap_V_axis], axis=1)

        episode_start_t_plus_1_without_boundary = episode_starts_t[:, 1:, :]
        bootstrap_episode_starts_axis = bootstrap_episode_starts[:, None, :]
        episode_start_t_plus_1 = np.concatenate([episode_start_t_plus_1_without_boundary, bootstrap_episode_starts_axis], axis=1)

        # gamma_t = gamma (1 - xi_t).
        not_terminal_t = 1.0 - episode_start_t_plus_1
        gamma_t = self.gamma * not_terminal_t

        # delta_t^V = rho_t [r_t + gamma_t V_(t+1) - V_t].
        one_step_return_t = rewards_t + gamma_t * V_t_plus_1
        td_error_t = one_step_return_t - V_t
        delta_V_t = rho_t * td_error_t

        # v_t - V_t = delta_t^V + gamma_t c_t [v_(t+1) - V_(t+1)].
        v_t = np.empty_like(V_t)
        v_minus_V_t_plus_1 = np.zeros((V_t.shape[0], V_t.shape[2]), dtype=V_t.dtype)
        for t in reversed(range(self.buffer_size)):
            discounted_trace_t = gamma_t[:, t, :] * c_t[:, t, :] * v_minus_V_t_plus_1
            v_minus_V_t = delta_V_t[:, t, :] + discounted_trace_t
            v_t[:, t, :] = V_t[:, t, :] + v_minus_V_t
            v_minus_V_t_plus_1 = v_minus_V_t

        # q_t = r_t + gamma_t v_(t+1).
        v_t_plus_1_without_bootstrap = v_t[:, 1:, :]
        v_t_plus_1 = np.concatenate([v_t_plus_1_without_bootstrap, bootstrap_V_axis], axis=1)
        q_t = rewards_t + gamma_t * v_t_plus_1

        # A_t = rho_t [q_t - V_t].
        q_minus_V_t = q_t - V_t
        A_t = rho_t * q_minus_V_t
        return v_t, A_t

    def _current_rollout(self) -> StoredRollout:
        """Return the current rollout in time-major format [H, N_E, ...]."""
        if self.generator_ready:
            if self._current_rollout_cache is None:
                raise RuntimeError("Current rollout cache is empty")
            return self._current_rollout_cache

        return StoredRollout(
            observations=self.observations.copy(),
            actions=self.actions.copy(),
            rewards=self.rewards.copy(),
            episode_starts=self.episode_starts.copy(),
            values=self.values.copy(),
            log_probs=self.log_probs.copy(),
            behavior_mean=self.behavior_mean.copy(),
            behavior_std=self.behavior_std.copy(),
            advantages=self.advantages.copy(),
            returns=self.returns.copy(),
        )

    def _prepare_window(self) -> None:
        """Flatten rollouts only when PPO requests minibatches."""
        if self.generator_ready:
            return

        current_rollout = self._current_rollout()
        self._current_rollout_cache = current_rollout
        rollouts = [current_rollout, *self.history]

        self.observations = np.concatenate([self.swap_and_flatten(rollout.observations) for rollout in rollouts])
        self.actions = np.concatenate([self.swap_and_flatten(rollout.actions) for rollout in rollouts])
        self.rewards = np.concatenate([self.swap_and_flatten(rollout.rewards) for rollout in rollouts])
        self.episode_starts = np.concatenate([self.swap_and_flatten(rollout.episode_starts) for rollout in rollouts])
        self.values = np.concatenate([self.swap_and_flatten(rollout.values) for rollout in rollouts])
        self.log_probs = np.concatenate([self.swap_and_flatten(rollout.log_probs) for rollout in rollouts])
        self.behavior_mean = np.concatenate([self.swap_and_flatten(rollout.behavior_mean) for rollout in rollouts])
        self.behavior_std = np.concatenate([self.swap_and_flatten(rollout.behavior_std) for rollout in rollouts])
        self.advantages = np.concatenate([self.swap_and_flatten(rollout.advantages) for rollout in rollouts])
        self.returns = np.concatenate([self.swap_and_flatten(rollout.returns) for rollout in rollouts])

        rollout_size = self.buffer_size * self.n_envs
        self.window_id = np.repeat(np.arange(self.n_rollouts, dtype=np.int64), rollout_size)
        self.generator_ready = True

    def _get_samples(self, batch_inds: np.ndarray, env=None) -> PoserRolloutBufferSamples:
        samples = super()._get_samples(batch_inds, env=env)
        return PoserRolloutBufferSamples(
            observations=samples.observations,
            actions=samples.actions,
            old_values=samples.old_values,
            old_log_prob=samples.old_log_prob,
            behavior_mean=self.to_torch(self.behavior_mean[batch_inds]),
            behavior_std=self.to_torch(self.behavior_std[batch_inds]),
            advantages=samples.advantages,
            returns=samples.returns,
            window_id=self.to_torch(self.window_id[batch_inds]),
        )
