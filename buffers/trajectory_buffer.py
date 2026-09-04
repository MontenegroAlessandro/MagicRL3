from collections.abc import Generator
from typing import List, Optional, Union, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import BaseBuffer

try:
    import psutil
except ImportError:
    psutil = None


class TrajectoryBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    rewards: th.Tensor
    episode_starts: th.Tensor
    returns: th.Tensor


class TrajectoryBuffer(BaseBuffer):
    """
    Buffer for trajectory-based on-policy algorithms (e.g., REINFORCE, FDPG).

    Each of the n_envs parallel environments collects exactly one complete trajectory
    per iteration. Trajectories may have different lengths across environments.
    Returns-to-go are computed per trajectory after collection is done.

    Calling get() yields a single batch with all transitions from all environments
    concatenated together (no mini-batching, no padding).

    :param buffer_size: Upper bound on trajectory length (used only as a soft guard)
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gamma: Discount factor for returns-to-go
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gamma = gamma
        self.reset()

    def reset(self) -> None:
        # Ragged per-env storage: one list of steps per environment
        self._obs: List[List[np.ndarray]] = [[] for _ in range(self.n_envs)]
        self._actions: List[List[np.ndarray]] = [[] for _ in range(self.n_envs)]
        self._rewards: List[List[float]] = [[] for _ in range(self.n_envs)]
        self._episode_starts: List[List[float]] = [[] for _ in range(self.n_envs)]
        self._returns: List[Optional[np.ndarray]] = [None] * self.n_envs
        self._traj_lengths: np.ndarray = np.zeros(self.n_envs, dtype=np.int64)

        # Populated by get() for optional indexed access after flattening
        self._flat_obs: Optional[np.ndarray] = None
        self._flat_actions: Optional[np.ndarray] = None
        self._flat_rewards: Optional[np.ndarray] = None
        self._flat_episode_starts: Optional[np.ndarray] = None
        self._flat_returns: Optional[np.ndarray] = None

        super().reset()  # sets pos=0, full=False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        env_indices: Optional[np.ndarray] = None,
    ) -> None:
        """
        Append one transition for each active environment.

        :param obs: Observations, shape (n_active, *obs_shape)
        :param action: Actions, shape (n_active, *action_shape)
        :param reward: Rewards, shape (n_active,)
        :param episode_start: Episode-start flags, shape (n_active,)
        :param env_indices: Indices of the active environments within [0, n_envs).
                            Defaults to all n_envs environments.
        """
        if env_indices is None:
            env_indices = np.arange(self.n_envs)

        n_active = len(env_indices)

        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((n_active, *self.obs_shape))

        action = action.reshape((n_active, self.action_dim))

        for i, env_idx in enumerate(env_indices):
            self._obs[env_idx].append(np.array(obs[i], copy=True))
            self._actions[env_idx].append(np.array(action[i], copy=True))
            self._rewards[env_idx].append(float(reward[i]))
            self._episode_starts[env_idx].append(float(episode_start[i]))
            self._traj_lengths[env_idx] += 1

    def compute_returns(self) -> None:
        """
        Compute returns-to-go for every collected trajectory and mark the buffer as full.
        Must be called once all trajectories have been fully collected.
        """
        lengths = self._traj_lengths  # (n_envs,)
        max_len = int(lengths.max()) if lengths.any() else 0

        if max_len == 0:
            self._returns = [np.empty(0, dtype=np.float32)] * self.n_envs
            self.full = True
            return

        # (n_envs, max_len) — zeros beyond each trajectory's end
        padded = np.zeros((self.n_envs, max_len), dtype=np.float32)
        for i in range(self.n_envs):
            padded[i, : lengths[i]] = self._rewards[i]

        if self.gamma == 0.0:
            returns_2d = padded
        else:
            # build the vector of discounts (gamma^0, gamma^1, ..., gamma^(max_len-1))
            gamma_powers = self.gamma ** np.arange(max_len, dtype=np.float64)  # (max_len,)
            # compute the rewards with dicsounts
            scaled = padded.astype(np.float64) * gamma_powers      # (n_envs, max_len)
            # do the reverse sum place  
            # [:, ::-1] is actually reversing the rows
            suffix = np.cumsum(scaled[:, ::-1], axis=1)[:, ::-1]               # (n_envs, max_len)
            # scale by the actual discount factor at time t to get the correct returns-to-go
            # (the cumsum accumulates unwanted gamma factors in each element due to the multiplication by gamma_powers)
            returns_2d = (suffix / gamma_powers).astype(np.float32)                                  # (n_envs, max_len)
            # NOTE: we use float64 just as a guard against numerical instabilities for the gamma division

        self._returns = [returns_2d[i, : lengths[i]] for i in range(self.n_envs)]
        self.full = True

    def get(self) -> Generator[TrajectoryBufferSamples, None, None]:
        """
        Yield a single batch containing all transitions from all trajectories.
        Calls compute_returns() automatically if not already done.
        """
        if not self.full:
            self.compute_returns()

        active = [i for i in range(self.n_envs) if self._traj_lengths[i] > 0]
        assert active, "No trajectories collected — call add() before get()"

        self._flat_obs = np.concatenate(
            [np.array(self._obs[i]) for i in active], axis=0
        )
        self._flat_actions = np.concatenate(
            [np.array(self._actions[i]) for i in active], axis=0
        ).astype(np.float32, copy=False)
        self._flat_rewards = np.concatenate(
            [np.array(self._rewards[i], dtype=np.float32) for i in active]
        )
        self._flat_episode_starts = np.concatenate(
            [np.array(self._episode_starts[i], dtype=np.float32) for i in active]
        )
        self._flat_returns = np.concatenate(
            [self._returns[i] for i in active]
        )

        data = (
            self._flat_obs,
            self._flat_actions,
            self._flat_rewards,
            self._flat_episode_starts,
            self._flat_returns,
        )
        yield TrajectoryBufferSamples(*tuple(map(self.to_torch, data)))

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> TrajectoryBufferSamples:
        """
        Index-based sampling into the flattened buffer.
        Requires get() to have been called first so the flat arrays are populated.
        """
        assert self._flat_obs is not None, "Call get() before _get_samples()"
        data = (
            self._flat_obs[batch_inds],
            self._flat_actions[batch_inds],
            self._flat_rewards[batch_inds],
            self._flat_episode_starts[batch_inds],
            self._flat_returns[batch_inds],
        )
        return TrajectoryBufferSamples(*tuple(map(self.to_torch, data)))
