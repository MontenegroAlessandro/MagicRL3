from collections import deque
from collections.abc import Generator
from typing import List, Optional, Union, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.vec_env import VecNormalize

from buffers.trajectory_buffer import TrajectoryBuffer


class MultiTrajectoryBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    rewards: th.Tensor
    episode_starts: th.Tensor
    returns: th.Tensor
    on_policy_mask: th.Tensor   # 1.0 = current rollout, 0.0 = history
    window_id: th.Tensor        # 0 = current, 1 = previous, ..., w-1 = oldest


class MultiTrajectoryBuffer(TrajectoryBuffer):
    """
    Extension of TrajectoryBuffer that retains the last `window_length` rollouts.

    window_length = 1 is identical to TrajectoryBuffer.
    window_length > 1 keeps the previous window_length-1 completed rollouts and
    concatenates them with the current one in get(), exposing on_policy_mask and
    window_id so the training algorithm can distinguish on- vs off-policy samples.

    :param window_length: Total number of rollouts to keep (current + history).
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gamma: float = 0.99,
        n_envs: int = 1,
        window_length: int = 1,
    ):
        self.window_length = window_length
        # history holds up to window_length-1 past rollouts as flat numpy dicts
        self.history: deque = deque(maxlen=max(0, window_length - 1))
        self._combined_flat: dict = {}
        super().__init__(buffer_size, observation_space, action_space, device, gamma, n_envs)

    def reset(self) -> None:
        # Snapshot the completed rollout into history before clearing.
        if self.full and self.window_length > 1:
            active = [i for i in range(self.n_envs) if self._traj_lengths[i] > 0]
            if active:
                entry = {
                    "observations": np.concatenate(
                        [np.array(self._obs[i]) for i in active], axis=0
                    ),
                    "actions": np.concatenate(
                        [np.array(self._actions[i]) for i in active], axis=0
                    ).astype(np.float32, copy=False),
                    "rewards": np.concatenate(
                        [np.array(self._rewards[i], dtype=np.float32) for i in active]
                    ),
                    "episode_starts": np.concatenate(
                        [np.array(self._episode_starts[i], dtype=np.float32) for i in active]
                    ),
                    "returns": np.concatenate([self._returns[i] for i in active]),
                }
                self.history.appendleft(entry)

        self._combined_flat.clear()
        super().reset()

    def get(self) -> Generator[MultiTrajectoryBufferSamples, None, None]:
        if not self.full:
            self.compute_returns()

        active = [i for i in range(self.n_envs) if self._traj_lengths[i] > 0]
        assert active, "No trajectories collected — call add() before get()"

        current = {
            "observations": np.concatenate(
                [np.array(self._obs[i]) for i in active], axis=0
            ),
            "actions": np.concatenate(
                [np.array(self._actions[i]) for i in active], axis=0
            ).astype(np.float32, copy=False),
            "rewards": np.concatenate(
                [np.array(self._rewards[i], dtype=np.float32) for i in active]
            ),
            "episode_starts": np.concatenate(
                [np.array(self._episode_starts[i], dtype=np.float32) for i in active]
            ),
            "returns": np.concatenate([self._returns[i] for i in active]),
        }

        current_size = current["observations"].shape[0]

        if self.window_length == 1 or len(self.history) == 0:
            self._combined_flat = current
            self._combined_flat["on_policy_mask"] = np.ones(current_size, dtype=np.float32)
            self._combined_flat["window_id"] = np.zeros(current_size, dtype=np.float32)
        else:
            keys = ["observations", "actions", "rewards", "episode_starts", "returns"]
            self._combined_flat = {
                k: np.concatenate([current[k]] + [h[k] for h in self.history], axis=0)
                for k in keys
            }

            total_size = self._combined_flat["observations"].shape[0]

            on_policy_mask = np.zeros(total_size, dtype=np.float32)
            on_policy_mask[:current_size] = 1.0
            self._combined_flat["on_policy_mask"] = on_policy_mask

            window_id = np.zeros(current_size, dtype=np.float32)
            for i, h in enumerate(self.history):
                window_id = np.concatenate(
                    [window_id, np.full(h["observations"].shape[0], i + 1, dtype=np.float32)]
                )
            self._combined_flat["window_id"] = window_id

        # Populate parent flat arrays so _get_samples stays consistent
        self._flat_obs = self._combined_flat["observations"]
        self._flat_actions = self._combined_flat["actions"]
        self._flat_rewards = self._combined_flat["rewards"]
        self._flat_episode_starts = self._combined_flat["episode_starts"]
        self._flat_returns = self._combined_flat["returns"]

        data = (
            self._combined_flat["observations"],
            self._combined_flat["actions"],
            self._combined_flat["rewards"],
            self._combined_flat["episode_starts"],
            self._combined_flat["returns"],
            self._combined_flat["on_policy_mask"],
            self._combined_flat["window_id"],
        )
        yield MultiTrajectoryBufferSamples(*tuple(map(self.to_torch, data)))

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> MultiTrajectoryBufferSamples:
        assert self._combined_flat, "Call get() before _get_samples()"
        data = (
            self._combined_flat["observations"][batch_inds],
            self._combined_flat["actions"][batch_inds],
            self._combined_flat["rewards"][batch_inds],
            self._combined_flat["episode_starts"][batch_inds],
            self._combined_flat["returns"][batch_inds],
            self._combined_flat["on_policy_mask"][batch_inds],
            self._combined_flat["window_id"][batch_inds],
        )
        return MultiTrajectoryBufferSamples(*tuple(map(self.to_torch, data)))
