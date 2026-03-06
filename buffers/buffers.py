import copy
from collections import deque
from collections.abc import Generator
from typing import NamedTuple, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class RTRolloutBufferSamples(NamedTuple):
    """Extended rollout buffer samples with an on-policy mask."""
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    on_policy_mask: th.Tensor  # 1.0 for current rollout, 0.0 for past


class MultiRolloutBuffer(RolloutBuffer):
    def __init__(self, *args, window_size=1, **kwargs):
        """
        window_size = 1 means standard PPO buffer.
        window_size > 1 means we will keep data from the past `window_size-1` iterations.
        """
        self.window_length = window_size
        self.history = deque(maxlen=max(0, window_size - 1))
        self._combined_tensors = {}

        # super class init
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        if self.full and self.window_length > 1:
            self.history.append({
                "observations": self.observations.copy(),
                "actions": self.actions.copy(),
                "values": self.values.copy(),
                "log_probs": self.log_probs.copy(),
                "advantages": self.advantages.copy(),
                "returns": self.returns.copy(),
            })

        # free memory for the combined tensor
        self._combined_tensors.clear()

        # reset
        super().reset()

    def get(self, batch_size=None):
        assert self.full, "Rollout buffer must be full before sampling from it"

        if not self.generator_ready:
            _tensor_names = ["observations", "actions", "values", "log_probs", "advantages", "returns"]

            # Flatten current buffer arrays
            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

            current_size = self.__dict__["observations"].shape[0]

            # If no history, just link the combined tensors to the current arrays
            if self.window_length == 1 or len(self.history) == 0:
                for tensor in _tensor_names:
                    self._combined_tensors[tensor] = self.__dict__[tensor]
                # All samples are on-policy
                self._combined_tensors["on_policy_mask"] = np.ones(current_size, dtype=np.float32)
            else:
                for tensor in _tensor_names:
                    tensors_to_concat = [self.__dict__[tensor]] + [h[tensor] for h in self.history]
                    self._combined_tensors[tensor] = np.concatenate(tensors_to_concat, axis=0)

                # Build on-policy mask: 1.0 for current, 0.0 for past
                total_size = self._combined_tensors["observations"].shape[0]
                mask = np.zeros(total_size, dtype=np.float32)
                mask[:current_size] = 1.0
                self._combined_tensors["on_policy_mask"] = mask

        # Yield minibatches from the combined dataset
        total_size = self._combined_tensors["observations"].shape[0]
        indices = np.random.permutation(total_size)

        if batch_size is None:
            batch_size = total_size

        start_idx = 0
        while start_idx < total_size:
            yield self._get_combined_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_combined_samples(self, batch_inds: np.ndarray) -> RTRolloutBufferSamples:
        """Helper to extract batches from our custom combined dictionary."""
        data = (
            self._combined_tensors["observations"][batch_inds],
            self._combined_tensors["actions"][batch_inds].astype(np.float32, copy=False),
            self._combined_tensors["values"][batch_inds].flatten(),
            self._combined_tensors["log_probs"][batch_inds].flatten(),
            self._combined_tensors["advantages"][batch_inds].flatten(),
            self._combined_tensors["returns"][batch_inds].flatten(),
            self._combined_tensors["on_policy_mask"][batch_inds],
        )
        return RTRolloutBufferSamples(*tuple(map(self.to_torch, data)))