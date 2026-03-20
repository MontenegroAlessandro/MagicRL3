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
    """
    Extended rollout buffer exposing:
    1.  on_policy_mask: a binary mask indicating which samples are from the current iteration (1.0) vs 
        past iterations (0.0).
    2.  all_log_probs: (optional) log probabilities under all the behavioral policies in the window. 
        The shape is (batch_size, window_size). It is needed just for balance heuristic corrections.
    """
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    on_policy_mask: th.Tensor 
    all_log_probs: Optional[th.Tensor] = None  


class MultiRolloutBuffer(RolloutBuffer):
    def __init__(self, *args, window_size=1, use_bh: bool = False, **kwargs):
        """
        window_size = 1 means standard PPO buffer.
        window_size > 1 means we will keep data from the past `window_size-1` iterations.
        """
        self.window_length = window_size
        self.history = deque(maxlen=max(0, window_size - 1))
        self._combined_tensors = {}
        self.use_bh = use_bh and window_size > 1

        # super class init
        super().__init__(*args, **kwargs)

        # allocate the matrix of log probs for the BH
        self._current_all_log_probs = None
        if self.use_bh:
            self._current_all_log_probs = np.full(
                (self.buffer_size, self.n_envs, self.window_length),
                fill_value=-np.inf,
                dtype=np.float32,
            )
            # j-th column has the $\log \pi_{j}$ for the current rollout

    def store_current_log_probs(self, log_probs: np.ndarray) -> None:
        """
        Store the log prob of the current policy for the current rollout into column 0 of the all_log_probs matrix.
        This is already called by SB3, but we are exposing it to deal with the new buffer.
        """
        if not self.use_bh:
            return
        
        self._current_all_log_probs[:, :, 0] = log_probs
    
    def update_past_log_probs(self, log_probs: np.ndarray, history_idx: int) -> None:
        """
        Fill "log pi new" for past data. This is called after having collected a new rollout. 
        """
        if not self.use_bh:
            return
        
        entry = self.history[history_idx]
        alp = entry["all_log_probs"]          # (n_flat, n_cols)
        new_col = log_probs.reshape(-1, 1)    # (n_flat, 1)

        if alp.shape[1] < self.window_length:
            # still building ... just append
            entry["all_log_probs"] = np.concatenate([alp, new_col], axis=1)
        else:
            # already at window_length 
            # keep col 0 (behavioral)
            # drop col 1 (oldest non-behavioral), shift cols 2.. left, append new col
            entry["all_log_probs"] = np.concatenate(
                [alp[:, :1], alp[:, 2:], new_col], axis=1
            )

    def update_current_cross_log_probs(self, log_probs: np.ndarray, policy_idx: int) -> None:
        """
        Fill column `policy_idx` of _current_all_log_probs with log probs from
        a past policy evaluated on the current rollout's data.
        """
        if not self.use_bh:
            return
        
        err_msg = f"policy_idx must be in [1, window_length-1], got {policy_idx}"
        assert 0 < policy_idx < self.window_length, err_msg
        self._current_all_log_probs[:, :, policy_idx] = log_probs

    def old_reset(self) -> None:
        if self.full and self.window_length > 1:
            # first of all we need to flatten all_log_probs
            flat_all_log_probs = self._current_all_log_probs.reshape(
                self.buffer_size * self.n_envs, self.window_length
            )

            # keep just the filled columns (column 0 is always filled)
            n_filled = 1 + len(self.history)  # current + how many past we have
            flat_all_log_probs = flat_all_log_probs[:, :n_filled]

            # insert into the history
            self.history.append({
                "observations": self.observations.copy(),
                "actions": self.actions.copy(),
                "values": self.values.copy(),
                "log_probs": self.log_probs.copy(),
                "advantages": self.advantages.copy(),
                "returns": self.returns.copy(),
                "all_log_probs": flat_all_log_probs,
            })

        # reset the current log probs matrix for the next iteration
        self._current_all_log_probs.fill(-np.inf)

        # free memory for the combined tensor
        self._combined_tensors.clear()

        # reset
        super().reset()

    def reset(self) -> None:
        if self.full and self.window_length > 1:
            entry = {
                "observations": self.observations.copy(),
                "actions": self.actions.copy(),
                "values": self.values.copy(),
                "log_probs": self.log_probs.copy(),
                "advantages": self.advantages.copy(),
                "returns": self.returns.copy(),
            }
            if self.use_bh:
                flat_all_log_probs = self._current_all_log_probs.reshape(
                    self.buffer_size * self.n_envs, self.window_length
                )
                # Only keep filled columns (column 0 always filled; 1..w-1 filled lazily)
                n_filled = 1 + len(self.history)
                entry["all_log_probs"] = flat_all_log_probs[:, :n_filled].copy()

            self.history.append(entry)

        if self.use_bh and hasattr(self, "_current_all_log_probs") and self._current_all_log_probs is not None:
            self._current_all_log_probs[:] = -np.inf

        self._combined_tensors.clear()
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
                
                if self.use_bh:
                    flat_current_all_log_probs = self._current_all_log_probs.reshape(
                        current_size, self.window_length
                    )
                    # all_log_probs for current data: just column 0
                    self._combined_tensors["all_log_probs"] = flat_current_all_log_probs[:, :1]
            else:
                for tensor in _tensor_names:
                    tensors_to_concat = [self.__dict__[tensor]] + [h[tensor] for h in self.history]
                    self._combined_tensors[tensor] = np.concatenate(tensors_to_concat, axis=0)

                # Build on-policy mask: 1.0 for current, 0.0 for past
                total_size = self._combined_tensors["observations"].shape[0]
                mask = np.zeros(total_size, dtype=np.float32)
                mask[:current_size] = 1.0
                self._combined_tensors["on_policy_mask"] = mask

                if self.use_bh:
                    flat_current_all_log_probs = self._current_all_log_probs.reshape(
                        current_size, self.window_length
                    )
                    # Pad all_log_probs matrices to the same width (window_length) before concat.
                    # Current data has window_length columns; historical data may have fewer
                    # if we haven't accumulated w iterations yet.
                    padded = [flat_current_all_log_probs]
                    for h in self.history:
                        h_lp = h["all_log_probs"]           # (n_flat, n_cols)
                        n_missing = self.window_length - h_lp.shape[1]
                        if n_missing > 0:
                            pad = np.full(
                                (h_lp.shape[0], n_missing), -np.inf, dtype=np.float32
                            )
                            h_lp = np.concatenate([h_lp, pad], axis=1)
                        padded.append(h_lp)
                    self._combined_tensors["all_log_probs"] = np.concatenate(padded, axis=0)

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
        # all_log_probs is only present when use_bh=True; fall back to old_log_prob reshaped
        all_log_probs = (
            self._combined_tensors["all_log_probs"][batch_inds]
            if self.use_bh
            else self._combined_tensors["log_probs"][batch_inds].flatten().reshape(-1, 1)
        )
        data = (
            self._combined_tensors["observations"][batch_inds],
            self._combined_tensors["actions"][batch_inds].astype(np.float32, copy=False),
            self._combined_tensors["values"][batch_inds].flatten(),
            self._combined_tensors["log_probs"][batch_inds].flatten(),
            self._combined_tensors["advantages"][batch_inds].flatten(),
            self._combined_tensors["returns"][batch_inds].flatten(),
            self._combined_tensors["on_policy_mask"][batch_inds],
            all_log_probs,
        )
        return RTRolloutBufferSamples(*tuple(map(self.to_torch, data)))