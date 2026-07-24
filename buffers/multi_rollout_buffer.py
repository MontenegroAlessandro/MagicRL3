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


# this is the data structure that arrives to the policy in train()
class RTRolloutBufferSamples(NamedTuple):
    """
    Extended rollout buffer exposing:
    1.  on_policy_mask: a binary mask indicating which samples are from the current iteration (1.0) vs 
        past iterations (0.0).
    2.  all_log_probs: (optional) log probabilities under all the behavioral policies in the window. 
        The shape is (batch_size, window_size). It is needed just for balance heuristic corrections.
    """
    observations: th.Tensor  # (n_steps * n_envs, obs_shape)
    actions: th.Tensor  # (n_steps * n_envs, action_shape)
    old_values: th.Tensor  # (n_steps * n_envs)
    old_log_prob: th.Tensor  # (n_steps * n_envs)
    advantages: th.Tensor  # (n_steps * n_envs)
    returns: th.Tensor  # (n_steps * n_envs)
    on_policy_mask: th.Tensor  # (n_steps * n_envs) binary mask: 1.0 for current iteration's samples, 0.0 for past iterations
    window_id: th.Tensor  # (n_steps * n_envs) integer in [0, window_size-1] indicating which policy in the window generated the sample; 0 means current policy, 1 means previous policy, etc.
    all_log_probs: Optional[th.Tensor] = None  # (n_steps * n_envs, window_size) log probs under all behavioral policies in the window; only present if use_bh=True
    

class MultiRolloutBuffer(RolloutBuffer):
    def __init__(self, *args, window_size=1, use_bh: bool = False, balanced_batches: bool = False, **kwargs):
        """
        window_size = 1 means standard PPO buffer.
        window_size > 1 means we will keep data from the past `window_size-1` iterations.
        """
        self.window_length = window_size
        self.history = deque(maxlen=max(0, window_size - 1))
        self.use_bh = use_bh and window_size > 1
        self.balanced_batches = balanced_batches and window_size > 1
        self._combined_tensors = {}

        # super class init
        super().__init__(*args, **kwargs)

    def set_current_policy(self, policy):
        # Store a frozen copy of the policy directly on the correct device
        policy_copy = copy.deepcopy(policy)
        policy_copy.to(policy.device)
        policy_copy.set_training_mode(False)

        self._current_policy = policy_copy

    def update_all_log_probs(self):
        """
        Current situation: 
        - current policy π_k is stored in self._current_policy
        - in self we have the current rollout_k (observations, actions, returns, etc)
        - history = [rollout_{k-1}, rollout_{k-2}, ..., rollout_{k-w+1}]  -->  length = w-1
        - history = [π_{k-1}, π_{k-2}, ..., π_{k-w+1}]  -->  length = w-1
        - rollout_j["all_log_probs"] = [log π_{k-1}, log π_{k-2}, ..., log π_{k-w+1}, log π_{k-w}]  -->  length = w

        Step 1:
        - we want to fill self._current_all_log_probs, which has shape (n_steps, n_envs, window_size)
        - self._current_all_log_probs = [log π_k, log π_{k-1}, log π_{k-2}, ..., log π_{k-w+1}]

        Step 2:
        - we want to update all_log_probs in the history with the new policy π_k evaluated on past data
        - rollout_j["all_log_probs"] = [log π_k, log π_{k-1}, log π_{k-2}, ..., log π_{k-w+1}] for each j in history
        """
        
        if not self.use_bh:
            return

        assert self._current_policy is not None, "Policy must be set before update_all_log_probs()"

        # Step 0: get flattened current rollout
        obs = self.swap_and_flatten(self.observations)
        actions = self.swap_and_flatten(self.actions)

        # Step 1: fill current rollout matrix
        self._fill_current_all_log_probs(obs, actions)

        # Step 2: update history
        self._update_history_all_log_probs()

    def _fill_current_all_log_probs(self, obs, actions):
        """
        Fill self._current_all_log_probs with:
        [π_k, π_{k-1}, ..., π_{k-w+1}]
        """
        # Column 0 → current policy
        log_prob = self._eval_log_prob(self._current_policy, obs, actions)
        self._current_all_log_probs[:, :, 0] = self._unflatten_and_swap(log_prob, self.buffer_size, self.n_envs)

        # Columns 1..w-1 → past policies
        for i, entry in enumerate(self.history):
            if "policy" not in entry:
                continue

            past_policy = entry["policy"]
            log_prob = self._eval_log_prob(past_policy, obs, actions)

            self._current_all_log_probs[:, :, i + 1] = self._unflatten_and_swap(log_prob, self.buffer_size, self.n_envs)

    def _unflatten_and_swap(self, arr, n_steps, n_envs):
        # utility function to revert the flattening and swapping done in swap_and_flatten by SB3
        return arr.reshape(n_envs, n_steps, *arr.shape[1:]).swapaxes(0, 1)

    def _update_history_all_log_probs(self):
        """
        Prepend log π_k to each history entry and truncate.
        history[j]["all_log_probs"] = [log π_k, log π_{k-1}, log π_{k-2}, ..., log π_{k-w+1}] for each j
        """
        for entry in self.history:
            if "policy" not in entry or "all_log_probs" not in entry:
                continue

            obs = entry["observations"]  # already flattened
            actions = entry["actions"]  # already flattened

            log_prob = self._eval_log_prob(self._current_policy, obs, actions)
            log_prob = log_prob.reshape(-1, 1)

            # prepend π_k
            entry["all_log_probs"] = np.concatenate(
                [log_prob, entry["all_log_probs"]],
                axis=1,
            )

            # truncate to window size
            entry["all_log_probs"] = entry["all_log_probs"][:, :self.window_length]

    def _eval_log_prob(self, policy, obs, actions):
        """
        Evaluate log π(a|s) for a given policy on numpy inputs.
        Returns a flat numpy array.
        """
        device = policy.device

        obs_t = th.as_tensor(obs).to(device)
        actions_t = th.as_tensor(actions).to(device)

        with th.no_grad():
            _, log_prob, _ = policy.evaluate_actions(obs_t, actions_t)

        return log_prob.cpu().numpy()

    def recompute_advantages(self, policy) -> None:
        """
        When this method is called, then we recompute the advantage estimates fro the data in the window.
        This is achieved by employing VTRACE (with trunctation hard-coded to 1.0 for simplicity).
        Notice that this procedure will be done also for the on-policy data, since they where computed via GAE if the
        flag for activating this method is set to False.
        Notice that the return estimates too will no longer be stale and can in principle be used to fit the critic.
        """
        assert self.generator_ready, "Call get() first to build _combined_tensors"

        obs = self._combined_tensors["observations"]
        actions = self._combined_tensors["actions"]
        old_log_probs = self._combined_tensors["log_probs"].flatten()
        rewards = self._combined_tensors["rewards"].flatten()
        episode_starts = self._combined_tensors["episode_starts"].flatten()

        obs_t = th.as_tensor(obs).to(policy.device)
        actions_t = th.as_tensor(actions).to(policy.device)

        with th.no_grad():
            fresh_values_t, new_log_prob_t, _ = policy.evaluate_actions(obs_t, actions_t)

        fresh_values = fresh_values_t.cpu().numpy().flatten()
        new_log_probs = new_log_prob_t.cpu().numpy().flatten()

        # V-TRACE IS ratios clipped to 1.0: rho_bar_t = min(1, pi_current / pi_behavioral)
        rho_bar = np.minimum(1.0, np.exp(new_log_probs - old_log_probs))

        n = self.buffer_size   # n_steps per rollout
        m = self.n_envs
        n_rollouts = obs.shape[0] // (n * m)

        # Reshape to (n_rollouts, n_envs, n_steps); each rollout chunk is [env0_steps, env1_steps, ...]
        rho = rho_bar.reshape(n_rollouts, m, n)
        V = fresh_values.reshape(n_rollouts, m, n)
        R = rewards.reshape(n_rollouts, m, n)
        # episode_starts[t] == 1 marca il PRIMO passo di un nuovo episodio: serve a
        # mascherare il bootstrap attraverso i confini di episodio dentro il rollout.
        ep_starts = episode_starts.reshape(n_rollouts, m, n)

        # Backward pass — GAE with V-TRACE IS correction (matches reference gae_vtrace).
        #
        # Advantage:    A_t = δ'_t + γλ · non_term_{t+1} · c_{t+1} · A_{t+1}
        # Value target: v_t = c_t · A_t + V(s_t)           (≡ rtg = adv * ratio_trunc + V)
        #
        # where δ'_t = r_t + γ · non_term_{t+1} · V(s_{t+1}) - V(s_t)  (raw TD error, no IS weight on first term)
        #       c_t  = min(1, π_k(a_t|s_t) / π_behavioral(a_t|s_t))
        #       non_term_{t+1} = 1 - episode_starts[t+1]  (0 se t è terminale -> niente bootstrap)
        v_trace = np.zeros_like(V)
        adv = np.zeros_like(V)

        V_next = np.zeros((n_rollouts, m))   # V(s_T) = 0  (bootstrap)
        A_next = np.zeros((n_rollouts, m))   # A_T    = 0  (no future steps)
        c_next = np.zeros((n_rollouts, m))   # c_T    = 0  (boundary)

        for t in reversed(range(n)):
            if t == n - 1:
                non_term = np.zeros((n_rollouts, m))      # confine di rollout: V(s_T)=0, come prima
            else:
                non_term = 1.0 - ep_starts[:, :, t + 1]   # 0 se t era terminale -> niente bootstrap

            delta_t = R[:, :, t] + self.gamma * non_term * V_next - V[:, :, t]
            adv[:, :, t] = delta_t + self.gamma * self.gae_lambda * non_term * c_next * A_next
            v_trace[:, :, t] = rho[:, :, t] * adv[:, :, t] + V[:, :, t]
            c_next = rho[:, :, t]
            A_next = adv[:, :, t]
            V_next = V[:, :, t]

        self._combined_tensors["advantages"] = adv.reshape(-1, 1)
        self._combined_tensors["returns"] = v_trace.reshape(-1, 1)
    
    def reset(self) -> None:
        if self.full and self.window_length > 1:
            entry = {
                "observations": self.observations.copy(),
                "actions": self.actions.copy(),
                "rewards": self.rewards.copy(),
                "values": self.values.copy(),
                "log_probs": self.log_probs.copy(),
                "advantages": self.advantages.copy(),
                "returns": self.returns.copy(),
                "episode_starts": self.episode_starts.copy(),
            }
            if self.use_bh:
                entry["policy"] = self._current_policy
                entry["all_log_probs"] = self._current_all_log_probs.copy() 

            self.history.appendleft(entry)

        self._current_policy = None
        self._current_all_log_probs = np.full(
            (self.buffer_size, self.n_envs, self.window_length),
            fill_value=-np.inf,
            dtype=np.float32,
        )
        self._combined_tensors.clear()
        super().reset()

    def _yield_unbalanced(self, batch_size, window_id=None):
        """Yields minibatches without window balancing, optionally filtered by window_id."""
        total_size = self._combined_tensors["observations"].shape[0]

        if window_id is not None:
            valid_mask = self._combined_tensors["window_id"] == window_id
            candidate_indices = np.where(valid_mask)[0]
        else:
            candidate_indices = np.arange(total_size)

        indices = np.random.permutation(candidate_indices)
        n_samples = len(indices)
        if batch_size is None:
            batch_size = n_samples

        start_idx = 0
        while start_idx < n_samples:
            yield self._get_combined_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def get(self, batch_size=None, window_id: Optional[int] = None):
        '''
        Current situation:
        - in self we have the current rollout (observations, actions, returns, etc)
        - self.all_log_probs = [log π_k, log π_{k-1}, ..., log π_{k-w+1}] for the current rollout
        - history = [rollout_{k-1}, rollout_{k-2}, ..., rollout_{k-w+1}]  -->  length = w-1
        - history = [π_{k-1}, π_{k-2}, ..., π_{k-w+1}]  -->  length = w-1
        - rollout_j["all_log_probs"] = [log π_k, log π_{k-1}, ..., log π_{k-w+1}] for each j

        This function concateates the current rollout with the historical rollouts
        Then it yields minibatches from the combined dataset
        '''

        assert self.full, "Rollout buffer must be full before sampling from it"

        _tensor_names = ["observations", "actions", "rewards", "values", "log_probs", "advantages", "returns", "episode_starts"]

        if not self.generator_ready:
            # Flatten current buffer arrays
            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            if self.use_bh:
                self._current_all_log_probs = self.swap_and_flatten(self._current_all_log_probs)
            self.generator_ready = True

            current_size = self.__dict__["observations"].shape[0]

            # If no history, just link the combined tensors to the current arrays
            if self.window_length == 1 or len(self.history) == 0:
                for tensor in _tensor_names:
                    self._combined_tensors[tensor] = self.__dict__[tensor]

                # All samples are on-policy
                self._combined_tensors["on_policy_mask"] = np.ones(current_size, dtype=np.float32)
                self._combined_tensors["window_id"] = np.zeros(current_size, dtype=np.float32)

                if self.use_bh:
                    self._combined_tensors["all_log_probs"] = self._current_all_log_probs

            else:
                for tensor in _tensor_names:
                    tensors_to_concat = [self.__dict__[tensor]] + [h[tensor] for h in self.history]
                    self._combined_tensors[tensor] = np.concatenate(tensors_to_concat, axis=0)

                # Build on-policy mask: 1.0 for current, 0.0 for past
                total_size = self._combined_tensors["observations"].shape[0]
                mask = np.zeros(total_size, dtype=np.float32)
                mask[:current_size] = 1.0
                self._combined_tensors["on_policy_mask"] = mask

                # Build window_id: 0 for current, 1 for previous, ... , w-1 for oldest in the window
                window_ids = np.zeros(current_size, dtype=np.float32)  # current rollout = 0
                for i, h in enumerate(self.history):
                    size = h["observations"].shape[0]
                    window_ids = np.concatenate([window_ids, np.full(size, i + 1, dtype=np.float32)])
                self._combined_tensors["window_id"] = window_ids

                if self.use_bh:
                    all_log_probs_to_concat = [self._current_all_log_probs] + [h["all_log_probs"] for h in self.history]
                    self._combined_tensors["all_log_probs"] = np.concatenate(all_log_probs_to_concat, axis=0)

        # Yield minibatches from the combined dataset
        n_windows = 1 + len(self.history) if self.window_length > 1 else 1

        # fallback
        if (not self.balanced_batches
                or window_id is not None
                or n_windows == 1
                or batch_size is None):
            yield from self._yield_unbalanced(batch_size, window_id=window_id)
            return

        # Balanced minibatch sampling: equal representation per window
        per_window = batch_size // n_windows
        if per_window == 0:
            raise ValueError(
                f"batch_size={batch_size} is too small for n_windows={n_windows}. "
                f"Need batch_size >= {n_windows}."
            )

        # Shuffle each window's indices independently
        window_indices = []
        for wid in range(n_windows):
            mask = self._combined_tensors["window_id"] == wid
            candidates = np.where(mask)[0]
            window_indices.append(np.random.permutation(candidates))

        # Number of complete balanced minibatches limited by the smallest window
        n_minibatches = min(len(wi) for wi in window_indices) // per_window

        for mb in range(n_minibatches):
            batch = np.concatenate([
                wi[mb * per_window: (mb + 1) * per_window]
                for wi in window_indices
            ])
            batch = np.random.permutation(batch)  # shuffle within minibatch
            yield self._get_combined_samples(batch)



    def _get_combined_samples(self, batch_inds: np.ndarray) -> RTRolloutBufferSamples:
        data = (
            self._combined_tensors["observations"][batch_inds],
            self._combined_tensors["actions"][batch_inds].astype(np.float32, copy=False),
            self._combined_tensors["values"][batch_inds].flatten(),
            self._combined_tensors["log_probs"][batch_inds].flatten(),
            self._combined_tensors["advantages"][batch_inds].flatten(),
            self._combined_tensors["returns"][batch_inds].flatten(),
            self._combined_tensors["on_policy_mask"][batch_inds],
            self._combined_tensors["window_id"][batch_inds],
        )

        if self.use_bh:
            data += (self._combined_tensors["all_log_probs"][batch_inds],)

        return RTRolloutBufferSamples(*tuple(map(self.to_torch, data)))