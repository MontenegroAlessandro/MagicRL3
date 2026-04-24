from collections import deque
import warnings
from typing import Any, ClassVar, Literal, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import FloatSchedule, explained_variance
from stable_baselines3 import PPO

IS_WEIGHT_TYPE = Literal["naive", "bh"]

class RT_PPO(PPO):
    """PPO extension that reuses data from past iterations."""
    def __init__(
            self,
            on_policy_critic: bool = True,
            is_weight_type: IS_WEIGHT_TYPE = "naive",
            sequential_window_training: bool = False,
            fresh_adv: bool = False,
            on_policy_masking: bool = False,
            *args,
            **kwargs,
        ):
        """
        Args: the arguments for PPO.
        sequential_window_training: if True, train() iterates windows in recency order
            (window_id=0 first, then 1, ...). KL early stopping breaks the current
            window's mini-batch loop but the training continues with the next window.
            If False (default), all windows are mixed together as before.
        """
        super().__init__(*args, **kwargs)

        self.on_policy_critic = on_policy_critic
        self.is_weight_type = is_weight_type
        self.sequential_window_training = sequential_window_training
        self.fresh_adv = fresh_adv
        self.on_policy_masking = on_policy_masking

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        """
        After the standard rollout collection we:
          1. Store log π_current into column 0 of the buffer's all_log_probs.
          2. Evaluate all past policies on the current rollout → fill columns 1..w-1.
          3. Evaluate the current policy on all historical rollouts → append a new column.
          4. Save the current policy state_dict to _policy_history (of the buffer).
        """
        result = super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)

        if self.rollout_buffer.window_length > 1 and self.is_weight_type == "bh":
            self.rollout_buffer.set_current_policy(self.policy)
            self.rollout_buffer.update_all_log_probs()

        return result

    def _compute_ratio(
        self,
        log_prob: th.Tensor,
        rollout_data,
    ) -> th.Tensor:
        """Returns the IS weight π_new / π_behavior.
        naive: standard on-policy ratio (π_new / π_old).
        bh: π_new divided by the geometric mean of all historical policies in the window,
            so that off-policy samples are re-weighted relative to the mixture behavior policy.
        """
        if self.is_weight_type == "naive" or self.rollout_buffer.window_length == 1:
            return th.exp(log_prob - rollout_data.old_log_prob)

        all_log_probs = rollout_data.all_log_probs  # (batch, w)

        # mask valid entries (-inf = invalid)
        valid_mask = th.isfinite(all_log_probs)  # (batch, w)

        # count valid policies per sample
        valid_counts = valid_mask.sum(dim=1)  # (batch,)

        # avoid division by zero (just in case)
        valid_counts = th.clamp(valid_counts, min=1)

        # logsumexp automatically ignores -inf
        log_sum_exp = th.logsumexp(all_log_probs, dim=1)

        # normalize using ONLY valid policies
        log_mean = log_sum_exp - th.log(valid_counts.float())

        # final ratio
        ratio = th.exp(log_prob - log_mean)

        return ratio

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer (spanning across different iterations).
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # In sequential mode we iterate windows in recency order (0 = most recent).
        # In mixed mode we use a single pass with window_id=None (existing behaviour).
        if self.sequential_window_training:
            n_windows = 1 + len(self.rollout_buffer.history)
            window_ids = list(range(n_windows))
        else:
            window_ids = [None]

        # update advantages
        if self.fresh_adv:
            _gen = self.rollout_buffer.get(self.batch_size)  # just triggers generator_ready
            next(_gen)
            del _gen
            if self.rollout_buffer.window_length > 1:
                self.rollout_buffer.recompute_advantages(self.policy)

        # --- initial diagnostics (before training) ---
        initial_clip_fractions_by_window: dict[int, list[float]] = {}
        initial_kl_by_window: dict[int, list[float]] = {}
        initial_abs_ratio_by_window: dict[int, list[float]] = {}
        with th.no_grad():
            for rollout_data in self.rollout_buffer.get(batch_size=None, window_id=None):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()
                _, log_prob, _ = self.policy.evaluate_actions(rollout_data.observations, actions)
                # clip fraction uses the actual training ratio (BH or naive) to mirror what gets clipped in the loss;
                # KL and abs(1-ratio) use the naive ratio to measure raw policy shift regardless of IS scheme.
                initial_ratio = self._compute_ratio(log_prob, rollout_data)
                naive_ratio = th.exp(log_prob - rollout_data.old_log_prob)
                clipped = (th.abs(initial_ratio - 1) > clip_range).float()
                wids_in_batch = rollout_data.window_id
                for w in wids_in_batch.unique():
                    w_int = int(w.item())
                    w_mask = wids_in_batch == w
                    w_naive = naive_ratio[w_mask]
                    initial_clip_fractions_by_window.setdefault(w_int, []).append(clipped[w_mask].mean().item())
                    initial_kl_by_window.setdefault(w_int, []).append(((w_naive - 1) - th.log(w_naive)).mean().item())  # Schulman approx. reverse KL
                    initial_abs_ratio_by_window.setdefault(w_int, []).append((w_naive - 1).abs().mean().item())

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            kl_triggered_windows = set()  # tracks which window_ids hit the KL threshold this epoch

            for wid in window_ids:
                # Do a complete pass on the rollout buffer (optionally filtered by window)
                for rollout_data in self.rollout_buffer.get(self.batch_size, window_id=wid):
                    actions = rollout_data.actions
                    if isinstance(self.action_space, spaces.Discrete):
                        # Convert discrete action from float to long
                        actions = rollout_data.actions.long().flatten()

                    values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                    values = values.flatten()
                    # Normalize advantage
                    advantages = rollout_data.advantages
                    # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                    # advantage normalization made just on on-policy data
                    if self.normalize_advantage and len(advantages) > 1:
                        on_mask = rollout_data.on_policy_mask.bool()
                        if on_mask.sum() > 1 and self.on_policy_masking:
                            adv_mean = advantages[on_mask].mean()
                            adv_std = advantages[on_mask].std() + 1e-8
                        else:
                            adv_mean, adv_std = advantages.mean(), advantages.std() + 1e-8
                        advantages = (advantages - adv_mean) / adv_std

                    # ratio between old and new policy, should be one at the first iteration
                    ratio = self._compute_ratio(log_prob, rollout_data)

                    # clipped surrogate loss
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                    pg_losses.append(policy_loss.item())
                    clip_fractions.append((th.abs(ratio - 1) > clip_range).float().mean().item())

                    if self.clip_range_vf is None:
                        # No clipping
                        values_pred = values
                    else:
                        # Clip the difference between old and new value
                        # NOTE: this depends on the reward scaling
                        values_pred = rollout_data.old_values + th.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                        )
                    # Value loss using the TD(gae_lambda) target
                    # NOTE: we just use newer data for the mse computation
                    on_policy_mask = rollout_data.on_policy_mask
                    n_on_policy = on_policy_mask.sum().item()
                    if n_on_policy > 0 and self.on_policy_critic:
                        value_errors = (rollout_data.returns - values_pred) ** 2
                        value_loss = (value_errors * on_policy_mask).sum() / n_on_policy
                    elif not self.on_policy_critic:
                        value_errors = (rollout_data.returns - values_pred) ** 2
                        value_loss = value_errors.mean()
                    else:
                        value_loss = th.tensor(0.0, device=values.device)
                    value_losses.append(value_loss.item())

                    # Entropy loss favor exploration
                    if entropy is not None:
                        on_mask = rollout_data.on_policy_mask.bool()
                        if on_mask.sum() > 0 and self.on_policy_masking:
                            entropy_loss = -entropy[on_mask].mean()
                        else:
                            entropy_loss = -entropy.mean()

                    entropy_losses.append(entropy_loss.item())

                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                    # Calculate approximate form of reverse KL Divergence for early stopping.
                    # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                    # and Schulman blog: http://joschu.net/blog/kl-approx.html
                    # NOTE: it is the Schulman's approximation, it incorporates a bit of variance
                    # reduction by exploiting the second-order taylor expansion of the KL.
                    with th.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        if self.sequential_window_training:
                            # All samples in the mini-batch belong to window `wid`, so we
                            # compute KL over all of them regardless of on_policy_mask.
                            approx_kl_div = th.mean(
                                (th.exp(log_ratio) - 1) - log_ratio
                            ).cpu().numpy()
                        else:
                            # Mixed mode: guard against inflated KL from off-policy samples
                            # by computing KL only over the on-policy portion.
                            on_policy_mask_bool = rollout_data.on_policy_mask.bool()
                            if on_policy_mask_bool.sum() > 0 and self.on_policy_masking:
                                log_ratio_on = log_ratio[on_policy_mask_bool]
                                approx_kl_div = th.mean(
                                    (th.exp(log_ratio_on) - 1) - log_ratio_on
                                ).cpu().numpy()
                            else:
                                # No on-policy samples in this minibatch — skip early stopping
                                approx_kl_div = 0.0

                    if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                        if self.verbose >= 1:
                            if self.sequential_window_training:
                                print(
                                    f"Early stopping at epoch {epoch}, window {wid} "
                                    f"due to reaching max kl: {approx_kl_div:.2f}"
                                )
                            else:
                                print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                        if self.sequential_window_training:
                            # Record the violation and move on to the next (older) window.
                            kl_triggered_windows.add(wid)
                        else:
                            continue_training = False
                        break

                    # Optimization step
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()

            # In sequential mode, only stop training if every window hit the KL threshold.
            if self.sequential_window_training and kl_triggered_windows == set(window_ids):
                continue_training = False

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Default PPO logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates)
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        # Initial diagnostics
        for wid in sorted(initial_clip_fractions_by_window.keys()):
            self.logger.record(f"diagnostics_clip/initial_clip_fraction_window_{wid}", np.mean(initial_clip_fractions_by_window[wid]))
        if initial_clip_fractions_by_window:
            self.logger.record("diagnostics_clip/initial_clip_fraction_mean", np.mean([f for fracs in initial_clip_fractions_by_window.values() for f in fracs]))
        for wid in sorted(initial_kl_by_window.keys()):
            self.logger.record(f"diagnostics_kl/initial_kl_window_{wid}", np.mean(initial_kl_by_window[wid]))
        if initial_kl_by_window:
            self.logger.record("diagnostics_kl/initial_kl_mean", np.mean([v for vals in initial_kl_by_window.values() for v in vals]))
        for wid in sorted(initial_abs_ratio_by_window.keys()):
            self.logger.record(f"diagnostics_abs_ratio/initial_window_{wid}", np.mean(initial_abs_ratio_by_window[wid]))
        if initial_abs_ratio_by_window:
            self.logger.record("diagnostics_abs_ratio/initial_mean", np.mean([v for vals in initial_abs_ratio_by_window.values() for v in vals]))

        # Final diagnostics (single pass with the updated policy)
        clip_fractions_by_window: dict[int, list[float]] = {}
        kl_by_window: dict[int, list[float]] = {}
        abs_ratio_by_window: dict[int, list[float]] = {}
        with th.no_grad():
            for rollout_data in self.rollout_buffer.get(batch_size=None, window_id=None):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()
                _, log_prob, _ = self.policy.evaluate_actions(rollout_data.observations, actions)
                naive_ratio = th.exp(log_prob - rollout_data.old_log_prob)
                clipped = (th.abs(naive_ratio - 1) > clip_range).float()
                wids_in_batch = rollout_data.window_id
                for w in wids_in_batch.unique():
                    w_int = int(w.item())
                    w_mask = wids_in_batch == w
                    w_naive = naive_ratio[w_mask]
                    clip_fractions_by_window.setdefault(w_int, []).append(clipped[w_mask].mean().item())
                    kl_by_window.setdefault(w_int, []).append(((w_naive - 1) - th.log(w_naive)).mean().item())  # Schulman approx. reverse KL
                    abs_ratio_by_window.setdefault(w_int, []).append((w_naive - 1).abs().mean().item())

        for wid in sorted(clip_fractions_by_window.keys()):
            self.logger.record(f"diagnostics_clip/clip_fraction_window_{wid}", np.mean(clip_fractions_by_window[wid]))
        if clip_fractions_by_window:
            self.logger.record("diagnostics_clip/clip_fraction_mean", np.mean([f for fracs in clip_fractions_by_window.values() for f in fracs]))
        for wid in sorted(kl_by_window.keys()):
            self.logger.record(f"diagnostics_kl/kl_window_{wid}", np.mean(kl_by_window[wid]))
        if kl_by_window:
            self.logger.record("diagnostics_kl/kl_mean", np.mean([v for vals in kl_by_window.values() for v in vals]))
        for wid in sorted(abs_ratio_by_window.keys()):
            self.logger.record(f"diagnostics_abs_ratio/final_window_{wid}", np.mean(abs_ratio_by_window[wid]))
        if abs_ratio_by_window:
            self.logger.record("diagnostics_abs_ratio/final_mean", np.mean([v for vals in abs_ratio_by_window.values() for v in vals]))
