"""POSER: PPO with diagnostics over a window of recent rollouts."""

from typing import Literal, Optional

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance

from .utils.diagnostics import (
    advantages_by_window,
    approx_kl,
    clip_fraction,
    compute_naive_ratios,
    mean_abs_deviation,
    normalized_ess,
    ratio_variance,
)


class POSER(PPO):
    """
    omega-PPO-U trained on a window of rollouts, with per-window diagnostics.
    """

    def __init__(
        self,
        *args,
        weight_type: Optional[str] = None,
        weighted_critic: bool = False,
        weight_discard_threshold: Optional[float] = None,
        optimization_stopping_threshold: Optional[float] = None,
        optimization_stopping_strategy: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.weight_type = weight_type
        self.weighted_critic = weighted_critic
        self.weight_discard_threshold = weight_discard_threshold
        self.optimization_stopping_threshold = optimization_stopping_threshold
        self.optimization_stopping_strategy = optimization_stopping_strategy

        super().__init__(*args, **kwargs)

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        """Collect a rollout and store its behavior Gaussian."""
        rollout_complete = super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
        if rollout_complete:
            rollout_buffer.record_behavior_distribution(self.policy)
        return rollout_complete

    def _compute_weights(self, rollout_data, weight_type=None) -> th.Tensor:
        """Compute uniform or variance-based POSER weights."""
        weight_type = weight_type or self.weight_type

        windows = rollout_data.window_id.flatten()
        unique_windows, samples_per_window = th.unique(windows, sorted=True, return_counts=True)
        number_of_windows = unique_windows.numel()

        if weight_type == "uniform":
            uniform_weight = 1.0 / number_of_windows
            return th.full(size=(number_of_windows,), fill_value=uniform_weight, dtype=rollout_data.behavior_mean.dtype, device=rollout_data.behavior_mean.device)

        if weight_type == "variance-based":
            with th.no_grad():
                target_distribution = self.policy.get_distribution(rollout_data.observations).distribution
                target_mean = target_distribution.mean
                target_std = target_distribution.stddev
                behavior_mean = rollout_data.behavior_mean
                behavior_std = rollout_data.behavior_std

                target_variance = target_std.square()
                behavior_variance = behavior_std.square()
                renyi_denominator = 2.0 * behavior_variance - target_variance
                log_normalization_term = 2.0 * th.log(behavior_std) - th.log(target_std) - 0.5 * th.log(renyi_denominator)
                mean_difference = target_mean - behavior_mean
                quadratic_term = mean_difference.square() / renyi_denominator

                action_dimensions = tuple(range(1, target_mean.ndim))
                sample_log_d2 = (log_normalization_term + quadratic_term).sum(dim=action_dimensions)

                log_unnormalized_weights = []
                for window, sample_count in zip(unique_windows, samples_per_window):
                    samples_from_window = sample_log_d2[windows == window]
                    sample_count = sample_count.to(sample_log_d2.dtype)
                    log_sum_d2 = th.logsumexp(samples_from_window, dim=0)
                    log_mean_d2 = log_sum_d2 - th.log(sample_count)
                    log_inverse_radius = 0.5 * (th.log(sample_count) - log_mean_d2)
                    log_unnormalized_weights.append(log_inverse_radius)

                log_unnormalized_weights = th.stack(log_unnormalized_weights)
                return th.softmax(log_unnormalized_weights, dim=0)

    def train(self) -> None:
        """Update the policy using PPO and log diagnostics before and after."""
        # This is PPO.train() from Stable-Baselines3 2.7.1. POSER-specific
        # diagnostics are explicitly marked and do not affect the update.

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        # Recompute historical V-trace targets before diagnostics flatten the buffer.
        self.rollout_buffer.recompute_advantages(self.policy)

        # Compute the window weights once on the complete rollout window.
        complete_rollout_data = self.rollout_buffer.get_all()
        window_weights = self._compute_weights(complete_rollout_data)

        # POSER diagnostics before the PPO update.
        self._log_diagnostics(clip_range, "pre")

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # Weighted POSER clipped surrogate: sum_i w_i L_i.
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                clipped_objective = th.min(policy_loss_1, policy_loss_2)
                unique_windows = th.unique(rollout_data.window_id, sorted=True)
                batch_window_weights = window_weights[unique_windows.long()]
                window_objectives = []
                for window in unique_windows:
                    samples_from_window = rollout_data.window_id == window
                    window_objectives.append(clipped_objective[samples_from_window].mean())
                policy_loss = -th.sum(batch_window_weights * th.stack(window_objectives))

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction_value = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction_value)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_errors = F.mse_loss(rollout_data.returns, values_pred, reduction="none")
                if self.weighted_critic:
                    window_value_losses = []
                    for window in unique_windows:
                        samples_from_window = rollout_data.window_id == window
                        window_value_losses.append(value_errors[samples_from_window].mean())
                    value_loss = th.sum(
                        batch_window_weights * th.stack(window_value_losses)
                    )
                else:
                    value_loss = value_errors.mean()
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        # POSER diagnostics after the PPO update.
        self._log_diagnostics(clip_range, "post")



    def _log_diagnostics(self, clip_range: float, stage: Literal["pre", "post"]) -> None:
        """Log ratio diagnostics and, before training, advantage diagnostics."""

        # weights diagnostics
        rollout_data = self.rollout_buffer.get_all()
        window_weights_variance = self._compute_weights(rollout_data, weight_type="variance")

        for window_id, weight in enumerate(window_weights_variance):
            self.logger.record(f"diag/weight/variance_based/{stage}_w{window_id}", weight.item())

        # ratio diagnostics
        ratios = compute_naive_ratios(self.policy, rollout_data, self.action_space)

        window_ids = rollout_data.window_id.long()
        unique_windows = th.unique(window_ids, sorted=True)
        subsets = {
            f"w{window_id.item()}": ratios[window_ids == window_id]
            for window_id in unique_windows
        }
        subsets["mean"] = ratios

        for suffix, ratio in subsets.items():
            self.logger.record(f"diag/clip_fraction/{stage}_{suffix}", clip_fraction(ratio, clip_range).item())
            self.logger.record(f"diag/approx_kl/{stage}_{suffix}", approx_kl(ratio).item())
            self.logger.record(f"diag/abs_ratio/{stage}_{suffix}", mean_abs_deviation(ratio).item())
            self.logger.record(f"diag/normalized_ess/{stage}_{suffix}", normalized_ess(ratio).item())
            self.logger.record(f"diag/ratio_variance/{stage}_{suffix}", ratio_variance(ratio).item())



