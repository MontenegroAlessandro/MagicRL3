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


WEIGHT_TYPES = ("uniform", "variance-based")
DISCARD_POLICIES = ("oldest", "highest_decay")


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
        ess_decay_threshold: Optional[float] = None,
        discard_policy: str = "oldest",
        **kwargs,
    ) -> None:
        if weight_type not in WEIGHT_TYPES:
            raise ValueError(f"weight_type must be one of {WEIGHT_TYPES}, got {weight_type!r}")
        if ess_decay_threshold is not None and not (0.0 < ess_decay_threshold <= 1.0):
            raise ValueError(f"ess_decay_threshold must be in (0, 1], got {ess_decay_threshold!r}")
        if discard_policy not in DISCARD_POLICIES:
            raise ValueError(f"discard_policy must be one of {DISCARD_POLICIES}, got {discard_policy!r}")

        self.weight_type = weight_type
        self.weighted_critic = weighted_critic
        self.weight_discard_threshold = weight_discard_threshold
        self.ess_decay_threshold = ess_decay_threshold
        self.discard_policy = discard_policy
        self._ess_at_theta_k: Optional[th.Tensor] = None

        super().__init__(*args, **kwargs)

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        """Collect a rollout and store its behavior Gaussian."""
        rollout_complete = super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
        if rollout_complete:
            rollout_buffer.record_behavior_distribution(self.policy)
        return rollout_complete

    def _compute_log_mean_d2(self, rollout_data) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Per-window log-mean of the per-sample 2-Renyi divergence pi_theta || pi_{k-i}.

        Shared by the variance-based POSER weights and by ESS-decay early stopping --
        both are transforms of the same d_2 values, computed with a single forward pass.
        Returns (unique_windows, samples_per_window, log_mean_d2_by_window).
        """
        windows = rollout_data.window_id.flatten()
        unique_windows, samples_per_window = th.unique(windows, sorted=True, return_counts=True)

        with th.no_grad():
            target_distribution = self.policy.get_distribution(rollout_data.observations).distribution
            target_mean = target_distribution.mean
            target_std = target_distribution.stddev
            behavior_mean = rollout_data.behavior_mean
            behavior_std = rollout_data.behavior_std

            target_variance = target_std.square()
            behavior_variance = behavior_std.square()
            renyi_denominator = 2.0 * behavior_variance - target_variance
            # D_2 is genuinely infinite where target_variance >= 2 * behavior_variance
            # (the ratio's second moment doesn't exist there); log(non-positive) would
            # otherwise NaN this whole tensor. Substitute a placeholder to compute
            # safely, then force those entries to +inf so they still drive the
            # affected window's weight to exactly 0 instead of poisoning the softmax.
            valid = renyi_denominator > 0
            safe_denominator = th.where(valid, renyi_denominator, th.ones_like(renyi_denominator))
            log_normalization_term = 2.0 * th.log(behavior_std) - th.log(target_std) - 0.5 * th.log(safe_denominator)
            mean_difference = target_mean - behavior_mean
            quadratic_term = mean_difference.square() / safe_denominator

            action_dimensions = tuple(range(1, target_mean.ndim))
            per_dim_log_d2 = th.where(valid, log_normalization_term + quadratic_term, th.full_like(safe_denominator, float("inf")))
            sample_log_d2 = per_dim_log_d2.sum(dim=action_dimensions)

            log_mean_d2_by_window = []
            for window, sample_count in zip(unique_windows, samples_per_window):
                samples_from_window = sample_log_d2[windows == window]
                sample_count = sample_count.to(sample_log_d2.dtype)
                log_sum_d2 = th.logsumexp(samples_from_window, dim=0)
                log_mean_d2_by_window.append(log_sum_d2 - th.log(sample_count))

        return unique_windows, samples_per_window, th.stack(log_mean_d2_by_window)

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
            _, samples_per_window, log_mean_d2_by_window = self._compute_log_mean_d2(rollout_data)
            log_sample_count = th.log(samples_per_window.to(log_mean_d2_by_window.dtype))
            log_unnormalized_weights = 0.5 * (log_sample_count - log_mean_d2_by_window)
            return th.softmax(log_unnormalized_weights, dim=0)

        raise ValueError(f"weight_type must be one of {WEIGHT_TYPES}, got {weight_type!r}")

    def _compute_ess(self, rollout_data) -> th.Tensor:
        """Per-window effective sample size ESS_i(theta) = N_i / d_2(pi_theta || pi_{theta_k-i})."""
        _, samples_per_window, log_mean_d2_by_window = self._compute_log_mean_d2(rollout_data)
        n_per_window = samples_per_window.to(log_mean_d2_by_window.dtype)
        return n_per_window / log_mean_d2_by_window.exp()

    def _current_ratio(self, rollout_data) -> th.Tensor:
        """pi_theta_k(a|s) / pi_{k-i}(a|s), theta_k = the policy at the start of this train() call."""
        actions = rollout_data.actions
        if isinstance(self.action_space, spaces.Discrete):
            actions = actions.long().flatten()

        was_training = self.policy.training
        self.policy.set_training_mode(False)
        try:
            with th.no_grad():
                _, log_prob, _ = self.policy.evaluate_actions(rollout_data.observations, actions)
                return th.exp(log_prob - rollout_data.old_log_prob)
        finally:
            self.policy.set_training_mode(was_training)

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

        # Samples/behavior stats are fixed for this train() call; only theta moves.
        complete_rollout_data = self.rollout_buffer.get_all()

        # Advantage-normalization stats: a single self-normalized IS estimate over the
        # whole window, evaluated at theta_k (before any gradient step this call).
        # recompute_advantages() already put every window's advantage on the current
        # critic's scale, so one shared (mean, std) is right; the samples themselves
        # are still collected off-policy, hence the importance weighting.
        adv_mean = adv_std = None
        if self.normalize_advantage:
            is_weights = self._current_ratio(complete_rollout_data)
            total_weight = is_weights.sum()
            adv_mean = (is_weights * complete_rollout_data.advantages).sum() / total_weight
            adv_var = (is_weights * (complete_rollout_data.advantages - adv_mean) ** 2).sum() / total_weight
            adv_std = adv_var.sqrt() + 1e-8

        # ESS_i(theta_k): the baseline every later epoch's ESS-decay check (and the
        # diag/ess_decay/* diagnostics below) is measured against. theta_k = the
        # policy right now, before any gradient step this call.
        self._ess_at_theta_k = self._compute_ess(complete_rollout_data)

        # POSER diagnostics before the PPO update.
        self._log_diagnostics(clip_range, "pre")

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        early_stop_epoch = self.n_epochs
        ess_decay_min = float("inf")
        discarded_window = False
        discarded_window_id = float("nan")

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            # Recompute against the current theta once per epoch: it moved at every
            # minibatch step of the previous epoch. Per-minibatch recomputation was
            # tried and cost ~n_minibatches x more compute than the update itself.
            need_d2 = self.weight_type == "variance-based" or self.ess_decay_threshold is not None
            if need_d2:
                _, samples_per_window, log_mean_d2_by_window = self._compute_log_mean_d2(complete_rollout_data)

            if self.weight_type == "variance-based":
                log_sample_count = th.log(samples_per_window.to(log_mean_d2_by_window.dtype))
                window_weights = th.softmax(0.5 * (log_sample_count - log_mean_d2_by_window), dim=0)
            else:
                window_weights = self._compute_weights(complete_rollout_data)

            # ESS-decay early stopping: gates whether to enter this epoch at all: it
            # does not interrupt one already in progress. theta here is the iterate
            # produced by the previous epoch's updates (or theta_k, for epoch 0), so at
            # epoch 0 decay is always exactly 1 and this can never trigger there.
            if self.ess_decay_threshold is not None:
                n_per_window = samples_per_window.to(log_mean_d2_by_window.dtype)
                ess_this_epoch = n_per_window / log_mean_d2_by_window.exp()
                decay = ess_this_epoch / self._ess_at_theta_k
                epoch_min_decay = decay.min().item()
                ess_decay_min = min(ess_decay_min, epoch_min_decay)

                if epoch_min_decay < self.ess_decay_threshold:
                    early_stop_epoch = epoch
                    if self.verbose >= 1:
                        print(f"ESS-decay early stopping before epoch {epoch}: min decay {epoch_min_decay:.3f} < {self.ess_decay_threshold}")
                    break

            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage with the whole-window statistics computed above.
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - adv_mean) / adv_std

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

                # Approximate reverse KL divergence, logged as train/approx_kl below.
                # Stopping is handled entirely by ESS-decay above, not by this: see
                # issue #417 / PR #419 on DLR-RM/stable-baselines3 and Schulman's blog
                # (http://joschu.net/blog/kl-approx.html) for the estimator itself.
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1

        # Window-discard policy: fires only once the window is at capacity, matching
        # the exact point reset()'s own maxlen rotation would otherwise trigger --
        # never during warm-up (n_rollouts < window_size), where nothing should be
        # evicted yet. Runs regardless of how the epoch loop ended (early stop or all
        # n_epochs completed), respecting discard_policy: "oldest" reproduces the
        # natural FIFO rotation exactly (a no-op relative to letting reset() handle
        # it); "highest_decay" instead evicts by decay_i(theta) = ESS_i(theta) /
        # ESS_i(theta_k) at this exact exit point (freshly computed, not reused from
        # the loop above), the same quantity the stopping check above uses.
        if self.ess_decay_threshold is not None and self.rollout_buffer.n_rollouts >= self.rollout_buffer.window_size:
            if self.discard_policy == "oldest":
                window_to_discard = self.rollout_buffer.n_rollouts - 1
            else:  # "highest_decay"
                _, samples_per_window, log_mean_d2_at_exit = self._compute_log_mean_d2(complete_rollout_data)
                ess_at_exit = samples_per_window.to(log_mean_d2_at_exit.dtype) / log_mean_d2_at_exit.exp()
                decay_at_exit = ess_at_exit / self._ess_at_theta_k
                window_to_discard = int(decay_at_exit.argmin().item())

            # window 0 is the current, in-progress rollout: never discardable (nothing
            # archived to remove yet). If it has the least ESS, there is genuinely
            # nothing to evict this call.
            discarded_window = window_to_discard > 0
            if discarded_window:
                self.rollout_buffer.discard_window(window_to_discard)
                discarded_window_id = float(window_to_discard)
            if self.verbose >= 1:
                outcome = f"discarded window {window_to_discard}" if discarded_window else "window 0 has the least ESS, nothing to discard"
                print(f"POSER window management (policy={self.discard_policy}): {outcome}")

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

        self.logger.record("train/early_stop_epoch", early_stop_epoch)
        self.logger.record("train/ess_decay_min", ess_decay_min if ess_decay_min != float("inf") else float("nan"))
        self.logger.record("train/discarded_window", discarded_window)
        self.logger.record("train/discarded_window_id", discarded_window_id)

        # POSER diagnostics after the PPO update.
        self._log_diagnostics(clip_range, "post")



    def _log_diagnostics(self, clip_range: float, stage: Literal["pre", "post"]) -> None:
        """Log ratio diagnostics and, before training, advantage diagnostics."""

        # weights diagnostics
        rollout_data = self.rollout_buffer.get_all()
        window_weights_variance = self._compute_weights(rollout_data, weight_type="variance-based")

        for window_id, weight in enumerate(window_weights_variance):
            self.logger.record(f"diag/weight/variance_based/{stage}_w{window_id}", weight.item())

        # ESS-decay diagnostics: ESS_i(theta) / ESS_i(theta_k), 1.0 everywhere at "pre"
        # by construction (no gradient step has moved theta away from theta_k yet).
        ess_decay = self._compute_ess(rollout_data) / self._ess_at_theta_k
        for window_id, decay in enumerate(ess_decay):
            self.logger.record(f"diag/ess_decay/{stage}_w{window_id}", decay.item())

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



