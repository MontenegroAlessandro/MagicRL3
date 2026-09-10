"""POSER: PPO with diagnostics over a window of recent rollouts."""

import copy
from typing import Optional

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.utils import explained_variance, obs_as_tensor

from .utils.poser_diagnostics import critic_statistics, ratio_statistics


WEIGHT_TYPES = ("uniform", "d2_rad", "d2")
D2_WEIGHT_TYPES = ("d2_rad", "d2")
DISCARD_POLICIES = ("oldest", "highest_decay")
CLIP_RANGE_ADAPTATIONS = ("none", "weighted", "weighted_geppo")


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
        psr_threshold: Optional[float] = None,
        discard_policy: str = "oldest",
        clip_range_adaptation: str = "none",
        debug: bool = False,
        **kwargs,
    ) -> None:
        if weight_type not in WEIGHT_TYPES:
            raise ValueError(f"weight_type must be one of {WEIGHT_TYPES}, got {weight_type!r}")
        if psr_threshold is not None and psr_threshold < 0.0:
            raise ValueError(f"psr_threshold must be non-negative, got {psr_threshold!r}")
        if discard_policy not in DISCARD_POLICIES:
            raise ValueError(f"discard_policy must be one of {DISCARD_POLICIES}, got {discard_policy!r}")
        if clip_range_adaptation not in CLIP_RANGE_ADAPTATIONS:
            raise ValueError(
                "clip_range_adaptation must be one of "
                f"{CLIP_RANGE_ADAPTATIONS}, got {clip_range_adaptation!r}"
            )

        self.weight_type = weight_type
        self.weighted_critic = weighted_critic
        self.weight_discard_threshold = weight_discard_threshold
        self.psr_threshold = psr_threshold
        self.discard_policy = discard_policy
        self.clip_range_adaptation = clip_range_adaptation
        self.debug = debug
        self._ess_at_theta_k: Optional[th.Tensor] = None
        self._diagnostic_step = 0

        super().__init__(*args, **kwargs)
        if self.use_sde:
            raise ValueError(
                "POSER requires a state-independent Gaussian standard deviation; "
                "set use_sde=False"
            )

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)  # type: ignore[arg-type]
                actions, values, log_probs = self.policy(obs_tensor)
                behavior_distribution = self.policy.get_distribution(obs_tensor).distribution
                behavior_mean = behavior_distribution.mean
                behavior_std = behavior_distribution.stddev

            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                behavior_mean=behavior_mean,
                behavior_std=behavior_std,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def _compute_log_mean_d2(self, rollout_data) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Per-window log-mean of the per-sample 2-Renyi divergence pi_theta || pi_{k-i}.

        Shared by the d2_rad POSER weights and by PSR early stopping --
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
        """Compute uniform or D2-based POSER weights."""
        weight_type = weight_type or self.weight_type

        windows = rollout_data.window_id.flatten()
        unique_windows, samples_per_window = th.unique(windows, sorted=True, return_counts=True)
        number_of_windows = unique_windows.numel()

        if weight_type == "uniform":
            uniform_weight = 1.0 / number_of_windows
            return th.full(size=(number_of_windows,), fill_value=uniform_weight, dtype=rollout_data.behavior_mean.dtype, device=rollout_data.behavior_mean.device)

        if weight_type in D2_WEIGHT_TYPES:
            _, samples_per_window, log_mean_d2_by_window = self._compute_log_mean_d2(rollout_data)
            return self._compute_d2_weights(
                samples_per_window, log_mean_d2_by_window, weight_type
            )

        raise ValueError(f"weight_type must be one of {WEIGHT_TYPES}, got {weight_type!r}")

    @staticmethod
    def _compute_d2_weights(
        samples_per_window: th.Tensor,
        log_mean_d2_by_window: th.Tensor,
        weight_type: str,
    ) -> th.Tensor:
        """Normalize either sqrt(N_i / d2_i) or N_i / d2_i across windows."""
        log_sample_count = th.log(
            samples_per_window.to(log_mean_d2_by_window.dtype)
        )
        log_effective_sample_size = log_sample_count - log_mean_d2_by_window
        exponent = 0.5 if weight_type == "d2_rad" else 1.0
        return th.softmax(exponent * log_effective_sample_size, dim=0)

    @staticmethod
    def _compute_psr_value(
        samples_per_window: th.Tensor,
        log_mean_d2_by_window: th.Tensor,
        weight_type: str,
    ) -> th.Tensor:
        """Return sum_i w_i(theta) * sqrt(d2_i(theta) - 1).

        The computation stays in log space and avoids an indeterminate zero
        times infinity for D2-based weights when a divergence is infinite.
        """
        log_d2 = log_mean_d2_by_window.clamp_min(0.0)
        # log(1 - exp(-x)); this is -inf at x=0, as required.
        log_one_minus_inverse_d2 = th.log(-th.expm1(-log_d2))

        if weight_type == "uniform":
            log_terms = (
                0.5 * log_d2
                + 0.5 * log_one_minus_inverse_d2
                - th.log(
                    th.as_tensor(
                        log_d2.numel(), dtype=log_d2.dtype, device=log_d2.device
                    )
                )
            )
        else:
            log_sample_count = th.log(samples_per_window.to(log_d2.dtype))
            exponent = 0.5 if weight_type == "d2_rad" else 1.0
            weight_logits = exponent * (log_sample_count - log_d2)
            log_weight_normalizer = th.logsumexp(weight_logits, dim=0)

            # Combine log(weight_i) and log(sqrt(d2_i - 1)) algebraically,
            # so d2_rad remains well-defined as log_d2 tends to infinity.
            log_terms = exponent * log_sample_count
            if exponent != 0.5:
                log_terms = log_terms + (0.5 - exponent) * log_d2
            log_terms = (
                log_terms
                + 0.5 * log_one_minus_inverse_d2
                - log_weight_normalizer
            )

        return th.exp(th.logsumexp(log_terms, dim=0))

    def _compute_clip_ranges(
        self, clip_range: float, window_weights: th.Tensor
    ) -> th.Tensor:
        """Return per-rollout clip ranges, ordered from newest to oldest."""
        rollout_ages = th.arange(
            1,
            window_weights.numel() + 1,
            dtype=window_weights.dtype,
            device=window_weights.device,
        )
        if self.clip_range_adaptation == "none":
            return th.full_like(window_weights, clip_range)
        clip_range_step = clip_range / th.sum(window_weights * rollout_ages)
        if self.clip_range_adaptation == "weighted_geppo":
            return th.full_like(window_weights, clip_range_step)
        return clip_range_step * rollout_ages

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

        # Copy historical targets before V-trace mutates them, keeping [H, N_E]
        # alignment. Measure the change once; these targets then stay fixed.
        previous_targets = [rollout.returns.copy() for rollout in self.rollout_buffer.history]
        self.rollout_buffer.recompute_advantages(self.policy)
        target_changes = {
            window_id: float(np.sqrt(np.mean(
                (rollout.returns.astype(np.float64) - previous.astype(np.float64)) ** 2
            )))
            for window_id, (previous, rollout) in enumerate(
                zip(previous_targets, self.rollout_buffer.history), start=1
            )
        }
        del previous_targets

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

        # ESS_i(theta_k): baseline for diagnostics and highest-decay eviction.
        # theta_k is the policy right now, before any gradient step this call.
        d2_stats_at_theta_k = self._compute_log_mean_d2(complete_rollout_data)
        _, samples_per_window_at_theta_k, log_mean_d2_at_theta_k = d2_stats_at_theta_k
        self._ess_at_theta_k = (
            samples_per_window_at_theta_k.to(log_mean_d2_at_theta_k.dtype)
            / log_mean_d2_at_theta_k.exp()
        )

        reference_policy = None
        if self.clip_range_adaptation == "weighted_geppo":
            reference_policy = copy.deepcopy(self.policy)
            reference_policy.set_training_mode(False)

        d2_stats = d2_stats_at_theta_k

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        discarded_window = False
        approx_kl_divs = []
        last_loss = float("nan")

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            # The preceding snapshot already evaluated D2 at this theta.
            _, samples_per_window, log_mean_d2_by_window = d2_stats

            if self.weight_type in D2_WEIGHT_TYPES:
                window_weights = self._compute_d2_weights(
                    samples_per_window, log_mean_d2_by_window, self.weight_type
                )
            else:
                window_weights = self._compute_weights(complete_rollout_data)
            window_clip_ranges = self._compute_clip_ranges(
                clip_range, window_weights
            )

            # PSR gates entry, including epoch zero. A blocked entry still logs
            # candidate weights/ranges, as requested, on its own flat step.
            psr_value = None
            psr_triggered = False
            if self.psr_threshold is not None:
                psr_value = self._compute_psr_value(
                    samples_per_window, log_mean_d2_by_window, self.weight_type
                ).item()
                psr_triggered = psr_value > self.psr_threshold

            if epoch == 0 or psr_triggered:
                self._log_diagnostics(
                    complete_rollout_data,
                    epoch=epoch,
                    window_weights=window_weights,
                    window_clip_ranges=window_clip_ranges,
                    d2_stats=d2_stats,
                    reference_policy=reference_policy,
                    target_changes=target_changes if epoch == 0 else None,
                    psr_value=psr_value,
                    psr_triggered=psr_triggered,
                )
            if psr_triggered:
                if self.verbose >= 1:
                    print(f"PSR early stopping before epoch {epoch}: "
                          f"{psr_value:.6g} > {self.psr_threshold}")
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

                # Weighted POSER clipped surrogate. GePPO clips the full
                # pi_theta/pi_behavior ratio around pi_k/pi_behavior.
                sample_clip_ranges = window_clip_ranges[
                    rollout_data.window_id.long()
                ]
                clip_center = th.ones_like(ratio)
                if reference_policy is not None:
                    with th.no_grad():
                        _, reference_log_prob, _ = reference_policy.evaluate_actions(
                            rollout_data.observations, actions
                        )
                    clip_center = th.exp(
                        reference_log_prob - rollout_data.old_log_prob
                    )
                policy_loss_1 = advantages * ratio
                clipped_ratio = th.clamp(
                    ratio,
                    clip_center - sample_clip_ranges,
                    clip_center + sample_clip_ranges,
                )
                policy_loss_2 = advantages * clipped_ratio
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
                clip_fraction_value = th.mean(
                    (th.abs(ratio - clip_center) > sample_clip_ranges).float()
                ).item()
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
                last_loss = loss.item()

                # Approximate reverse KL divergence, logged as train/approx_kl below.
                # Stopping is handled entirely by PSR above, not by this: see
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
            # Measure the updated policy with the weights and bounds just used,
            # before any window is discarded. Reuse D2 for the next epoch.
            d2_stats = self._log_diagnostics(
                complete_rollout_data,
                epoch=epoch + 1,
                window_weights=window_weights,
                window_clip_ranges=window_clip_ranges,
                reference_policy=reference_policy,
            )

        # Window-discard policy: fires only once the window is at capacity, matching
        # the exact point reset()'s own maxlen rotation would otherwise trigger --
        # never during warm-up (n_rollouts < window_size), where nothing should be
        # evicted yet. Runs regardless of how the epoch loop ended (early stop or all
        # n_epochs completed), and independently of PSR early stopping. "oldest"
        # reproduces the natural FIFO rotation; "highest_decay" evicts by
        # decay_i(theta) = ESS_i(theta) / ESS_i(theta_k) at this exact exit point.
        if self.rollout_buffer.n_rollouts >= self.rollout_buffer.window_size:
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
            if self.verbose >= 1:
                outcome = f"discarded window {window_to_discard}" if discarded_window else "window 0 has the least ESS, nothing to discard"
                print(f"POSER window management (policy={self.discard_policy}): {outcome}")

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )

        # Logs. An epoch-0 stop legitimately leaves all minibatch lists empty.
        def mean_or_nan(values):
            return np.mean(values) if values else float("nan")
        self.logger.record("train/entropy_loss", mean_or_nan(entropy_losses))
        self.logger.record("train/policy_gradient_loss", mean_or_nan(pg_losses))
        self.logger.record("train/value_loss", mean_or_nan(value_losses))
        self.logger.record("train/approx_kl", mean_or_nan(approx_kl_divs))
        self.logger.record("train/clip_fraction", mean_or_nan(clip_fractions))
        self.logger.record("train/loss", last_loss)
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        if self.debug:
            _, _, final_log_d2 = d2_stats
            self.logger.record("train/normalized_ess", (-final_log_d2).exp().mean().item())

    @th.no_grad()
    def _log_diagnostics(
        self,
        rollout_data,
        *,
        epoch: int,
        window_weights: th.Tensor,
        window_clip_ranges: th.Tensor,
        d2_stats=None,
        reference_policy=None,
        target_changes: Optional[dict[int, float]] = None,
        psr_value: Optional[float] = None,
        psr_triggered: bool = False,
    ):
        """Dump one full-window snapshot without flushing pending SB3 metrics.

        Epoch zero and PSR-blocked entries use candidate weights/bounds; completed
        epochs use the frozen weights/bounds actually used by the optimizer.
        Raw D2, its square root and ESS describe the snapshot's policy.
        """
        was_training = self.policy.training
        self.policy.set_training_mode(False)
        try:
            if d2_stats is None:
                d2_stats = self._compute_log_mean_d2(rollout_data)
            windows, counts, log_d2 = d2_stats
            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.long().flatten()
            values, log_prob, _ = self.policy.evaluate_actions(rollout_data.observations, actions)
            log_ratio = log_prob - rollout_data.old_log_prob
            clip_center = th.ones_like(log_ratio)
            if reference_policy is not None:
                _, reference_log_prob, _ = reference_policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                clip_center = (reference_log_prob - rollout_data.old_log_prob).exp()
        finally:
            self.policy.set_training_mode(was_training)

        # A separate record buffer shares the configured writers, but does not own
        # or close them. Standard SB3 keys remain pending at their usual env step.
        diagnostic_logger = Logger(self.logger.dir, self.logger.output_formats)
        diagnostic_logger.set_level(self.logger.level)
        record = diagnostic_logger.record
        record("diag/time/epoch", epoch)
        record("diag/time/flat_step", self._diagnostic_step)
        record("diag/time/env_step", self.num_timesteps)

        # Exponentiate in double precision; sqrt(D) is computed directly from
        # log(D), so it can remain finite even when D itself overflows.
        diagnostic_log_d2 = log_d2.double()
        window_ids = rollout_data.window_id.long()
        sample_ranges = window_clip_ranges[window_ids]
        for index, window in enumerate(windows):
            window_id = int(window.item())
            suffix = f"w{window_id:02d}"
            mask = window_ids == window_id
            record(f"diag/d2_rad/{suffix}", (0.5 * diagnostic_log_d2[index]).exp().item())
            record(f"diag/d2/{suffix}", diagnostic_log_d2[index].exp().item())
            record(f"diag/weight/{suffix}", window_weights[window_id].item())
            record(f"diag/clip_range/{suffix}", window_clip_ranges[window_id].item())
            record(f"diag/ess_analytic/{suffix}", (-log_d2[index]).exp().item())
            stats = ratio_statistics(log_ratio[mask], clip_center[mask], sample_ranges[mask])
            for name, value in stats.items():
                record(f"diag/{name}/{suffix}", value)
            stats = critic_statistics(
                values.flatten()[mask], rollout_data.returns[mask], min_target_variance=1e-8
            )
            for name, value in stats.items():
                record(f"diag/critic/{name}/{suffix}", value)

        global_stats = ratio_statistics(log_ratio, clip_center, sample_ranges)
        for name in ("abs_eps", "clip_fraction"):
            record(f"diag/{name}/global", global_stats[name])
        for name in ("ratio_variance", "clipped_ratio_variance"):
            record(f"diag/{name}", global_stats[name])
        if target_changes is not None:
            for window_id, value in target_changes.items():
                record(f"diag/critic/vtrace_target_change_rms/w{window_id:02d}", value)
        # Observe PSR even when its stopping gate is disabled.
        if psr_value is None:
            psr_value = self._compute_psr_value(counts, log_d2, self.weight_type).item()
        record("diag/psr_value", psr_value)
        record("diag/psr_triggered", int(psr_triggered))
        if self.debug:
            record("diag/normalized_ess", (-log_d2).exp().mean().item())

        diagnostic_logger.dump(step=self._diagnostic_step)
        self._diagnostic_step += 1
        return d2_stats
