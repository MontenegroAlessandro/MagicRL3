from collections import deque
import copy
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

from .utils.adaptive_lr import AdaptiveLRScheduler
from .utils.diagnostics import (
    Ratios,
    advantages_by_window,
    approx_kl,
    clip_fraction,
    collect_ratios,
    mean_abs_deviation,
    mean_abs_ratio_gap,
    normalized_ess,
    ratio_variance,
    sign_flip_fraction,
    spearman_corr,
    standardize_by_window,
    window_moments,
)

IS_WEIGHT_TYPE = Literal["naive", "bh"]

class RT_PPO(PPO):
    """PPO extension that reuses data from past iterations."""
    def __init__(
            self,
            on_policy_critic: bool = True,
            is_weight_type: IS_WEIGHT_TYPE = "naive",
            fresh_adv: bool = False,
            on_policy_masking: bool = False,
            geppo_clip: bool = False,
            adaptive_lr: bool = False,
            adaptive_lr_alpha: float = 0.03,
            adaptive_lr_beta: float = 0.5,
            *args,
            **kwargs,
        ):
        """
        Args: the arguments for PPO.
        """
        # initialize PPO standard parameters
        super().__init__(*args, **kwargs)

        # new parameters
        self.on_policy_critic = on_policy_critic
        self.is_weight_type = is_weight_type
        self.fresh_adv = fresh_adv
        self.on_policy_masking = on_policy_masking
        self.geppo_clip = geppo_clip
        self.adaptive_lr_scheduler = AdaptiveLRScheduler(adaptive_lr, adaptive_lr_alpha, adaptive_lr_beta, self.learning_rate)

    def _update_learning_rate(self, optimizers) -> None:
        # Overrides BaseAlgorithm._update_learning_rate to fold in the
        # GePPO-style adaptive scale tracked by self.adaptive_lr_scheduler.
        # `self` here is always this RT_PPO instance: no inheritance from
        # another class is involved, this is a plain method call.
        if self.adaptive_lr_scheduler.enabled:
            self.adaptive_lr_scheduler.update_learning_rate(self, optimizers)
        else:
            super()._update_learning_rate(optimizers)

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        """
        After the standard rollout collection we:
          1. Store log π_current into column 0 of the buffer's all_log_probs.
          2. Evaluate all past policies on the current rollout → fill columns 1..w-1.
          3. Evaluate the current policy on all historical rollouts → append a new column.
          4. Save the current policy state_dict to _policy_history (of the buffer).
        """
        # Standard SB3 rollout collection
        result = super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)

        rollout_trajectories = env.num_envs + int(rollout_buffer.episode_starts[1:].sum())
        completed_trajectories = int(rollout_buffer.episode_starts[1:].sum()) + int(self._last_episode_starts.sum())
        self.logger.record("rollout/num_total_trajectories", rollout_trajectories)
        self.logger.record("rollout/num_completed_trajectories", completed_trajectories)


        if self.rollout_buffer.window_length > 1 and self.is_weight_type == "bh":
            self.rollout_buffer.set_current_policy(self.policy)
            self.rollout_buffer.update_all_log_probs()

        return result

    def _compute_ratio(
        self,
        log_prob: th.Tensor,
        rollout_data,
    ) -> th.Tensor:

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

    def _current_ratio(self, data) -> th.Tensor:
        """
        The IS ratio the loss multiplies the advantages by, evaluated at the current policy
        on the samples of `data`: r = pi_k/pi_{k-i} (naive) or pi_k/mean_j pi_{k-j} (BH),
        since it is called before any gradient step. With naive weighting it is exactly 1 on
        window 0, whose samples pi_k collected itself; with BH weighting it is not.
        """
        actions = data.actions
        if isinstance(self.action_space, spaces.Discrete):
            actions = actions.long().flatten()
        was_training = self.policy.training
        self.policy.set_training_mode(False)
        with th.no_grad():
            _, log_prob, _ = self.policy.evaluate_actions(data.observations, actions)
            ratio = self._compute_ratio(log_prob, data)
        self.policy.set_training_mode(was_training)
        return ratio

    # ------------------------------------------------------------------ #
    # diagnostics: no impact on the algorithm, only on what gets logged   #
    # ------------------------------------------------------------------ #

    def _log_initial_diagnostics(self, ratios: Ratios, clip_range: float, use_bh: bool) -> None:
        """
        Diagnostics of the policy before the update, for each window i and pooled over all
        the samples ("mean"): clip fraction, Schulman approx. reverse KL, mean |r - 1|,
        normalized ESS and variance of r.
        The naive ratio r = pi/pi_{k-i} carries all of them; with BH weighting the same
        statistics are logged for the BH ratio too, and the clip fraction is measured on
        it because that is the ratio the loss actually clips.
        """
        # the same statistics are logged on the samples of each window and on all of them together
        subsets = {f"window_{i}": ratios.of(i) for i in ratios.windows}
        subsets["mean"] = ratios

        for suffix, subset in subsets.items():
            r = subset.naive     # r    = pi / pi_{k-i}
            r_bh = subset.bh     # r_BH = pi / mean_j pi_{k-j}
            self.logger.record(f"diagnostics_clip/initial_clip_fraction_{suffix}", clip_fraction(r_bh, clip_range).item())
            self.logger.record(f"diagnostics_kl/initial_kl_{suffix}", approx_kl(r).item())
            self.logger.record(f"diagnostics_abs_ratio/initial_{suffix}", mean_abs_deviation(r).item())
            self.logger.record(f"diagnostics_ess/initial_naive_ess_{suffix}", normalized_ess(r).item())
            self.logger.record(f"diagnostics_var/initial_naive_ratio_var_{suffix}", ratio_variance(r).item())
            if use_bh:
                self.logger.record(f"diagnostics_kl/initial_kl_bh_{suffix}", approx_kl(r_bh).item())
                self.logger.record(f"diagnostics_abs_ratio/initial_bh_{suffix}", mean_abs_deviation(r_bh).item())
                self.logger.record(f"diagnostics_ess/initial_bh_ess_{suffix}", normalized_ess(r_bh).item())
                self.logger.record(f"diagnostics_var/initial_bh_ratio_var_{suffix}", ratio_variance(r_bh).item())

    def _log_final_diagnostics(self, ratios: Ratios, clip_range: float, use_bh: bool) -> None:
        """
        Diagnostics of the updated policy: same statistics as the initial pass, except that
        the clip fraction is measured on the naive ratio.
        """
        # the same statistics are logged on the samples of each window and on all of them together
        subsets = {f"window_{i}": ratios.of(i) for i in ratios.windows}
        subsets["mean"] = ratios

        for suffix, subset in subsets.items():
            r = subset.naive     # r    = pi / pi_{k-i}
            r_bh = subset.bh     # r_BH = pi / mean_j pi_{k-j}
            self.logger.record(f"diagnostics_clip/clip_fraction_{suffix}", clip_fraction(r, clip_range).item())
            self.logger.record(f"diagnostics_kl/kl_{suffix}", approx_kl(r).item())
            self.logger.record(f"diagnostics_abs_ratio/final_{suffix}", mean_abs_deviation(r).item())
            self.logger.record(f"diagnostics_ess/final_naive_ess_{suffix}", normalized_ess(r).item())
            self.logger.record(f"diagnostics_var/final_naive_ratio_var_{suffix}", ratio_variance(r).item())
            if use_bh:
                self.logger.record(f"diagnostics_kl/kl_bh_{suffix}", approx_kl(r_bh).item())
                self.logger.record(f"diagnostics_abs_ratio/final_bh_{suffix}", mean_abs_deviation(r_bh).item())
                self.logger.record(f"diagnostics_ess/final_bh_ess_{suffix}", normalized_ess(r_bh).item())
                self.logger.record(f"diagnostics_var/final_bh_ratio_var_{suffix}", ratio_variance(r_bh).item())

    def _log_advantage_diagnostics(
        self,
        stale_advantages: Optional[np.ndarray],
        adv_mean_by_window: Optional[th.Tensor],
        adv_std_by_window: Optional[th.Tensor],
    ) -> None:
        """
        Diagnostics of the advantages that multiply the ratio in the loss.

        Always logged, for each window i and pooled over the buffer ("mean"): the raw
        mu_i and sigma_i, in the units of the critic that produced them. They say whether
        the windows sit on the same scale and, with fresh advantages, whether the VTRACE
        recomputation actually realigns them.

        With fresh advantages, the stale advantages are also compared to the recomputed
        ones, each side carrying its own normalization, i.e. exactly the two vectors that
        enter (or would have entered) the gradient:
          - sign flip: fraction of samples the update pushes the other way;
          - Spearman: how much the recomputation reshuffles their ordering.
        recompute_advantages() leaves window 0 untouched, so its flip = 0 and rho = 1 are
        a sanity check, not a measurement, and stay out of the pooled values.
        """
        adv, window = advantages_by_window(self.rollout_buffer)
        n_windows = 1 + len(self.rollout_buffer.history)
        means, stds = window_moments(adv, window, n_windows)
        for i in range(n_windows):
            self.logger.record(f"diagnostics_adv/adv_mean_window_{i}", means[i])
            self.logger.record(f"diagnostics_adv/adv_std_window_{i}", stds[i])
        self.logger.record("diagnostics_adv/adv_mean_mean", float(adv.mean()))
        self.logger.record("diagnostics_adv/adv_std_mean", float(adv.std(ddof=1)))

        if stale_advantages is None:  # nothing was recomputed: no before/after to compare
            return

        stale_means, stale_stds = window_moments(stale_advantages, window, n_windows)
        for i in range(n_windows):
            self.logger.record(f"diagnostics_adv/stale_adv_mean_window_{i}", stale_means[i])
            self.logger.record(f"diagnostics_adv/stale_adv_std_window_{i}", stale_stds[i])

        if adv_mean_by_window is not None:
            # the stale side with the per-window statistics it would have been normalized
            # with, the fresh side with the ones the update is using now (+1e-8 as there)
            old = standardize_by_window(stale_advantages, window, stale_means, stale_stds + 1e-8)
            new = standardize_by_window(adv, window,
                                        adv_mean_by_window.cpu().numpy(),
                                        adv_std_by_window.cpu().numpy())
        else:
            old, new = stale_advantages, adv  # normalize_advantage=False: raw advantages

        flips, rhos = [], []
        for i in range(n_windows):
            mask = window == i
            flip = sign_flip_fraction(old[mask], new[mask])
            rho = spearman_corr(old[mask], new[mask])
            self.logger.record(f"diagnostics_adv/sign_flip_window_{i}", flip)
            self.logger.record(f"diagnostics_adv/spearman_window_{i}", rho)
            if i > 0:
                flips.append(flip)
                rhos.append(rho)
        if flips:
            # every window holds the same number of samples, so the mean over the windows
            # is the sign flip fraction pooled over all the recomputed samples
            self.logger.record("diagnostics_adv/sign_flip_mean", float(np.mean(flips)))
            self.logger.record("diagnostics_adv/spearman_mean", float(np.mean(rhos)))

    # ------------------------------------------------------------------ #

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

        early_stop_condition_total = 0
        early_stop_condition_true = 0
        approx_kl_divs = []

        # Recompute the advantages of the past windows with the current critic
        stale_advantages = None  # kept only for the diagnostics, see _log_advantage_diagnostics
        if self.fresh_adv:
            _gen = self.rollout_buffer.get(self.batch_size)  # just triggers generator_ready
            next(_gen)
            del _gen
            if self.rollout_buffer.window_length > 1:
                stale_advantages, _ = advantages_by_window(self.rollout_buffer)
                self.rollout_buffer.recompute_advantages(self.policy)

        # Advantage normalization statistics, computed once on the whole buffer.
        # With fresh advantages every window is estimated under the current critic, so all of
        # them share a single (mu, sigma), but the samples come from pi_{k-i}, so it is the
        # IS-weighted one; with stale advantages each window gets its own, since it carries
        # the offset and the scale of the critic that produced it.
        n_windows = 1 + len(self.rollout_buffer.history)
        adv_mean_by_window = adv_std_by_window = None
        if self.normalize_advantage:
            all_data = next(self.rollout_buffer.get(batch_size=None))  # the whole buffer in one batch
            all_advantages, all_windows = all_data.advantages, all_data.window_id.long()
            if self.fresh_adv:
                # Each sample enters the loss multiplied by its ratio, so the moments of the
                # objective are the self-normalized IS ones. Weights and advantages come from
                # the same pass: get() reshuffles at every call, pairing two passes would
                # match each weight with the wrong sample.
                weights = self._current_ratio(all_data)
                total_weight = weights.sum()
                mean = (weights * all_advantages).sum() / total_weight
                var = (weights * (all_advantages - mean) ** 2).sum() / total_weight
                adv_mean_by_window = mean.repeat(n_windows)
                adv_std_by_window = (var.sqrt() + 1e-8).repeat(n_windows)
            else:
                adv_mean_by_window = th.stack([all_advantages[all_windows == i].mean() for i in range(n_windows)])
                adv_std_by_window = th.stack([all_advantages[all_windows == i].std() + 1e-8 for i in range(n_windows)])

        self._log_advantage_diagnostics(stale_advantages, adv_mean_by_window, adv_std_by_window)

        # Diagnostics of the policy before any gradient step
        use_bh = self.is_weight_type == "bh" and self.rollout_buffer.window_length > 1
        initial_ratios = collect_ratios(
            self.policy, self.rollout_buffer, self.action_space,
            bh_ratio_fn=self._compute_ratio if use_bh else None,
        )
        self._log_initial_diagnostics(initial_ratios, clip_range, use_bh)

        # GePPO-style clipping: freeze a snapshot of the policy as it is right now
        # (theta_k, before any gradient step in this train() call). During the
        # epoch loop we evaluate it on each minibatch to get pi_{theta_k}(a|s),
        # which together with old_log_prob (pi_{theta_{k-i}}(a|s), the policy that
        # actually collected the sample) generalizes the "1" in the clip bounds.
        # The adaptive LR needs the same snapshot: its TV estimate (Lemma 3)
        # pairs pi/pi_{k-i} and pi_k/pi_{k-i} on the same (s,a), so both ratios
        # must be evaluated on the same minibatch (buffer.get() shuffles, so
        # pairing final vs initial diagnostics passes element-wise would be wrong).
        reference_policy = None
        if self.geppo_clip or self.adaptive_lr_scheduler.enabled:
            reference_policy = copy.deepcopy(self.policy)
            reference_policy.set_training_mode(False)

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                on_policy_mask = rollout_data.on_policy_mask
                on_mask = on_policy_mask.bool()
                # Normalize advantage with the statistics of the window each sample comes from
                # (the same ones for every sample when the advantages are fresh)
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    sample_window = rollout_data.window_id.long()
                    advantages = (advantages - adv_mean_by_window[sample_window]) / adv_std_by_window[sample_window]

                # ratio between old and new policy, should be one at the first iteration
                ratio = self._compute_ratio(log_prob, rollout_data)

                # clipped surrogate loss
                if self.geppo_clip:
                    # generalize the "1" to pi_{theta_k}(a|s) / pi_{theta_{k-i}}(a|s)
                    with th.no_grad():
                        _, ref_log_prob, _ = reference_policy.evaluate_actions(rollout_data.observations, actions)
                    clip_center = th.exp(ref_log_prob - rollout_data.old_log_prob)
                else:
                    clip_center = 1.0
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, clip_center - clip_range, clip_center + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

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
                value_errors = (rollout_data.returns - values_pred) ** 2
                n_on_policy = on_policy_mask.sum().item()
                if not self.on_policy_critic:
                    value_loss = value_errors.mean()
                elif n_on_policy > 0:
                    value_loss = (value_errors * on_policy_mask).sum() / n_on_policy
                else:
                    value_loss = th.tensor(0.0, device=values.device)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is not None:
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
                    if on_mask.sum() > 0 and self.on_policy_masking:
                        # Guard against inflated KL from off-policy samples
                        # by computing KL only over the on-policy portion.
                        log_ratio_on = log_ratio[on_mask]
                        approx_kl_div = th.mean((th.exp(log_ratio_on) - 1) - log_ratio_on).cpu().numpy()
                    else:
                        # No on-policy samples in this minibatch — skip early stopping
                        approx_kl_div = 0.0
                    approx_kl_divs.append(float(approx_kl_div))

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    early_stop_condition_total += 1
                    early_stop_condition_true += 1
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    continue_training = False
                    break
                elif self.target_kl is not None:
                    early_stop_condition_total += 1

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        approx_kl_mean = float(np.mean(approx_kl_divs)) if len(approx_kl_divs) > 0 else 0.0
        self.logger.record("train/approx_kl_window_all_mean", approx_kl_mean)
        self.logger.record("train/approx_kl_window_all_count", len(approx_kl_divs))

        early_stop_true_pct = (
            100.0 * early_stop_condition_true / early_stop_condition_total
            if early_stop_condition_total > 0 else 0.0
        )
        self.logger.record("debug/early_stopping_condition_true_pct_window_all", early_stop_true_pct)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates)
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        # Diagnostics of the updated policy
        final_ratios = collect_ratios(
            self.policy, self.rollout_buffer, self.action_space,
            bh_ratio_fn=self._compute_ratio if use_bh else None,
            reference_policy=reference_policy if self.adaptive_lr_scheduler.enabled else None,
        )
        self._log_final_diagnostics(final_ratios, clip_range, use_bh)

        # --- adaptive learning rate (GePPO Algorithm 1) ---
        if self.adaptive_lr_scheduler.enabled:
            # E|r - r_k| pooled over all the samples: the implicit nu matches the training
            # objective, which mixes windows proportionally to their sample count
            # (same nu in surrogate and penalty, as required by Theorem 1).
            mean_abs_ratio_diff = mean_abs_ratio_gap(final_ratios.naive, final_ratios.reference)
            self.adaptive_lr_scheduler.update(mean_abs_ratio_diff.item(), clip_range)
