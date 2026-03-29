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

        # deque for the history of policies to be considered when BH is used
        window_size = self.rollout_buffer.window_length  # set by MultiRolloutBuffer
        if self.is_weight_type == "bh" and window_size > 1:
            self._policy_history: deque = deque(maxlen=window_size - 1)
        else:
            self._policy_history = None


    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        """
        After the standard rollout collection we:
          1. Store log π_current into column 0 of the buffer's all_log_probs.
          2. Evaluate all past policies on the current rollout → fill columns 1..w-1.
          3. Evaluate the current policy on all historical rollouts → append a new column.
          4. Save the current policy state_dict to _policy_history.
        """
        # Standard SB3 rollout collection
        result = super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)

        if self.rollout_buffer.window_length > 1 and self.is_weight_type == "bh":
            self._fill_cross_log_probs()

        return result

    @th.no_grad()
    def _fill_cross_log_probs(self) -> None:
        """
        After a new rollout D_k has been collected:
          - Column 0 of the current buffer is already filled (SB3 stores log_probs during rollout).
          - We evaluate π_{k-1}, ..., π_{k-w+1} on D_k  → columns 1..len(history) of current buffer.
          - We evaluate π_k on D_{k-1}, ..., D_{k-w+1}  → new column appended to each history entry.
        """
        buf = self.rollout_buffer
        self.policy.set_training_mode(False)

        # store the log probs of the current policy for the current rollout into column 0 of the all_log_probs matrix
        buf.store_current_log_probs(buf.log_probs)

        # evaluate past policies on new data
        obs_flat = buf.observations.reshape(-1, *buf.observations.shape[2:])   # (N, obs_dim)
        act_flat = buf.actions.reshape(-1, *buf.actions.shape[2:])             # (N, act_dim)

        obs_tensor = th.tensor(obs_flat, device=self.device)
        act_tensor = th.tensor(act_flat, dtype=th.float32, device=self.device)
        if isinstance(self.action_space, spaces.Discrete):
            act_tensor = act_tensor.long().flatten()

        for policy_idx, past_state_dict in enumerate(reversed(self._policy_history), start=1):
            # Load past policy weights temporarily
            current_state_dict = {k: v.clone() for k, v in self.policy.state_dict().items()}
            self.policy.load_state_dict(past_state_dict)

            _, log_probs_past, _ = self.policy.evaluate_actions(obs_tensor, act_tensor)
            log_probs_past_np = log_probs_past.cpu().numpy().reshape(
                buf.buffer_size, buf.n_envs
            )
            buf.update_current_cross_log_probs(log_probs_past_np, policy_idx)

            # Restore current policy
            self.policy.load_state_dict(current_state_dict)

        # evaluate current policy on past data
        for hist_idx, entry in enumerate(buf.history):
            hist_obs = entry["observations"].reshape(-1, *buf.observations.shape[2:])
            hist_act = entry["actions"].reshape(-1, *buf.actions.shape[2:])

            hist_obs_tensor = th.tensor(hist_obs, device=self.device)
            hist_act_tensor = th.tensor(hist_act, dtype=th.float32, device=self.device)
            if isinstance(self.action_space, spaces.Discrete):
                hist_act_tensor = hist_act_tensor.long().flatten()

            _, log_probs_current, _ = self.policy.evaluate_actions(
                hist_obs_tensor, hist_act_tensor
            )
            buf.update_past_log_probs(
                log_probs_current.cpu().numpy(), history_idx=hist_idx
            )

        self._policy_history.append(
            {k: v.cpu().clone() for k, v in self.policy.state_dict().items()}
        )

        self.policy.set_training_mode(True)

    def _compute_ratio(
        self,
        log_prob: th.Tensor,
        rollout_data,
    ) -> th.Tensor:
        """
        Compute the IS ratio for the policy loss.
        """
        if self.is_weight_type == "naive" or self.rollout_buffer.window_length == 1:
            return th.exp(log_prob - rollout_data.old_log_prob)
        
        all_log_probs = rollout_data.all_log_probs          # (batch, w)
        w = all_log_probs.shape[1]

        # Mask -inf columns (not yet filled) so they don't contribute to logsumexp
        # Replace -inf with a very large negative number that still registers as zero
        # probability but avoids NaN gradients.
        valid_mask = ~th.isinf(all_log_probs)               # (batch, w)
        n_valid = valid_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)

        # logsumexp over valid columns only
        masked = all_log_probs.masked_fill(~valid_mask, -1e9)
        log_sum_pi = th.logsumexp(masked, dim=1)            # (batch,)

        # BH ratio
        log_ratio_bh = log_prob + th.log(n_valid.squeeze(1)) - log_sum_pi
        return th.exp(log_ratio_bh)

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
                # ratio = th.exp(log_prob - rollout_data.old_log_prob)
                ratio = self._compute_ratio(log_prob, rollout_data)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
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
                    # NOTE: for the RT-PPO, we apply the early stopping criterion to all data, not just the on-policy 
                    # data, since the old log probs of the off-policy data will lead a.s. to high KL divergence
                    # this is done to defend against unlucky sampling
                    on_policy_mask = rollout_data.on_policy_mask.bool()
                    if on_policy_mask.sum() > 0:
                        log_ratio_on_policy = log_ratio[on_policy_mask]
                        approx_kl_div = th.mean(
                            (th.exp(log_ratio_on_policy) - 1) - log_ratio_on_policy
                        ).cpu().numpy()
                    else:
                        # No on-policy samples in this minibatch — skip early stopping
                        approx_kl_div = 0.0

                    # approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
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

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

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

        self.logger.record("train/n_updates", self._n_updates)
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)