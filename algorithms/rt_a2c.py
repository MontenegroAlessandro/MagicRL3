import warnings
from typing import Any, ClassVar, Optional, TypeVar, Union, Literal
from collections import deque

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import FloatSchedule, explained_variance
from stable_baselines3 import A2C

IS_WEIGHT_TYPE = Literal["naive", "bh"]

class RT_A2C(A2C):
    """A2C extension that reuses data from past iterations."""
    def __init__(
            self,
            on_policy_critic: bool = True,
            is_weight_type: IS_WEIGHT_TYPE = "naive",
            *args,
            **kwargs,
        ):
        """
        Args: the arguments for A2C.
        """
        # initialize A2C standard parameters
        super().__init__(*args, **kwargs)

        # new parameters
        self.on_policy_critic = on_policy_critic
        self.is_weight_type = is_weight_type



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
        log_mean = log_sum_exp - th.log(valid_counts)

        # final ratio
        ratio = th.exp(log_prob - log_mean)

        return ratio



    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()

            values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
            values = values.flatten()

            # Normalize advantage (not present in the original implementation)
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # ratio between old and new policy, should be one at the first iteration
            # ratio = th.exp(log_prob - rollout_data.old_log_prob)
            ratio = self._compute_ratio(log_prob, rollout_data)

            # Policy gradient loss
            policy_loss = -(ratio.detach() * advantages * log_prob).mean()
            # policy_loss = -(ratio * advantages).mean()  # should be the same!?

            # Value loss using the TD(gae_lambda) target
            # NOTE: we just use newer data for the mse computation
            on_policy_mask = rollout_data.on_policy_mask
            n_on_policy = on_policy_mask.sum().item()
            if n_on_policy > 0 and self.on_policy_critic:
                value_errors = (rollout_data.returns - values) ** 2
                value_loss = (value_errors * on_policy_mask).sum() / n_on_policy
            elif not self.on_policy_critic:
                value_errors = (rollout_data.returns - values) ** 2
                value_loss = value_errors.mean()
            else:
                value_loss = th.tensor(0.0, device=values.device)

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

            # Optimization step
            self.policy.optimizer.zero_grad()
            loss.backward()

            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        # logging ratio for each window rollout for debugging purposes
        for i in range(self.rollout_buffer.window_length):
            if not (rollout_data.window_id == i).any():
                continue
            ratio_i = ratio[rollout_data.window_id == i]
            eps = th.abs(ratio_i - 1.0)
            self.logger.record(f"mean_ratio/over_window_{i}", ratio_i.mean().item())
            self.logger.record(f"max_ratio/over_window_{i}", ratio_i.max().item())
            self.logger.record(f"min_ratio/over_window_{i}", ratio_i.min().item())
            self.logger.record(f"std_ratio/over_window_{i}", ratio_i.std().item())
            self.logger.record(f"mean_|ratio-1|/over_window_{i}", eps.mean().item())



        # ======== DEBUGGING ==========
        if False:
            print("\n========== Debugging ==========")
            # print how many ratio > window_size per window_id
            print("\nRatio statistics per window_id:")
            for i in range(self.rollout_buffer.window_length):
                if not (rollout_data.window_id == i).any():
                    continue
                ratio_i = ratio[rollout_data.window_id == i]
                n_large = (ratio_i > self.rollout_buffer.window_length).sum().item()
                total = ratio_i.shape[0]
                # proprietà di ratio_i
                print(f"window_id={i}: n_large={n_large}/{total} ({n_large/total:.2%}), mean={ratio_i.mean():.3f}, std={ratio_i.std():.3f}, max={ratio_i.max():.3f}, min={ratio_i.min():.3f}") 
                
            # print log_probs and old_log_probs preview
            print("\nLog probabilities preview:")
            for i in range(min(5, log_prob.shape[0])):
                print(f"idx={i}, log_prob={log_prob[i].item():.3f}, old_log_prob={rollout_data.old_log_prob[i].item():.3f}, all_log_probs={rollout_data.all_log_probs[i].cpu().numpy()}")

            # print rollout_data sample that has ratio > 3.5
            print("\nFirst 5 samples with ratio > 3.5:")
            large_ratio_mask = ratio > 3.5
            if large_ratio_mask.any():
                for i in range(min(5, large_ratio_mask.sum().item())):
                    idx = th.where(large_ratio_mask)[0][i].item()
                    print(f"window_id={rollout_data.window_id[idx].item()}, ratio={ratio[idx].item():.3f}, log_prob={log_prob[idx].item():.3f}, old_log_prob={rollout_data.old_log_prob[idx].item():.3f}, all_log_probs={rollout_data.all_log_probs[idx].cpu().numpy()}")
