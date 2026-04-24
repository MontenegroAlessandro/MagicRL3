import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3 import PPO


class MyPPO(PPO):
    """PPO wrapper that adds diagnostic logging after training."""

    def train(self) -> None:
        super().train()

        clip_range = self.clip_range(self._current_progress_remaining)

        clip_fracs = []
        naive_is_max = []
        naive_is_mean = []
        naive_ess_sum_w  = 0.0
        naive_ess_sum_w2 = 0.0
        naive_kl = []

        with th.no_grad():
            for rollout_data in self.rollout_buffer.get(batch_size=None):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()
                _, log_prob, _ = self.policy.evaluate_actions(rollout_data.observations, actions)
                naive_ratio = th.exp(log_prob - rollout_data.old_log_prob)
                clip_fracs.append((th.abs(naive_ratio - 1) > clip_range).float().mean().item())
                naive_is_max.append(naive_ratio.max().item())
                naive_is_mean.append(naive_ratio.mean().item())
                naive_ess_sum_w  += naive_ratio.sum().item()
                naive_ess_sum_w2 += (naive_ratio ** 2).sum().item()
                naive_kl.append(((naive_ratio - 1) - th.log(naive_ratio)).mean().item())

        ess = naive_ess_sum_w ** 2 / naive_ess_sum_w2 if naive_ess_sum_w2 > 0 else 0.0
        self.logger.record("diagnostics/clip_fraction_window_0", np.mean(clip_fracs))
        self.logger.record("diagnostics/clip_fraction_mean", np.mean(clip_fracs))
        self.logger.record("diagnostics/naive_is_weight_max_window_0", max(naive_is_max))
        self.logger.record("diagnostics/naive_is_weight_mean_window_0", np.mean(naive_is_mean))
        self.logger.record("diagnostics/naive_is_weight_max", max(naive_is_max))
        self.logger.record("diagnostics/naive_is_weight_mean", np.mean(naive_is_mean))
        self.logger.record("diagnostics/naive_ess_window_0", ess)
        self.logger.record("diagnostics/naive_ess", ess)
        self.logger.record("diagnostics/naive_kl_window_0", np.mean(naive_kl))
        self.logger.record("diagnostics/naive_kl", np.mean(naive_kl))
