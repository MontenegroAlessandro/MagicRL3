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
        kl_vals = []
        abs_ratio_vals = []

        with th.no_grad():
            for rollout_data in self.rollout_buffer.get(batch_size=None):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()
                _, log_prob, _ = self.policy.evaluate_actions(rollout_data.observations, actions)
                naive_ratio = th.exp(log_prob - rollout_data.old_log_prob)
                clip_fracs.append((th.abs(naive_ratio - 1) > clip_range).float().mean().item())
                kl_vals.append(((naive_ratio - 1) - th.log(naive_ratio)).mean().item())
                abs_ratio_vals.append((naive_ratio - 1).abs().mean().item())

        self.logger.record("diagnostics_clip/clip_fraction_window_0", np.mean(clip_fracs))
        self.logger.record("diagnostics_clip/clip_fraction_mean", np.mean(clip_fracs))
        self.logger.record("diagnostics_kl/kl_window_0", np.mean(kl_vals))
        self.logger.record("diagnostics_kl/kl_mean", np.mean(kl_vals))
        self.logger.record("diagnostics_abs_ratio/final_window_0", np.mean(abs_ratio_vals))
        self.logger.record("diagnostics_abs_ratio/final_mean", np.mean(abs_ratio_vals))
