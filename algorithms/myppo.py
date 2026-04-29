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
        ratio_lo_vals = []
        ratio_hi_vals = []
        all_ratios: list[th.Tensor] = []

        self.policy.set_training_mode(False)
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
                if (naive_ratio < 1).any():
                    ratio_lo_vals.append(naive_ratio[naive_ratio < 1].mean().item())
                if (naive_ratio > 1).any():
                    ratio_hi_vals.append(naive_ratio[naive_ratio > 1].mean().item())
                all_ratios.append(naive_ratio.cpu())

        self.logger.record("diagnostics_clip/post_w0", np.mean(clip_fracs))
        self.logger.record("diagnostics_clip/post_mean", np.mean(clip_fracs))
        self.logger.record("diagnostics_kl/post_naive_w0", np.mean(kl_vals))
        self.logger.record("diagnostics_kl/post_naive_mean", np.mean(kl_vals))
        self.logger.record("diagnostics_abs_ratio/post_naive_w0", np.mean(abs_ratio_vals))
        self.logger.record("diagnostics_abs_ratio/post_naive_mean", np.mean(abs_ratio_vals))
        if ratio_lo_vals:
            self.logger.record("diagnostics_ratio_lo/post_naive_w0", np.mean(ratio_lo_vals))
            self.logger.record("diagnostics_ratio_lo/post_naive_mean", np.mean(ratio_lo_vals))
        if ratio_hi_vals:
            self.logger.record("diagnostics_ratio_hi/post_naive_w0", np.mean(ratio_hi_vals))
            self.logger.record("diagnostics_ratio_hi/post_naive_mean", np.mean(ratio_hi_vals))
        if all_ratios:
            all_r = th.cat(all_ratios)
            ess = (all_r.sum() ** 2 / (all_r ** 2).sum()) / len(all_r)
            self.logger.record("diagnostics_ess/post_naive_w0", ess.item())
            self.logger.record("diagnostics_ess/post_naive_mean", ess.item())
            self.logger.record("diagnostics_ratio_var/post_naive_w0", all_r.var().item())
            self.logger.record("diagnostics_ratio_var/post_naive_mean", all_r.var().item())
            self._log_ratio_histogram("diagnostics_ratio_dist/post_naive_all", all_r)
        self.policy.set_training_mode(True)

    def _log_ratio_histogram(self, key: str, ratios: th.Tensor) -> None:
        """Log ratio distribution via TensorBoard add_histogram (wandb syncs it as density heatmap)."""
        data = ratios.float().cpu().numpy()
        for fmt in self.logger.output_formats:
            if hasattr(fmt, "writer"):
                fmt.writer.add_histogram(key, data, self.num_timesteps)
                break