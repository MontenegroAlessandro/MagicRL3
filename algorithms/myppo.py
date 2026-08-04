from stable_baselines3 import PPO

from .adaptive_lr import AdaptiveLRScheduler
from .diagnostics import (
    approx_kl,
    clip_fraction,
    collect_ratios,
    mean_abs_deviation,
    normalized_ess,
    ratio_variance,
)


class MyPPO(PPO):
    """PPO wrapper that adds diagnostic logging after training."""

    def __init__(
        self,
        adaptive_lr: bool = False,
        adaptive_lr_alpha: float = 0.03,
        adaptive_lr_beta: float = 0.5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.adaptive_lr_scheduler = AdaptiveLRScheduler(adaptive_lr, adaptive_lr_alpha, adaptive_lr_beta, self.learning_rate)

    def _update_learning_rate(self, optimizers) -> None:
        # Overrides BaseAlgorithm._update_learning_rate. `self` here is
        # always this MyPPO instance — plain method call, no mixin involved.
        if self.adaptive_lr_scheduler.enabled:
            self.adaptive_lr_scheduler.update_learning_rate(self, optimizers)
        else:
            super()._update_learning_rate(optimizers)

    def train(self) -> None:
        super().train()

        clip_range = self.clip_range(self._current_progress_remaining)

        # Diagnostics of the updated policy on r = pi/pi_old. There is a single rollout,
        # so the per-window metrics and the pooled ones coincide.
        r = collect_ratios(self.policy, self.rollout_buffer, self.action_space).naive
        for suffix in ("window_0", "mean"):
            self.logger.record(f"diagnostics_clip/clip_fraction_{suffix}", clip_fraction(r, clip_range).item())
            self.logger.record(f"diagnostics_kl/kl_{suffix}", approx_kl(r).item())
            self.logger.record(f"diagnostics_abs_ratio/final_{suffix}", mean_abs_deviation(r).item())
            self.logger.record(f"diagnostics_ess/final_naive_ess_{suffix}", normalized_ess(r).item())
            self.logger.record(f"diagnostics_var/final_naive_ratio_var_{suffix}", ratio_variance(r).item())

        if self.adaptive_lr_scheduler.enabled:
            self.adaptive_lr_scheduler.update(mean_abs_deviation(r).item(), clip_range)