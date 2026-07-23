"""GePPO-style adaptive learning rate (Queeney et al., 2021, Algorithm 1).

`AdaptiveLRScheduler` is a plain object owned by the algorithm (RT_PPO /
MyPPO), holding its own state (`scale`, `enabled`, `base_lr`, ...). Host
classes call `self.adaptive_lr_scheduler.update_learning_rate(self,
optimizers)` from their own `_update_learning_rate` override, passing
themselves in since the scheduler still needs one SB3-native attribute
(`logger`) that lives on the host, not on the scheduler.
"""
from stable_baselines3.common.utils import update_learning_rate

__all__ = ["AdaptiveLRScheduler"]


class AdaptiveLRScheduler:
    """Tracks a multiplicative scale on top of the base SB3 LR schedule.

    When `enabled=False`, `scaled_lr`/`update` are no-ops, so the algorithm's
    learning rate behaves exactly as without this feature.
    """

    def __init__(self, enabled: bool, alpha: float, beta: float, learning_rate=None):
        if enabled and callable(learning_rate):
            raise ValueError(
                "adaptive_lr requires a constant learning_rate (a plain float): "
                "the GePPO-style adaptive scale multiplies a fixed base LR, so a "
                "callable/decaying learning_rate schedule is not supported."
            )
        self.enabled = enabled
        self.alpha = alpha
        self.beta = beta
        self.scale = 1.0
        self.base_lr = float(learning_rate) if enabled else None

    def scaled_lr(self, base_lr: float) -> float:
        return base_lr * self.scale if self.enabled else base_lr

    def update(self, mean_abs_ratio_diff: float, clip_range: float) -> None:
        """Adjust `scale` for the *next* update from this update's realized
        TV distance estimate (Lemma 2/3 of the paper, before the 1/2 factor).
        """
        if not self.enabled:
            return
        tv_estimate = 0.5 * mean_abs_ratio_diff
        target_tv = clip_range / 2
        if tv_estimate > target_tv:
            self.scale /= (1 + self.alpha)
        elif tv_estimate < self.beta * target_tv:
            self.scale *= (1 + self.alpha)

    def update_learning_rate(self, model, optimizers) -> None:
        """Drop-in replacement for `model._update_learning_rate`, applying this
        scheduler's `scale` to the constant base LR captured at construction.
        Only call this when `self.enabled` — callers already guard on that.

        Because `adaptive_lr` enforces a constant `learning_rate` (see
        `__init__`), the effective LR is simply `base_lr * scale`, matching the
        paper's recursive eta update (eta <- eta/(1+alpha) or eta*(1+alpha)).

        `model` must expose `logger` (set up by SB3 before training starts).
        """
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        lr = self.scaled_lr(self.base_lr)
        model.logger.record("train/learning_rate", lr)
        model.logger.record("train/adaptive_lr_scale", self.scale)
        for optimizer in optimizers:
            update_learning_rate(optimizer, lr)
