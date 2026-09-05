"""Read-only statistics for POSER's full-rollout diagnostic snapshots."""

import torch as th


def ratio_statistics(
    log_ratio: th.Tensor,
    clip_center: th.Tensor,
    clip_range: th.Tensor,
) -> dict[str, float]:
    """Use the loss's sample-specific bounds, including GePPO centers.

    Variances retain the sample-variance convention of the previous logger.
    ESS is evaluated in log space to avoid overflow in squared IS ratios.
    """
    # Match the loss's arithmetic for ratios, bounds and boundary comparisons;
    # use double precision only for the statistical reductions.
    log_ratio = log_ratio.detach().flatten()
    ratio = log_ratio.exp()
    center = clip_center.detach().flatten()
    width = clip_range.detach().flatten()
    clipped_ratio = ratio.clamp(center - width, center + width).double()
    outside = (ratio - center).abs() > width
    ratio = ratio.double()
    n = ratio.numel()
    log_ratio = log_ratio.double()
    log_ess = 2 * th.logsumexp(log_ratio, 0) - th.logsumexp(2 * log_ratio, 0)
    return {
        "abs_eps": (ratio - 1).abs().mean().item(),
        "clip_fraction": outside.double().mean().item(),
        "ratio_variance": ratio.var().item() if n > 1 else float("nan"),
        "clipped_ratio_variance": clipped_ratio.var().item() if n > 1 else float("nan"),
        "ess_empirical": (log_ess.exp() / n).item(),
    }


def critic_statistics(
    values: th.Tensor,
    targets: th.Tensor,
    *,
    min_target_variance: float,
) -> dict[str, float]:
    """Evaluate current, unclipped predictions against frozen training targets."""
    targets = targets.detach().double().flatten()
    errors = values.detach().double().flatten() - targets
    target_variance = targets.var(correction=0).item()
    return {
        "normalized_mse": (
            (errors.square().mean() / target_variance).item()
            if target_variance > min_target_variance else float("nan")
        ),
        "bias": errors.mean().item(),
    }
