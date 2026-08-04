"""
Diagnostics on the importance-sampling ratios.

Nothing here takes part in the update: this module only produces the quantities that
get logged, plus the E|r - r_k| that the adaptive learning rate consumes.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import torch as th
from gymnasium import spaces


@dataclass
class Ratios:
    """
    Importance-sampling ratios of one full pass over the buffer.
    All the tensors are aligned sample by sample:

        naive[n]     = pi(a_n|s_n) / pi_{k-i}(a_n|s_n)             with i = window[n]
        bh[n]        = pi(a_n|s_n) / mean_j pi_{k-j}(a_n|s_n)      (equal to naive[n] without BH weighting)
        reference[n] = pi_k(a_n|s_n) / pi_{k-i}(a_n|s_n)           pi_k = policy before the update
        window[n]    = i, which past policy collected the sample (0 = the most recent one)
    """
    naive: th.Tensor
    bh: th.Tensor
    reference: Optional[th.Tensor]
    window: th.Tensor

    @property
    def windows(self) -> list[int]:
        """The window indices present in the buffer, most recent first."""
        return sorted(int(w) for w in self.window.unique())

    def of(self, window_id: int) -> "Ratios":
        """The same ratios, restricted to the samples collected by pi_{k-window_id}."""
        mask = self.window == window_id
        return Ratios(
            naive=self.naive[mask],
            bh=self.bh[mask],
            reference=self.reference[mask] if self.reference is not None else None,
            window=self.window[mask],
        )


# --------------------------------------------------------------------------- #
# statistics of a set of ratios r (1-D tensor)                                 #
# --------------------------------------------------------------------------- #

def clip_fraction(r: th.Tensor, eps: float) -> th.Tensor:
    """P(|r - 1| > eps): fraction of samples the clipped surrogate objective clips."""
    return (th.abs(r - 1) > eps).float().mean()


def approx_kl(r: th.Tensor) -> th.Tensor:
    """E[(r - 1) - log r]: Schulman's estimator of KL(pi_old || pi), unbiased and non-negative."""
    return ((r - 1) - th.log(r)).mean()


def mean_abs_deviation(r: th.Tensor) -> th.Tensor:
    """E|r - 1|: how far the ratios sit from the on-policy value."""
    return (r - 1).abs().mean()


def normalized_ess(r: th.Tensor) -> th.Tensor:
    """(sum r)^2 / (N * sum r^2): effective sample size as a fraction of N, in (0, 1]."""
    return r.sum() ** 2 / (r.numel() * (r ** 2).sum())


def ratio_variance(r: th.Tensor) -> th.Tensor:
    """Var[r]."""
    return r.var()


def mean_abs_ratio_gap(r: th.Tensor, r_k: th.Tensor) -> th.Tensor:
    """
    E|r - r_k| with r = pi/pi_{k-i} and r_k = pi_k/pi_{k-i} on the same samples:
    the total-variation estimate of GePPO Lemma 3, which drives the adaptive learning rate.
    """
    return (r - r_k).abs().mean()


# --------------------------------------------------------------------------- #

def collect_ratios(
    policy,
    buffer,
    action_space,
    bh_ratio_fn: Optional[Callable[[th.Tensor, Any], th.Tensor]] = None,
    reference_policy=None,
) -> Ratios:
    """
    One full pass over the buffer, evaluating `policy` on every stored sample.

    bh_ratio_fn: (log_prob, rollout_data) -> BH ratio; without it the BH ratio is the naive one.
    reference_policy: pi_k; without it Ratios.reference is None.
    """
    naive_parts, bh_parts, ref_parts, window_parts = [], [], [], []

    was_training = policy.training
    policy.set_training_mode(False)
    with th.no_grad():
        for rollout_data in buffer.get(batch_size=None):
            actions = rollout_data.actions
            if isinstance(action_space, spaces.Discrete):
                actions = actions.long().flatten()
            _, log_prob, _ = policy.evaluate_actions(rollout_data.observations, actions)

            naive_ratio = th.exp(log_prob - rollout_data.old_log_prob)
            naive_parts.append(naive_ratio.cpu())
            bh_parts.append(bh_ratio_fn(log_prob, rollout_data).cpu() if bh_ratio_fn is not None else naive_ratio.cpu())
            if reference_policy is not None:
                _, ref_log_prob, _ = reference_policy.evaluate_actions(rollout_data.observations, actions)
                ref_parts.append(th.exp(ref_log_prob - rollout_data.old_log_prob).cpu())
            # plain SB3 buffers hold a single rollout, i.e. one window
            window_parts.append(getattr(rollout_data, "window_id", th.zeros_like(naive_ratio)).cpu())
    policy.set_training_mode(was_training)

    return Ratios(
        naive=th.cat(naive_parts),
        bh=th.cat(bh_parts),
        reference=th.cat(ref_parts) if ref_parts else None,
        window=th.cat(window_parts),
    )


# --------------------------------------------------------------------------- #
# statistics of the advantages                                                 #
# --------------------------------------------------------------------------- #

def advantages_by_window(buffer) -> tuple[np.ndarray, np.ndarray]:
    """
    The advantages currently stored in the buffer, plus the window each sample comes from.

    Read from the combined tensors and not from buffer.get(), which shuffles: two
    snapshots taken at different moments (before and after the VTRACE recomputation)
    must stay aligned sample by sample to be comparable.
    """
    if not buffer.generator_ready:
        next(buffer.get(batch_size=None))  # get() is what builds the combined tensors
    adv = buffer._combined_tensors["advantages"].reshape(-1).astype(np.float64)
    window = buffer._combined_tensors["window_id"].astype(int)
    return adv, window


def window_moments(adv: np.ndarray, window: np.ndarray, n_windows: int) -> tuple[np.ndarray, np.ndarray]:
    """
    (mu_i, sigma_i) of the advantages of each window, in the units they were estimated in.
    ddof=1 matches torch's std, i.e. the statistics the update itself normalizes with.
    """
    means = np.array([adv[window == i].mean() for i in range(n_windows)])
    stds = np.array([adv[window == i].std(ddof=1) for i in range(n_windows)])
    return means, stds


def standardize_by_window(adv: np.ndarray, window: np.ndarray,
                          means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """(A - mu_i) / sigma_i with the statistics of the window each sample belongs to."""
    return (adv - means[window]) / stds[window]


def sign_flip_fraction(old: np.ndarray, new: np.ndarray) -> float:
    """
    Fraction of samples whose (normalized) advantage changes sign: the update pushes
    those actions the opposite way. The sign is taken as A >= 0, so exactly-zero
    advantages fall on one side instead of counting as a third state.
    """
    if old.size == 0:
        return float("nan")
    return float(np.mean((old >= 0) != (new >= 0)))


def _ranks(x: np.ndarray) -> np.ndarray:
    """Ranks of x. Ties are broken arbitrarily: float advantages practically never tie."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = np.arange(x.size, dtype=np.float64)
    return ranks


def spearman_corr(old: np.ndarray, new: np.ndarray) -> float:
    """
    Spearman rank correlation between the two sets of advantages: how much the
    recomputation reshuffles their ordering. It is invariant to the per-window
    normalization (a positive affine map), so it measures the reordering alone.
    """
    if old.size < 2:
        return float("nan")
    return float(np.corrcoef(_ranks(old), _ranks(new))[0, 1])
