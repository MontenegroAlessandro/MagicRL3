"""Tests for ZDPG / ZDPG-S (algorithms/zdpg.py), finite and geometric horizons.

Run from the repo root with: python -m pytest tests/test_zdpg.py
"""

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3.common.env_util import make_vec_env

import envs  # noqa: F401  (registers LQR-v0)
from algorithms import FDPG, ZDPG

A_MATRIX = np.array([[0.9, 0.2], [0.0, 0.8]])
INIT_STATE = np.array([1.0, -1.0])
K_TEST = th.tensor([[-0.3, 0.1], [0.05, -0.2]])

# Deterministic toy 2x2 LQR (run/environment/lqr_toy_2x2.yaml), horizon set per test.
LQR_KWARGS = dict(
    A=A_MATRIX.tolist(), B=[[1.0, 0.0], [0.0, 1.0]], Q=[[1.0, 0.0], [0.0, 1.0]], R=[[1.0, 0.0], [0.0, 1.0]],
    noise=0.0, init_dist="fixed", init_state=INIT_STATE.tolist(),
)


class ToyEnv(gym.Env):
    """Same system as the toy LQR, plus an optional cubic action cost c * sum(a^3) (so
    that Q is not quadratic in the action) and an optional termination at a fixed step."""

    def __init__(self, c=0.0, terminate_at=None, render_mode=None):
        self.c, self.terminate_at = c, terminate_at
        self.observation_space = spaces.Box(-np.inf, np.inf, (2,), np.float64)
        self.action_space = spaces.Box(-10.0, 10.0, (2,), np.float64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state, self.k = INIT_STATE.copy(), 0
        return self.state.copy(), {}

    def step(self, action):
        a = np.ravel(np.asarray(action, dtype=float))
        s = self.state
        reward = -(s @ s + a @ a + self.c * np.sum(a**3))
        self.state = A_MATRIX @ s + a
        self.k += 1
        terminated = self.terminate_at is not None and self.k >= self.terminate_at
        return self.state.copy(), float(reward), terminated, False, {}


if "ZDPGToy-v0" not in gym.registry:
    gym.register("ZDPGToy-v0", entry_point=ToyEnv)


def exact_gradient(K, horizon, gamma, c=0.0, terminate_at=None):
    """grad_K of sum_{t < min(horizon, terminate_at)} gamma^t r_t for a = K s, via autograd."""
    K = K.clone().double().requires_grad_(True)
    A, s = th.tensor(A_MATRIX), th.tensor(INIT_STATE)
    J = th.zeros((), dtype=th.float64)
    steps = horizon if terminate_at is None else min(horizon, terminate_at)
    for t in range(steps):
        a = K @ s
        J = J - gamma**t * (s @ s + a @ a + c * (a**3).sum())
        s = A @ s + a
    J.backward()
    return K.grad.flatten()


def make_zdpg(env_id="LQR-v0", env_kwargs=None, n_envs=8, n_steps=10, gamma=1.0, sigma=0.1,
              mode="standard", sampling_mode="normal", horizon_mode="auto", seed=0, **kwargs):
    env_kwargs = dict(LQR_KWARGS, max_episode_steps=n_steps) if env_kwargs is None else env_kwargs
    env = make_vec_env(env_id, n_envs=n_envs, seed=seed, env_kwargs=env_kwargs)
    model = ZDPG(
        "MlpPolicy", env, n_steps=n_steps, gamma=gamma, sigma=sigma, batch_size=n_envs, mode=mode,
        sampling_mode=sampling_mode, horizon_mode=horizon_mode, env_id=env_id, env_kwargs=env_kwargs,
        max_grad_norm=None, policy_kwargs=dict(net_arch=kwargs.pop("net_arch", []), **kwargs.pop("policy_kwargs", {})),
        seed=seed, device="cpu", **kwargs,
    )
    if model.policy.net_arch == []:
        with th.no_grad():
            model.policy.action_net.weight.copy_(K_TEST)
    return model


def collect(model):
    if not hasattr(model, "_test_callback"):
        _, model._test_callback = model._setup_learn(total_timesteps=10**12, callback=None)
    assert model.collect_rollouts(model.env, model._test_callback, model.rollout_buffer, model.n_steps)


def zdpg_estimates(model, n_updates):
    out = []
    for _ in range(n_updates):
        collect(model)
        objective = model._objective()
        out.append(th.autograd.grad(objective, [model.policy.action_net.weight])[0].flatten().double())
        model._sample = None
    return th.stack(out)


def fdpg_estimates(model, n_updates):
    out = []
    for _ in range(n_updates):
        collect(model)
        _, terms, _, _ = model._train_objective_batched_step()
        objective = terms.mean() / model.sigma
        out.append(th.autograd.grad(objective, [model.policy.action_net.weight])[0].flatten().double())
    return th.stack(out)


def mean_and_se(estimates):
    return estimates.mean(0), estimates.std(0) / np.sqrt(len(estimates))


def assert_statistically_equal(mean_a, se_a, mean_b, se_b=None, n_se=4.5):
    se = se_a if se_b is None else th.sqrt(se_a**2 + se_b**2)
    deviation = ((mean_a - mean_b).abs() / se).max().item()
    assert deviation < n_se, f"{mean_a} vs {mean_b}: {deviation:.2f} standard errors apart"


# ------------------------------------------------------------------------ finite


@pytest.mark.parametrize("gamma", (1.0, 0.9))
@pytest.mark.parametrize("sampling_mode", ("normal", "sphere"))
def test_finite_matches_step_fdpg_and_true_gradient(gamma, sampling_mode):
    """Finite-horizon ZDPG is an unbiased single-term sample of Step-FDPG's time sum, so
    both have the same mean -- here, on an LQR (Q quadratic in a), the exact gradient."""
    H, sigma = 10, 0.1
    zdpg = make_zdpg(n_envs=1000, n_steps=H, gamma=gamma, sigma=sigma, sampling_mode=sampling_mode, seed=1)
    assert zdpg.resolved_horizon_mode == "finite"
    z_mean, z_se = mean_and_se(zdpg_estimates(zdpg, 40))

    env_kwargs = dict(LQR_KWARGS, max_episode_steps=H)
    fdpg = FDPG(
        "MlpPolicy", make_vec_env("LQR-v0", n_envs=100, seed=2, env_kwargs=env_kwargs), n_steps=H, gamma=gamma,
        sigma=sigma, batch_size=100, mode="step", sampling_mode=sampling_mode, sampling_strategy="step",
        env_id="LQR-v0", env_kwargs=env_kwargs, max_grad_norm=None, policy_kwargs=dict(net_arch=[]), seed=2,
        device="cpu",
    )
    with th.no_grad():
        fdpg.policy.action_net.weight.copy_(K_TEST)
    f_mean, f_se = mean_and_se(fdpg_estimates(fdpg, 10))

    g_true = exact_gradient(K_TEST, H, gamma)
    assert z_se.norm() < 0.05 * g_true.norm()  # the comparison is not vacuous
    assert_statistically_equal(z_mean, z_se, f_mean, f_se)
    assert_statistically_equal(z_mean, z_se, g_true)


@pytest.mark.parametrize("mode", ("standard", "symmetric"))
def test_bias_vanishes_as_sigma_goes_to_zero(mode):
    """With a cubic action cost Q is not quadratic in a, so a finite sigma biases the
    estimate (by O(sigma^2) for both modes); the bias vanishes as sigma -> 0."""
    H, gamma, c = 6, 1.0, 0.5
    env_kwargs = dict(c=c, max_episode_steps=H)
    g_true = exact_gradient(K_TEST, H, gamma, c=c)
    deviations = {}
    for sigma in (1.0, 0.05):
        model = make_zdpg("ZDPGToy-v0", env_kwargs, n_envs=500, n_steps=H, gamma=gamma, sigma=sigma, mode=mode, seed=3)
        mean, se = mean_and_se(zdpg_estimates(model, 20))
        deviations[sigma] = ((mean - g_true).abs() / se).max().item()
    assert deviations[1.0] > 10  # clearly biased at sigma = 1
    assert deviations[0.05] < 4.5  # statistically unbiased at small sigma


def test_termination_gives_zero_gradient_and_stays_unbiased():
    H, terminate_at, gamma = 10, 4, 1.0
    env_kwargs = dict(terminate_at=terminate_at, max_episode_steps=H)
    model = make_zdpg("ZDPGToy-v0", env_kwargs, n_envs=400, n_steps=H, gamma=gamma, seed=4)

    collect(model)
    s = model._sample
    assert np.array_equal(s["valid"], s["t"] < terminate_at)
    assert (~s["valid"]).any() and s["valid"].any()
    # Samples beyond termination have no branch rollout at all and zero coefficient.
    assert (model._branch_buffer._traj_lengths[~s["valid"]] == 0).all()
    # With the valid samples' differences zeroed, what is left (the samples beyond
    # termination, whatever their perturbation) contributes exactly nothing.
    q_plus = s["q_plus"].copy()
    s["q_plus"] = np.where(s["valid"], s["q_minus"], 123.0)
    grad = th.autograd.grad(model._objective(), [model.policy.action_net.weight])[0]
    assert grad.abs().max() == 0
    s["q_plus"] = q_plus
    # Every env step is counted: reference (terminated at step 4) plus branches.
    assert model.num_timesteps == model.rollout_buffer._traj_lengths.sum() + model._branch_buffer._traj_lengths.sum()
    model._sample = None

    mean, se = mean_and_se(zdpg_estimates(model, 20))
    assert_statistically_equal(mean, se, exact_gradient(K_TEST, H, gamma, terminate_at=terminate_at))


# --------------------------------------------------------------------- geometric


def test_geometric_rejects_gamma_one():
    with pytest.raises(ValueError, match="gamma < 1"):
        make_zdpg(gamma=1.0, horizon_mode="geometric")


def test_auto_resolution():
    assert make_zdpg(gamma=0.9).resolved_horizon_mode == "finite"  # env has a time limit
    assert make_zdpg("ZDPGToy-v0", {}, gamma=0.9).resolved_horizon_mode == "geometric"  # no time limit
    assert make_zdpg("ZDPGToy-v0", {}, gamma=1.0).resolved_horizon_mode == "finite"  # gamma = 1
    assert make_zdpg(gamma=0.9, horizon_mode="geometric").resolved_horizon_mode == "geometric"


def test_geometric_horizons_independent():
    gamma = 0.8
    model = make_zdpg("ZDPGToy-v0", {}, gamma=gamma, n_steps=1000)
    t, end, _ = model._sample_branch_times(400_000)
    t_q = end - t - 1
    for draw in (t, t_q):
        assert abs(draw.mean() - gamma / (1 - gamma)) < 0.03
        assert draw.min() == 0
    assert abs(np.corrcoef(t, t_q)[0, 1]) < 0.01


def test_geometric_symmetric_branches_share_t_q():
    model = make_zdpg("ZDPGToy-v0", {}, n_envs=64, gamma=0.8, n_steps=1000, mode="symmetric", seed=5)
    collect(model)
    M, s = model.n_envs, model._sample
    lengths = model._branch_buffer._traj_lengths
    # Both branches run exactly to the same end time T_s + T_Q + 1 ...
    assert np.array_equal(lengths[:M], lengths[M:])
    assert (lengths[:M] > s["t"]).all()
    # ... while the reference only runs up to the branch state.
    assert np.array_equal(model.rollout_buffer._traj_lengths, s["t"] + 1)


def test_symmetric_finite_reference_stops_at_branch_state():
    H = 30
    model = make_zdpg(n_envs=32, n_steps=H, gamma=0.99, mode="symmetric", seed=12)
    collect(model)
    t, M = model._sample["t"], model.n_envs
    assert np.array_equal(model.rollout_buffer._traj_lengths, t + 1)
    assert (model._branch_buffer._traj_lengths == H).all()
    assert model.num_timesteps == (t + 1).sum() + 2 * M * H
    model.train()
    assert "train/mean_return" not in model.logger.name_to_value


def test_geometric_truncation_warns_and_is_logged():
    model = make_zdpg("ZDPGToy-v0", dict(max_episode_steps=5), n_envs=64, gamma=0.9,
                      n_steps=5, horizon_mode="geometric", seed=6)
    with pytest.warns(UserWarning, match="truncated"):
        collect(model)
    assert model._sample["truncated"].mean() > 0.3
    model.train()
    assert model.logger.name_to_value["train/truncated_fraction"] > 0.3


@pytest.mark.parametrize("mode", ("standard", "symmetric"))
def test_geometric_unbiased_for_infinite_horizon(mode):
    """gamma < 1, H = infinity (no time limit; n_steps only a cap that is ~never hit)."""
    gamma = 0.5
    model = make_zdpg("ZDPGToy-v0", {}, n_envs=1000, n_steps=200, gamma=gamma, sigma=0.1, mode=mode, seed=7)
    assert model.resolved_horizon_mode == "geometric"
    mean, se = mean_and_se(zdpg_estimates(model, 20))
    g_true = exact_gradient(K_TEST, 200, gamma)
    assert se.norm() < 0.05 * g_true.norm()
    assert_statistically_equal(mean, se, g_true)


# --------------------------------------------------------- branching and training


def _assert_prefix_identical(model):
    s, M = model._sample, model.n_envs
    width = model._branch_buffer.n_envs
    assert s["valid"].any()
    for j in range(width):
        i = j % M
        if not s["valid"][i]:
            continue
        t = s["t"][i]
        ref_obs = np.array(model.rollout_buffer._obs[i][: t + 1])
        branch_obs = np.array(model._branch_buffer._obs[j][: t + 1])
        assert np.array_equal(ref_obs, branch_obs), f"branch {j} prefix differs from reference {i}"
        assert np.array_equal(np.array(model.rollout_buffer._actions[i][:t]),
                              np.array(model._branch_buffer._actions[j][:t]))
        # The branch action differs from the nominal one exactly at step t.
        assert not np.array_equal(model.rollout_buffer._actions[i][t], model._branch_buffer._actions[j][t])


@pytest.mark.parametrize("mode", ("standard", "symmetric"))
def test_branch_prefix_bit_identical_lqr(mode):
    model = make_zdpg(n_envs=16, n_steps=30, gamma=0.99, mode=mode, seed=8)
    collect(model)
    _assert_prefix_identical(model)


@pytest.mark.parametrize("mode", ("standard", "symmetric"))
def test_branch_prefix_bit_identical_mujoco(mode):
    pytest.importorskip("mujoco")
    model = make_zdpg("Hopper-v5", {}, n_envs=4, n_steps=100, gamma=0.99, mode=mode, seed=9, net_arch=[32, 32])
    collect(model)
    _assert_prefix_identical(model)
    assert model.num_timesteps == model.rollout_buffer._traj_lengths.sum() + model._branch_buffer._traj_lengths.sum()


def test_train_ascends_along_estimate():
    model = make_zdpg(n_envs=16, learning_rate=0.01, policy_kwargs=dict(optimizer_class=th.optim.SGD), seed=10)
    collect(model)
    weight = model.policy.action_net.weight
    g = th.autograd.grad(model._objective(), [weight])[0]
    before = weight.detach().clone()
    model.train()
    assert th.allclose(weight.detach(), before + 0.01 * g, atol=1e-6)


def test_seeding_reproducible():
    estimates = [zdpg_estimates(make_zdpg(n_envs=8, seed=11), 2) for _ in range(2)]
    assert th.equal(*estimates)
