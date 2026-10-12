"""Tests for the parameter-space finite-difference baseline (algorithms/bastani_fd.py).

Run from the repo root with: python -m pytest tests/test_bastani_fd.py
"""

import numpy as np
import pytest
import torch as th
from stable_baselines3.common.env_util import make_vec_env
from torch.nn.utils import parameters_to_vector

import envs  # noqa: F401  (registers LQR-v0)
from algorithms import BastaniFD

SAMPLING_MODES = ("normal", "sphere")

# Deterministic toy 2x2 LQR (run/environment/lqr_toy_2x2.yaml): no process noise and a
# fixed initial state, so the return is a deterministic, differentiable function of K.
LQR_KWARGS = dict(
    A=[[0.9, 0.2], [0.0, 0.8]],
    B=[[1.0, 0.0], [0.0, 1.0]],
    Q=[[1.0, 0.0], [0.0, 1.0]],
    R=[[1.0, 0.0], [0.0, 1.0]],
    noise=0.0,
    init_dist="fixed",
    init_state=[1.0, -1.0],
    max_episode_steps=50,
)
HORIZON = 50
GAMMA = 0.99


def make_model(n_envs=1, sampling_mode="normal", sigma=1e-3, seed=0, use_crn=False, **kwargs):
    env = make_vec_env("LQR-v0", n_envs=n_envs, seed=seed, env_kwargs=LQR_KWARGS)
    return BastaniFD(
        "MlpPolicy",
        env,
        n_steps=HORIZON,
        gamma=GAMMA,
        sigma=sigma,
        batch_size=n_envs,
        sampling_mode=sampling_mode,
        use_crn=use_crn,
        max_grad_norm=None,
        policy_kwargs=dict(net_arch=[], **kwargs.pop("policy_kwargs", {})),
        seed=seed,
        device="cpu",
        **kwargs,
    )


def collect(model):
    _, callback = model._setup_learn(total_timesteps=10**12, callback=None)
    assert model.collect_rollouts(model.env, callback, model.rollout_buffer, model.n_steps)
    return model._gradient_estimate.clone()


def theta_of(model):
    return parameters_to_vector(model._search_parameters(model.policy)).detach().clone()


def lqr_return(K: th.Tensor) -> th.Tensor:
    """Discounted H-step return of a = K s on the toy LQR, differentiable in K."""
    A = th.tensor(LQR_KWARGS["A"], dtype=th.float64)
    B = th.tensor(LQR_KWARGS["B"], dtype=th.float64)
    Q = th.tensor(LQR_KWARGS["Q"], dtype=th.float64)
    R = th.tensor(LQR_KWARGS["R"], dtype=th.float64)
    s = th.tensor(LQR_KWARGS["init_state"], dtype=th.float64)
    K = K.to(th.float64)
    J = th.zeros((), dtype=th.float64)
    for t in range(HORIZON):
        u = K @ s
        J = J - GAMMA**t * (s @ Q @ s + u @ R @ u)
        s = A @ s + B @ u
    return J


# ---------------------------------------------------------------- analytic objectives


@pytest.mark.parametrize("sampling_mode", SAMPLING_MODES)
def test_recovers_gradient_of_quadratic(sampling_mode):
    """Averaging many directions recovers grad J; on a quadratic the two-sided difference
    is exact for any sigma, so this isolates the q(nu) scaling from any bias."""
    model = make_model(sampling_mode=sampling_mode, sigma=0.7)
    dim, n_directions = 6, 400_000
    gen = th.Generator().manual_seed(1)
    M = th.randn(dim, dim, generator=gen)
    A = M @ M.T + th.eye(dim)
    b = th.randn(dim, generator=gen)
    theta = th.randn(dim, generator=gen)

    def J(thetas):
        return -0.5 * th.einsum("ni,ij,nj->n", thetas, A, thetas) + thetas @ b

    nu, q_nu = model._sample_directions(n_directions, dim)
    if sampling_mode == "sphere":
        assert th.allclose(nu.norm(dim=1), th.ones(n_directions), atol=1e-5)
    g_hat = model._estimate_gradient(J(theta + model.sigma * nu), J(theta - model.sigma * nu), q_nu)
    g_true = -A @ theta + b
    assert (g_hat - g_true).norm() / g_true.norm() < 0.02


@pytest.mark.parametrize("sampling_mode", SAMPLING_MODES)
def test_bias_vanishes_as_sigma_goes_to_zero(sampling_mode):
    """On a non-quadratic J the estimator is biased for finite sigma (it estimates the
    gradient of the smoothed J); with the same directions, the bias vanishes as O(sigma^2)."""
    dim, n_directions = 5, 400_000
    gen = th.Generator().manual_seed(2)
    theta = th.randn(dim, generator=gen, dtype=th.float64)

    def J(thetas):
        return (thetas**3).sum(-1) / 3 + th.sin(thetas).sum(-1)

    g_true = theta**2 + th.cos(theta)
    model = make_model(sampling_mode=sampling_mode)
    nu, q_nu = model._sample_directions(n_directions, dim)
    nu, q_nu = nu.double(), q_nu.double()
    # Same directions with the exact directional derivative: what remains is pure bias.
    g_exact_fd = ((nu @ g_true)[:, None] * q_nu).mean(0)

    biases = []
    for sigma in (0.5, 0.1, 0.02, 0.004):
        model.sigma = sigma
        g_hat = model._estimate_gradient(J(theta + sigma * nu), J(theta - sigma * nu), q_nu)
        biases.append((g_hat - g_exact_fd).norm().item())
    assert biases[0] > 1e-2  # finite sigma really is biased here
    for coarse, fine in zip(biases, biases[1:]):
        assert fine < coarse / 10  # sigma / 5  ->  bias / 25
    assert biases[-1] < 1e-4
    # ... and the Monte-Carlo average of the smallest-sigma estimate matches grad J.
    assert (g_hat - g_true).norm() / g_true.norm() < 0.02


# ----------------------------------------------------------- end to end on the LQR


@pytest.mark.parametrize("sampling_mode", SAMPLING_MODES)
def test_lqr_estimate_matches_true_gradient(sampling_mode):
    n_envs, n_updates = 256, 20
    model = make_model(n_envs=n_envs, sampling_mode=sampling_mode, sigma=1e-3)
    K = th.tensor([[-0.2, 0.1], [0.05, -0.3]])
    with th.no_grad():
        model.policy.action_net.weight.copy_(K)
    theta_before = theta_of(model)

    estimates = [collect(model) for _ in range(n_updates)]
    g_hat = th.stack(estimates).mean(0).double()

    K_true = K.clone().double().requires_grad_(True)
    lqr_return(K_true).backward()
    g_true = K_true.grad.flatten()  # action_net.weight is the only searched parameter
    assert model._search_parameter_names(model.policy) == ["action_net.weight"]
    assert (g_hat - g_true).norm() / g_true.norm() < 0.05
    # theta is bit-identical after collection: perturbations never touch the nominal policy.
    assert th.equal(theta_of(model), theta_before)


def test_rollouts_are_deterministic_and_use_perturbed_parameters():
    """Every + / - return equals the noiseless return of mu_{theta +/- sigma nu_i}, so no
    action noise is injected and sub-env i really plays its own direction."""
    n_envs, sigma = 8, 0.05
    model = make_model(n_envs=n_envs, sigma=sigma)
    recorded = {}
    sample = model._sample_directions

    def recording_sample(n, d):
        recorded["nu"], recorded["q_nu"] = sample(n, d)
        return recorded["nu"], recorded["q_nu"]

    model._sample_directions = recording_sample
    theta = theta_of(model)
    g_hat = collect(model)

    j_plus = th.stack([lqr_return((theta + sigma * v).reshape(2, 2)) for v in recorded["nu"]])
    j_minus = th.stack([lqr_return((theta - sigma * v).reshape(2, 2)) for v in recorded["nu"]])
    buffer_minus = th.tensor([model.rollout_buffer._returns[i][0] for i in range(n_envs)], dtype=th.float64)
    assert th.allclose(buffer_minus, j_minus, rtol=1e-5)
    assert th.allclose(th.tensor(model._mean_return_plus, dtype=th.float64), j_plus.mean(), rtol=1e-5)
    expected = ((j_plus - j_minus) / (2 * sigma))[:, None] * recorded["q_nu"].double()
    assert th.allclose(g_hat.double(), expected.mean(0), rtol=1e-3, atol=1e-3)


def test_timestep_accounting_two_trajectories_per_direction():
    n_envs = 4
    model = make_model(n_envs=n_envs)
    collect(model)
    assert model.num_timesteps == 2 * n_envs * HORIZON


def test_train_ascends_along_estimate():
    model = make_model(n_envs=4, learning_rate=0.1, policy_kwargs=dict(optimizer_class=th.optim.SGD))
    g_hat = collect(model)
    theta = theta_of(model)
    model.train()
    assert th.allclose(theta_of(model), theta + 0.1 * g_hat, atol=1e-6)


@pytest.mark.parametrize("use_crn", (False, True))
def test_seeding_reproducible_and_crn(use_crn):
    seeds = []
    for _ in range(2):
        model = make_model(n_envs=3, seed=7, use_crn=use_crn)
        _, callback = model._setup_learn(total_timesteps=10**12, callback=None)
        rollout_seeds = []
        reset_env = model._reset_env

        def recording_reset(env):
            obs = reset_env(env)
            rollout_seeds.append(model._episode_seeds.copy())
            return obs

        model._reset_env = recording_reset
        model.collect_rollouts(model.env, callback, model.rollout_buffer, model.n_steps)
        seeds.append((rollout_seeds, model._gradient_estimate.clone()))

    (seeds_a, g_a), (seeds_b, g_b) = seeds
    assert th.equal(g_a, g_b)
    assert all(np.array_equal(x, y) for x, y in zip(seeds_a, seeds_b))
    plus_seeds, minus_seeds = seeds_a
    # Same base-seed stream and draw pattern as FDPG's reference reset.
    assert plus_seeds[0] == np.random.default_rng(7).integers(0, 2**31 - 1)
    assert np.array_equal(plus_seeds, minus_seeds) == use_crn


def test_rejects_mismatched_batch_size():
    env = make_vec_env("LQR-v0", n_envs=2, seed=0, env_kwargs=LQR_KWARGS)
    with pytest.raises(ValueError, match="batch_size"):
        BastaniFD("MlpPolicy", env, n_steps=HORIZON, batch_size=3, policy_kwargs=dict(net_arch=[]))
