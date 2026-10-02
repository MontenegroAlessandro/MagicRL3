"""Zeroth-order Deterministic Policy Gradient (ZDPG), Kumar et al. 2020.

Zeroth-order Deterministic Policy Gradient: https://arxiv.org/abs/2006.07314
Algorithm 1 (ZDPG), Algorithm 2 (Q-function sampler) and Algorithm 3 (discounted
state sampler) of that paper, plus a finite-horizon variant (also valid for gamma = 1).
"""

import warnings
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, unwrap_vec_normalize

from algorithms.trajectory_onpolicy_method import TrajectoryOnPolicyAlgorithm
from buffers import TrajectoryBuffer
from policies import ActorOnlyPolicy

SelfZDPG = TypeVar("SelfZDPG", bound="ZDPG")

MODES = ("standard", "symmetric")
"""
The two two-point gradient representations of Lemma 1 (eq. 6).
1.  standard: ZDPG, [Q(s, a + sigma * u) - Q(s, a)] / sigma.
2.  symmetric: ZDPG-S, [Q(s, a + sigma * u) - Q(s, a - sigma * u)] / (2 * sigma).
"""

SAMPLING_MODES = ("normal", "sphere")
"""
Distribution of the action perturbation u (same as FDPG).
1.  normal: u ~ N(0_{d_A}, I_{d_A}), q(u) = u.
2.  sphere: u ~ Unif(S^{d_A-1}), q(u) = d_A * u.
"""

HORIZON_MODES = ("auto", "finite", "geometric")
"""
Which objective the estimator targets.
1.  finite: J = E[sum_{t<H} gamma^t r_t] with H = min(env time limit, n_steps), the
    objective of FDPG/REINFORCE; gamma = 1 allowed. Branch time t ~ Unif{0..H-1},
    Q = discounted return-to-go from t to H (or termination), weight H * gamma^t.
2.  geometric: the paper's infinite-horizon discounted objective (gamma < 1). Branch
    time T_s ~ Geom(1 - gamma), Q = undiscounted sum of T_Q + 1 rewards with an
    independent T_Q ~ Geom(1 - gamma), weight 1 / (1 - gamma). n_steps and the env time
    limit only act as a safety cap: draws reaching it are truncated (biased), warned
    about once and logged as train/truncated_fraction.
3.  auto: finite if gamma = 1 or the env has a time limit, geometric otherwise.
"""

STATE_MATCH_ATOL = 1e-5
"""
Tolerance used when checking that a branch sub-env replayed its reference's trajectory
exactly. Replaying the same reset seed with the same deterministic policy has to
reproduce the very same states, so any real mismatch is an environment whose randomness
is not determined by its reset seed, not a numerical artifact.
"""


class ZDPG(TrajectoryOnPolicyAlgorithm):
    """
    Zeroth-order Deterministic Policy Gradient (Kumar et al., 2020).

    One update draws M = batch_size = n_envs independent samples. Sample i follows the
    deterministic policy mu_theta from s_0 in the training env (the reference rollout),
    picks a branch time t_i (see HORIZON_MODES) and an action perturbation u_i (see
    SAMPLING_MODES), and estimates the Q-function at s_{t_i} with the perturbed first
    action mu_theta(s_{t_i}) + sigma u_i against the nominal one (standard) or against
    mu_theta(s_{t_i}) - sigma u_i (symmetric), following mu_theta afterwards. Then

        g = 1/M sum_i w(t_i) grad_theta mu_theta(s_{t_i})^T (Q+_i - Q-_i) / (sigma or 2 sigma) q(u_i),

    computed as a single vector-Jacobian product. The perturbation lives in action space
    only, and no critic is involved: the Q-values come straight from rollouts.

    Branching: a side pool of M (standard) or 2M (symmetric) sub-envs resets with the
    reference's own reset seeds and replays mu_theta for t_i steps, landing exactly on
    s_{t_i} (verified) with the same env RNG state, before playing its branch action. In
    standard mode the reference rollout itself is the nominal branch: it provides both
    s_{t_i} and Q-_i, so only one extra rollout per sample is needed.

    A sample whose episode terminated before reaching t_i contributes a zero gradient
    and still counts in the average over M (unbiased: the terminal state is absorbing).

    :param sigma: Positive action-perturbation radius (mu in the paper).
    :param batch_size: M, the number of samples per update; must equal env.num_envs.
    :param mode: "standard" (ZDPG) or "symmetric" (ZDPG-S), see MODES.
    :param sampling_mode: "normal" or "sphere", see SAMPLING_MODES.
    :param horizon_mode: "auto", "finite" or "geometric", see HORIZON_MODES.
    :param env_id: Gym id used to build the branch pool.
    :param env_kwargs: Extra kwargs for that pool's environments; must match the ones
        of the training env, otherwise the replayed states cannot agree.

    Cost per update (finite mode), including the replayed prefixes: at most 2 * M * H
    steps for standard (reference to H + one branch to H), and sum_i (t_i + 1) + 2 * M * H
    for symmetric (reference only up to s_{t_i} + two branches to H), i.e. about
    2.5 * M * H on average. In symmetric mode train/mean_return is not logged, since the
    reference rollouts stop at the branch state; use eval/* instead.
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorOnlyPolicy,
    }

    def __init__(
        self,
        policy: Union[str, type[ActorOnlyPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        n_steps: int = 1000,
        gamma: float = 0.99,
        sigma: float = 0.1,
        batch_size: int = 1,
        mode: str = "standard",
        sampling_mode: str = "normal",
        horizon_mode: str = "auto",
        env_id: str = None,
        env_kwargs: Optional[dict[str, Any]] = None,
        max_grad_norm: Optional[float] = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[TrajectoryBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        perturbed_env_wrapper_class=None,
        perturbed_env_wrapper_kwargs=None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        if sigma is None or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("[ZDPG] sigma must be finite and strictly positive")
        if mode not in MODES:
            raise ValueError(f"[ZDPG] mode must be one of {MODES}, but got {mode}")
        if sampling_mode not in SAMPLING_MODES:
            raise ValueError(f"[ZDPG] sampling_mode must be one of {SAMPLING_MODES}, but got {sampling_mode}")
        if horizon_mode not in HORIZON_MODES:
            raise ValueError(f"[ZDPG] horizon_mode must be one of {HORIZON_MODES}, but got {horizon_mode}")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("[ZDPG] batch_size must be a positive integer")
        if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 1:
            raise ValueError("[ZDPG] n_steps must be a positive integer")
        if not np.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("[ZDPG] gamma must be between 0 and 1")
        if horizon_mode == "geometric" and gamma >= 1:
            # Geom(1 - gamma) is almost surely infinite for gamma = 1.
            raise ValueError("[ZDPG] horizon_mode='geometric' needs gamma < 1; use 'finite' for gamma = 1")
        if use_sde:
            raise ValueError("[ZDPG] the policy is deterministic; use_sde must be False")

        policy_kwargs = dict(policy_kwargs or {})
        # The policy is a deterministic map s -> a; there is no action distribution to learn.
        policy_kwargs["learn_std"] = False
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            max_grad_norm=max_grad_norm,
            collect_deterministic_rollouts=True,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
            supported_action_spaces=(spaces.Box,),
        )

        self.sigma = float(sigma)
        self.batch_size = batch_size
        self.mode = mode
        self.sampling_mode = sampling_mode
        self.horizon_mode = horizon_mode
        self.env_id = env_id
        self.env_kwargs = env_kwargs if env_kwargs is not None else {}
        self.perturbed_env_wrapper_class = perturbed_env_wrapper_class
        self.perturbed_env_wrapper_kwargs = perturbed_env_wrapper_kwargs or {}
        self._n_updates = 0

        self.name = "ZDPG" if mode == "standard" else "ZDPG-S"

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        if self.n_envs != self.batch_size:
            raise ValueError(
                f"[ZDPG] env.num_envs must equal batch_size: got {self.n_envs} envs but "
                f"batch_size={self.batch_size}. Build the training env with n_envs=batch_size "
                f"(one sub-env per sample)."
            )
        # Checked here rather than in __init__ so that load() -- which rebuilds the model
        # with default arguments before restoring the saved ones -- still works.
        if self.env_id is None:
            raise ValueError("[ZDPG] env_id must be provided to build the branch pool")

        # Same dedicated RNG streams as FDPG (env reseeding, perturbation sampling), plus
        # one for the branch times, all seeded from self.seed and independent of the
        # global numpy/torch streams.
        self._episode_seed_rng = np.random.default_rng(self.seed)
        self._perturbation_rng = None
        if self.seed is not None:
            self._perturbation_rng = th.Generator(device=self.device)
            self._perturbation_rng.manual_seed(self.seed)
        self._horizon_rng = np.random.default_rng(np.random.SeedSequence(self.seed).spawn(1)[0])

        # Sub-env j is the branch of sample j % M; in symmetric mode j < M is the + branch
        # and j >= M the - branch, in standard mode there is only the + branch.
        self._branch_width = self.n_envs * (2 if self.mode == "symmetric" else 1)
        self._q_env = make_vec_env(
            self.env_id,
            n_envs=self._branch_width,
            env_kwargs=self.env_kwargs,
            vec_env_cls=DummyVecEnv,
            wrapper_class=self.perturbed_env_wrapper_class,
            wrapper_kwargs=self.perturbed_env_wrapper_kwargs,
        )
        self._branch_buffer = self.rollout_buffer_class(
            self.n_steps, self.observation_space, self.action_space,
            device=self.device, gamma=self.gamma, n_envs=self._branch_width,
        )
        # NOTE: the pool is not seeded here; every iteration reseeds it to replay the
        # reset seeds of the reference rollout (see _collect_branches).

        # Horizon: the env's time limit (TimeLimit wrapper) capped by n_steps.
        self._time_limit = self._q_env.get_attr("spec")[0].max_episode_steps
        self._horizon = min(self.n_steps, self._time_limit) if self._time_limit else self.n_steps
        if self.horizon_mode == "auto":
            finite = self.gamma >= 1 or self._time_limit is not None
            self._resolved_horizon_mode = "finite" if finite else "geometric"
        else:
            self._resolved_horizon_mode = self.horizon_mode
        if self._resolved_horizon_mode == "geometric" and self.gamma >= 1:
            raise ValueError("[ZDPG] horizon_mode='geometric' needs gamma < 1; use 'finite' for gamma = 1")
        if self.verbose >= 1:
            print(f"[ZDPG] horizon_mode={self.horizon_mode} resolved to "
                  f"'{self._resolved_horizon_mode}' (H={self._horizon}, gamma={self.gamma})")
        self._warned_truncation = False

        self._action_low = th.as_tensor(self.action_space.low, device=self.device)
        self._action_high = th.as_tensor(self.action_space.high, device=self.device)

        self._episode_seeds = None
        self._step_caps = None
        self._sample = None

        if self.env is not None:
            self._check_normalization(self.env)

    @property
    def resolved_horizon_mode(self) -> str:
        return self._resolved_horizon_mode

    @staticmethod
    def _check_normalization(env: VecEnv) -> None:
        normalizer = unwrap_vec_normalize(env)
        if normalizer is not None:
            raise ValueError(
                "[ZDPG] the branch pool is built from raw environments, so a "
                "VecNormalize training env would compare normalized reference states "
                "and returns against unnormalized rollouts. Drop VecNormalize."
            )

    def _reset_env(self, env: VecEnv) -> np.ndarray:
        # Explicit seed drawn from our own dedicated RNG (see _setup_model) rather than
        # env.seed(None), which would silently fall back to the global numpy RNG stream.
        # The seeds are recorded because the branch pool has to replay them.
        base_seed = int(self._episode_seed_rng.integers(0, 2**31 - 1))
        self._episode_seeds = np.array(env.seed(base_seed))
        return env.reset()

    def _rollout_step_caps(self) -> Optional[np.ndarray]:
        return self._step_caps

    def _clip(self, actions: th.Tensor) -> th.Tensor:
        """Same action-bound handling as the base collector, applied to the branches so
        that the Q-values always refer to actions the system can execute."""
        return th.max(th.min(actions, self._action_high), self._action_low)

    def _sample_perturbations(self, n_samples: int) -> tuple[th.Tensor, th.Tensor]:
        """Draw u (n_samples, action_dim) and q(u), with FDPG's sampling conventions."""
        action_dim = get_action_dim(self.action_space)
        u = th.randn(n_samples, action_dim, device=self.device, generator=self._perturbation_rng)
        if self.sampling_mode == "normal":
            q_u = u
        elif self.sampling_mode == "sphere":
            u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            q_u = action_dim * u
        else:
            raise ValueError(f"[ZDPG] sampling_mode must be one of {SAMPLING_MODES}, but got {self.sampling_mode}")
        return u, q_u

    def _sample_branch_times(self, n_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Draw each sample's branch time t and end time e (the Q-rollouts cover steps
        t, ..., e - 1), independently across samples.

        - finite:    t ~ Unif{0, ..., H-1}, e = H.
        - geometric: t = T_s, e = T_s + T_Q + 1 with T_s, T_Q ~ Geom(1 - gamma) on
                     {0, 1, ...}, independent (the paper's Algorithm 1 reuses a single
                     draw for both, which biases the estimator). Both branches of a
                     sample share e, i.e. T_Q.

        :return: (t, e, truncated), truncated flagging geometric draws that the horizon
            cap H cuts short (e > H).
        """
        H = self._horizon
        if self._resolved_horizon_mode == "finite":
            t = self._horizon_rng.integers(0, H, size=n_samples)
            end = np.full(n_samples, H, dtype=np.int64)
            return t, end, np.zeros(n_samples, dtype=bool)
        # numpy counts trials until the first success, so the number of failures is one less.
        t = self._horizon_rng.geometric(1.0 - self.gamma, size=n_samples) - 1
        t_q = self._horizon_rng.geometric(1.0 - self.gamma, size=n_samples) - 1
        end = t + t_q + 1
        return t, end, end > H

    def _collect_branches(self, t: np.ndarray, end: np.ndarray, valid: np.ndarray,
                          states: np.ndarray, u: th.Tensor) -> TrajectoryBuffer:
        """
        Algorithm 2 for every sample at once, in lockstep: branch sub-env j (sample
        i = j % M) resets with the reference's reset seed, replays mu_theta for t_i
        steps (landing on s_{t_i}, checked), plays mu_theta(s_{t_i}) +/- sigma u_i at
        step t_i and follows mu_theta until step end_i (capped at H) or termination.
        Samples that are not valid never start, and cost nothing.
        """
        env = self._q_env
        width, M = env.num_envs, self.n_envs
        sample = np.arange(width) % M
        sign = np.where(np.arange(width) < M, 1.0, -1.0)
        caps = np.minimum(end, self._horizon)[sample]
        branch_t = t[sample]

        self._branch_buffer.reset()
        # Same reset seeds as the reference: same initial state and same env RNG stream.
        env._seeds = [int(self._episode_seeds[i]) for i in sample]
        obs = env.reset()
        active = valid[sample].copy()

        perturbation = (self.sigma * u)[sample] * th.as_tensor(sign, dtype=u.dtype, device=self.device)[:, None]

        for k in range(int(caps[active].max()) if active.any() else 0):
            branching = active & (branch_t == k)
            if branching.any():
                deviation = np.abs(np.asarray(obs)[branching] - states[sample[branching]]).max()
                if not np.isfinite(deviation) or deviation > STATE_MATCH_ATOL:
                    raise ValueError(
                        f"[ZDPG] replaying the reference rollout in the branch pool landed "
                        f"{deviation:.3e} away from the reference state (tolerance "
                        f"{STATE_MATCH_ATOL:.0e}). Q-values must be evaluated at the very state "
                        f"the gradient is taken at, so this environment cannot be used: its "
                        f"randomness is not fully determined by the reset seed, or the pool's "
                        f"env_id/env_kwargs/wrappers differ from the training env's."
                    )
            # One forward pass per group of M sub-envs: the same batch shape as the reference
            # rollout's, so that the replayed prefix is bit-identical (BLAS kernels, and with
            # them float rounding, depend on the matrix shape).
            obs_tensor = obs_as_tensor(obs, self.device)
            with th.no_grad():
                mu = th.cat([self.policy(chunk, deterministic=True)[0] for chunk in obs_tensor.split(M)])
            mask =th.as_tensor(branching, device=self.device)[:, None]
            actions = self._clip(mu + perturbation * mask).cpu().numpy()
            new_obs, rewards, dones, _ = env.step(actions)

            active_indices = np.where(active)[0]
            self._branch_buffer.add(
                obs[active_indices], actions[active_indices], rewards[active_indices],
                np.zeros(len(active_indices)), env_indices=active_indices,
            )
            # Branch interaction is real environment cost: count it, replayed prefix included.
            self.num_timesteps += int(active.sum())
            active &= ~dones
            active &= (k + 1) < caps
            obs = new_obs
            if not active.any():
                break

        lengths = self._branch_buffer._traj_lengths
        if (valid[sample] & (lengths <= branch_t)).any():
            raise ValueError(
                "[ZDPG] an episode ended while replaying the reference rollout in the "
                "branch pool, although the reference did not. Check that the pool's "
                "env_id/env_kwargs/wrappers match the training env's."
            )
        self._branch_buffer.compute_returns()
        return self._branch_buffer

    def _q_value(self, buffer: TrajectoryBuffer, index: int, t: int) -> float:
        """Q estimate of a trajectory from its step t: the discounted return-to-go
        (finite, same convention as FDPG) or the undiscounted reward sum (geometric)."""
        if self._resolved_horizon_mode == "finite":
            return float(buffer._returns[index][t])
        return float(np.sum(buffer._rewards[index][t:], dtype=np.float64))

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: TrajectoryBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect the M reference rollouts (via the base class, so that episode statistics,
        callbacks and timestep accounting stay the repo's), then the branch rollouts.
        """
        self._check_normalization(env)
        self._sample = None
        M = env.num_envs
        t, end, truncated = self._sample_branch_times(M)

        # Each reference only runs as long as it is used; a sample whose branch time is
        # already beyond the cap never starts.
        # - symmetric: both Q-values come from the branches, so the reference only has to
        #   reach s_{t_i} (t_i + 1 steps, the last one recording s_{t_i}).
        # - standard, geometric: the reference is the nominal branch up to its end time.
        # - standard, finite: the reference is the nominal branch up to H (no cap needed).
        if self.mode == "symmetric":
            self._step_caps = np.where(t >= self._horizon, 0, t + 1)
        elif self._resolved_horizon_mode == "geometric":
            self._step_caps = np.where(t >= self._horizon, 0, np.minimum(end, self._horizon))
        try:
            horizon = min(n_rollout_steps, self._horizon)
            if not super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps=horizon):
                return False
        finally:
            self._step_caps = None

        # s_{t_i} is reached only if the trajectory has more than t_i steps; otherwise it
        # terminated before (or the geometric draw was cut by the cap): zero gradient.
        lengths = rollout_buffer._traj_lengths
        valid = lengths > t
        states = np.zeros((M, *self.observation_space.shape), dtype=self.observation_space.dtype)
        for i in np.where(valid)[0]:
            states[i] = rollout_buffer._obs[i][t[i]]

        u, q_u = self._sample_perturbations(M)
        branches = self._collect_branches(t, end, valid, states, u)

        q_plus, q_minus = np.zeros(M), np.zeros(M)
        for i in np.where(valid)[0]:
            q_plus[i] = self._q_value(branches, i, t[i])
            if self.mode == "symmetric":
                q_minus[i] = self._q_value(branches, M + i, t[i])
            else:
                # The reference rollout is the nominal branch.
                q_minus[i] = self._q_value(rollout_buffer, i, t[i])

        if truncated.any() and not self._warned_truncation:
            warnings.warn(
                f"[ZDPG] geometric horizon draws exceed the horizon cap H={self._horizon} "
                f"(n_steps / env time limit) and are truncated, which biases the estimator. "
                f"Increase n_steps or use horizon_mode='finite'. The truncated fraction is "
                f"logged as train/truncated_fraction.",
                UserWarning,
            )
            self._warned_truncation = True

        self._sample = dict(t=t, valid=valid, truncated=truncated, states=states, q_u=q_u,
                            q_plus=q_plus, q_minus=q_minus)
        return True

    def _objective(self) -> th.Tensor:
        """
        1/M sum_i w(t_i) <mu_theta(s_i), (Q+_i - Q-_i) / (sigma or 2 sigma) q(u_i)>, whose
        gradient is the ZDPG estimate (a vector-Jacobian product through mu_theta at the
        branch states only). Invalid samples carry a zero coefficient but count in M.
        """
        s = self._sample
        denominator = 2 * self.sigma if self.mode == "symmetric" else self.sigma
        if self._resolved_horizon_mode == "finite":
            weights = self._horizon * self.gamma ** s["t"].astype(np.float64)
        else:
            weights = np.full(len(s["t"]), 1.0 / (1.0 - self.gamma))
        coefficients = np.where(s["valid"], weights * (s["q_plus"] - s["q_minus"]) / denominator, 0.0)

        mean_actions, _ = self.policy(obs_as_tensor(s["states"], self.device), deterministic=True)
        vjp_vectors = th.as_tensor(coefficients, dtype=mean_actions.dtype, device=self.device)[:, None] * s["q_u"]
        return (mean_actions * vjp_vectors).sum(-1).mean()

    def train(self) -> None:
        if self._sample is None:
            raise RuntimeError("[ZDPG] collect_rollouts() must run before train()")
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        objective = self._objective()
        loss = -objective

        self.policy.optimizer.zero_grad()
        loss.backward()
        gradient_norm = th.norm(
            th.stack([p.grad.norm() for p in self.policy.parameters() if p.grad is not None])
        ).item()
        if self.max_grad_norm is not None:
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

        s = self._sample
        valid = s["valid"]
        self._n_updates += 1
        # Not excluded from tensorboard (unlike most "info" fields): this needs to be a
        # real synced metric so wandb can plot other train/* curves against it as a custom
        # x-axis (parameter updates rather than env timesteps).
        self.logger.record("train/n_updates", self._n_updates)
        self.logger.record("train/policy_loss", loss.item())
        self.logger.record("train/objective", objective.item())
        self.logger.record("train/n_valid_trajectories", int(valid.sum()))
        if self._resolved_horizon_mode == "finite" and self.mode == "standard":
            # The reference rollouts are complete H-step trajectories of the nominal policy
            # (in symmetric mode they stop at the branch state, so there is no such return).
            ref_returns = [
                self.rollout_buffer._returns[i][0]
                for i in range(self.n_envs) if self.rollout_buffer._traj_lengths[i] > 0
            ]
            self.logger.record("train/mean_return", float(np.mean(ref_returns)))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/gradient_norm", gradient_norm)
        if valid.any():
            self.logger.record("train/mean_q_plus", float(s["q_plus"][valid].mean()))
            self.logger.record("train/mean_q_minus", float(s["q_minus"][valid].mean()))
            self.logger.record("train/mean_q_difference", float((s["q_plus"] - s["q_minus"])[valid].mean()))
        self.logger.record("train/mean_t", float(s["t"].mean()))
        self.logger.record("train/truncated_fraction", float(s["truncated"].mean()))
        self.logger.record("train/sigma", self.sigma)
        self.logger.record("train/horizon_mode", self._resolved_horizon_mode, exclude="tensorboard")

        self._sample = None

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + [
            "_q_env", "_branch_buffer", "_sample", "_episode_seeds", "_step_caps",
        ]

    def learn(
        self: SelfZDPG,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "ZDPG",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfZDPG:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
