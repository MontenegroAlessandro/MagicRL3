"""Zeroth-order Deterministic Policy Gradient (ZDPG), Kumar et al. 2020.

Zeroth-order Deterministic Policy Gradient: https://arxiv.org/abs/2006.07314
Algorithm 1 (ZDPG), Algorithm 2 (Q-function sampler) and Algorithm 3 (discounted
state sampler) of that paper.
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
1.  standard: ZDPG, [Q(s, a + mu * u) - Q(s, a)] / mu.
2.  symmetric: ZDPG-S, [Q(s, a + mu * u) - Q(s, a - mu * u)] / (2 * mu).
"""

STATE_MATCH_ATOL = 1e-5
"""
Tolerance used when checking that a Q-pool sub-env replayed its reference's state
sampler trajectory exactly. Replaying the same reset seed with the same deterministic
policy has to reproduce the very same states, so any real mismatch is an environment
whose randomness is not determined by its reset seed, not a numerical artifact.
"""


class ZDPG(TrajectoryOnPolicyAlgorithm):
    """
    Zeroth-order Deterministic Policy Gradient (Kumar et al., 2020), Algorithm 1.

    One iteration draws a random horizon T_Q ~ Geom(1 - gamma), a state
    s_t ~ (1 - gamma) rho^{pi_theta} (Algorithm 3) and an action-space perturbation
    u ~ N(0, I_p), estimates the Q-function at the perturbed and at the nominal
    initial action with N random-horizon rollout pairs (Algorithm 2), and ascends

        g_t = 1 / (1 - gamma) * grad_theta pi_theta(s_t) * (Q+ - Q-) / mu * u.

    The perturbation lives in the p-dimensional action space, never in the
    d-dimensional parameter space, and no critic is involved: the Q-values come
    straight from truncated rollouts of the deterministic policy.

    Batching: the repo's n_envs parallel environments provide n_envs independent
    (s_t, u) draws per iteration, whose quasi-gradients are averaged (the mini-batch
    setting of Theorem 2). T_Q is drawn once per iteration and shared across the
    batch, exactly as Algorithm 1 uses a single T_Q for both samplers.

    Q-function sampling requires restarting the system at s_t with a chosen initial
    action, which the paper obtains from a simulator that can reset and reproduce the
    same stochastic environment (Section 3). Here that is done by replaying: every
    sub-env of the side pool resets with its reference's own reset seed and replays
    the T_Q deterministic policy steps of the state sampler, which lands it exactly on
    s_t. The replayed states are verified against the reference ones, so an
    environment whose randomness does not follow from its reset seed fails loudly
    instead of silently biasing the estimator.

    :param mu: Smoothing parameter (mu > 0), the action-perturbation radius.
    :param n_rollouts: N, the Monte-Carlo rollout pairs averaged per state (N = 1 is
        no variance reduction). Only pays off in environments with randomness beyond
        the initial state, since the pairs are otherwise identical by construction.
    :param mode: "standard" (ZDPG) or "symmetric" (ZDPG-S), see MODES.
    :param use_crn: Give the two rollouts of a pair the same post-branch environment
        noise (common random numbers). The pair already starts from the very same
        state; this extends the sharing to the rest of the rollout.
    :param env_id: Gym id used to build the Q-sampling pool of 2 * N * n_envs sub-envs.
    :param env_kwargs: Extra kwargs for that pool's environments; must match the ones
        of the training env, otherwise the replayed states cannot agree.
    :param gamma: Must be in (0, 1): it is the parameter of the geometric horizons
        that make both samplers unbiased, so gamma = 1 has no meaning here.

    All other arguments follow TrajectoryOnPolicyAlgorithm. Actions are always
    deterministic and log_std is frozen, as the policy is a deterministic map.
    One iteration costs at most n_envs * (1 + 4 * N) * T_Q + 2 * N * n_envs
    transitions -- the state sampler, then each pool sub-env's replay and its own
    T_Q + 1 Q-rollout steps -- so a complete update may overshoot learn()'s
    timestep budget.
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
        mu: float = 0.05,
        n_rollouts: int = 1,
        mode: str = "standard",
        use_crn: bool = True,
        env_id: str = None,
        env_kwargs: Optional[dict[str, Any]] = None,
        max_grad_norm: Optional[float] = None,
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
        if not np.isfinite(mu) or mu <= 0:
            raise ValueError("[ZDPG] mu must be finite and strictly positive")
        if mode not in MODES:
            raise ValueError(f"[ZDPG] mode must be one of {MODES}, but got {mode}")
        if not isinstance(n_rollouts, int) or isinstance(n_rollouts, bool) or n_rollouts < 1:
            raise ValueError("[ZDPG] n_rollouts must be a positive integer")
        if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 1:
            raise ValueError("[ZDPG] n_steps must be a positive integer")
        if not np.isfinite(gamma) or not 0 < gamma < 1:
            # Both samplers draw their horizon from Geom(1 - gamma): gamma = 1 gives an
            # almost surely infinite horizon and gamma = 0 a degenerate one-step problem.
            raise ValueError("[ZDPG] gamma must be strictly between 0 and 1")
        if use_sde:
            raise ValueError("[ZDPG] the policy is deterministic; use_sde must be False")

        policy_kwargs = dict(policy_kwargs or {})
        # The policy is a deterministic map s -> a; there is no action distribution to learn.
        policy_kwargs["learn_std"] = False
        # Algorithm 1's update is theta + alpha * g, i.e. plain stochastic gradient ascent.
        policy_kwargs.setdefault("optimizer_class", th.optim.SGD)
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

        self.mu = float(mu)
        self.n_rollouts = n_rollouts
        self.mode = mode
        self.use_crn = use_crn
        self.env_id = env_id
        self.env_kwargs = env_kwargs if env_kwargs is not None else {}
        self.perturbed_env_wrapper_class = perturbed_env_wrapper_class
        self.perturbed_env_wrapper_kwargs = perturbed_env_wrapper_kwargs or {}
        self._n_updates = 0
        self._n_iterations = 0

        self.name = "ZDPG" if mode == "standard" else "ZDPG-S"

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # Dedicated RNG streams for the three stochastic ingredients of the estimator
        # (random horizons, env seeding, action perturbations), all spawned from
        # self.seed. Deliberately NOT drawn from the global numpy/torch streams, so that
        # re-running with the same seed stays bit-identical by construction rather than
        # by accident of what else happens to consume those streams.
        horizon_seed, episode_seed, perturbation_seed = np.random.SeedSequence(self.seed).spawn(3)
        self._horizon_rng = np.random.default_rng(horizon_seed)
        self._episode_seed_rng = np.random.default_rng(episode_seed)
        self._perturbation_rng = np.random.default_rng(perturbation_seed)

        # Checked here rather than in __init__ so that load() -- which rebuilds the model
        # with default arguments before restoring the saved ones -- still works.
        if self.env_id is None:
            raise ValueError("[ZDPG] env_id must be provided to build the Q-sampling pool")

        # Sub-env (i * N + n) * 2 + b estimates Q for reference state i, rollout pair n,
        # branch b (0 = perturbed initial action, 1 = nominal one).
        self._group_width = 2 * self.n_rollouts
        self._q_env = make_vec_env(
            self.env_id,
            n_envs=self.n_envs * self._group_width,
            env_kwargs=self.env_kwargs,
            vec_env_cls=DummyVecEnv,
            wrapper_class=self.perturbed_env_wrapper_class,
            wrapper_kwargs=self.perturbed_env_wrapper_kwargs,
        )
        # NOTE: the pool is not seeded here; every iteration reseeds it to replay the
        # reset seeds of the state sampler (see _collect_q_estimates).
        self._can_reseed_noise = True

        self._action_low = th.as_tensor(self.action_space.low, device=self.device)
        self._action_high = th.as_tensor(self.action_space.high, device=self.device)

        self._episode_seeds = None
        self._states = None
        self._u = None
        self._q_hat = None
        self._q_difference = None
        self._valid = None
        self._horizon = 0
        self._horizon_truncated = False

        if self.env is not None:
            self._check_normalization(self.env)

    @staticmethod
    def _check_normalization(env: VecEnv) -> None:
        normalizer = unwrap_vec_normalize(env)
        if normalizer is not None:
            raise ValueError(
                "[ZDPG] the Q-sampling pool is built from raw environments, so a "
                "VecNormalize training env would compare normalized reference states "
                "and returns against unnormalized rollouts. Drop VecNormalize."
            )

    def _reset_env(self, env: VecEnv) -> np.ndarray:
        # Explicit seed drawn from our own dedicated RNG (see _setup_model) rather than
        # env.seed(None), which would silently fall back to the global numpy RNG stream.
        # The seeds are recorded because the Q-sampling pool has to replay them.
        base_seed = int(self._episode_seed_rng.integers(0, 2**31 - self.n_envs))
        self._episode_seeds = np.array(env.seed(base_seed))
        return env.reset()

    def _clip(self, actions: th.Tensor) -> th.Tensor:
        """Same action-bound handling as the base collector, applied to both branches so
        that the two Q-values of a pair always refer to actions the system can execute."""
        return th.max(th.min(actions, self._action_high), self._action_low)

    def _policy_actions(self, obs: np.ndarray) -> np.ndarray:
        """pi_theta(s), clipped: the action the deterministic policy plays at s."""
        with th.no_grad():
            actions, _ = self.policy(obs_as_tensor(obs, self.device), deterministic=True)
        return self._clip(actions).cpu().numpy()

    def _step_pool(self, actions: np.ndarray, active: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        One lockstep step of the Q-sampling pool. Rewards of sub-envs that already
        reached a terminal state are zeroed (absorbing convention) and their steps are
        not counted: only transitions that actually feed the estimator are charged to
        the training budget, as everywhere else in the repo.

        :return: (observations, masked rewards, updated active mask)
        """
        obs, rewards, dones, _ = self._q_env.step(actions)
        self.num_timesteps += int(active.sum())
        return obs, rewards * active, active & ~dones

    def _reseed_pool_noise(self) -> None:
        """
        Give every (reference state, rollout pair) its own post-branch environment noise.

        All sub-envs of a group replay one and the same reset seed to reach s_t, so
        without this the N Monte-Carlo repetitions would be bit-identical and averaging
        them would reduce no variance at all. Only the environment's RNG is replaced,
        never its state. Under use_crn the two branches of a pair keep the same stream,
        which is the "reproduction of the same stochastic environment" of Section 3;
        otherwise each branch gets an independent one.
        """
        if not self._can_reseed_noise:
            return
        for pair in range(self.n_envs * self.n_rollouts):
            plus_seed = int(self._episode_seed_rng.integers(0, 2**31 - 1))
            minus_seed = plus_seed if self.use_crn else int(self._episode_seed_rng.integers(0, 2**31 - 1))
            for branch, noise_seed in enumerate((plus_seed, minus_seed)):
                try:
                    self._q_env.envs[2 * pair + branch].unwrapped.np_random = np.random.default_rng(noise_seed)
                except AttributeError:
                    warnings.warn(
                        "[ZDPG] the environment does not expose a Gymnasium np_random "
                        "generator, so the rollouts of a state cannot be decorrelated. "
                        "With deterministic dynamics this is harmless (the pairs are "
                        "exact anyway); otherwise use n_rollouts=1 and use_crn=True.",
                        UserWarning,
                    )
                    self._can_reseed_noise = False
                    return

    def _check_replayed_states(self, obs: np.ndarray, states: np.ndarray, active: np.ndarray) -> None:
        """Every still-active sub-env must sit exactly on its reference's s_t."""
        if not active.any():
            return
        expected = np.repeat(states, self._group_width, axis=0)[active]
        deviation = np.abs(np.asarray(obs)[active] - expected).max()
        if not np.isfinite(deviation) or deviation > STATE_MATCH_ATOL:
            raise ValueError(
                f"[ZDPG] replaying the state sampler in the Q-sampling pool landed "
                f"{deviation:.3e} away from the reference state (tolerance "
                f"{STATE_MATCH_ATOL:.0e}). Q-values must be evaluated at the very state "
                f"the gradient is taken at, so this environment cannot be used: its "
                f"randomness is not fully determined by the reset seed, or the pool's "
                f"env_id/env_kwargs/wrappers differ from the training env's."
            )

    def _collect_q_estimates(self, states: np.ndarray, u: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """
        Algorithm 2, run for every (reference state, rollout pair, branch) at once.

        Each sub-env first replays its reference's state sampler trajectory to reach
        s_t, then plays its branch's initial action a_0 and follows pi_theta for the
        remaining T_Q steps, accumulating the undiscounted reward sum
        R(s_0, a_0) + ... + R(s_T, a_T) -- an unbiased Q-estimate precisely because the
        horizon is geometric.

        :return: the raw Q-estimates, shape (n_envs, N, 2), the last axis being the
            perturbed and the nominal branch of each rollout pair.
        """
        env = self._q_env
        horizon, group = self._horizon, self._group_width

        # Sub-env j replays the reset seed of reference j // group, so the deterministic
        # steps below reproduce that reference's trajectory state by state.
        env._seeds = [int(self._episode_seeds[j // group]) for j in range(env.num_envs)]
        obs = env.reset()
        active = np.repeat(valid, group)

        # --- replay phase: land on s_t (no reward is collected, this is Algorithm 3) ---
        for _ in range(horizon):
            obs, _, active = self._step_pool(self._policy_actions(obs), active)
        if (np.repeat(valid, group) & ~active).any():
            raise ValueError(
                "[ZDPG] an episode ended while replaying the state sampler in the "
                "Q-sampling pool, although the reference trajectory did not. Check that "
                "the pool's env_id/env_kwargs/wrappers match the training env's."
            )
        self._check_replayed_states(obs, states, active)
        self._reseed_pool_noise()

        # --- initial actions: a_0 = pi_theta(s_t) + mu * u on the + branch ---
        with th.no_grad():
            mean_actions, _ = self.policy(obs_as_tensor(states, self.device), deterministic=True)
        perturbation = self.mu * th.as_tensor(u, dtype=mean_actions.dtype, device=self.device)
        plus_actions = self._clip(mean_actions + perturbation).cpu().numpy()
        if self.mode == "symmetric":
            minus_actions = self._clip(mean_actions - perturbation).cpu().numpy()
        else:
            minus_actions = self._clip(mean_actions).cpu().numpy()

        actions = np.empty((env.num_envs, get_action_dim(self.action_space)), dtype=plus_actions.dtype)
        actions[0::2] = np.repeat(plus_actions, self.n_rollouts, axis=0)
        actions[1::2] = np.repeat(minus_actions, self.n_rollouts, axis=0)

        # --- Q phase: the perturbed first action, then T_Q steps of pi_theta ---
        q_hat = np.zeros(env.num_envs, dtype=np.float64)
        for t in range(horizon + 1):
            obs, rewards, active = self._step_pool(actions, active)
            q_hat += rewards
            if not active.any():
                break
            if t < horizon:
                actions = self._policy_actions(obs)

        return q_hat.reshape(self.n_envs, self.n_rollouts, 2)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: TrajectoryBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Draw this iteration's random horizon, the batch of discounted-distribution states
        (Algorithm 3, run in the training env through the base collector so that the
        episode statistics, callbacks and timestep accounting stay the repo's), the
        action perturbations, and the Q-estimates of both branches (Algorithm 2).
        """
        self._check_normalization(env)
        self._q_difference = None

        # T_Q ~ Geom(1 - gamma) on {0, 1, ...}: numpy counts trials until the first
        # success, so the number of failures is one less. Capped at the repo's horizon.
        raw_horizon = int(self._horizon_rng.geometric(1.0 - self.gamma)) - 1
        self._horizon = min(raw_horizon, n_rollout_steps)
        self._horizon_truncated = raw_horizon > n_rollout_steps

        if not super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps=self._horizon):
            return False

        # The base collector leaves _last_obs on the state reached after T_Q steps,
        # which is exactly Algorithm 3's output s_T.
        states = np.array(self._last_obs, copy=True)
        if self._horizon == 0:
            valid = np.ones(env.num_envs, dtype=bool)
        else:
            # A sub-env whose episode ended before completing the T_Q steps was auto-reset
            # by the VecEnv, so its observation belongs to a fresh episode rather than to
            # the discounted state distribution: it sits out this iteration. Terminating
            # exactly on the last step is caught by _last_episode_starts.
            completed = rollout_buffer._traj_lengths == self._horizon
            valid = completed & ~np.asarray(self._last_episode_starts, dtype=bool)

        u = self._perturbation_rng.standard_normal((env.num_envs, get_action_dim(self.action_space)))

        if valid.any():
            q_hat = self._collect_q_estimates(states, u, valid)
        else:
            q_hat = np.zeros((env.num_envs, self.n_rollouts, 2))

        self._states = states
        self._u = u
        self._valid = valid
        self._q_hat = q_hat
        # The N rollout pairs are the Monte-Carlo variance reduction of Algorithm 1.
        q_plus, q_minus = q_hat[:, :, 0].mean(axis=1), q_hat[:, :, 1].mean(axis=1)
        self._q_plus, self._q_minus = q_plus, q_minus
        # Lemma 1's two representations only differ by which action the second rollout
        # starts from, and by the width of the difference quotient.
        denominator = 2 * self.mu if self.mode == "symmetric" else self.mu
        self._q_difference = (q_plus - q_minus) / denominator
        return True

    def train(self) -> None:
        if self._q_difference is None:
            raise RuntimeError("[ZDPG] collect_rollouts() must run before train()")

        self._n_iterations += 1
        valid = self._valid
        n_valid = int(valid.sum())

        if n_valid > 0:
            # Counted only here: an iteration whose states were all discarded applies no
            # optimizer step, so counting it would stretch the x-axis with empty ticks.
            self._n_updates += 1
            self.policy.set_training_mode(True)
            self._update_learning_rate(self.policy.optimizer)

            # grad_theta <pi_theta(s_t), (Q+ - Q-) / mu * u> is exactly Algorithm 1's
            # Psi_t * (Q+ - Q-) / mu * u, so the quasi-gradient is obtained from the
            # repo's usual backward()/optimizer.step() path. Only pi_theta carries a
            # gradient: the states, the perturbations and the Q-values are data.
            mean_actions, _ = self.policy(
                obs_as_tensor(self._states[valid], self.device), deterministic=True
            )
            weights = th.as_tensor(
                self._q_difference[valid, None] * self._u[valid],
                dtype=mean_actions.dtype,
                device=self.device,
            )
            objective = (mean_actions * weights).sum(-1).mean() / (1.0 - self.gamma)
            loss = -objective

            self.policy.optimizer.zero_grad()
            loss.backward()
            gradient_norm = th.norm(
                th.stack([p.grad.norm() for p in self.policy.parameters() if p.grad is not None])
            ).item()
            if self.max_grad_norm is not None:
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            self.logger.record("train/policy_loss", loss.item())
            self.logger.record("train/objective", objective.item())
            self.logger.record("train/gradient_norm", gradient_norm)
            self.logger.record("train/mean_q_plus", float(self._q_plus[valid].mean()))
            self.logger.record("train/mean_q_minus", float(self._q_minus[valid].mean()))
            self.logger.record("train/mean_q_difference", float(self._q_difference[valid].mean()))
            # Spread of the N rollouts of a branch: zero means the repetitions carried no
            # new randomness, so n_rollouts > 1 is buying no variance reduction at all.
            self.logger.record("train/q_rollout_std", float(self._q_hat[valid].std(axis=1).mean()))

        # Not excluded from tensorboard (unlike most "info" fields): this needs to be a
        # real synced metric so wandb can plot other train/* curves against it as a custom
        # x-axis (parameter updates rather than env timesteps).
        self.logger.record("train/n_updates", self._n_updates)
        self.logger.record("train/n_iterations", self._n_iterations)
        self.logger.record("train/n_valid_states", n_valid)
        self.logger.record("train/horizon", self._horizon)
        self.logger.record("train/horizon_truncated", float(self._horizon_truncated))
        self.logger.record("train/mu", self.mu)
        self.logger.record("train/n_rollouts", self.n_rollouts)

        self._q_difference = None

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + [
            "_q_env", "_states", "_u", "_valid", "_q_hat", "_q_plus", "_q_minus",
            "_q_difference", "_episode_seeds",
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
