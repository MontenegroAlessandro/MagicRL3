from typing import Any, ClassVar, Optional, TypeVar, Union

import torch as th
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.env_util import make_vec_env

from algorithms.trajectory_onpolicy_method import TrajectoryOnPolicyAlgorithm
from buffers import TrajectoryBuffer
from policies import ActorOnlyPolicy

SelfFDPG = TypeVar("SelfFDPG", bound="FDPG")

MODES = ("step", "trajectory")
"""
Different modes for the perturbation to apply.
1.  step: it means that the underlying deterministic policy is perturbed just in one step. The number of steps per train
    call is <= (n_steps + 1) * n_envs * n_steps (i.e., one full trajectory for the nominal deterministic policy, plus 
    one for each step).
2. trajectory: it means that the underlying deterministic policy is perturbed at every step. The number of steps per 
    train call is <= 2 * n_envs * n_steps (i.e., one full trajectory for the nominal deterministic policy, plus 
    one full trajectory for each perturbed policy).
"""

SAMPLING_MODES = ("normal", "sphere")
"""
Different modes for the perturbation to apply.
1.  normal: the noise is sampled from a normal distribution N(0_{d_{A}},I_{d_{A}}).
2.  sphere: the noise is sampled from a uniform distribution on the surface of a unit sphere S^{d_{A}-1}.
"""

SAMPLING_STRATEGIES = ("step", "trajectory")
"""
Different strategies for sampling the perturbations.
1.  step: the noise is sampled iid for each step and trajectory.
2.  trajectory: the noise is sampled to be applied to the entire trajectory.
"""

class FDPG(TrajectoryOnPolicyAlgorithm):
    """
    Finite Difference Policy Gradient (FDPG).
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
        mode: str = "step", # !
        sampling_mode: str = "normal", # !
        sampling_strategy: str = "step", # !
        env_id: str = None,
        env_kwargs: Optional[dict[str, Any]] = None,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[TrajectoryBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        use_crn: bool = False,
        perturbed_env_wrapper_class=None,   # <-- optional, see note below
        perturbed_env_wrapper_kwargs=None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            max_grad_norm=max_grad_norm,
            collect_deterministic_rollouts=True, # force the sampler to use the deterministic policy 
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(spaces.Box),
        )

        self.sigma = sigma
        self._n_updates = 0

        err_msg = f"[FDPG] mode must be one of {MODES}, but got {mode}"
        assert mode in MODES, err_msg
        self.mode = mode

        err_msg = f"[FDPG] sampling_mode must be one of {SAMPLING_MODES}, but got {sampling_mode}"
        assert sampling_mode in SAMPLING_MODES, err_msg
        self.sampling_mode = sampling_mode

        err_msg = f"[FDPG] sampling_strategy must be one of {SAMPLING_STRATEGIES}, but got {sampling_strategy}"
        assert sampling_strategy in SAMPLING_STRATEGIES, err_msg
        self.sampling_strategy = sampling_strategy

        err_msg = f"[FDPG] batch_size must be at least 1, but got {batch_size}"
        assert batch_size >= 1, err_msg
        self.batch_size = batch_size

        # env info
        err_msg = "[FDPG] env_id must be provided to build the perturbed-env pool"
        assert env_id is not None, err_msg
        self.env_id = env_id
        self.env_kwargs = env_kwargs if env_kwargs is not None else {}
        # step mode needs one sub-env per (reference trajectory, candidate perturbation
        # timestep) pair, so every trajectory's candidates can be collected in one batched
        # rollout; trajectory mode needs one sub-env per reference trajectory.
        self.perturbed_width = self.n_steps * self.batch_size if mode == "step" else self.batch_size


        self.use_crn = use_crn
        """
        If True: perturbed rollouts replay the SAME environment randomness as their
        reference trajectory (common random numbers) -- lower variance, at the cost
        of forcing identical seeds across the perturbed sub-envs. If False (default):
        each perturbed rollout gets its own independently-drawn seed. The estimator
        stays unbiased either way -- CRN only removes environment-noise variance
        from the g - b comparison, it isn't required for correctness.
        """
        self.perturbed_env_wrapper_class = perturbed_env_wrapper_class
        self.perturbed_env_wrapper_kwargs = perturbed_env_wrapper_kwargs or {}

        if _init_setup_model:
            self._setup_model()

        # set up the method's name for logging purposes
        self.name = "FDPG"
        if mode == "step":
            self.name = "Step-" + self.name
        elif mode == "trajectory":
            self.name = "Trajectory-" + self.name
        if sampling_mode == "normal":
            self.name = self.name + "-normal"
        elif sampling_mode == "sphere":
            self.name = self.name + "-sphere"
        if sampling_strategy == "step":
            self.name = self.name + "-step"
        elif sampling_strategy == "trajectory":
            self.name = self.name + "-trajectory"

    def _setup_model(self) -> None:
        super()._setup_model()

        err_msg = (
            f"[FDPG] env.num_envs must equal batch_size: got {self.n_envs} envs "
            f"but batch_size={self.batch_size}. Build the training env with "
            f"n_envs=batch_size (one sub-env per reference trajectory)."
        )
        assert self.n_envs == self.batch_size, err_msg

        # Dedicated RNG streams for the two stochastic ingredients of the estimator
        # (per-iteration env reseeding, perturbation sampling), seeded once from
        # self.seed. Deliberately NOT drawing from the global numpy/torch RNG streams
        # for these: this decouples reproducibility from exactly what else happens to
        # consume those global streams elsewhere in the process (e.g. gSDE noise,
        # future code changes), so re-running with the same seed stays bit-identical
        # by construction rather than "by accident of call order".
        self._episode_seed_rng = np.random.default_rng(self.seed)
        self._perturbation_rng = None
        if self.seed is not None:
            self._perturbation_rng = th.Generator(device=self.device)
            self._perturbation_rng.manual_seed(self.seed)

        self._perturbed_env = make_vec_env(
            self.env_id,
            n_envs=self.perturbed_width,
            env_kwargs=self.env_kwargs,
            vec_env_cls=DummyVecEnv,
            wrapper_class=self.perturbed_env_wrapper_class,
            wrapper_kwargs=self.perturbed_env_wrapper_kwargs,
        )
        self._perturbed_buffer = self.rollout_buffer_class(
            self.n_steps, self.observation_space, self.action_space,
            device=self.device, gamma=self.gamma, n_envs=self.perturbed_width,
        )
        # NOTE: the perturbed env is not seeded since it will be seeded when collecting the perturbed rollout.

    def _sample_perturbations(self, n_trajectories: int, horizon: int) -> tuple[th.Tensor, th.Tensor]:
        """
        Draw the perturbation vectors u_0, ..., u_{H-1} (and their associated q(u_t)) needed
        for one train() call, batched over all n_trajectories reference trajectories at once.

        Returns (u, q_u), each of shape (n_trajectories, horizon, action_dim). Row i is drawn
        independently of row j (i != j) -- every reference trajectory gets its own perturbation
        draw(s). Within one row, `self.sampling_strategy` controls the H entries along the
        horizon axis:
        - "step":       iid draws, one per (trajectory, timestep).
        - "trajectory": a single draw per trajectory, broadcast across the whole horizon
                        via a zero-stride view (`.expand`) -- NOT materialized H times, so
                        this costs n_trajectories * action_dim samples, not
                        n_trajectories * horizon * action_dim.
        Only the first L_i entries along the horizon axis will actually get used for trajectory
        i, where L_i <= horizon is that trajectory's real collected length; slicing is the
        caller's responsibility.
        """
        action_dim = get_action_dim(self.action_space)
        draw_horizon = horizon if self.sampling_strategy == "step" else 1

        u = th.randn(n_trajectories, draw_horizon, action_dim, device=self.device, generator=self._perturbation_rng)
        # we sample from a N(0,1) n_trajectories elements. in each element there are draw_horizon vectors of
        # dimension action_dim

        if self.sampling_mode == "normal":
            q_u = u
        elif self.sampling_mode == "sphere":
            u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            # computes the norm of each action perturbation
            q_u = action_dim * u
        else:
            raise ValueError(f"[FDPG] sampling_mode must be one of {SAMPLING_MODES}, but got {self.sampling_mode}")

        if self.sampling_strategy == "trajectory":
            # just replicate the perturbation for the horizon
            u = u.expand(n_trajectories, horizon, action_dim)
            q_u = q_u.expand(n_trajectories, horizon, action_dim)

        return u, q_u

    def _reset_env(self, env: VecEnv) -> np.ndarray:
        # Explicit seed drawn from our own dedicated RNG (see _setup_model) rather than
        # env.seed(None), which would silently fall back to the global numpy RNG stream.
        base_seed = int(self._episode_seed_rng.integers(0, 2**31 - 1))
        self._episode_seeds = np.array(env.seed(base_seed))  # records one seed per env
        return env.reset()

    def _collect_perturbed_rollouts_batch(self, u: th.Tensor, q_u: th.Tensor) -> TrajectoryBuffer:
        """
        Trajectory mode only: one sub-env per reference trajectory, all `batch_size`
        perturbed rollouts collected in lockstep -- replaces `batch_size` sequential
        single-env calls with one call that batches the policy forward pass.
        """
        env = self._perturbed_env
        width = env.num_envs
        assert width == self.batch_size, (
            f"[FDPG] trajectory mode's batched collector needs a width-{self.batch_size} "
            f"perturbed env pool, got {width}"
        )

        self._perturbed_buffer.reset()

        if self.use_crn:
            env._seeds = [int(s) for s in self._episode_seeds]   # sub-env i replays reference trajectory i's randomness
        else:
            env.seed(int(self._episode_seed_rng.integers(0, 2**31 - 1)))  # auto-increment -> fresh, distinct per sub-env
        obs = env.reset()

        low = th.as_tensor(self.action_space.low, device=self.device)
        high = th.as_tensor(self.action_space.high, device=self.device)
        active = np.ones(width, dtype=bool)

        for t in range(self.n_steps):
            obs_tensor = obs_as_tensor(obs, self.device)
            with th.no_grad():
                mu, _ = self.policy(obs_tensor, deterministic=True)   # ONE batched forward pass, all `width` trajectories

            perturbation = self.sigma * u[:, t, :]   # (width, action_dim) -- trajectory i's own u, at step t
            clipped = th.max(th.min(mu + perturbation, high), low).cpu().numpy()
            new_obs, rewards, dones, infos = env.step(clipped)

            active_indices = np.where(active)[0]
            self._perturbed_buffer.add(
                obs[active_indices], clipped[active_indices], rewards[active_indices],
                np.zeros(len(active_indices)), env_indices=active_indices,
            )
            active &= ~dones
            obs = new_obs
            if not active.any():
                break

        self._perturbed_buffer.compute_returns()
        self.num_timesteps += int(self._perturbed_buffer._traj_lengths.sum())
        return self._perturbed_buffer

    def _collect_perturbed_rollouts_batch_step(self, u: th.Tensor, q_u: th.Tensor) -> TrajectoryBuffer:
        """
        Step mode, batched: one sub-env per (reference trajectory, candidate perturbation
        timestep) pair -- width = batch_size * n_steps, sub-env `i * n_steps + t` being
        trajectory i's candidate t. Sub-env `i * n_steps + t` is perturbed ONLY at its own
        local timestep t (zero before AND after) and plays mu_theta(s) exactly like the
        reference everywhere else -- replaces `batch_size` sequential width-n_steps rollouts
        with one call that batches the policy forward pass across all of them at once.
        """
        env = self._perturbed_env
        width = env.num_envs
        assert width == self.batch_size * self.n_steps, (
            f"[FDPG] step mode's batched collector needs a width-{self.batch_size * self.n_steps} "
            f"perturbed env pool (one sub-env per (trajectory, candidate timestep) pair), got {width}"
        )

        self._perturbed_buffer.reset()

        if self.use_crn:
            # sub-env i*n_steps + t replays reference trajectory i's own randomness, for every t
            env._seeds = [int(s) for s in np.repeat(self._episode_seeds, self.n_steps)]
        else:
            # one fresh base seed per trajectory i, auto-incremented across its n_steps
            # candidates -- matches what a bare env.seed(base_seed) call assigns to a
            # width-n_steps pool, just drawn once per trajectory instead of once per call.
            base_seeds = self._episode_seed_rng.integers(0, 2**31 - 1, size=self.batch_size)
            env._seeds = [
                int(base_seeds[i]) + t for i in range(self.batch_size) for t in range(self.n_steps)
            ]
        obs = env.reset()

        low = th.as_tensor(self.action_space.low, device=self.device)
        high = th.as_tensor(self.action_space.high, device=self.device)
        active = np.ones(width, dtype=bool)

        # sub-env i*n_steps + c is perturbed only once the shared loop time t reaches c.
        candidate_offsets = th.arange(self.batch_size, device=self.device) * self.n_steps

        for t in range(self.n_steps):
            obs_tensor = obs_as_tensor(obs, self.device)
            with th.no_grad():
                mu, _ = self.policy(obs_tensor, deterministic=True)   # ONE batched forward pass, all `width` sub-envs

            perturbation = th.zeros_like(mu)
            perturbed_idx = candidate_offsets + t   # sub-env i*n_steps + t, for every trajectory i
            perturbation[perturbed_idx] = self.sigma * u[:, t, :]

            clipped = th.max(th.min(mu + perturbation, high), low).cpu().numpy()
            new_obs, rewards, dones, infos = env.step(clipped)

            active_indices = np.where(active)[0]
            self._perturbed_buffer.add(
                obs[active_indices],
                clipped[active_indices],  # not a problem, the action will not be used for gradient computation
                rewards[active_indices],
                np.zeros(len(active_indices)),
                env_indices=active_indices,
            )
            active &= ~dones
            obs = new_obs
            if not active.any():
                break

        self._perturbed_buffer.compute_returns()

        # perturbed-rollout interaction also has a real environment cost -- count it
        # towards the training budget even though it bypasses the main callback.
        self.num_timesteps += int(self._perturbed_buffer._traj_lengths.sum())

        return self._perturbed_buffer

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: TrajectoryBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect the batch_size reference trajectories (via the base class, using the
        deterministic policy), then their perturbed counterpart(s) in one batched call.
        Both the b-side (read directly out of `self.rollout_buffer`, which stays valid
        until the next call) and g-side extraction are left to train(), which batches
        them -- and the forward passes they feed -- across the whole batch at once instead
        of one trajectory (or candidate timestep) at a time.
        """
        continue_training = super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
        if not continue_training:
            return False

        self._u, self._q_u = self._sample_perturbations(self.batch_size, self.n_steps)

        if self.mode == "trajectory":
            self._collect_perturbed_rollouts_batch(self._u, self._q_u)
        else:
            self._collect_perturbed_rollouts_batch_step(self._u, self._q_u)

        return True

    def _extract_batched_terms(
        self, buffer: TrajectoryBuffer, q_u: th.Tensor, valid_idx: np.ndarray
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, int]:
        """
        b-side (both modes) and g-side ("trajectory" mode) extraction: pulls EVERY
        trajectory named by `valid_idx` out of `buffer` at once (all t = 0, ..., L-1 of
        each), concatenated into flat (total_steps, ...) tensors, plus `traj_idx` mapping
        each row back to its trajectory's position within `valid_idx` (0, ..., len(valid_idx)
        - 1). This lets the caller run ONE batched policy forward pass and ONE
        `scatter_add_` over all trajectories, instead of looping over them one at a time
        (each with its own forward pass).

        :param buffer: rollout_buffer (b-side, both modes) or self._perturbed_buffer
            (g-side, "trajectory" mode only -- "step" mode's g-side uses
            `_extract_diagonal_terms_batched` instead, since it needs one state per
            candidate timestep rather than the whole sub-trajectory).
        :param q_u: (batch_size, horizon, action_dim) perturbation weights for the whole batch.
        :param valid_idx: env indices to include, e.g. trajectories that are non-empty in
            BOTH the reference and perturbed buffers (a trajectory that's empty in either one
            has no (g, b) pair to contribute and must be excluded from both sides identically).
        :return: (states, returns_to_go, q_u, gamma_pow, traj_idx, n_valid).
        """
        lengths_np = buffer._traj_lengths[valid_idx]
        states = buffer.to_torch(np.concatenate([np.array(buffer._obs[i]) for i in valid_idx], axis=0))
        returns = buffer.to_torch(
            np.concatenate([buffer._returns[i] for i in valid_idx], axis=0).astype(np.float32, copy=False)
        )
        qu = th.cat([q_u[i, :length, :] for i, length in zip(valid_idx, lengths_np)], dim=0)

        n_valid = len(valid_idx)
        lengths = th.as_tensor(lengths_np, dtype=th.long, device=self.device)
        traj_idx = th.repeat_interleave(th.arange(n_valid, device=self.device), lengths)
        starts = th.cat([th.zeros(1, dtype=th.long, device=self.device), lengths.cumsum(0)[:-1]])
        local_t = th.arange(states.shape[0], device=self.device) - starts[traj_idx]
        gamma_pow = self.gamma ** local_t.to(th.float32)

        return states, returns, qu, gamma_pow, traj_idx, n_valid

    def _extract_diagonal_terms_batched(
        self, buffer: TrajectoryBuffer, q_u: th.Tensor, valid_idx: np.ndarray
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, int]:
        """
        "step" mode g-side: `buffer` holds one sub-trajectory per (reference trajectory,
        candidate perturbation timestep) pair -- sub-env `i * n_steps + t` is trajectory
        i's candidate t. For each i in `valid_idx` and each of its candidates t, pulls
        sub-env `i * n_steps + t`'s OWN state and return-to-go at ITS local index t (the
        diagonal) -- i.e. tilde_s_t^t and R(tilde_tau^t_{t:}). Candidates whose episode
        ended before ever reaching local step t are dropped: the perturbation was never
        actually applied for them. Concatenates every valid (i, t) pair across the whole
        batch into flat tensors, plus `traj_idx` mapping each row back to i's position
        within `valid_idx` -- so multiple candidate t's for the same i scatter-add into
        that single trajectory's g-term, exactly like `_extract_batched_terms` does for
        the b-side and for "trajectory" mode's g-side.

        :param buffer: self._perturbed_buffer, width == batch_size * n_steps.
        :param q_u: (batch_size, n_steps, action_dim) perturbation weights for the whole batch.
        :param valid_idx: reference-trajectory indices i to include -- each must have at
            least one valid candidate (the caller is responsible for checking this).
        :return: (states, returns_to_go, q_u, gamma_pow, traj_idx, n_valid).
        """
        lengths = buffer._traj_lengths.reshape(self.batch_size, self.n_steps)
        t_range = np.arange(self.n_steps)

        all_states, all_returns, all_qu, all_gamma_pow, all_traj_idx = [], [], [], [], []
        for pos, i in enumerate(valid_idx):
            valid_t = t_range[lengths[i] > t_range]
            sub_env_idx = i * self.n_steps + valid_t
            all_states.append(np.stack([buffer._obs[e][t] for e, t in zip(sub_env_idx, valid_t)]))
            all_returns.append(np.array([buffer._returns[e][t] for e, t in zip(sub_env_idx, valid_t)], dtype=np.float32))
            all_qu.append(q_u[i, valid_t, :])
            all_gamma_pow.append(self.gamma ** valid_t.astype(np.float32))
            all_traj_idx.append(np.full(len(valid_t), pos, dtype=np.int64))

        states = buffer.to_torch(np.concatenate(all_states, axis=0))
        returns = buffer.to_torch(np.concatenate(all_returns, axis=0))
        qu = th.cat(all_qu, dim=0)
        gamma_pow = buffer.to_torch(np.concatenate(all_gamma_pow, axis=0))
        traj_idx = th.as_tensor(np.concatenate(all_traj_idx, axis=0), dtype=th.long, device=self.device)

        return states, returns, qu, gamma_pow, traj_idx, len(valid_idx)

    def _objective_from_terms(
        self,
        b_terms: tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, int],
        g_terms: tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, int],
    ) -> tuple[int, th.Tensor, np.ndarray, np.ndarray]:
        """
        Shared by both modes: given each side's already-flattened (states, returns_to_go,
        q_u, gamma_pow, traj_idx, n_valid), runs exactly ONE forward pass per side and ONE
        `scatter_add_` per side to reduce to per-trajectory sums, then pairs them into the
        (g - b) objective. Mathematically identical to running the g/b computation
        trajectory-by-trajectory and stacking the results -- just without the Python loop
        around the (expensive) policy forward passes.
        """
        n_valid = b_terms[-1]
        b_states, b_returns, b_qu, b_gamma_pow, b_traj_idx, _ = b_terms
        g_states, g_returns, g_qu, g_gamma_pow, g_traj_idx, _ = g_terms

        # mu_theta(s) computed WITH grad -- deterministic=True selects the mean action
        # (no sampling noise), but the graph back to theta stays intact.
        mu_b, _ = self.policy(b_states, deterministic=True)
        mu_g, _ = self.policy(g_states, deterministic=True)

        # sum_t gamma^t R(.) * <mu_theta(s_t), q(u_t)> per trajectory, via scatter_add_
        # instead of a Python-level per-trajectory .sum().
        weighted_b = b_gamma_pow * b_returns * (mu_b * b_qu).sum(-1)
        weighted_g = g_gamma_pow * g_returns * (mu_g * g_qu).sum(-1)
        obj_b = th.zeros(n_valid, device=self.device).scatter_add_(0, b_traj_idx, weighted_b)
        obj_g = th.zeros(n_valid, device=self.device).scatter_add_(0, g_traj_idx, weighted_g)

        objective_terms = obj_g - obj_b
        return n_valid, objective_terms, obj_g.detach().cpu().numpy(), obj_b.detach().cpu().numpy()

    def _train_objective_batched(self) -> tuple[int, th.Tensor, np.ndarray, np.ndarray]:
        """
        "trajectory" mode: b-side and g-side are both whole sub-trajectories, so both are
        extracted with `_extract_batched_terms`.
        """
        b_lengths = self.rollout_buffer._traj_lengths
        g_lengths = self._perturbed_buffer._traj_lengths
        # A trajectory contributes only if BOTH its reference and perturbed rollout are
        # non-empty -- the (g_i, b_i) pair has to come from the same i on both sides.
        valid_idx = np.where((b_lengths > 0) & (g_lengths > 0))[0]
        if len(valid_idx) == 0:
            return 0, th.empty(0, device=self.device), np.empty(0), np.empty(0)

        b_terms = self._extract_batched_terms(self.rollout_buffer, self._q_u, valid_idx)
        g_terms = self._extract_batched_terms(self._perturbed_buffer, self._q_u, valid_idx)
        return self._objective_from_terms(b_terms, g_terms)

    def _train_objective_batched_step(self) -> tuple[int, th.Tensor, np.ndarray, np.ndarray]:
        """
        "step" mode: b-side is still a whole sub-trajectory (`_extract_batched_terms`), but
        the g-side only needs each candidate timestep's diagonal entry
        (`_extract_diagonal_terms_batched`) out of the width-(batch_size * n_steps)
        perturbed buffer.
        """
        b_lengths = self.rollout_buffer._traj_lengths
        g_lengths = self._perturbed_buffer._traj_lengths.reshape(self.batch_size, self.n_steps)
        # A trajectory contributes only if its reference rollout is non-empty AND at least
        # one of its candidate timesteps actually got perturbed before the episode ended.
        has_valid_candidate = (g_lengths > np.arange(self.n_steps)[None, :]).any(axis=1)
        valid_idx = np.where((b_lengths > 0) & has_valid_candidate)[0]
        if len(valid_idx) == 0:
            return 0, th.empty(0, device=self.device), np.empty(0), np.empty(0)

        b_terms = self._extract_batched_terms(self.rollout_buffer, self._q_u, valid_idx)
        g_terms = self._extract_diagonal_terms_batched(self._perturbed_buffer, self._q_u, valid_idx)
        return self._objective_from_terms(b_terms, g_terms)

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        if self.mode == "trajectory":
            n_valid, objective_terms, g_values, b_values = self._train_objective_batched()
        else:
            n_valid, objective_terms, g_values, b_values = self._train_objective_batched_step()

        err_msg = "[FDPG] no valid (reference, perturbed) trajectory pair to train on this iteration"
        assert n_valid > 0, err_msg

        # mean over the batch of reference trajectories, then the 1/sigma factor shared by g and b
        objective = objective_terms.mean() / self.sigma
        loss = -objective

        self.policy.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

        ref_returns = [
            self.rollout_buffer._returns[i][0]
            for i in range(self.batch_size) if self.rollout_buffer._traj_lengths[i] > 0
        ]

        self._n_updates += 1
        # Not excluded from tensorboard (unlike most "info" fields): this needs to be a
        # real synced metric so wandb can plot other train/* curves against it as a custom
        # x-axis (parameter updates rather than env timesteps).
        self.logger.record("train/n_updates", self._n_updates)
        self.logger.record("train/policy_loss", loss.item())
        self.logger.record("train/objective", objective.item())
        self.logger.record("train/mean_g", float(np.mean(g_values)))
        self.logger.record("train/mean_b", float(np.mean(b_values)))
        self.logger.record("train/n_valid_trajectories", n_valid)
        self.logger.record("train/mean_return", float(np.mean(ref_returns)))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

    def learn(
        self: SelfFDPG,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "FDPG",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfFDPG:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
