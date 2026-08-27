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
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
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
        # step mode needs one sub-env per candidate perturbation timestep;
        # trajectory mode perturbs a single trajectory end-to-end.
        self.perturbed_width = self.n_steps if mode == "step" else 1

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

        self._perturbed_env = DummyVecEnv([
            lambda: gym.make(self.env_id, **self.env_kwargs) for _ in range(self.perturbed_width)
        ])
        self._perturbed_buffer = self.rollout_buffer_class(
            self.n_steps, self.observation_space, self.action_space,
            device=self.device, gamma=self.gamma, n_envs=self.perturbed_width,
        )

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

    def _collect_perturbed_rollout(
        self, traj_seed: int, u_i: th.Tensor, q_u_i: th.Tensor
    ) -> TrajectoryBuffer:
        """
        Detached rollout for ONE reference trajectory: replays it under CRN and injects the
        perturbation(s) specified by self.mode.
        - mode == "trajectory": width == 1. The single sub-env is perturbed at EVERY
                                timestep t with sigma * u_i[t].
        - mode == "step":       width == self.n_steps. Sub-env i is perturbed ONLY at its
                                own timestep t == i -- zero before i, zero after i -- and
                                plays mu_theta(s) exactly like the reference everywhere else.
        Returns the detached buffer with compute_returns() already called.
        """
        env = self._perturbed_env
        width = env.num_envs

        if self.mode == "trajectory":
            assert width == 1, f"[FDPG] trajectory mode needs a width-1 perturbed env, got {width}"
        elif self.mode == "step":
            assert width == self.n_steps, (
                f"[FDPG] step mode needs a width-{self.n_steps} perturbed env "
                f"(one sub-env per candidate perturbation timestep), got {width}"
            )

        self._perturbed_buffer.reset()

        env.seed(traj_seed)
        if hasattr(env, "_seeds"):
            env._seeds = [traj_seed] * width  # CRN: identical seed across all sub-envs
        obs = env.reset()

        low = th.as_tensor(self.action_space.low, device=self.device)
        high = th.as_tensor(self.action_space.high, device=self.device)
        active = np.ones(width, dtype=bool)

        for t in range(self.n_steps):
            obs_tensor = obs_as_tensor(obs, self.device)
            with th.no_grad():
                mu, _ = self.policy(obs_tensor, deterministic=True)

            if self.mode == "trajectory":
                # applied always -- every step of the single perturbed trajectory
                perturbation = self.sigma * u_i[t].unsqueeze(0)  # (1, action_dim)
            else:
                # applied ONLY at sub-env t's own timestep t -- zero before AND after.
                perturbation = th.zeros_like(mu)
                perturbation[t] = self.sigma * u_i[t]

            clipped = th.max(th.min(mu + perturbation, high), low).cpu().numpy()
            new_obs, rewards, dones, infos = env.step(clipped)

            active_indices = np.where(active)[0]
            self._perturbed_buffer.add(
                obs[active_indices], clipped[active_indices], rewards[active_indices],
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
        deterministic policy), then, for each one, collect its perturbed counterpart(s)
        under CRN and immediately extract the (state, return, q(u)) terms train() needs
        for the g-side of the g - b estimator. The b-side is read directly out of
        `self.rollout_buffer` inside train() since it stays valid until the next call.
        """
        continue_training = super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
        if not continue_training:
            return False

        self._u, self._q_u = self._sample_perturbations(self.batch_size, self.n_steps)

        self._g_terms: list[Optional[tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]]] = []
        for i in range(self.batch_size):
            traj_seed = int(self._episode_seeds[i])
            perturbed_buffer = self._collect_perturbed_rollout(traj_seed, self._u[i], self._q_u[i])
            if self.mode == "trajectory":
                terms = self._extract_terms(perturbed_buffer, 0, self._q_u[i])
            else:
                terms = self._extract_diagonal_terms(perturbed_buffer, self._q_u[i])
            self._g_terms.append(terms)

        return True

    def _extract_terms(
        self, buffer: TrajectoryBuffer, env_idx: int, q_u_i: th.Tensor
    ) -> Optional[tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]]:
        """
        Pull one full trajectory (all t = 0, ..., L-1) out of `buffer` at `env_idx`, paired
        with the perturbation weights q(u_t) at those same timesteps. Used for the b-side
        (reference trajectory) and for the g-side in "trajectory" mode (single perturbed
        trajectory). Returns (states, returns_to_go, q_u, gamma_pow), or None if the
        trajectory is empty. Values are materialized (copied) so they stay valid even after
        `buffer` is reset/reused by a later call.
        """
        length = int(buffer._traj_lengths[env_idx])
        if length == 0:
            return None

        states = buffer.to_torch(np.array(buffer._obs[env_idx]))
        returns = buffer.to_torch(buffer._returns[env_idx].astype(np.float32, copy=False))
        t_idx = th.arange(length, dtype=th.long, device=self.device)
        return states, returns, q_u_i[t_idx], self.gamma ** t_idx.to(th.float32)

    def _extract_diagonal_terms(
        self, buffer: TrajectoryBuffer, q_u_i: th.Tensor
    ) -> Optional[tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]]:
        """
        "step" mode g-side: `buffer` holds one sub-trajectory per candidate perturbation
        timestep t (sub-env t == candidate t). For each t, pull sub-env t's OWN state and
        return-to-go at ITS local index t (the diagonal) -- i.e. tilde_s_t^t and
        R(tilde_tau^t_{t:}) from the PDF. Candidate t's whose episode ended before ever
        reaching local step t (buffer._traj_lengths[t] <= t) are dropped: the perturbation
        was never actually applied for them. Returns None if no candidate survived.
        """
        lengths = buffer._traj_lengths
        valid_t = [t for t in range(buffer.n_envs) if lengths[t] > t]
        if not valid_t:
            return None

        states = buffer.to_torch(np.stack([buffer._obs[t][t] for t in valid_t]))
        returns = buffer.to_torch(np.array([buffer._returns[t][t] for t in valid_t], dtype=np.float32))
        t_idx = th.tensor(valid_t, dtype=th.long, device=self.device)
        return states, returns, q_u_i[t_idx], self.gamma ** t_idx.to(th.float32)

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        objective_terms = []
        g_values, b_values = [], []

        for i in range(self.batch_size):
            b_terms = self._extract_terms(self.rollout_buffer, i, self._q_u[i])
            g_terms = self._g_terms[i]
            if b_terms is None or g_terms is None:
                continue

            b_states, b_returns, b_qu, b_gamma_pow = b_terms
            g_states, g_returns, g_qu, g_gamma_pow = g_terms

            # mu_theta(s) computed WITH grad -- deterministic=True selects the mean action
            # (no sampling noise), but the graph back to theta stays intact.
            mu_b, _ = self.policy(b_states, deterministic=True)
            mu_g, _ = self.policy(g_states, deterministic=True)

            # sum_t gamma^t R(.) * <mu_theta(s_t), q(u_t)> -- a scalar whose gradient w.r.t.
            # theta is exactly sum_t gamma^t R(.) * grad_theta mu_theta(s_t)^T q(u_t), i.e.
            # sigma * b (resp. sigma * g) via a single vector-Jacobian product per side.
            obj_b = (b_gamma_pow * b_returns * (mu_b * b_qu).sum(-1)).sum()
            obj_g = (g_gamma_pow * g_returns * (mu_g * g_qu).sum(-1)).sum()

            objective_terms.append(obj_g - obj_b)
            g_values.append(obj_g.item())
            b_values.append(obj_b.item())

        err_msg = "[FDPG] no valid (reference, perturbed) trajectory pair to train on this iteration"
        assert objective_terms, err_msg

        # mean over the batch of reference trajectories, then the 1/sigma factor shared by g and b
        objective = th.stack(objective_terms).mean() / self.sigma
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
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/policy_loss", loss.item())
        self.logger.record("train/objective", objective.item())
        self.logger.record("train/mean_g", float(np.mean(g_values)))
        self.logger.record("train/mean_b", float(np.mean(b_values)))
        self.logger.record("train/n_valid_trajectories", len(objective_terms))
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
