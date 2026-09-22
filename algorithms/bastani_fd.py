"""Parameter-space finite differences from Bastani (AISTATS 2020), Sections 3 and 6.

Sample Complexity of Estimating the Policy Gradient for Nearly Deterministic
Dynamical Systems: https://proceedings.mlr.press/v108/bastani20a.html
"""

from copy import deepcopy
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.vec_env import VecEnv, unwrap_vec_normalize

from algorithms.trajectory_onpolicy_method import TrajectoryOnPolicyAlgorithm
from buffers import TrajectoryBuffer
from policies import ActorOnlyPolicy

SelfBastaniFD = TypeVar("SelfBastaniFD", bound="BastaniFD")


class BastaniFD(TrajectoryOnPolicyAlgorithm):
    """Two-sided finite differences of the deterministic policy's total return.

    For each direction v, collect n_envs trajectories at theta + fd_step * v
    and another n_envs at theta - fd_step * v. Their mean returns give
    g_v = (mean(J_plus) - mean(J_minus)) / (2 * fd_step) * v.

    ``coordinate`` sums g_v over all parameter basis vectors (Section 3).
    ``sphere`` uses random unit vectors, as in Section 6, with the gradient
    normalization d_theta / n_directions. Since E[v v^T] = I / d_theta,
    merely averaging g_v would estimate a gradient scaled by 1 / d_theta.
    For finite fd_step this estimates the gradient of J smoothed over the
    parameter-space ball of that radius, rather than the exact gradient of J.

    Uses the existing actor, trajectory collector/buffer, learning loop,
    callbacks, clipping, and optimizer API. SGD is the default, as in the
    experiments; policy_kwargs may explicitly select another optimizer.

    :param fd_step: Positive parameter perturbation radius (lambda in the paper).
    :param mode: "coordinate" (Section 3) or "sphere" (Section 6).
    :param n_directions: Directions averaged in sphere mode; coordinate mode
        always evaluates every parameter and requires n_directions=1.
    :param use_crn: Replay reset seeds for each +/- pair (simulator variance
        reduction mentioned in Section 3). Requires envs that honor reset seeds
        for all their randomness. Default False uses independent samples.
    :param gamma: Defaults to 1, the paper's undiscounted finite-horizon return.
        Values below 1 extend the estimator to the repo's discounted objective.

    All other arguments follow TrajectoryOnPolicyAlgorithm. Actions are always
    deterministic, and log_std is frozen/excluded from the parameter search.
    One update costs at most 2 * n_envs * n_steps * number_of_directions
    transitions; a complete update may overshoot learn()'s timestep budget.
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
        gamma: float = 1.0,
        fd_step: float = 0.01,
        mode: str = "coordinate",
        n_directions: int = 1,
        use_crn: bool = False,
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
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        if not np.isfinite(fd_step) or fd_step <= 0:
            raise ValueError("fd_step must be finite and strictly positive")
        if mode not in ("coordinate", "sphere"):
            raise ValueError("mode must be 'coordinate' or 'sphere'")
        if not isinstance(n_directions, int) or isinstance(n_directions, bool) or n_directions < 1:
            raise ValueError("n_directions must be a positive integer")
        if mode == "coordinate" and n_directions != 1:
            raise ValueError("n_directions only applies to sphere mode; use 1 for coordinate mode")
        if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 1:
            raise ValueError("n_steps must be a positive integer")
        if not np.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        if use_sde:
            raise ValueError("BastaniFD uses deterministic actions; use_sde must be False")

        policy_kwargs = dict(policy_kwargs or {})
        policy_kwargs["learn_std"] = False
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
        self.fd_step = float(fd_step)
        self.mode = mode
        self.n_directions = n_directions
        self.use_crn = use_crn
        self._n_updates = 0

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        # Keep the nominal policy available to predict/evaluation/checkpoint
        # callbacks even while collecting a perturbed policy's trajectories.
        self._perturbed_policy = deepcopy(self.policy)
        if not self._search_parameters(self.policy):
            raise ValueError("BastaniFD requires trainable deterministic actor parameters")
        self._gradient_estimate = None
        self._rollout_seed = None
        # Preserve these streams on save/load; independent of callback/global RNGs.
        if not hasattr(self, "_episode_seed_rng"):
            episode_seed, direction_seed = np.random.SeedSequence(self.seed).spawn(2)
            self._episode_seed_rng = np.random.default_rng(episode_seed)
            self._direction_rng = np.random.default_rng(direction_seed)
        if self.env is not None:
            self._check_normalization(self.env)

    @staticmethod
    def _search_parameters(policy: ActorOnlyPolicy) -> list[th.nn.Parameter]:
        return [p for name, p in policy.named_parameters() if p.requires_grad and name != "log_std"]

    @staticmethod
    def _check_normalization(env: VecEnv) -> None:
        normalizer = unwrap_vec_normalize(env)
        if normalizer is not None and normalizer.training and (normalizer.norm_obs or normalizer.norm_reward):
            raise ValueError(
                "BastaniFD requires fixed observation/reward transforms across +/- rollouts. "
                "Disable VecNormalize or freeze its statistics with training=False."
            )

    def _get_rollout_policy(self) -> ActorOnlyPolicy:
        return self._perturbed_policy

    def _reset_env(self, env: VecEnv) -> np.ndarray:
        env.seed(self._rollout_seed)
        return env.reset()

    def _directions(self, theta: th.Tensor):
        if self.mode == "coordinate":
            # Stream basis vectors; never allocate a d_theta by d_theta matrix.
            for k in range(theta.numel()):
                direction = th.zeros_like(theta)
                direction[k] = 1
                yield direction
        else:
            for _ in range(self.n_directions):
                direction = th.as_tensor(
                    self._direction_rng.standard_normal(theta.numel()),
                    dtype=theta.dtype,
                    device=theta.device,
                )
                yield direction / direction.norm()

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: TrajectoryBuffer,
        n_rollout_steps: int,
    ) -> bool:
        self._check_normalization(env)
        self._gradient_estimate = None
        self._perturbed_policy.load_state_dict(self.policy.state_dict())
        parameters = self._search_parameters(self._perturbed_policy)
        theta = parameters_to_vector(self._search_parameters(self.policy)).detach().clone()
        gradient = th.zeros_like(theta)
        plus_returns, minus_returns = [], []

        try:
            for direction in self._directions(theta):
                pair_seed = int(self._episode_seed_rng.integers(0, 2**31 - env.num_envs))
                returns = []
                for sign in (1, -1):
                    self._rollout_seed = pair_seed
                    if sign == -1 and not self.use_crn:
                        self._rollout_seed = int(self._episode_seed_rng.integers(0, 2**31 - env.num_envs))
                    with th.no_grad():
                        vector_to_parameters(theta + sign * self.fd_step * direction, parameters)
                    # Reuse the collector, including action clipping, episode caps,
                    # termination masks, timestep accounting and all callbacks.
                    if not super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps):
                        return False
                    returns.append(float(np.mean([
                        rollout_buffer._returns[i][0] for i in range(env.num_envs)
                    ], dtype=np.float64)))
                plus_returns.append(returns[0])
                minus_returns.append(returns[1])
                gradient.add_(direction, alpha=(returns[0] - returns[1]) / (2 * self.fd_step))
        finally:
            # Also restore the working copy on early stop or environment errors.
            self._perturbed_policy.load_state_dict(self.policy.state_dict())
            self._rollout_seed = None

        if self.mode == "sphere":
            # E[v v^T] = I / d_theta for a uniform unit vector. Restore the
            # gradient scale after averaging the sampled directional derivatives.
            gradient.mul_(theta.numel() / self.n_directions)
        if not th.isfinite(gradient).all():
            raise ValueError("Non-finite BastaniFD gradient; check rewards and fd_step")
        self._gradient_estimate = gradient
        self._mean_return_plus = float(np.mean(plus_returns))
        self._mean_return_minus = float(np.mean(minus_returns))
        self._directions_evaluated = len(plus_returns)
        differences = np.asarray(plus_returns) - np.asarray(minus_returns)
        self._mean_abs_return_difference = float(np.abs(differences).mean())
        self._zero_difference_fraction = float(np.mean(differences == 0))
        return True

    def train(self) -> None:
        if self._gradient_estimate is None:
            raise RuntimeError("Collect a complete finite-difference estimate before train()")
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        parameters = self._search_parameters(self.policy)
        self.policy.optimizer.zero_grad(set_to_none=True)
        offset = 0
        for parameter in parameters:
            size = parameter.numel()
            # Optimizers minimize; negate the return gradient for ascent.
            parameter.grad = -self._gradient_estimate[offset:offset + size].reshape_as(parameter).clone()
            offset += size
        if self.max_grad_norm is not None:
            th.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
        self.policy.optimizer.step()

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates)
        self.logger.record("train/gradient_norm", self._gradient_estimate.norm().item())
        self.logger.record("train/mean_return_plus", self._mean_return_plus)
        self.logger.record("train/mean_return_minus", self._mean_return_minus)
        self.logger.record("train/mean_return", (self._mean_return_plus + self._mean_return_minus) / 2)
        self.logger.record("train/mean_abs_return_difference", self._mean_abs_return_difference)
        self.logger.record("train/zero_difference_fraction", self._zero_difference_fraction)
        self.logger.record("train/fd_step", self.fd_step)
        self.logger.record("train/n_directions", self._directions_evaluated)
        self.logger.record("train/n_parameters", offset)
        self._gradient_estimate = None

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["_perturbed_policy", "_gradient_estimate", "_rollout_seed"]

    def learn(
        self: SelfBastaniFD,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "BastaniFD",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfBastaniFD:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
