"""Parameter-space finite differences from Bastani (AISTATS 2020), Section 6.

Sample Complexity of Estimating the Policy Gradient for Nearly Deterministic
Dynamical Systems: https://proceedings.mlr.press/v108/bastani20a.html
"""

from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.func import functional_call, vmap
from torch.nn.utils import parameters_to_vector

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.vec_env import VecEnv, unwrap_vec_normalize

from algorithms.trajectory_onpolicy_method import TrajectoryOnPolicyAlgorithm
from buffers import TrajectoryBuffer
from policies import ActorOnlyPolicy

SelfBastaniFD = TypeVar("SelfBastaniFD", bound="BastaniFD")

SAMPLING_MODES = ("normal", "sphere")
"""
Distribution of the parameter-space direction nu (d_Theta = number of searched policy parameters).
1.  normal: nu ~ N(0_{d_Theta}, I_{d_Theta}), q(nu) = nu.
2.  sphere: nu ~ Unif(S^{d_Theta-1}), q(nu) = d_Theta * nu.
"""


class _MeanActor(th.nn.Module):
    """Deterministic action path of ActorOnlyPolicy.forward(deterministic=True), without
    the distribution object (whose argument validation is not vmap-compatible)."""

    def __init__(self, policy: ActorOnlyPolicy):
        super().__init__()
        self.policy = policy

    def forward(self, obs: th.Tensor) -> th.Tensor:
        features = self.policy.extract_features(obs, self.policy.features_extractor)
        return self.policy.action_net(self.policy.mlp_extractor.forward_actor(features))


class _PerEnvParameterPolicy:
    """
    Rollout-time view of the nominal policy in which sub-env i acts deterministically
    with its own parameter vector theta_i, i.e. a_t^i = mu_{theta_i}(s_t^i). The nominal
    parameters are never written to: theta_i is only bound for the forward pass through
    `functional_call`, and all sub-envs are evaluated in ONE batched pass via `vmap`.
    Exposes the subset of the policy interface used by the base trajectory collector.
    """

    def __init__(self, policy: ActorOnlyPolicy, names: list[str]):
        self.policy = policy
        self.mean_actor = _MeanActor(policy)
        self.names = names
        self.shapes = [p.shape for name, p in policy.named_parameters() if name in names]
        self._batched_params: Optional[dict[str, th.Tensor]] = None

    @property
    def squash_output(self) -> bool:
        return self.policy.squash_output

    def unscale_action(self, action: np.ndarray) -> np.ndarray:
        return self.policy.unscale_action(action)

    def set_training_mode(self, mode: bool) -> None:
        self.policy.set_training_mode(mode)

    def set_parameters(self, thetas: th.Tensor) -> None:
        """:param thetas: (n_envs, d_Theta), row i is sub-env i's flattened parameter vector."""
        params, offset = {}, 0
        for name, shape in zip(self.names, self.shapes):
            size = int(np.prod(shape))
            params["policy." + name] = thetas[:, offset:offset + size].reshape(thetas.shape[0], *shape)
            offset += size
        self._batched_params = params

    def __call__(self, obs: th.Tensor, deterministic: bool = True) -> tuple[th.Tensor, None]:
        assert deterministic, "BastaniFD rollouts are deterministic"
        assert self._batched_params is not None, "call set_parameters() before rolling out"

        def single(params: dict[str, th.Tensor], single_obs: th.Tensor) -> th.Tensor:
            # Parameters not in `params` (log_std, a frozen bias) keep their nominal values.
            return functional_call(self.mean_actor, params, (single_obs.unsqueeze(0),), strict=False).squeeze(0)

        actions = vmap(single)(self._batched_params, obs)
        return actions.reshape((-1, *self.policy.action_space.shape)), None


class BastaniFD(TrajectoryOnPolicyAlgorithm):
    """Two-sided parameter-space finite differences of the deterministic policy's return.

    One update draws batch_size = n_envs directions nu_1, ..., nu_N over the flattened
    policy parameters (see SAMPLING_MODES) and collects two width-N rollouts of the
    deterministic policy: sub-env i plays theta + sigma * nu_i in the first and
    theta - sigma * nu_i in the second, giving the returns J+_i and J-_i. Then

        g = 1/N sum_i (J+_i - J-_i) / (2 * sigma) * q(nu_i).

    Each direction costs exactly two trajectories (2N per update, the same count as
    Trajectory-FDPG with batch_size=N). No action noise is injected anywhere and no
    grad_theta mu_theta is used: this is a black-box estimator. For finite sigma it is
    unbiased for the gradient of J smoothed over the Gaussian / ball of radius sigma.

    Returns, horizon capping, episode termination, action clipping, env reseeding and
    timestep accounting all go through the shared trajectory collector, exactly as for
    FDPG's reference rollouts.

    :param sigma: Positive parameter-perturbation radius (lambda in the paper).
    :param batch_size: N, the number of directions per update; must equal env.num_envs.
    :param sampling_mode: "normal" or "sphere", see SAMPLING_MODES.
    :param use_crn: If True, the + and - rollouts replay the same reset seeds (common
        random numbers). If False (default), the - rollout gets an independent seed.

    The returns J+ / J- belong to the perturbed policies; the nominal theta is never
    rolled out during training, so train/mean_return = mean (J+ + J-) / 2 and
    rollout/* stats come from perturbed policies. Use eval/* to compare methods.
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
        sampling_mode: str = "normal",
        use_crn: bool = False,
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
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        if sigma is None or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be finite and strictly positive")
        if sampling_mode not in SAMPLING_MODES:
            raise ValueError(f"sampling_mode must be one of {SAMPLING_MODES}, but got {sampling_mode}")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 1:
            raise ValueError("n_steps must be a positive integer")
        if not np.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        if use_sde:
            raise ValueError("BastaniFD uses deterministic actions; use_sde must be False")

        policy_kwargs = dict(policy_kwargs or {})
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
        self.sampling_mode = sampling_mode
        self.use_crn = use_crn
        self._n_updates = 0

        if _init_setup_model:
            self._setup_model()

        # set up the method's name for logging purposes
        self.name = "BastaniFD-" + sampling_mode

    def _setup_model(self) -> None:
        super()._setup_model()

        if self.n_envs != self.batch_size:
            raise ValueError(
                f"env.num_envs must equal batch_size: got {self.n_envs} envs but "
                f"batch_size={self.batch_size}. Build the training env with n_envs=batch_size "
                f"(one sub-env per direction)."
            )
        names = self._search_parameter_names(self.policy)
        if not names:
            raise ValueError("BastaniFD requires trainable deterministic actor parameters")
        self._rollout_policy = _PerEnvParameterPolicy(self.policy, names)
        self._gradient_estimate = None
        self._rollout_seed = None

        # Same dedicated RNG streams as FDPG (env reseeding, perturbation sampling),
        # seeded once from self.seed and independent of the global numpy/torch streams.
        self._episode_seed_rng = np.random.default_rng(self.seed)
        self._perturbation_rng = None
        if self.seed is not None:
            self._perturbation_rng = th.Generator(device=self.device)
            self._perturbation_rng.manual_seed(self.seed)

        if self.env is not None:
            self._check_normalization(self.env)

    @staticmethod
    def _search_parameter_names(policy: ActorOnlyPolicy) -> list[str]:
        return [name for name, p in policy.named_parameters() if p.requires_grad and name != "log_std"]

    def _search_parameters(self, policy: ActorOnlyPolicy) -> list[th.nn.Parameter]:
        names = set(self._search_parameter_names(policy))
        return [p for name, p in policy.named_parameters() if name in names]

    @staticmethod
    def _check_normalization(env: VecEnv) -> None:
        normalizer = unwrap_vec_normalize(env)
        if normalizer is not None and normalizer.training and (normalizer.norm_obs or normalizer.norm_reward):
            raise ValueError(
                "BastaniFD requires fixed observation/reward transforms across +/- rollouts. "
                "Disable VecNormalize or freeze its statistics with training=False."
            )

    def _get_rollout_policy(self) -> _PerEnvParameterPolicy:
        return self._rollout_policy

    def _reset_env(self, env: VecEnv) -> np.ndarray:
        self._episode_seeds = np.array(env.seed(self._rollout_seed))  # records one seed per env
        return env.reset()

    def _sample_directions(self, n_directions: int, dim: int) -> tuple[th.Tensor, th.Tensor]:
        """Draw nu (n_directions, dim) and q(nu), with FDPG's sampling conventions."""
        nu = th.randn(n_directions, dim, device=self.device, generator=self._perturbation_rng)
        if self.sampling_mode == "normal":
            q_nu = nu
        elif self.sampling_mode == "sphere":
            nu = nu / nu.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            q_nu = dim * nu
        else:
            raise ValueError(f"sampling_mode must be one of {SAMPLING_MODES}, but got {self.sampling_mode}")
        return nu, q_nu

    def _estimate_gradient(self, j_plus: th.Tensor, j_minus: th.Tensor, q_nu: th.Tensor) -> th.Tensor:
        """g = 1/N sum_i (J+_i - J-_i) / (2 sigma) * q(nu_i)."""
        differences = (j_plus - j_minus).to(q_nu.dtype) / (2 * self.sigma)
        return (differences[:, None] * q_nu).mean(dim=0)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: TrajectoryBuffer,
        n_rollout_steps: int,
    ) -> bool:
        self._check_normalization(env)
        self._gradient_estimate = None
        with th.no_grad():
            theta = parameters_to_vector(self._search_parameters(self.policy)).detach().clone()
            nu, q_nu = self._sample_directions(env.num_envs, theta.numel())

        returns = []
        try:
            for sign in (1, -1):
                # Same draw pattern as FDPG: one seed for the first rollout, and one more
                # for the second only without CRN (with CRN it replays the same seeds).
                if sign == 1 or not self.use_crn:
                    self._rollout_seed = int(self._episode_seed_rng.integers(0, 2**31 - 1))
                with th.no_grad():
                    self._rollout_policy.set_parameters(theta + sign * self.sigma * nu)
                # Reuse the collector, including action clipping, episode caps,
                # termination masks, timestep accounting and all callbacks.
                if not super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps):
                    return False
                returns.append(th.as_tensor(
                    np.array([rollout_buffer._returns[i][0] for i in range(env.num_envs)], dtype=np.float64),
                    device=self.device,
                ))
        finally:
            self._rollout_policy._batched_params = None
            self._rollout_seed = None

        j_plus, j_minus = returns
        gradient = self._estimate_gradient(j_plus, j_minus, q_nu)
        if not th.isfinite(gradient).all():
            raise ValueError("Non-finite BastaniFD gradient; check rewards and sigma")
        self._gradient_estimate = gradient
        self._mean_return_plus = float(j_plus.mean())
        self._mean_return_minus = float(j_minus.mean())
        differences = (j_plus - j_minus).cpu().numpy()
        self._mean_abs_return_difference = float(np.abs(differences).mean())
        self._zero_difference_fraction = float(np.mean(differences == 0))
        return True

    def train(self) -> None:
        if self._gradient_estimate is None:
            raise RuntimeError("Collect a complete finite-difference estimate before train()")
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        parameters = self._search_parameters(self.policy)
        self.policy.optimizer.zero_grad()
        offset = 0
        for parameter in parameters:
            size = parameter.numel()
            # Optimizers minimize; negate the return gradient for ascent.
            parameter.grad = -self._gradient_estimate[offset:offset + size].reshape_as(parameter).clone()
            offset += size
        if self.max_grad_norm is not None:
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates)
        self.logger.record("train/n_valid_trajectories", self.batch_size)
        self.logger.record("train/mean_return", (self._mean_return_plus + self._mean_return_minus) / 2)
        self.logger.record("train/gradient_norm", self._gradient_estimate.norm().item())
        self.logger.record("train/mean_return_plus", self._mean_return_plus)
        self.logger.record("train/mean_return_minus", self._mean_return_minus)
        self.logger.record("train/mean_abs_return_difference", self._mean_abs_return_difference)
        self.logger.record("train/zero_difference_fraction", self._zero_difference_fraction)
        self.logger.record("train/sigma", self.sigma)
        self.logger.record("train/n_parameters", offset)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self._gradient_estimate = None

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["_rollout_policy", "_gradient_estimate", "_rollout_seed"]

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
