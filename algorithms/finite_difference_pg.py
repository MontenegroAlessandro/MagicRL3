from typing import Any, ClassVar, Optional, TypeVar, Union

import torch as th
import numpy as np
from gymnasium import spaces

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import obs_as_tensor

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
        mode: str = "step", # !
        sampling_mode: str = "normal", # !
        sampling_strategy: str = "step", # !
        max_grad_norm: float = 0.5,
        ent_coef: float = 0.0,
        normalize_returns: bool = False,
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

        self.ent_coef = ent_coef
        self.normalize_returns = normalize_returns
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

        if _init_setup_model:
            self._setup_model()

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        # get() yields one batch with all trajectories concatenated in env order
        # notice that by construction the buffer will have n_envs nominal trajectories (i.e., collected via the deterministic policy)
        for rollout_data in self.rollout_buffer.get():
            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.long().flatten()

            log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)

            # --- trajectory view ---
            # Assign each step to its trajectory so we can reduce per trajectory.
            # _traj_lengths[i] = number of steps in trajectory i (0 for inactive envs).
            lengths = th.tensor(
                self.rollout_buffer._traj_lengths[self.rollout_buffer._traj_lengths > 0],
                dtype=th.long, device=self.device,
            )                                                           # (N,)
            N = len(lengths)
            traj_idx = th.repeat_interleave(th.arange(N, device=self.device), lengths)  # (total_steps,)

            # Sum log π(a_t|s_t) over the steps of each trajectory: shape (N,)
            log_prob_sums = th.zeros(N, device=self.device).scatter_add_(0, traj_idx, log_prob)

            # G(τ_i) = return-to-go at t=0 = total discounted return of trajectory i
            starts = th.cat([th.zeros(1, dtype=th.long, device=self.device), lengths.cumsum(0)[:-1]])
            G = rollout_data.returns[starts]                            # (N,)

            if self.normalize_returns:
                G = (G - G.mean()) / (G.std() + 1e-8)

            # L = -(1/N) Σ_i [ G(τ_i) · Σ_t log π(a_t^i | s_t^i) ]
            policy_loss = -(G * log_prob_sums).mean()

            # entropy regularization (averaged over steps, not summed, to be length-invariant)
            entropy_loss = -(-log_prob if entropy is None else entropy).mean()

            loss = policy_loss + self.ent_coef * entropy_loss

            self.policy.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None:
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/mean_return", G.mean().item())
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
