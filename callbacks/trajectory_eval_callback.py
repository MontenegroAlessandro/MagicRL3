import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.callbacks import EventCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv


def collect_discounted_returns(model, env, n_steps, deterministic=True):
    """
    Collect exactly one capped trajectory per env in `env` — so env.num_envs
    IS the trajectory count, no rounds needed. Read-only w.r.t. `model`.
    """
    policy = model.policy
    policy.set_training_mode(False)
    gamma = model.gamma
    n_envs = env.num_envs

    obs = env.reset()
    active = np.ones(n_envs, dtype=bool)
    discounted_returns = np.zeros(n_envs, dtype=np.float64)
    discount = 1.0
    steps = 0

    with th.no_grad():
        while active.any() and steps < n_steps:
            obs_tensor = obs_as_tensor(obs, model.device)
            actions, _ = policy(obs_tensor, deterministic=deterministic)
            actions = actions.cpu().numpy()

            clipped_actions = actions
            if isinstance(model.action_space, spaces.Box):
                if policy.squash_output:
                    clipped_actions = policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(actions, model.action_space.low, model.action_space.high)

            obs, rewards, dones, infos = env.step(clipped_actions)

            discounted_returns += active * discount * rewards
            discount *= gamma
            active &= ~dones
            steps += 1

    return discounted_returns


class TrajectoryEvalCallback(EventCallback):
    def __init__(self, eval_env, n_eval_episodes: int = 10, eval_freq: int = 10_000,
                 deterministic: bool = True, verbose: int = 0):
        super().__init__(verbose=verbose)
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])
        assert eval_env.num_envs == n_eval_episodes, (
            f"eval_env has {eval_env.num_envs} parallel envs but n_eval_episodes="
            f"{n_eval_episodes}; make eval_env's n_envs match, or drop n_eval_episodes "
            f"and just read it off eval_env.num_envs."
        )
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self._last_eval_timestep = None

    def _evaluate(self) -> None:
        returns = collect_discounted_returns(
            self.model, self.eval_env, n_steps=self.model.n_steps, deterministic=self.deterministic,
        )
        mean_return, std_return = float(returns.mean()), float(returns.std())

        if self.verbose >= 1:
            print(f"Eval num_timesteps={self.num_timesteps}, "
                  f"mean_discounted_return={mean_return:.2f} +/- {std_return:.2f}")

        self.logger.record("eval/mean_discounted_return", mean_return)
        self.logger.record("eval/std_discounted_return", std_return)
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        self.logger.dump(self.num_timesteps)

        self._last_eval_timestep = self.num_timesteps

    def _on_training_start(self) -> None:
        if self.eval_freq > 0:
            self._evaluate()

    def _on_step(self) -> bool:
        # `num_timesteps` can jump by more than one environment step per call
        # (vectorized envs, and algorithms like FDPG that fold extra rollout
        # batches into num_timesteps without a matching on_step() call), so we
        # can't check for an exact multiple of eval_freq — instead fire as soon
        # as we've advanced eval_freq steps since the last evaluation.
        if self.eval_freq > 0 and self.num_timesteps - self._last_eval_timestep >= self.eval_freq:
            self._evaluate()

        return True

    def _on_training_end(self) -> None:
        # self.num_timesteps is only kept in sync with self.model.num_timesteps
        # inside on_step(); algorithms that fold extra steps into num_timesteps
        # outside of on_step() (e.g. FDPG's perturbed rollouts) can leave it
        # stale by the time training stops, so resync before deciding.
        self.num_timesteps = self.model.num_timesteps
        if self.eval_freq > 0 and self.num_timesteps != self._last_eval_timestep:
            self._evaluate()