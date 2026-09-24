from .grid_world import *
from .lqr import LQR
from gymnasium.envs.registration import register

register(
    id="GridWorld-v0",
    entry_point="envs.grid_world:GridWorld",
    max_episode_steps=1000,
)

register(
    id="GridWorldWalls-v0",
    entry_point="envs.grid_world:GridWorldWalls",
    max_episode_steps=1000,
)
# LQR is registered once, with no parameters baked in: the system (dimensions,
# matrices, noise) is chosen at construction and the horizon is a time limit like
# for any other environment, e.g.
#   gym.make("LQR-v0", state_dim=3, action_dim=2, max_episode_steps=50)
#   make_vec_env("LQR-v0", n_envs=8, env_kwargs=dict(state_dim=3, noise=0.1))
# The discount belongs to the algorithm; LQR's closed-form helpers take it as an
# explicit argument.
register(
    id="LQR-v0",
    entry_point="envs.lqr:LQR",
)
