from .grid_world import *
from gymnasium.envs.registration import register

register(
    id="GridWorld-v0",
    entry_point="envs.grid_world:GridWorld",
    max_episode_steps=1000,
)