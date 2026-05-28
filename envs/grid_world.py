import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple

class GridWorld(gym.Env):
    """
    Continuous gridworld environment compatible with Stable Baselines 3.
    State space: (x, y) position of the agent within [0, grid_dimension].
    Goal: center of the grid.
    Action space: polar commands [radius, angle], radius in [0, max_radius], angle in [-pi, pi].
    Reward: negative euclidean distance to the goal (dense penalty; 0 at goal).
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
            self, render_mode=None,
            grid_dimension: Tuple[float, float] = (10, 10),
            max_radius: float = .1,
            starting_state: Optional[Tuple[float, float]] = None,
            randomize_starting_state: bool = False,
            goal_tolerance: float = .1,
            ):
        super().__init__()

        self.name = "GridWorld"

        self.starting_state = np.array(starting_state, dtype=np.float32) if starting_state is not None else None
        self.randomize_starting_state = randomize_starting_state

        assert goal_tolerance >= 0, f"[{self.name}] goal_tolerance must be non-negative"
        self.goal_tolerance = goal_tolerance

        assert np.array(grid_dimension).shape == (2,), f"[{self.name}] grid_dimension must be a tuple of size 2"
        assert np.all(np.array(grid_dimension) > 0), f"[{self.name}] grid_dimension must be positive"
        self.grid_dimension = grid_dimension

        assert max_radius >= 0, f"[{self.name}] max_radius must be non-negative"
        self.max_radius = max_radius

        self.observation_space = spaces.Box(
            low=np.zeros(2), high=np.array(grid_dimension), shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([0, -np.pi]), high=np.array([max_radius, np.pi]), shape=(2,), dtype=np.float32
        )

        self.goal_position = np.array(grid_dimension, dtype=np.float32) / 2.0

        self.render_mode = render_mode
        self._state = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.randomize_starting_state:
            self._state = self.np_random.uniform(
                low=self.observation_space.low, high=self.observation_space.high
            ).astype(np.float32)
        elif self.starting_state is not None:
            self._state = self.starting_state.copy()
        else:
            self._state = np.zeros(2, dtype=np.float32)
        return self._state, {}
    
    def _check_absorbed(self):
        return np.linalg.norm(self._state - self.goal_position) < self.goal_tolerance
    
    def _compute_next_state(self, action):
        if self._check_absorbed():
            return self._state  
        radius = np.clip(action[0], 0, self.max_radius)
        angle = (action[1] + np.pi) % (2 * np.pi) - np.pi
        candidate_state = self._state + radius * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        return np.clip(candidate_state, self.observation_space.low, self.observation_space.high)

    def step(self, action):
        self._state = self._compute_next_state(action)
        obs = self._state.astype(np.float32)
        reward = float(-np.linalg.norm(self._state - self.goal_position))
        terminated = self._check_absorbed()
        return obs, reward, terminated, False, {}

    def render(self):
        if self.render_mode == "human":
            print(f"State: {self._state}")

    def close(self):
        pass