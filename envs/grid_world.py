import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple

# TODO: implement the rendering

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

class GridWorldWalls(GridWorld):
    """
    GridWorld with a U-shaped obstacle (soft walls) around the goal.

    The U opens upward (+y). Three penalty zones surround the goal:
      left arm  : x ∈ (0.35W, 0.45W),  y ∈ (0.35H, 0.65H)
      right arm : x ∈ (0.55W, 0.65W),  y ∈ (0.35H, 0.65H)
      bottom bar: x ∈ (0.35W, 0.65W),  y ∈ (0.25H, 0.35H)
    The goal at (0.5W, 0.5H) lies in the U interior and is never inside a wall.
    The agent must enter from above (y > 0.65H) to reach the goal.

    Reward: w0 * r_distance + w1 * r_obstacle
      r_distance : -||state - goal||       (dense distance penalty)
      r_obstacle : -1 if inside a wall zone, 0 otherwise
    Both components are returned in info under "r_distance" / "r_obstacle".
    """
    def __init__(
            self, render_mode=None,
            grid_dimension: Tuple[float, float] = (10, 10),
            max_radius: float = .1,
            starting_state: Optional[Tuple[float, float]] = None,
            randomize_starting_state: bool = False,
            goal_tolerance: float = .1,
            reward_weights: Tuple[float, float] = (1.0, 1.0),
            ):
        super().__init__(
            render_mode,
            grid_dimension,
            max_radius,
            starting_state,
            randomize_starting_state,
            goal_tolerance,
        )
        self.name = "GridWorldWalls"
        assert len(reward_weights) == 2, f"[{self.name}] reward_weights must be a tuple of size 2"
        self.reward_weights = np.array(reward_weights, dtype=np.float32)

    def _check_wall(self) -> bool:
        x, y = self._state
        W, H = self.grid_dimension
        in_left_arm   = (W * 0.35 < x < W * 0.45) and (H * 0.35 < y < H * 0.65)
        in_right_arm  = (W * 0.55 < x < W * 0.65) and (H * 0.35 < y < H * 0.65)
        in_bottom_bar = (W * 0.35 < x < W * 0.65) and (H * 0.25 < y < H * 0.35)
        return in_left_arm or in_right_arm or in_bottom_bar

    def step(self, action):
        self._state = self._compute_next_state(action)
        obs = self._state.astype(np.float32)

        r_distance = float(-np.linalg.norm(self._state - self.goal_position))
        r_obstacle = -1.0 if self._check_wall() else 0.0
        reward = float(self.reward_weights @ np.array([r_distance, r_obstacle], dtype=np.float32))

        terminated = self._check_absorbed()
        info = {"r_distance": r_distance, "r_obstacle": r_obstacle}
        return obs, reward, terminated, False, info