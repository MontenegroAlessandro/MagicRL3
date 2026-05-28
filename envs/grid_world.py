import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class GridWorld(gym.Env):
    """
    Continuous gridworld environment compatible with Stable Baselines 3.
    State space: (x, y) position of the agent within [0, grid_dimension].
    Goal: center of the grid.
    Action space: Cartesian displacement [dx, dy], each in [-max_radius, max_radius].
    Reward: negative euclidean distance to the goal (dense penalty; 0 at goal).
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
            self, render_mode=None,
            grid_dimension: Tuple[float, float] = (10, 10),
            max_radius: float = .1,
            starting_state: Optional[Tuple[float, float]] = None,
            randomize_starting_state: bool = True,
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
            low=np.full(2, -max_radius), high=np.full(2, max_radius), shape=(2,), dtype=np.float32
        )

        self.goal_position = np.array(grid_dimension, dtype=np.float32) / 2.0

        assert render_mode is None or render_mode in self.metadata["render_modes"], (
            f"[{self.name}] render_mode must be one of {self.metadata['render_modes']}"
        )
        self.render_mode = render_mode
        self._state = None

        # rendering handles — None until first render() call
        self._fig = None
        self._ax = None
        self._agent_artist = None
        self._traj_artist = None
        # trajectory buffer and episode counter: only allocated when rendering is active
        self._trajectory: Optional[deque] = deque(maxlen=300) if render_mode else None
        self._episode: int = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.randomize_starting_state:
            # self._state = self.np_random.uniform(
            #     low=self.observation_space.low, high=self.observation_space.high
            # ).astype(np.float32)
            self._state = np.array(np.random.choice([0, 1],size=2) * self.grid_dimension, dtype=np.float32)
        elif self.starting_state is not None:
            self._state = self.starting_state.copy()
        else:
            self._state = np.zeros(2, dtype=np.float32)
        if self._trajectory is not None:
            self._trajectory.clear()
            self._episode += 1
        if self.render_mode == "human":
            self.render()
        return self._state, {}

    def _check_absorbed(self):
        # return bool(np.linalg.norm(self._state - self.goal_position, ord=1) <= self.goal_tolerance)
        return bool(np.linalg.norm(self._state - self.goal_position) <= self.goal_tolerance)

    def _compute_next_state(self, action):
        if self._check_absorbed():
            return self._state
        displacement = np.clip(action, -self.max_radius, self.max_radius).astype(np.float32)
        candidate_state = self._state + displacement
        return np.clip(candidate_state, self.observation_space.low, self.observation_space.high)

    def step(self, action):
        self._state = self._compute_next_state(action)
        obs = self._state.astype(np.float32)
        reward = float(-np.linalg.norm(self._state - self.goal_position))
        terminated = self._check_absorbed()
        if self._trajectory is not None:
            self._trajectory.append(self._state.copy())
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, False, {}

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _draw_static(self, ax) -> None:
        """Draw elements that never change (goal zone, goal marker).
        Called once at init; subclasses extend this to add obstacles."""
        tol_patch = mpatches.Circle(
            self.goal_position, self.goal_tolerance, color="#8cdc8c", zorder=2
        )
        ax.add_patch(tol_patch)
        ax.plot(*self.goal_position, "o", color="#1ea01e", markersize=12, zorder=3)

    def _update_title(self) -> None:
        dist = np.linalg.norm(self._state - self.goal_position)
        try:
            self._fig.canvas.manager.set_window_title(
                f"{self.name} | episode {self._episode} | dist {dist:.3f}"
            )
        except AttributeError:
            pass

    def _init_render(self) -> None:
        plt.ion()
        self._fig, self._ax = plt.subplots(figsize=(6, 6))
        self._fig.patch.set_facecolor("#f5f5f5")
        self._update_title()

        ax = self._ax
        W, H = self.grid_dimension
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")
        ax.set_facecolor("#dcdcdc")
        ax.set_xticks([])
        ax.set_yticks([])

        # static scene elements (goal, obstacles, …)
        self._draw_static(ax)

        # trajectory trail — updated via set_data() every frame
        (self._traj_artist,) = ax.plot(
            [], [], "-", color="#1e64d2", alpha=0.35, linewidth=1.5, zorder=3
        )
        # agent dot — updated via set_data() every frame
        (self._agent_artist,) = ax.plot(
            [], [], "o", color="#1e64d2", markersize=12,
            markeredgecolor="white", markeredgewidth=2, zorder=4,
        )
        plt.tight_layout(pad=0.5)

    def _draw_scene(self) -> None:
        """Update the per-frame dynamic artists (agent + trail).
        Static elements were drawn once in _init_render; no ax.clear() needed."""
        self._agent_artist.set_data([self._state[0]], [self._state[1]])
        if self._trajectory and len(self._trajectory) > 1:
            traj = np.array(self._trajectory)
            self._traj_artist.set_data(traj[:, 0], traj[:, 1])
        else:
            self._traj_artist.set_data([], [])

    def render(self):
        if self.render_mode is None:
            return
        if self._fig is None:
            self._init_render()

        self._draw_scene()
        self._update_title()

        if self.render_mode == "human":
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
            plt.pause(1.0 / self.metadata["render_fps"])
        else:
            self._fig.canvas.draw()
            buf = self._fig.canvas.tostring_rgb()
            w, h = self._fig.canvas.get_width_height()
            return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
            self._agent_artist = None
            self._traj_artist = None


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
            max_radius: float = .2,
            starting_state: Optional[Tuple[float, float]] = None,
            randomize_starting_state: bool = True,
            goal_tolerance: float = .1,
            reward_weights: Tuple[float, float] = (.1, 1.0),
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
        if self._trajectory is not None:
            self._trajectory.append(self._state.copy())
        if self.render_mode == "human":
            self.render()
        info = {
            "rewards": np.array([r_distance, r_obstacle], dtype=np.float32),
        }
        return obs, reward, terminated, False, info

    def _draw_static(self, ax) -> None:
        super()._draw_static(ax)
        W, H = self.grid_dimension
        # Wall zones: (x, y, width, height) matching _check_wall() bounds exactly
        wall_zones = [
            (W * 0.35, H * 0.35, W * 0.10, H * 0.30),  # left arm
            (W * 0.55, H * 0.35, W * 0.10, H * 0.30),  # right arm
            (W * 0.35, H * 0.25, W * 0.30, H * 0.10),  # bottom bar
        ]
        for (x, y, w, h) in wall_zones:
            ax.add_patch(mpatches.Rectangle(
                (x, y), w, h,
                linewidth=1.5, edgecolor="#a00000",
                facecolor="#d23232", alpha=0.55, zorder=1,
            ))
