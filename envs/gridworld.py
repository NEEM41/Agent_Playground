from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import numpy as np

from maps import MapSpec, EMPTY, WALL, TRAP, GOAL


# Actions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ACTION_TO_DELTA = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}


@dataclass
class GridWorldConfig:
    max_steps: int = 300

    # Rewards
    step_penalty: float = -0.01
    wall_penalty: float = -0.10
    trap_penalty: float = -0.20
    goal_reward: float = 1.00

    # If True, reaching goal ends episode immediately (recommended)
    terminate_on_goal: bool = True


class DiscreteActionSpace:
    """Tiny helper so you can do env.action_space.sample()."""

    def __init__(self, n: int, rng: Optional[np.random.Generator] = None):
        self.n = int(n)
        self._rng = rng or np.random.default_rng()

    def seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def sample(self) -> int:
        return int(self._rng.integers(0, self.n))


class GridWorld:
    """
    A simple cell-based gridworld.

    Grid codes (from maps.py):
      EMPTY=0, WALL=1, TRAP=2, GOAL=3

    State:
      - agent position (r, c)
      - step counter

    API:
      obs = reset(seed=...)
      obs, reward, done, info = step(action)
    """

    def __init__(self, map_spec: MapSpec, config: Optional[GridWorldConfig] = None):
        self.map_spec = map_spec
        self.cfg = config or GridWorldConfig()

        self.H, self.W = self.map_spec.shape

        # Copy base grid so we never mutate MapSpec.grid
        self._base_grid = np.array(self.map_spec.grid, copy=True)

        # RNG initialized in reset()
        self.rng: np.random.Generator = np.random.default_rng()

        # Action space
        self.action_space = DiscreteActionSpace(n=4, rng=self.rng)

        # Runtime state
        self.agent_pos: Tuple[int, int] = self.map_spec.start
        self.goal_pos: Tuple[int, int] = self.map_spec.goal
        self.step_count: int = 0
        self.done: bool = False

    # Public API
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.action_space = DiscreteActionSpace(n=4, rng=self.rng)

        self.agent_pos = self.map_spec.start
        self.goal_pos = self.map_spec.goal
        self.step_count = 0
        self.done = False

        return self._get_obs()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        if int(action) not in ACTION_TO_DELTA:
            raise ValueError(f"Invalid action {action}. Must be one of {list(ACTION_TO_DELTA.keys())}.")

        prev_pos = self.agent_pos
        dr, dc = ACTION_TO_DELTA[int(action)]
        nr, nc = prev_pos[0] + dr, prev_pos[1] + dc

        reward = 0.0
        info: Dict[str, Any] = {
            "hit_wall": False,
            "hit_trap": False,
            "reached_goal": False,
            "prev_pos": prev_pos,
        }

        # Step penalty always applies (encourages faster solutions)
        reward += float(self.cfg.step_penalty)

        # Check bounds
        if not self._in_bounds(nr, nc):
            # Treat OOB as wall
            reward += float(self.cfg.wall_penalty)
            info["hit_wall"] = True
            nr, nc = prev_pos  # stay put
        else:
            cell = int(self._base_grid[nr, nc])
            if cell == WALL:
                reward += float(self.cfg.wall_penalty)
                info["hit_wall"] = True
                nr, nc = prev_pos  # stay put

        # Apply move
        self.agent_pos = (nr, nc)

        # Check landing cell effects
        cell = int(self._base_grid[self.agent_pos])
        if cell == TRAP:
            reward += float(self.cfg.trap_penalty)
            info["hit_trap"] = True

        if self.agent_pos == self.goal_pos or cell == GOAL:
            reward += float(self.cfg.goal_reward)
            info["reached_goal"] = True
            if self.cfg.terminate_on_goal:
                self.done = True

        # Time limit termination
        self.step_count += 1
        if self.step_count >= int(self.cfg.max_steps):
            self.done = True

        obs = self._get_obs()
        return obs, float(reward), bool(self.done), info

    # Observation helpers
    def _get_obs(self) -> Dict[str, Any]:
        """
        MVP observation: return the static grid plus agent/goal positions.
        Later you can add:
          - local window (egocentric)
          - channels (walls/traps/goal/agent)
        """
        return {
            "grid": self._base_grid,         # (H, W) int32
            "agent_pos": self.agent_pos,     # (r, c)
            "goal_pos": self.goal_pos,       # (r, c)
            "step": self.step_count,
        }

    # Utility
    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.H and 0 <= c < self.W
    
    # Rendering to check
    def render_rgb(
    self,
    cell_size: int = 18,
    agent_pos: Optional[Tuple[int, int]] = None,
    agent_color: Tuple[int, int, int] = (60, 120, 220),
    sprite: str = "box",                 # "box" or "pacman"
    facing: str = "right",               # "right","left","up","down"
    ) -> np.ndarray:
        H, W = self.H, self.W
        img = np.ones((H, W, 3), dtype=np.uint8) * 255

        wall_color = np.array([0, 0, 0], dtype=np.uint8)
        trap_color = np.array([220, 60, 60], dtype=np.uint8)
        goal_color = np.array([60, 180, 60], dtype=np.uint8)

        grid = self._base_grid
        img[grid == WALL] = wall_color
        img[grid == TRAP] = trap_color
        img[grid == GOAL] = goal_color

        # upscale to pixel canvas
        if cell_size != 1:
            img = np.kron(img, np.ones((cell_size, cell_size, 1), dtype=np.uint8))

        # draw agent sprite into pixel canvas
        ar, ac = agent_pos if agent_pos is not None else self.agent_pos
        r0, c0 = ar * cell_size, ac * cell_size
        cell = img[r0:r0 + cell_size, c0:c0 + cell_size]

        if sprite == "box":
            cell[:, :] = np.array(agent_color, dtype=np.uint8)

        elif sprite == "pacman":
            # pacman is usually yellow; still allow override via agent_color
            col = np.array(agent_color, dtype=np.uint8)

            yy, xx = np.mgrid[0:cell_size, 0:cell_size]
            cy = (cell_size - 1) / 2.0
            cx = (cell_size - 1) / 2.0
            dy = yy - cy
            dx = xx - cx
            dist = np.sqrt(dx * dx + dy * dy)

            radius = (cell_size - 2) / 2.0
            circle = dist <= radius

            # angle in radians [-pi, pi]
            ang = np.arctan2(dy, dx)

            # mouth wedge centered on facing direction
            facing_angle = {
                "right": 0.0,
                "left": np.pi,
                "up": -np.pi / 2.0,
                "down": np.pi / 2.0,
            }.get(facing, 0.0)

            mouth_open = np.deg2rad(50)  # total mouth opening ~100 degrees
            # angular difference (wrap-safe)
            d = np.arctan2(np.sin(ang - facing_angle), np.cos(ang - facing_angle))
            mouth = np.abs(d) < (mouth_open / 2.0)

            pacman_mask = circle & (~mouth)

            # draw pacman
            cell[pacman_mask] = col

            # optional: tiny eye (cute)
            eye_y = int(cy - radius * 0.35)
            eye_x = int(cx + radius * 0.15)
            eye_r = max(1, cell_size // 12)
            yy2, xx2 = np.mgrid[0:cell_size, 0:cell_size]
            eye = (yy2 - eye_y) ** 2 + (xx2 - eye_x) ** 2 <= eye_r ** 2
            cell[eye] = np.array([30, 30, 30], dtype=np.uint8)

        else:
            raise ValueError(f"Unknown sprite '{sprite}'. Use 'box' or 'pacman'.")

        return img