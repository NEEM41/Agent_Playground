from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from envs.maps import MapSpec, WALL, TRAP, GOAL

# Actions (must match your agents)
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
class MultiGridWorldConfig:
    max_steps: int = 300

    # Rewards
    step_penalty: float = -0.01
    wall_penalty: float = -0.10
    trap_penalty: float = -0.5
    goal_reward: float = 1.00
    backtrack_penalty: float = -0.05   # discourage A<->B oscillation
    stay_penalty: float = -0.02        # discourage bumping walls/collisions

    # Termination behavior
    terminate_on_goal: bool = True          # if an agent reaches goal, mark that agent done
    terminate_on_any_goal: bool = False      # if any agent reaches goal, end episode for everyone (race)
    terminate_on_trap: bool = True         # if an agent steps on trap, mark that agent done

    # Collision handling
    block_on_collision: bool = False         # if multiple agents want same cell, all stay


class MultiGridWorld:
    """
    Multi-agent gridworld on a shared map.

    - N agents in same grid
    - step(actions): list[int] of length N
    - Observations are per-agent dicts compatible with your agents:
        obs_i["agent_pos"], obs_i["goal_pos"], obs_i["grid"], obs_i["local"], etc.

    Collision rules (simple + deterministic):
      1) Walls/OOB block movement for that agent.
      2) If two+ agents propose the same target cell -> all those agents stay.
      3) If two agents try to swap positions (A->Bpos and B->Apos) -> both stay.
    """

    def __init__(self, map_spec: MapSpec, n_agents: int, config: Optional[MultiGridWorldConfig] = None):
        assert n_agents >= 1
        self.map_spec = map_spec
        self.cfg = config or MultiGridWorldConfig()
        self.n_agents = int(n_agents)

        self.H, self.W = self.map_spec.shape
        self._base_grid = np.array(self.map_spec.grid, copy=True)

        self.rng: np.random.Generator = np.random.default_rng()

        # Runtime state
        self.agent_pos: List[Tuple[int, int]] = []
        self.agent_done: List[bool] = [False] * self.n_agents
        self.last_actions: List[int] = [RIGHT] * self.n_agents  # for rendering facing
        self.step_count: int = 0
        self.done: bool = False

        self.goal_pos: Tuple[int, int] = self.map_spec.goal
        self.prev_pos: List[Tuple[int, int]] = []

    # -------------------------
    # Public API
    # -------------------------
    def reset(self, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.done = False
        self.agent_done = [False] * self.n_agents
        self.last_actions = [RIGHT] * self.n_agents

        # Spawn agents: by default, agent 0 at map start; others near start (first free cells found).
        self.agent_pos = self._spawn_positions()
        self.prev_pos = list(self.agent_pos)

        return self._get_all_obs()

    def step(self, actions: List[int]) -> Tuple[List[Dict[str, Any]], List[float], bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        if len(actions) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} actions, got {len(actions)}")

        rewards = [0.0 for _ in range(self.n_agents)]
        info: Dict[str, Any] = {
            "reached_goal": [False] * self.n_agents,
            "hit_trap": [False] * self.n_agents,
            "hit_wall": [False] * self.n_agents,
            "collided": [False] * self.n_agents,
            "prev_pos": list(self.agent_pos),
        }

        # Step penalty for agents that are still active
        for i in range(self.n_agents):
            if not self.agent_done[i]:
                rewards[i] += float(self.cfg.step_penalty)

        # 1) Compute proposed moves with wall/OOB blocking
        proposed = list(self.agent_pos)  # default: stay
        for i, a in enumerate(actions):
            if self.agent_done[i]:
                continue

            a = int(a)
            if a not in ACTION_TO_DELTA:
                raise ValueError(f"Invalid action {a} for agent {i}")

            pr, pc = self.agent_pos[i]
            dr, dc = ACTION_TO_DELTA[a]
            nr, nc = pr + dr, pc + dc

            # bounds
            if not self._in_bounds(nr, nc):
                rewards[i] += float(self.cfg.wall_penalty)
                info["hit_wall"][i] = True
                proposed[i] = (pr, pc)
                continue

            # wall
            cell = int(self._base_grid[nr, nc])
            if cell == WALL:
                rewards[i] += float(self.cfg.wall_penalty)
                info["hit_wall"][i] = True
                proposed[i] = (pr, pc)
                continue

            proposed[i] = (nr, nc)

        # 2) Resolve collisions (same target cell)
        if self.cfg.block_on_collision:
            counts: Dict[Tuple[int, int], List[int]] = {}
            for i in range(self.n_agents):
                if self.agent_done[i]:
                    continue
                counts.setdefault(proposed[i], []).append(i)

            for cell, idxs in counts.items():
                if len(idxs) >= 2:
                    # # All these agents stay put
                    # for i in idxs:
                    #     proposed[i] = self.agent_pos[i]
                    #     info["collided"][i] = True
                    winner = int(self.rng.choice(idxs))
                    for i in idxs:
                        if i != winner:
                            proposed[i] = self.agent_pos[i]
                            info["collided"][i] = True

        # 3) Resolve swaps (A->B and B->A)
        # Simple pairwise check
        current = list(self.agent_pos)
        for i in range(self.n_agents):
            if self.agent_done[i]:
                continue
            for j in range(i + 1, self.n_agents):
                if self.agent_done[j]:
                    continue
                if proposed[i] == current[j] and proposed[j] == current[i] and proposed[i] != current[i]:
                    # block both
                    proposed[i] = current[i]
                    proposed[j] = current[j]
                    info["collided"][i] = True
                    info["collided"][j] = True

        # Apply positions + store last actions (for rendering facing)
        for i in range(self.n_agents):
            if self.agent_done[i]:
                continue

            old = self.agent_pos[i]
            new = proposed[i]

            # stayed in place => small penalty (walls/collisions lead here)
            if new == old:
                rewards[i] += float(self.cfg.stay_penalty)

            # backtrack => if you returned to where you were last step
            # (this catches A<->B oscillation)
            if self.prev_pos and new == self.prev_pos[i] and old != new:
                rewards[i] += float(self.cfg.backtrack_penalty)

            # update
            self.prev_pos[i] = old
            self.agent_pos[i] = new
            self.last_actions[i] = int(actions[i])

        # 4) Landing effects (trap/goal) per agent
        any_goal = False
        for i in range(self.n_agents):
            if self.agent_done[i]:
                continue

            cell = int(self._base_grid[self.agent_pos[i]])

            if cell == TRAP:
                rewards[i] += float(self.cfg.trap_penalty)
                info["hit_trap"][i] = True
                if self.cfg.terminate_on_trap:
                    self.agent_done[i] = True

            if self.agent_pos[i] == self.goal_pos or cell == GOAL:
                rewards[i] += float(self.cfg.goal_reward)
                info["reached_goal"][i] = True
                any_goal = True
                if self.cfg.terminate_on_goal:
                    self.agent_done[i] = True

        # 5) Episode termination
        self.step_count += 1
        if self.step_count >= int(self.cfg.max_steps):
            self.done = True

        # Race setting: stop for everyone if any reaches goal
        if self.cfg.terminate_on_any_goal and any_goal:
            self.done = True

        # Also done if all agents are done
        if all(self.agent_done):
            self.done = True

        obs = self._get_all_obs()
        return obs, rewards, bool(self.done), info

    # -------------------------
    # Observations
    # -------------------------
    def _get_all_obs(self) -> List[Dict[str, Any]]:
        agents_meta = [
            {"pos": self.agent_pos[j], "last_action": int(self.last_actions[j]), "done": bool(self.agent_done[j])}
            for j in range(self.n_agents)
        ]

        obs_list: List[Dict[str, Any]] = []
        for i in range(self.n_agents):
            obs_list.append(
                {
                    "grid": self._base_grid,
                    "agent_id": i,
                    "agent_pos": self.agent_pos[i],
                    "goal_pos": self.goal_pos,
                    "step": self.step_count,
                    "local": self.get_local_view(agent_id=i, radius=3),
                    "agents": agents_meta,
                    "done": bool(self.agent_done[i]),
                }
            )
        return obs_list

    def get_local_view(self, agent_id: int, radius: int = 3) -> np.ndarray:
        ar, ac = self.agent_pos[agent_id]
        k = 2 * radius + 1
        patch = np.full((k, k), WALL, dtype=np.int32)

        r0 = ar - radius
        c0 = ac - radius

        for i in range(k):
            rr = r0 + i
            if rr < 0 or rr >= self.H:
                continue
            for j in range(k):
                cc = c0 + j
                if 0 <= cc < self.W:
                    patch[i, j] = int(self._base_grid[rr, cc])

        # mark goal if within window
        gr, gc = self.goal_pos
        if abs(gr - ar) <= radius and abs(gc - ac) <= radius:
            patch[gr - r0, gc - c0] = GOAL

        return patch

    # -------------------------
    # Rendering
    # -------------------------
    def render_rgb(
        self,
        cell_size: int = 18,
        agent_colors: Optional[List[Tuple[int, int, int]]] = None,
        sprite: str = "box",  # "box" or "pacman"
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

        if cell_size != 1:
            img = np.kron(img, np.ones((cell_size, cell_size, 1), dtype=np.uint8))

        if agent_colors is None:
            agent_colors = [
                (60, 120, 220),   # BFS agent — blue
                (180, 80, 180),   # Greedy agent — purple
                (120, 120, 120),  # Random agent — gray
                (220, 140, 60),   # Q-learning agent — orange
            ]
        # extend if needed
        while len(agent_colors) < self.n_agents:
            agent_colors.append(tuple(int(x) for x in self.rng.integers(40, 220, size=3)))

        # draw agents on top
        for i in range(self.n_agents):
            if self.agent_done[i]:
                continue  # optional: don't render done agents
            ar, ac = self.agent_pos[i]
            r0, c0 = ar * cell_size, ac * cell_size
            cell = img[r0 : r0 + cell_size, c0 : c0 + cell_size]
            color = np.array(agent_colors[i], dtype=np.uint8)

            if sprite == "box":
                cell[:, :] = color
            elif sprite == "pacman":
                facing = self._action_to_facing(self.last_actions[i])
                self._draw_pacman(cell, color, facing=facing)
            else:
                raise ValueError(f"Unknown sprite '{sprite}'")

        return img

    def _action_to_facing(self, a: int) -> str:
        return {UP: "up", DOWN: "down", LEFT: "left", RIGHT: "right"}.get(int(a), "right")
    
    def _far_enough(self, p, positions, min_dist: int) -> bool:
        return all(abs(p[0]-q[0]) + abs(p[1]-q[1]) >= min_dist for q in positions)

    def _draw_pacman(self, cell: np.ndarray, col: np.ndarray, facing: str = "right") -> None:
        cell_size = cell.shape[0]
        yy, xx = np.mgrid[0:cell_size, 0:cell_size]
        cy = (cell_size - 1) / 2.0
        cx = (cell_size - 1) / 2.0
        dy = yy - cy
        dx = xx - cx
        dist = np.sqrt(dx * dx + dy * dy)

        radius = (cell_size - 2) / 2.0
        circle = dist <= radius

        ang = np.arctan2(dy, dx)
        facing_angle = {
            "right": 0.0,
            "left": np.pi,
            "up": -np.pi / 2.0,
            "down": np.pi / 2.0,
        }.get(facing, 0.0)

        mouth_open = np.deg2rad(50)
        d = np.arctan2(np.sin(ang - facing_angle), np.cos(ang - facing_angle))
        mouth = np.abs(d) < (mouth_open / 2.0)

        pacman_mask = circle & (~mouth)
        cell[pacman_mask] = col

        # tiny eye
        eye_y = int(cy - radius * 0.35)
        eye_x = int(cx + radius * 0.15)
        eye_r = max(1, cell_size // 12)
        yy2, xx2 = np.mgrid[0:cell_size, 0:cell_size]
        eye = (yy2 - eye_y) ** 2 + (xx2 - eye_x) ** 2 <= eye_r ** 2
        cell[eye] = np.array([30, 30, 30], dtype=np.uint8)

    # -------------------------
    # Utils
    # -------------------------
    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.H and 0 <= c < self.W

    def _spawn_positions(self) -> List[Tuple[int, int]]:
        positions = [self.map_spec.start]
        sr, sc = self.map_spec.start

        min_dist = 3  # <-- key: adjust (2-5). 3 works well for 20x10-ish maps.

        def far_enough(p):
            return all(abs(p[0]-q[0]) + abs(p[1]-q[1]) >= min_dist for q in positions)

        # Prefer near start but not too close to each other
        for r in range(self.H):
            for c in range(self.W):
                if len(positions) >= self.n_agents:
                    break
                if int(self._base_grid[r, c]) == WALL:
                    continue
                if (r, c) == self.goal_pos:
                    continue
                if (r, c) in positions:
                    continue
                if abs(r - sr) + abs(c - sc) <= 8 and far_enough((r, c)):
                    positions.append((r, c))
            if len(positions) >= self.n_agents:
                break

        # Fill anywhere if needed (still enforce far enough if possible)
        if len(positions) < self.n_agents:
            for r in range(self.H):
                for c in range(self.W):
                    if len(positions) >= self.n_agents:
                        break
                    if int(self._base_grid[r, c]) == WALL:
                        continue
                    if (r, c) == self.goal_pos:
                        continue
                    if (r, c) in positions:
                        continue
                    if far_enough((r, c)):
                        positions.append((r, c))
                if len(positions) >= self.n_agents:
                    break

        # If still not enough, relax distance constraint
        if len(positions) < self.n_agents:
            for r in range(self.H):
                for c in range(self.W):
                    if len(positions) >= self.n_agents:
                        break
                    if int(self._base_grid[r, c]) == WALL:
                        continue
                    if (r, c) == self.goal_pos:
                        continue
                    if (r, c) in positions:
                        continue
                    positions.append((r, c))
                if len(positions) >= self.n_agents:
                    break

        if len(positions) < self.n_agents:
            raise RuntimeError("Could not spawn all agents (map too blocked).")

        return positions

    