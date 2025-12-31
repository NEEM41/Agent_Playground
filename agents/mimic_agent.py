from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

from agents.base import BaseAgent

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_TO_DELTA = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

@dataclass
class MimicConfig:
    p_random: float = 0.05
    max_follow_dist: int = 12
    ignore_done_leaders: bool = True

class MimicAgent(BaseAgent):
    """
    Follows the nearest other agent by moving toward their POSITION.
    This produces visible "tailing" behavior immediately.
    """

    def __init__(self, cfg: MimicConfig = MimicConfig(), seed: Optional[int] = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def act(self, obs: Dict[str, Any], mode: str = "train") -> int:
        # small randomness keeps it from getting stuck in symmetric situations
        if self.rng.random() < self.cfg.p_random:
            return int(self.rng.integers(0, 4))

        agent_id = obs.get("agent_id", None)
        agents = obs.get("agents", None)
        my_pos = obs.get("agent_pos", None)
        grid = obs.get("grid", None)

        if agent_id is None or agents is None or my_pos is None:
            return int(self.rng.integers(0, 4))

        # find nearest leader (non-self; optionally non-done)
        best_pos = None
        best_d = None
        for j, aj in enumerate(agents):
            if j == agent_id:
                continue
            if self.cfg.ignore_done_leaders and bool(aj.get("done", False)):
                continue
            pos = aj.get("pos", None)
            if pos is None:
                continue
            d = abs(pos[0] - my_pos[0]) + abs(pos[1] - my_pos[1])
            if best_d is None or d < best_d:
                best_d = d
                best_pos = pos

        if best_pos is None or best_d is None or best_d > self.cfg.max_follow_dist:
            return int(self.rng.integers(0, 4))

        # choose action that reduces distance to best_pos
        r, c = my_pos
        tr, tc = best_pos

        candidates: List[int] = [UP, DOWN, LEFT, RIGHT]
        self.rng.shuffle(candidates)

        def manhattan(rr, cc):
            return abs(tr - rr) + abs(tc - cc)

        best_action = None
        best_score = manhattan(r, c)

        for a in candidates:
            dr, dc = ACTION_TO_DELTA[a]
            nr, nc = r + dr, c + dc

            # if grid is present, avoid stepping into walls
            if grid is not None:
                H, W = grid.shape
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if int(grid[nr, nc]) == 1:  # WALL
                    continue

            score = manhattan(nr, nc)
            if score < best_score:
                best_score = score
                best_action = a

        if best_action is None:
            # no improving move found; random step
            return int(self.rng.integers(0, 4))

        return int(best_action)