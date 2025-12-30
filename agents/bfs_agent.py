from __future__ import annotations

from collections import deque
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

# Must match your env action mapping:
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_TO_DELTA = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}
DELTA_TO_ACTION = {v: k for k, v in ACTION_TO_DELTA.items()}

# Must match your grid codes:
WALL = 1

class BFSAgent:
    """
    Shortest-path agent using BFS on the full grid.
    Recomputes path each step (fine for 30x30).
    """

    def __init__(self, fallback_action: int = UP):
        self.fallback_action = int(fallback_action)

    def act(self, obs: Dict[str, Any], deterministic: bool = True) -> int:
        grid: np.ndarray = obs["grid"]
        start: Tuple[int, int] = obs["agent_pos"]
        goal: Tuple[int, int] = obs["goal_pos"]

        path = self._bfs_path(grid, start, goal)
        if path is None or len(path) < 2:
            return self.fallback_action

        # path is [start, ..., goal]; take the next step
        (r0, c0) = path[0]
        (r1, c1) = path[1]
        dr, dc = (r1 - r0, c1 - c0)

        return DELTA_TO_ACTION.get((dr, dc), self.fallback_action)

    def _bfs_path(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        H, W = grid.shape
        if start == goal:
            return [start]

        q = deque([start])
        prev: dict[Tuple[int, int], Tuple[int, int]] = {}
        seen = {start}

        def neighbors(r: int, c: int):
            for dr, dc in ACTION_TO_DELTA.values():
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] != WALL:
                    yield (nr, nc)

        while q:
            cur = q.popleft()
            if cur == goal:
                break
            for nxt in neighbors(*cur):
                if nxt in seen:
                    continue
                seen.add(nxt)
                prev[nxt] = cur
                q.append(nxt)

        if goal not in seen:
            return None

        # reconstruct
        path = [goal]
        while path[-1] != start:
            path.append(prev[path[-1]])
        path.reverse()
        return path