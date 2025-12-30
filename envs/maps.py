from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np

# Cell codes (internal numeric representation)
EMPTY = 0
WALL  = 1
TRAP  = 2
GOAL  = 3
START = 4  # you can treat this as EMPTY after parsing


@dataclass(frozen=True)
class MapSpec:
    name: str
    grid: np.ndarray              # shape (H, W), dtype=int
    start: Tuple[int, int]        # (r, c)
    goal: Tuple[int, int]         # (r, c)

    @property
    def shape(self) -> Tuple[int, int]:
        return int(self.grid.shape[0]), int(self.grid.shape[1])


SYMBOL_TO_CODE: Dict[str, int] = {
    " ": EMPTY,
    ".": EMPTY,
    "#": WALL,
    "T": TRAP,
    "G": GOAL,
    "S": START,
}

def parse_ascii_map(name: str, ascii_map: str) -> MapSpec:
    """
    Parse a multiline ASCII map into a numeric grid.
    Allowed symbols:
      # wall, . or space empty, S start, G goal, T trap
    """
    lines = [line.rstrip("\n") for line in ascii_map.strip("\n").splitlines()]
    if not lines:
        raise ValueError("ascii_map is empty")

    # Normalize width
    W = max(len(line) for line in lines)
    H = len(lines)

    grid = np.zeros((H, W), dtype=np.int32)
    start: Optional[Tuple[int, int]] = None
    goal: Optional[Tuple[int, int]] = None

    for r, line in enumerate(lines):
        # pad short lines with spaces
        line = line.ljust(W, " ")
        for c, ch in enumerate(line):
            if ch not in SYMBOL_TO_CODE:
                raise ValueError(f"Unknown symbol '{ch}' at (r={r}, c={c})")
            code = SYMBOL_TO_CODE[ch]
            if code == START:
                if start is not None:
                    raise ValueError("Multiple start positions 'S' found")
                start = (r, c)
                grid[r, c] = EMPTY  # treat start as empty cell
            elif code == GOAL:
                if goal is not None:
                    raise ValueError("Multiple goal positions 'G' found")
                goal = (r, c)
                grid[r, c] = GOAL
            else:
                grid[r, c] = code

    if start is None:
        raise ValueError("No start position 'S' found")
    if goal is None:
        raise ValueError("No goal position 'G' found")

    # Safety checks
    if grid[start] == WALL:
        raise ValueError("Start is on a wall")
    if grid[goal] == WALL:
        raise ValueError("Goal is on a wall")

    return MapSpec(name=name, grid=grid, start=start, goal=goal)


# Example maps you can start with
MAPS: Dict[str, str] = {
    "tiny_corridor": r"""
####################
#S....#............#
#.#######.######...#
#.............#....#
#..######.#####.##.#
#..#....#.....#....#
#..#..T.#..T..#..G.#
####################
""",
}

def get_map(name: str) -> MapSpec:
    if name not in MAPS:
        raise KeyError(f"Unknown map '{name}'. Available: {list(MAPS.keys())}")
    return parse_ascii_map(name, MAPS[name])