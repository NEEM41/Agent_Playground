from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any
from typing import Optional
import numpy as np

# Must match your env action mapping:
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

class BaseAgent(ABC):
    @abstractmethod
    def act(self, obs: Dict[str, Any], mode: str = 'train') -> int:
        """Return an action int in {0,1,2,3}."""
        raise NotImplementedError

class RandomAgent(BaseAgent):
    def __init__(self, n_actions: int = 4, seed: Optional[int] = None):
        self.n_actions = int(n_actions)
        self.rng = np.random.default_rng(seed)

    def act(self, obs: Dict[str, Any], mode: str = 'train') -> int:
        # deterministic has no meaning here; always random
        return int(self.rng.integers(0, self.n_actions))

class GreedyGoalAgent(BaseAgent):
    """
    Moves toward the goal by Manhattan distance, ignoring obstacles.
    Great for sanity-checking env + rendering.
    """
    def act(self, obs: Dict[str, Any], mode: str = 'train') -> int:
        ar, ac = obs["agent_pos"]
        gr, gc = obs["goal_pos"]

        dr = gr - ar
        dc = gc - ac

        # Prefer moving in the dimension with larger absolute gap
        if abs(dr) >= abs(dc):
            if dr < 0:
                return UP
            elif dr > 0:
                return DOWN
        # else move horizontally
        if dc < 0:
            return LEFT
        elif dc > 0:
            return RIGHT

        # already at goal (should terminate)
        return UP