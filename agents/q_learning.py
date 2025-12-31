from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
from agents.base import BaseAgent

# Must match your env actions
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

@dataclass
class QLearningConfig:
    alpha: float = 0.25
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.995  # per episode

def sign(x: int) -> int:
    return (x > 0) - (x < 0)

class QLearningAgent(BaseAgent):
    """
    Tabular Q-learning using a local patch + coarse goal direction.
    State = (local_bytes, dr_sign, dc_sign)
    """
    def __init__(self, cfg: QLearningConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.eps = float(cfg.eps_start)
        self.Q: dict[Tuple[bytes, int, int], np.ndarray] = {}
        self.last_was_random = False # This is just to change soft exploration

    def _state(self, obs: Dict[str, Any]) -> Tuple[bytes, int, int]:
        local = obs["local"].astype(np.int8).tobytes()

        ar, ac = obs["agent_pos"]
        gr, gc = obs["goal_pos"]
        dr = sign(gr - ar)   # -1,0,1
        dc = sign(gc - ac)   # -1,0,1

        return (local, dr, dc)

    def _ensure(self, s: Tuple[bytes, int, int]) -> None:
        if s not in self.Q:
            self.Q[s] = np.zeros(4, dtype=np.float32)

    def _greedy_action(self, q: np.ndarray) -> int:
        m = float(q.max())
        best = np.flatnonzero(q == m)
        return int(self.rng.choice(best))

    def act(self, obs: Dict[str, Any], mode: str = "train") -> int:
        """
        mode:
          - "train": epsilon-greedy (exploration on)
          - "eval": greedy (no exploration)
          - "sample": soft sampling from Q (stochastic but informed)
        """
        s = self._state(obs)
        self._ensure(s)

        if mode == "eval":
            self.last_was_random = False
            return self._greedy_action(self.Q[s])

        if mode == "train":
            if self.rng.random() < self.eps:
                self.last_was_random = True
                return int(self.rng.integers(0, 4))
            self.last_was_random = False
            return self._greedy_action(self.Q[s])

        if mode == "sample":
            q = self.Q[s]
            z = np.exp(q - q.max())
            probs = z / z.sum()
            self.last_was_random = False
            return int(self.rng.choice(4, p=probs))

        raise ValueError(f"Unknown mode '{mode}'")

    def update(self, obs, action: int, reward: float, next_obs, done: bool) -> None:
        s = self._state(obs)
        ns = self._state(next_obs)
        self._ensure(s)
        self._ensure(ns)

        a = int(action)
        qsa = float(self.Q[s][a])
        target = reward if done else reward + self.cfg.gamma * float(np.max(self.Q[ns]))
        self.Q[s][a] = qsa + self.cfg.alpha * (target - qsa)

    def end_episode(self) -> None:
        self.eps = max(self.cfg.eps_end, self.eps * self.cfg.eps_decay)

    def save(self, path: str) -> None:
        # store keys as object array (works with bytes)
        keys = np.array(list(self.Q.keys()), dtype=object)
        vals = np.stack(list(self.Q.values()), axis=0) if self.Q else np.zeros((0, 4), dtype=np.float32)

        np.savez_compressed(
            path,
            Q_keys=keys,
            Q_values=vals,
            eps=np.array(self.eps, dtype=np.float32),
        )

    @classmethod
    def load(cls, path: str, cfg: QLearningConfig, seed: int = 0) -> "QLearningAgent":
        data = np.load(path, allow_pickle=True)
        agent = cls(cfg, seed=seed)
        agent.eps = float(data["eps"])

        keys = data["Q_keys"]
        vals = data["Q_values"]
        agent.Q = {tuple(k): vals[i] for i, k in enumerate(keys)}
        return agent

    # Example Usage of save, load method
    # agent.save("runs/q_policy_ep0500.npz")
    # agent2 = QLearningAgent.load("runs/q_policy_ep0500.npz", QLearningConfig())