import argparse
import os
import numpy as np
import imageio.v2 as imageio

from envs.maps import get_map
from envs.multi_gridworld import MultiGridWorld, MultiGridWorldConfig
from agents.base import RandomAgent, GreedyGoalAgent
from agents.bfs_agent import BFSAgent
from agents.q_learning import QLearningAgent, QLearningConfig
from agents.mimic_agnet import MimicConfig, MimicAgent

def save_mp4(path, frames, fps=12):
    imageio.mimsave(path, frames, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="tiny_corridor")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sprite", choices=["box", "pacman"], default="pacman")
    ap.add_argument("--out", default="assets/videos/multi_episode.mp4")

    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--max_seconds", type=float, default=20.0)
    ap.add_argument("--max_steps", type=int, default=None, help="Overrides max_seconds if provided")

    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    fps = int(args.fps)
    max_steps = int(args.max_steps) if args.max_steps is not None else int(np.ceil(args.max_seconds * fps))

    agents = [
        ("bfs", BFSAgent(), (60, 120, 220)),
        ("mimic", MimicAgent(MimicConfig()), (180, 80, 180)),
        # ("random", RandomAgent(seed=123), (120, 120, 120)),
        ("qlearn", QLearningAgent(QLearningConfig(), seed=0), (255, 215, 0)),
    ]
    agent_colors = [c for _, _, c in agents]

    env = MultiGridWorld(
        get_map(args.map),
        n_agents=len(agents),
        config=MultiGridWorldConfig(
            max_steps=max_steps,
            terminate_on_any_goal=False,  # <-- keep sim running even if someone reaches goal
            terminate_on_goal=True,       # agents reaching goal become done (stop moving)
            terminate_on_trap=False,
            block_on_collision=False,     # optional: keeps motion lively
        ),
    )

    obs_list = env.reset(seed=args.seed)

    frames = []
    done = False
    t = 0
    last_info = {}

    while not done and t < max_steps:
        frames.append(env.render_rgb(cell_size=18, agent_colors=agent_colors, sprite=args.sprite))

        actions = []
        for i, (name, agent, _color) in enumerate(agents):
            obs_i = obs_list[i]

            # If env marks this agent done, keep its last action (or any fixed action).
            # This avoids agents "twitching" after done.
            if obs_i.get("done", False):
                actions.append(3)  # RIGHT (arbitrary)
                continue

            mode = "eval"
            if name == "qlearn":
                mode = "sample"  # for motion; use "eval" once trained
            actions.append(agent.act(obs_i, mode=mode))

        obs_list, rewards, done, info = env.step(actions)
        last_info = info
        t += 1

    frames.append(env.render_rgb(cell_size=18, agent_colors=agent_colors, sprite=args.sprite))
    save_mp4(args.out, frames, fps=fps)

    print("Saved:", args.out)
    print("Steps:", t, "FPS:", fps, "Seconds:", t / fps)
    print("Info:", last_info)

if __name__ == "__main__":
    main()