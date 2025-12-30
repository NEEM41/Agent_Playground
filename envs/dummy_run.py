import argparse
import os
import numpy as np
import imageio.v2 as imageio

from envs.maps import get_map
from envs.gridworld import GridWorld, GridWorldConfig
from agents.base import RandomAgent, GreedyGoalAgent
from agents.bfs_agent import BFSAgent


AGENTS = [
    ("random", RandomAgent(seed=123), (120, 120, 120)),
    ("greedy", GreedyGoalAgent(),     (60, 120, 220)),
    ("bfs",    BFSAgent(),            (180, 80, 180)),
]

ACTION_TO_FACING = {0: "up", 1: "down", 2: "left", 3: "right"}

def rollout(env: GridWorld, agent, seed: int | None, max_steps: int, color, sprite: str):
    obs = env.reset(seed=seed)
    frames = []
    done = False
    last_action = 3  # default "right"

    while not done and obs["step"] < max_steps:
        facing = ACTION_TO_FACING.get(last_action, "right")
        frames.append(env.render_rgb(cell_size=18, agent_color=color, sprite=sprite, facing=facing))

        a = agent.act(obs, deterministic=True)
        last_action = int(a)
        obs, r, done, info = env.step(a)

    facing = ACTION_TO_FACING.get(last_action, "right")
    frames.append(env.render_rgb(cell_size=18, agent_color=color, sprite=sprite, facing=facing))
    return frames

def save_mp4(path, frames, fps=10):
    imageio.mimsave(
        path,
        frames,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="tiny_corridor")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--separate", action="store_true",
                    help="If set, save one MP4 per agent. Otherwise save one tiled MP4.")
    ap.add_argument("--sprite", choices=["box", "pacman"], default="box",
                    help="Agent sprite style.")
    ap.add_argument("--pacman_all_yellow", action="store_true",
                    help="If set and sprite=pacman, override agent colors with Pac-Man yellow.")
    args = ap.parse_args()

    m = get_map(args.map)

    envs = [GridWorld(m, GridWorldConfig(max_steps=args.max_steps)) for _ in AGENTS]

    # Use consistent output folder (pick ONE; don't mix ../ and assets)
    outdir = "./../assets/videos"
    os.makedirs(outdir, exist_ok=True)

    if args.separate:
        for (name, agent, color), env in zip(AGENTS, envs):
            use_color = (255, 215, 0) if (args.sprite == "pacman" and args.pacman_all_yellow) else color
            frames = rollout(env, agent, seed=args.seed, max_steps=args.max_steps, color=use_color, sprite=args.sprite)
            outpath = f"{outdir}/{args.map}_{name}_{args.sprite}.mp4"
            save_mp4(outpath, frames)
            print("Saved:", outpath)
    else:
        # Tiled video (side-by-side)
        all_frames = []

        for env in envs:
            env.reset(seed=args.seed)

        last_actions = [3 for _ in AGENTS]  # per-panel facing

        for t in range(args.max_steps):
            panels = []
            any_done = True

            for i, ((name, agent, color), env) in enumerate(zip(AGENTS, envs)):
                obs = env._get_obs()
                facing = ACTION_TO_FACING.get(last_actions[i], "right")
                use_color = (255, 215, 0) if (args.sprite == "pacman" and args.pacman_all_yellow) else color

                panels.append(env.render_rgb(cell_size=18, agent_color=use_color, sprite=args.sprite, facing=facing))

                if not env.done:
                    a = agent.act(obs, deterministic=True)
                    last_actions[i] = int(a)
                    env.step(a)

                any_done = any_done and env.done

            all_frames.append(np.concatenate(panels, axis=1))
            if any_done:
                break

        outpath = f"{outdir}/{args.map}_tiled_{args.sprite}.mp4"
        save_mp4(outpath, all_frames)
        print("Saved:", outpath)

if __name__ == "__main__":
    main()