import os
import argparse
import imageio.v2 as imageio

from envs.maps import get_map
from envs.gridworld import GridWorld, GridWorldConfig
from agents.q_learning import QLearningAgent, QLearningConfig

ACTION_TO_FACING = {0: "up", 1: "down", 2: "left", 3: "right"}

def save_mp4(path, frames, fps=12):
    imageio.mimsave(
        path,
        frames,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
    )

def rollout_video(env, agent, seed=0, max_steps=200, sprite="pacman", color=(255, 215, 0), mode="eval", cell_size=18):
    obs = env.reset(seed=seed)
    frames = []
    done = False
    last_action = 3
    last_info = {}

    while not done and obs["step"] < max_steps:
        frames.append(
            env.render_rgb(
                cell_size=cell_size,
                agent_color=color,
                sprite=sprite,
                facing=ACTION_TO_FACING.get(last_action, "right"),
            )
        )
        a = agent.act(obs, mode=mode)
        last_action = int(a)
        obs, r, done, info = env.step(a)
        last_info = info

    frames.append(
        env.render_rgb(
            cell_size=cell_size,
            agent_color=color,
            sprite=sprite,
            facing=ACTION_TO_FACING.get(last_action, "right"),
        )
    )

    success = bool(last_info.get("reached_goal", False))
    steps = int(obs["step"])
    return frames, success, steps, last_info

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=str, required=True, help="Path to .npz policy (e.g., runs/.../final_policy.npz)")
    ap.add_argument("--map", default="tiny_corridor")
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["eval", "sample"], default="eval")
    ap.add_argument("--sprite", choices=["box", "pacman"], default="pacman")
    ap.add_argument("--cell_size", type=int, default=18)
    ap.add_argument("--out", type=str, default=None, help="Output mp4 path. If omitted, writes to runs/infer/")
    args = ap.parse_args()

    # Load policy
    cfg = QLearningConfig()
    agent = QLearningAgent.load(args.policy, cfg, seed=0)

    # Build env
    env = GridWorld(get_map(args.map), GridWorldConfig(max_steps=args.max_steps))

    # Rollout + render
    frames, success, steps, info = rollout_video(
        env,
        agent,
        seed=args.seed,
        max_steps=args.max_steps,
        sprite=args.sprite,
        mode=args.mode,
        cell_size=args.cell_size,
    )

    # Output path
    if args.out is None:
        outdir = "runs/infer"
        os.makedirs(outdir, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.policy))[0]
        args.out = f"{outdir}/{base}__{args.map}__{args.mode}__seed{args.seed}.mp4"
    else:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    save_mp4(args.out, frames, fps=12)

    print(f"Saved: {args.out}")
    print(f"Result: success={success} steps={steps} info={info}")

if __name__ == "__main__":
    main()