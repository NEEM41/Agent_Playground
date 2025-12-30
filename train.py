import os
import csv
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

def eval_rollout(env, agent, seed=0, max_steps=200, sprite="pacman", color=(255, 215, 0), mode="eval"):
    """
    Runs one episode and returns frames + success + steps.
    mode="eval" recommended for stable snapshots.
    """
    obs = env.reset(seed=seed)
    frames = []
    done = False
    last_action = 3
    info = {}

    while not done and obs["step"] < max_steps:
        frames.append(
            env.render_rgb(
                cell_size=18,
                agent_color=color,
                sprite=sprite,
                facing=ACTION_TO_FACING.get(last_action, "right"),
            )
        )
        a = agent.act(obs, mode=mode)
        last_action = int(a)
        obs, r, done, info = env.step(a)

    frames.append(
        env.render_rgb(
            cell_size=18,
            agent_color=color,
            sprite=sprite,
            facing=ACTION_TO_FACING.get(last_action, "right"),
        )
    )

    return frames, bool(info.get("reached_goal", False)), int(obs["step"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="tiny_corridor")
    ap.add_argument("--run_name", default="qlearning_local7_v1")
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--eval_seed", type=int, default=0)
    ap.add_argument("--eval_mode", choices=["eval", "sample"], default="eval")
    ap.add_argument("--sprite", choices=["box", "pacman"], default="pacman")
    ap.add_argument("--save_policy", action="store_true", help="Whether to save policies every eval step")
    args = ap.parse_args()

    outdir = f"runs/{args.run_name}_{args.map}"
    videos_dir = f"{outdir}/videos"
    policies_dir = f"{outdir}/policies"
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(policies_dir, exist_ok=True)

    env = GridWorld(get_map(args.map), GridWorldConfig(max_steps=args.max_steps))
    agent = QLearningAgent(QLearningConfig(), seed=0)

    metrics_path = f"{outdir}/metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "train_return", "train_steps", "train_success", "epsilon"])

    for ep in range(1, args.episodes + 1):
        obs = env.reset()  # do NOT reseed each episode
        done = False
        total = 0.0
        last_info = {}

        while not done:
            a = agent.act(obs, mode="train")  # <-- exploration happens here
            next_obs, r, done, info = env.step(a)
            agent.update(obs, a, r, next_obs, done)
            obs = next_obs
            total += r
            last_info = info

        agent.end_episode()

        train_success = int(last_info.get("reached_goal", False))
        train_steps = int(obs["step"])

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, total, train_steps, train_success, agent.eps])

        if ep % args.eval_every == 0:
            if args.save_policy:
                policy_path = f"{policies_dir}/iter_{ep:04d}.npz"
                agent.save(policy_path)

            # Eval rollout video (separate env for clean eval)
            eval_env = GridWorld(get_map(args.map), GridWorldConfig(max_steps=args.max_steps))
            frames, eval_success, eval_steps = eval_rollout(
                eval_env,
                agent,
                seed=args.eval_seed,
                max_steps=args.max_steps,
                sprite=args.sprite,
                mode=args.eval_mode,
            )

            vid_path = f"{videos_dir}/iter_{ep:04d}.mp4"
            save_mp4(vid_path, frames, fps=12)

            print(
                f"[ep {ep}] train_return={total:.2f} train_success={train_success} "
                f"eps={agent.eps:.3f} | eval_success={eval_success} eval_steps={eval_steps} "
                f"| saved {vid_path}"
            )

    final_policy_path = f"{outdir}/policies/final_policy.npz"
    agent.save(final_policy_path)
    print("Saved final policy:", final_policy_path)

    print("Done. Metrics:", metrics_path)

if __name__ == "__main__":
    main()