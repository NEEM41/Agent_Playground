# train_multi_qlearning.py
# Usage examples:
#   python train_multi_qlearning.py --map tiny_corridor --run_name race_v1 --eval_every 200
#   python train_multi_qlearning.py --map trap_choice --run_name race_traps --terminate_on_trap --eval_every 200
#   python train_multi_qlearning.py --map tiny_corridor --run_name race_v1 --eval_every 200 --video_seconds 20 --fps 12

from __future__ import annotations

import os
import csv
import argparse
import imageio.v2 as imageio

from envs.maps import get_map
from envs.multi_gridworld import MultiGridWorld, MultiGridWorldConfig

from agents.bfs_agent import BFSAgent
from agents.base import RandomAgent, GreedyGoalAgent
from agents.q_learning import QLearningAgent, QLearningConfig
from agents.mimic_agent import MimicConfig, MimicAgent


def save_mp4(path, frames, fps=12):
    imageio.mimsave(path, frames, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p")


def eval_rollout(
    env: MultiGridWorld,
    agents,
    learner_index: int,
    seed: int,
    max_steps: int,
    sprite: str,
    colors,
    mode: str = "eval",
    cell_size: int = 18,
    stop_when_all_done: bool = True,
):
    """
    Run one evaluation episode and return frames + summary metrics.

    IMPORTANT: This should be used with a *video env config* where:
      terminate_on_any_goal=False
    so that the rollout continues after the first goal, and we can stop when
    everyone is done (or time cap is reached).
    """
    obs_list = env.reset(seed=seed)
    frames = []
    done = False
    t = 0

    reached_goal = [False] * len(agents)
    hit_trap = [False] * len(agents)
    collided = [0] * len(agents)

    while not done and t < max_steps:
        frames.append(env.render_rgb(cell_size=cell_size, agent_colors=colors, sprite=sprite))

        actions = []
        for i, agent in enumerate(agents):
            obs_i = obs_list[i]

            # If agent is already done, send a dummy action (env ignores done agents)
            if obs_i.get("done", False):
                actions.append(0)
                continue

            if i == learner_index:
                actions.append(agent.act(obs_i, mode=mode))
            else:
                actions.append(agent.act(obs_i, mode="eval"))

        obs_list, rewards, done, info = env.step(actions)

        # update stats
        for i in range(len(agents)):
            if info["reached_goal"][i]:
                reached_goal[i] = True
            if info["hit_trap"][i]:
                hit_trap[i] = True
            if info["collided"][i]:
                collided[i] += 1

        t += 1

        # For videos: stop when everyone is done (unless time cap hits first)
        if stop_when_all_done:
            if all(o.get("done", False) for o in obs_list):
                break

    frames.append(env.render_rgb(cell_size=cell_size, agent_colors=colors, sprite=sprite))

    return {
        "frames": frames,
        "steps": t,
        "reached_goal": reached_goal,
        "hit_trap": hit_trap,
        "collisions": collided,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--map", default="tiny_corridor")
    ap.add_argument("--run_name", default="multi_qlearn_v1")
    ap.add_argument("--episodes", type=int, default=1500)

    # TRAINING episode cap (race episodes are short, but still cap them)
    ap.add_argument("--max_steps", type=int, default=250)

    # Eval / video
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--eval_seed", type=int, default=0)
    ap.add_argument("--eval_mode", choices=["eval", "sample"], default="eval")

    ap.add_argument("--sprite", choices=["box", "pacman"], default="pacman")
    ap.add_argument("--cell_size", type=int, default=18)

    ap.add_argument("--save_policy", action="store_true", help="Save policy checkpoints every eval interval")
    ap.add_argument("--terminate_on_trap", action="store_true", help="If set, traps terminate an agent")

    # Video length control
    ap.add_argument("--video_seconds", type=float, default=20.0, help="Eval video max length in seconds")
    ap.add_argument("--fps", type=int, default=12, help="Video FPS (also used to compute video step cap)")

    # Collisions
    ap.add_argument("--block_on_collision", action="store_true", help="If set, enable collision blocking in env")

    args = ap.parse_args()

    outdir = f"runs/{args.run_name}_{args.map}"
    videos_dir = f"{outdir}/videos"
    policies_dir = f"{outdir}/policies"
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(policies_dir, exist_ok=True)

    # -------------------------
    # Agents
    # -------------------------
    bfs = BFSAgent()
    mimic = MimicAgent(MimicConfig(), seed=42)

    qcfg = QLearningConfig()
    learner = QLearningAgent(qcfg, seed=0)

    # NOTE: agent order must match colors and env indexing
    agents = [bfs, mimic, learner]
    learner_index = len(agents) - 1

    colors = [
        (60, 120, 220),   # bfs  - blue
        (60, 220, 120),   # mimic - green
        (255, 215, 0),    # learner - yellow
    ]

    # -------------------------
    # Two env configs:
    #   - train_cfg: race ends when ANYONE reaches goal (fast learning updates)
    #   - video_cfg: do NOT end on first goal; run until all agents are done or time cap
    # -------------------------
    train_cfg = MultiGridWorldConfig(
        max_steps=args.max_steps,
        terminate_on_any_goal=True,    # <-- race termination (TRAINING)
        terminate_on_goal=True,
        terminate_on_trap=args.terminate_on_trap,
        block_on_collision=bool(args.block_on_collision),
    )

    video_max_steps = int(args.fps * args.video_seconds)
    video_cfg = MultiGridWorldConfig(
        max_steps=video_max_steps,
        terminate_on_any_goal=False,   # <-- keep rolling after first goal (VIDEO)
        terminate_on_goal=True,
        terminate_on_trap=args.terminate_on_trap,
        block_on_collision=bool(args.block_on_collision),
    )

    env = MultiGridWorld(get_map(args.map), n_agents=len(agents), config=train_cfg)

    # -------------------------
    # Metrics file
    # -------------------------
    metrics_path = f"{outdir}/metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "episode",
                "train_return",
                "train_steps",
                "train_goal",
                "train_trap",
                "train_collisions",
                "epsilon",
            ]
        )

    # -------------------------
    # Training loop
    # -------------------------
    for ep in range(1, args.episodes + 1):
        obs_list = env.reset()  # do NOT reseed each episode
        done = False
        t = 0

        total_return = 0.0
        goal_hit = False
        trap_hit = False
        collisions = 0

        while not done and t < args.max_steps:
            actions = []
            learner_obs = None
            learner_action = None

            for i, agent in enumerate(agents):
                obs_i = obs_list[i]

                if obs_i.get("done", False):
                    actions.append(0)
                    continue

                if i == learner_index:
                    learner_obs = obs_i
                    a = agent.act(obs_i, mode="train")
                    learner_action = int(a)
                    actions.append(learner_action)
                else:
                    # Keep non-learners deterministic-ish during training
                    actions.append(agent.act(obs_i, mode="eval"))

            next_obs_list, rewards, done, info = env.step(actions)

            # learner update
            lr = float(rewards[learner_index])
            total_return += lr

            if learner_obs is not None and learner_action is not None:
                learner_next_obs = next_obs_list[learner_index]

                # For the learner, treat done if:
                # - it is done individually, OR
                # - episode ended (race ended / time cap)
                learner_done = bool(learner_next_obs.get("done", False)) #or bool(done)

                learner.update(learner_obs, learner_action, lr, learner_next_obs, learner_done)

            # stats
            if info["reached_goal"][learner_index]:
                goal_hit = True
            if info["hit_trap"][learner_index]:
                trap_hit = True
            if info["collided"][learner_index]:
                collisions += 1

            obs_list = next_obs_list
            t += 1

        learner.end_episode()

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    ep,
                    total_return,
                    t,
                    int(goal_hit),
                    int(trap_hit),
                    int(collisions),
                    learner.eps,
                ]
            )

        # -------------------------
        # Periodic eval video
        # -------------------------
        if ep % args.eval_every == 0:
            eval_env = MultiGridWorld(get_map(args.map), n_agents=len(agents), config=video_cfg)

            summary = eval_rollout(
                eval_env,
                agents,
                learner_index=learner_index,
                seed=args.eval_seed,
                max_steps=video_max_steps,
                sprite=args.sprite,
                colors=colors,
                mode=args.eval_mode,
                cell_size=args.cell_size,
                stop_when_all_done=True,
            )

            vid_path = f"{videos_dir}/iter_{ep:04d}.mp4"
            save_mp4(vid_path, summary["frames"], fps=args.fps)

            if args.save_policy:
                ckpt_path = f"{policies_dir}/iter_{ep:04d}.npz"
                learner.save(ckpt_path)

            print(
                f"[ep {ep}] learner_return={total_return:.2f} steps={t} "
                f"goal={int(goal_hit)} trap={int(trap_hit)} coll={collisions} eps={learner.eps:.3f} | "
                f"eval_steps={summary['steps']} eval_goal={int(summary['reached_goal'][learner_index])} "
                f"video_len={summary['steps']/max(1,args.fps):.1f}s -> {vid_path}"
            )

    # Save final policy
    final_policy_path = f"{outdir}/final_policy.npz"
    learner.save(final_policy_path)
    print("Saved final policy:", final_policy_path)
    print("Done. Metrics:", metrics_path)


if __name__ == "__main__":
    main()