# train_multi_qlearning.py
# 3 learners: default / explorer / cautious (+ BFS + Mimic)
#
# Examples:
#   python train_multi_qlearning.py --map tiny_corridor --run_name threelearners --eval_every 200 --save_policy
#   python train_multi_qlearning.py --map trap_choice --run_name threelearners_traps --terminate_on_trap --eval_every 200
#
# Notes:
# - Training env is "race": terminate_on_any_goal=True (fast episodes for learning updates)
# - Video/eval env runs up to --video_seconds or until all agents are done: terminate_on_any_goal=False
# - IMPORTANT FIX: learners use agent-specific done ONLY (not global done) in Q updates.

from __future__ import annotations

import os
import csv
import argparse
import imageio.v2 as imageio

from envs.maps import get_map
from envs.multi_gridworld import MultiGridWorld, MultiGridWorldConfig

from agents.bfs_agent import BFSAgent
from agents.mimic_agent import MimicConfig, MimicAgent
from agents.q_learning import QLearningAgent, QLearningConfig


def save_mp4(path, frames, fps=12):
    imageio.mimsave(path, frames, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p")


def eval_rollout(
    env: MultiGridWorld,
    agents,
    learner_indices,
    seed: int,
    max_steps: int,
    sprite: str,
    colors,
    modes_by_index: dict[int, str],
    cell_size: int = 18,
    stop_when_all_done: bool = True,
):
    """
    Eval/video rollout:
      - env should have terminate_on_any_goal=False so it keeps running
      - each agent uses modes_by_index[i] if provided, else "eval"
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
            if obs_i.get("done", False):
                actions.append(0)
                continue

            mode = modes_by_index.get(i, "eval")
            actions.append(agent.act(obs_i, mode=mode))

        obs_list, rewards, done, info = env.step(actions)

        for i in range(len(agents)):
            if info["reached_goal"][i]:
                reached_goal[i] = True
            if info["hit_trap"][i]:
                hit_trap[i] = True
            if info["collided"][i]:
                collided[i] += 1

        t += 1

        if stop_when_all_done and all(o.get("done", False) for o in obs_list):
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
    ap.add_argument("--run_name", default="three_learners_v1")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--max_steps", type=int, default=250)

    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--eval_seed", type=int, default=0)
    ap.add_argument("--sprite", choices=["box", "pacman"], default="pacman")
    ap.add_argument("--cell_size", type=int, default=18)

    ap.add_argument("--save_policy", action="store_true")
    ap.add_argument("--terminate_on_trap", action="store_true")
    ap.add_argument("--block_on_collision", action="store_true")

    ap.add_argument("--video_seconds", type=float, default=20.0)
    ap.add_argument("--fps", type=int, default=12)

    # Eval control for learners: "eval" (greedy) or "sample" (soft)
    ap.add_argument("--eval_mode", choices=["eval", "sample"], default="eval")

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

    # Mimic: keep it lively but not too chaotic
    mimic = MimicAgent(MimicConfig(p_random=0.05, max_follow_dist=12), seed=42)

    # Learner 1: default
    q_default = QLearningAgent(
        QLearningConfig(alpha=0.25, gamma=0.99, eps_start=1.0, eps_end=0.05, eps_decay=0.995),
        seed=0,
    )

    # Learner 2: explorer (more exploration for longer)
    q_explorer = QLearningAgent(
        QLearningConfig(alpha=0.20, gamma=0.99, eps_start=1.0, eps_end=0.10, eps_decay=0.998),
        seed=1,
    )

    # Learner 3: cautious (less random, more "settle"; plus recommend map reward penalties for traps)
    q_cautious = QLearningAgent(
        QLearningConfig(alpha=0.30, gamma=0.99, eps_start=0.60, eps_end=0.02, eps_decay=0.992),
        seed=2,
    )

    # Order matters: env obs_list index == this list index
    # TODO: Remove
    # agents = [bfs, mimic, q_default, q_explorer, q_cautious]
    agents = [q_default, q_explorer, q_cautious]
    # learner_indices = [2, 3, 4]
    learner_indices = [0, 1, 2]

    # Colors (must match agents list length)
    colors = [
        (60, 120, 220),   # bfs      - blue
        (60, 220, 120),   # mimic    - green
        (255, 215, 0),    # q_def    - yellow
        (220, 140, 60),   # explorer - orange
        (180, 80, 180),   # cautious - purple
    ]

    # -------------------------
    # Env configs
    # -------------------------
    train_cfg = MultiGridWorldConfig(
        max_steps=args.max_steps,
        terminate_on_any_goal=True,    # TRAIN: race ends when anyone finishes
        terminate_on_goal=True,
        terminate_on_trap=args.terminate_on_trap,
        block_on_collision=bool(args.block_on_collision),
    )

    video_max_steps = int(args.video_seconds * args.fps)
    video_cfg = MultiGridWorldConfig(
        max_steps=video_max_steps,
        terminate_on_any_goal=False,   # VIDEO: keep going until all done or time cap
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
                "steps",
                "q_default_return",
                "q_default_goal",
                "q_default_trap",
                "q_default_coll",
                "q_default_eps",
                "q_explorer_return",
                "q_explorer_goal",
                "q_explorer_trap",
                "q_explorer_coll",
                "q_explorer_eps",
                "q_cautious_return",
                "q_cautious_goal",
                "q_cautious_trap",
                "q_cautious_coll",
                "q_cautious_eps",
            ]
        )

    # Helper: map index -> agent object
    # TODO: Remove
    # learners = {2: q_default, 3: q_explorer, 4: q_cautious}
    learners = {0: q_default, 1: q_explorer, 2: q_cautious}

    # -------------------------
    # Training loop
    # -------------------------
    for ep in range(1, args.episodes + 1):
        obs_list = env.reset()  # do NOT reseed each episode
        done = False
        t = 0

        # per-learner episode accumulators
        ep_return = {i: 0.0 for i in learner_indices}
        ep_goal = {i: False for i in learner_indices}
        ep_trap = {i: False for i in learner_indices}
        ep_coll = {i: 0 for i in learner_indices}

        while not done and t < args.max_steps:
            actions = []
            transitions = {}  # i -> (obs_i, action_i)

            for i, agent in enumerate(agents):
                obs_i = obs_list[i]

                if obs_i.get("done", False):
                    actions.append(0)
                    continue

                if i in learners:
                    a = agent.act(obs_i, mode="train")
                    a = int(a)
                    actions.append(a)
                    transitions[i] = (obs_i, a)
                else:
                    # keep non-learners stable during training
                    actions.append(agent.act(obs_i, mode="eval"))

            next_obs_list, rewards, done, info = env.step(actions)

            # update each learner independently (IMPORTANT: done is per-agent only)
            for i, (o, a) in transitions.items():
                r = float(rewards[i])
                ep_return[i] += r

                no = next_obs_list[i]
                agent_done = bool(no.get("done", False))
                learners[i].update(o, a, r, no, agent_done)

            # stats per learner
            for i in learner_indices:
                if info["reached_goal"][i]:
                    ep_goal[i] = True
                if info["hit_trap"][i]:
                    ep_trap[i] = True
                if info["collided"][i]:
                    ep_coll[i] += 1

            obs_list = next_obs_list
            t += 1

        # decay eps per learner
        for i in learner_indices:
            learners[i].end_episode()

        # write metrics row
        #TODO: Remove
        with open(metrics_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    ep,
                    t,
                    ep_return[0],
                    int(ep_goal[0]),
                    int(ep_trap[0]),
                    int(ep_coll[0]),
                    learners[0].eps,
                    ep_return[1],
                    int(ep_goal[1]),
                    int(ep_trap[1]),
                    int(ep_coll[1]),
                    learners[1].eps,
                    ep_return[0],
                    int(ep_goal[2]),
                    int(ep_trap[2]),
                    int(ep_coll[2]),
                    learners[2].eps,
                ]
            )

        # periodic eval video
        if ep % args.eval_every == 0:
            eval_env = MultiGridWorld(get_map(args.map), n_agents=len(agents), config=video_cfg)

            # everyone deterministic, except optionally let learners be "sample" for lively motion
            modes_by_index = {i: "eval" for i in range(len(agents))}
            for i in learner_indices:
                modes_by_index[i] = args.eval_mode  # "eval" or "sample"

            summary = eval_rollout(
                eval_env,
                agents,
                learner_indices=learner_indices,
                seed=args.eval_seed,
                max_steps=video_max_steps,
                sprite=args.sprite,
                colors=colors,
                modes_by_index=modes_by_index,
                cell_size=args.cell_size,
                stop_when_all_done=True,
            )

            vid_path = f"{videos_dir}/iter_{ep:04d}.mp4"
            save_mp4(vid_path, summary["frames"], fps=args.fps)

            if args.save_policy:
                # Save only latest checkpoints for each learner (overwrite)
                learners[2].save(f"{policies_dir}/q_default_latest.npz")
                learners[3].save(f"{policies_dir}/q_explorer_latest.npz")
                learners[4].save(f"{policies_dir}/q_cautious_latest.npz")

            print(
                f"[ep {ep}] steps={t} | "
                f"def_ret={ep_return[0]:.2f} eps={learners[0].eps:.3f} goal={int(ep_goal[0])} | "
                f"exp_ret={ep_return[1]:.2f} eps={learners[1].eps:.3f} goal={int(ep_goal[1])} | "
                f"cau_ret={ep_return[2]:.2f} eps={learners[2].eps:.3f} goal={int(ep_goal[2])} | "
                f"video={summary['steps']/max(1,args.fps):.1f}s -> {vid_path}"
            )

    # Save final policies
    learners[0].save(f"{outdir}/q_default_final.npz")
    learners[1].save(f"{outdir}/q_explorer_final.npz")
    learners[2].save(f"{outdir}/q_cautious_final.npz")
    print("Saved final policies to:", outdir)
    print("Done. Metrics:", metrics_path)


if __name__ == "__main__":
    main()