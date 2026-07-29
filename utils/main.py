import argparse
from importlib import import_module
from pathlib import Path
import random

import gymnasium as gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

import mani_skill.envs.tasks


def load_planner(name: str):
    mod = name.replace("/", ".").removesuffix(".py").strip(".")
    if not mod.startswith("planners."):
        mod = f"planners.{mod}"
    return import_module(mod).planning


def main():
    parser = argparse.ArgumentParser(description="ManiSkill Planner Evaluator")
    parser.add_argument("--scene", "-s", default="MyRoboCasa-v1", help="Scene ID")
    parser.add_argument("--planner", "-p", default="myrobocasa_planner", help="Planner name or path")
    parser.add_argument("--num-episodes", "-n", type=int, default=100, help="Number of episodes")
    parser.add_argument("--start-seed", type=int, default=100, help="Start seed")
    parser.add_argument("--render-mode", "-r", default="rgb_array", choices=["rgb_array", "human"], help="Render mode")
    parser.add_argument("--output-video", "-o", default="videos/grid_validation.mp4", help="Video output path")
    parser.add_argument("--no-video", action="store_true", help="Disable video generation")
    parser.add_argument("--show", action="store_true", help="Show interactive window")
    args = parser.parse_args()

    planning_fn = load_planner(args.planner)
    env = gym.make(args.scene, render_mode=args.render_mode, obs_mode="rgb", robot_uids="ds_fetch", control_mode="pd_joint_pos")

    all_frames, successes = [], []

    print(f"=== EVALUATING {args.planner} ON {args.scene} ({args.num_episodes} EPISODES) ===")

    for i in range(args.num_episodes):
        seed = args.start_seed + i
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        record = not args.no_video and i < 16
        frames = []
        if record:
            r, s = env.reset, env.step
            env.reset = lambda *a, **k: (obs := r(*a, **k), frames.append(obs[0]["sensor_data"]["base_camera"]["rgb"][0].cpu().numpy()))[0]
            env.step = lambda act: (obs := s(act), frames.append(obs[0]["sensor_data"]["base_camera"]["rgb"][0].cpu().numpy()))[0]

        try:
            ok = bool(planning_fn(env, seed))
        except Exception as e:
            print(f"Ep {i+1} seed {seed} error: {e}")
            ok = False

        if record:
            env.reset, env.step = r, s
            all_frames.append(frames)

        successes.append(ok)
        print(f"Episode {i+1}/{args.num_episodes} (Seed {seed}): {'SUCCESS' if ok else 'FAILED'}")

    sr = sum(successes) / len(successes) * 100
    print(f"\n=== RESULTS ===\nSuccess Rate: {sr:.2f}% ({sum(successes)}/{len(successes)})\n")

    if all_frames:
        max_len = max(len(f) for f in all_frames)
        for f in all_frames:
            f.extend([f[-1]] * (max_len - len(f)))

        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        imgs = []
        for idx, ax in enumerate(axes.flat):
            ax.axis("off")
            if idx < len(all_frames):
                st = "SUCCESS" if successes[idx] else "FAILED"
                ax.set_title(f"Ep {idx+1}: {st}", color="green" if successes[idx] else "red", fontsize=9)
                imgs.append(ax.imshow(all_frames[idx][0]))
            else:
                imgs.append(ax.imshow(np.zeros_like(all_frames[0][0])))

        ani = animation.FuncAnimation(
            fig,
            lambda t: [im.set_data(all_frames[k][t]) for k, im in enumerate(imgs[:len(all_frames)])],
            frames=max_len,
            interval=40,
        )
        Path(args.output_video).parent.mkdir(parents=True, exist_ok=True)
        ani.save(args.output_video, writer="ffmpeg", fps=25)
        print(f"Video saved to {args.output_video}")
        if args.show:
            plt.show()

    env.close()


if __name__ == "__main__":
    main()
