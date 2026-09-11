"""Replay state-only ManiSkill trajectories with RGB rendering (post-pass).

Reads a state-mode RecordEpisode trajectory (flattened obs, env_states) and
reproduces it on another robot (e.g. upstream `fetch`) by setting env_states
step by step; renders RGB sensor observations every `--stride`-th step and
writes a NEW trajectory.h5 with dict-style obs (agent qpos/qvel + sensor rgb).
The planner and the source env are never touched.

Usage:
    uv run python -m utils.replay_rgb <source_run_dir> --output-dir <dir> \
        --robot fetch --stride 10 --fps 10
"""

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import torch

import gymnasium as gym

import planners.myrobocasa_takeitback_tray_planner  # noqa: F401 registers envs
from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.utils import common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--robot", default="fetch")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--episode", type=int, default=0,
                        help="trajectory index inside the source h5")
    args = parser.parse_args()

    src_h5_path = args.source_run_dir / "trajectory.h5"
    src_json = json.loads((args.source_run_dir / "trajectory.json").read_text())
    env_info = src_json["env_info"]
    episode = src_json["episodes"][args.episode]
    seed = episode.get("episode_seed")
    control_mode = episode.get("control_mode", env_info["env_kwargs"]["control_mode"])
    src_robot = env_info["env_kwargs"]["robot_uids"]

    env = gym.make(
        env_info["env_id"],
        num_envs=1,
        obs_mode="rgb",
        render_mode="rgb_array",
        control_mode=control_mode,
        robot_uids=args.robot,
        sim_config=env_info["env_kwargs"].get("sim_config"),
    )
    base_env = env.unwrapped
    obs, _ = env.reset(seed=seed, options={"reconfigure": True})

    with h5py.File(src_h5_path, "r") as src:
        traj = src[f"traj_{args.episode}"]
        states = trajectory_utils.dict_to_list_of_dicts(traj["env_states"])
        # env_states key articulations by the SOURCE robot's name; remap to the
        # replay robot (joint layout is identical across fetch variants)
        src_robot = src_json["env_info"]["env_kwargs"]["robot_uids"]
        if src_robot != args.robot:
            for s in states:
                art = s.get("articulations", {})
                if src_robot in art and src_robot != args.robot:
                    art[args.robot] = art.pop(src_robot)
        actions = traj["actions"][:]
        arrays = {
            name: traj[name][:]
            for name in ("rewards", "success", "terminated", "truncated")
        }
        rewards = arrays["rewards"]
        success = arrays["success"]
        terminated = arrays["terminated"]
        truncated = arrays["truncated"]
        src_flat_obs = traj["obs"][:] if traj["obs"].ndim == 2 else None
    T = len(actions)

    out_h5 = args.output_dir / "trajectory.h5"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_h5_path_tmp := args.output_dir / "trajectory.h5.tmp", "w") as dst:
        g = dst.create_group("traj_0", track_order=True)
        agent = g.create_group("obs/agent")
        qpos_ds = agent.create_dataset("qpos", shape=(0, 15), maxshape=(None, 15), dtype=np.float32)
        qvel_ds = agent.create_dataset("qvel", shape=(0, 15), maxshape=(None, 15), dtype=np.float32)
        actions_ds = g.create_dataset("actions", data=actions[::args.stride], dtype=np.float32)
        for name, arr in (
            ("rewards", rewards), ("success", success),
            ("terminated", terminated), ("truncated", truncated),
        ):
            g.create_dataset(name, data=arr[::args.stride], dtype=arr.dtype)
        # terminal flags: the source flips success on its final step, which is
        # not necessarily stride-aligned; carry the terminal values over
        for name in ("success", "terminated", "truncated"):
            g[name][-1] = bool(arrays[name][-1])
        sensor_names = [k for k in obs["sensor_data"].keys()]
        cam_groups = {}
        for cam in sensor_names:
            grp = g.create_group(f"obs/sensor_data/{cam}")
            n_rec = (T + 1 + args.stride - 1) // args.stride
            cam_groups[cam] = grp.create_dataset(
                "rgb",
                shape=(0, *obs["sensor_data"][cam]["rgb"].shape[-3:]),
                maxshape=(None, *obs["sensor_data"][cam]["rgb"].shape[-3:]),
                dtype=np.uint8,
                compression="gzip",
                compression_opts=5,
            )

        max_qerr = 0.0
        n_rec = 0
        robot = base_env.agent.robot
        src_qpos = src_flat_obs[:, :15] if src_flat_obs is not None else None
        # only recorded steps matter: set the state and render exactly there
        for t in range(0, T + 1, args.stride):
            base_env.set_state_dict(states[t])
            if src_qpos is not None:
                # cheap restore check: read articulation qpos directly (no render)
                qerr = float(
                    np.abs(np.asarray(robot.get_qpos()[0].cpu()) - src_qpos[t]).max()
                )
                max_qerr = max(max_qerr, qerr)
            obs_t = base_env.get_obs()
            qpos = common.to_numpy(obs_t["agent"]["qpos"])[0].astype(np.float32)
            qvel = common.to_numpy(obs_t["agent"]["qvel"])[0].astype(np.float32)
            for ds, row in ((qpos_ds, qpos), (qvel_ds, qvel)):
                ds.resize(ds.shape[0] + 1, axis=0)
                ds[-1] = row
            for cam, ds in cam_groups.items():
                frame = common.to_numpy(obs_t["sensor_data"][cam]["rgb"][0]).astype(np.uint8)
                ds.resize(ds.shape[0] + 1, axis=0)
                ds[-1] = frame
            n_rec += 1
        print(f"[replay] recorded {n_rec} frames, max qpos restore err {max_qerr:.2e}")
    out_h5_path_tmp.replace(out_h5)

    meta = dict(src_json)
    meta["episodes"] = [dict(episode, stride=args.stride, replay_robot=args.robot,
                             obs="rgb", elapsed_steps=(T - 1) // args.stride + 1)]
    meta["source_desc"] = (f"state-only demo replayed on {args.robot}; "
                           f"rgb rendered from env_states every {args.stride} steps")
    (args.output_dir / "trajectory.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    env.close()


if __name__ == "__main__":
    main()
