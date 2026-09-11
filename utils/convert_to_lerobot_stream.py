#!/usr/bin/env python3
"""Streaming ManiSkill HDF5 -> LeRobot v3.0 converter.

Same output layout as mani_skill.trajectory.convert_to_lerobot, but episodes
are processed one at a time (constant RAM): each episode's RGB frames are
encoded to video immediately and only per-episode dataframes plus running
statistics are kept, so 1000-episode merges cannot OOM.

Usage:
    uv run python -m utils.convert_to_lerobot_stream --traj-path M.h5 \
        --output-dir DIR --task-name "..." --fps 10 --image-size 128x128
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tyro

from mani_skill.trajectory.convert_to_lerobot import (
    create_directory_structure,
    create_video_from_frames,
    parse_image_size,
    process_episode,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Args:
    traj_path: str
    output_dir: str
    fps: int = 10
    task_name: str = "Unknown task"
    chunks_size: int = 100
    image_size: str = "128x128"
    robot_type: str = "fetch"


def iter_episodes(h5_file: Path):
    with h5py.File(h5_file, "r") as f:
        keys = sorted(k for k in f.keys() if k.startswith("traj_"))
        first = f[keys[0]]
        rgb_cameras = []
        if "obs/sensor_data" in first:
            for cam in first["obs/sensor_data"]:
                if "rgb" in first[f"obs/sensor_data/{cam}"]:
                    rgb_cameras.append(cam)
        state_dim = None
        if "obs/agent" in first and "qpos" in first["obs/agent"]:
            state_dim = first["obs/agent"]["qpos"].shape[1]
        for key in keys:
            traj = f[key]
            ep = {"actions": traj["actions"][:]}
            for cam in rgb_cameras:
                ep[f"rgb_{cam}"] = traj[f"obs/sensor_data/{cam}/rgb"][:]
            if state_dim:
                ep["robot_state"] = traj["obs/agent/qpos"][: len(ep["actions"])]
            yield ep, rgb_cameras, state_dim


class Moments:
    """Running mean/std/min/max accumulator."""

    def __init__(self):
        self.n = 0
        self.s = 0.0
        self.s2 = 0.0
        self.min = None
        self.max = None

    def add(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size == 0:
            return
        self.n += int(x.size)
        self.s += float(x.sum())
        self.s2 += float(np.square(x).sum())
        lo, hi = float(x.min()), float(x.max())
        self.min = lo if self.min is None else min(self.min, lo)
        self.max = hi if self.max is None else max(self.max, hi)

    def summary(self):
        mean = self.s / self.n
        std = float(np.sqrt(max(self.s2 / self.n - mean * mean, 0.0)))
        return {"mean": mean, "std": std, "min": self.min, "max": self.max, "n": self.n}


def scalar_stats_float(s, n):
    return {"mean": [s["mean"]], "std": [s["std"]],
            "max": [float(s["max"])], "min": [float(s["min"])], "count": [n]}


def scalar_stats_int(s, n):
    return {"mean": [s["mean"]], "std": [s["std"]],
            "max": [int(s["max"])], "min": [int(s["min"])], "count": [n]}


def vec_stats(moms, dim):
    return {
        "mean": [moms[i].summary()["mean"] for i in range(dim)],
        "std": [moms[i].summary()["std"] for i in range(dim)],
        "max": [moms[i].summary()["max"] for i in range(dim)],
        "min": [moms[i].summary()["min"] for i in range(dim)],
        "count": [moms[0].summary()["n"] // dim],
    }


def main(args: Args):
    input_path = Path(args.traj_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    base_path = Path(args.output_dir)
    image_width, image_height = parse_image_size(args.image_size)

    it = iter_episodes(input_path)
    first_ep, rgb_cameras, state_dim = next(it)
    action_dim = first_ep["actions"].shape[1]

    a_mom = [Moments() for _ in range(action_dim)]
    s_mom = [Moments() for _ in range(state_dim)] if state_dim else []
    cam_mom = {cam: [Moments() for _ in range(3)] for cam in rgb_cameras}
    ts_mom, fi_mom, ei_mom, ix_mom, ti_mom = Moments(), Moments(), Moments(), Moments(), Moments()

    episode_lengths = []
    episode_states = []
    dfs = []
    global_index = 0
    total_frames = 0
    ep_idx = -1

    def handle(ep_data):
        nonlocal ep_idx, global_index, total_frames
        ep_idx += 1
        df = process_episode(
            ep_data, ep_idx, state_dim is not None, args.fps,
            task_index=0, task_name=args.task_name,
        )
        length = len(df)
        df["index"] = range(global_index, global_index + length)
        global_index += length
        total_frames += length

        chunk_idx = ep_idx // args.chunks_size
        for cam in rgb_cameras:
            video_path = (base_path / "videos" / f"observation.images.{cam}" /
                          f"chunk-{chunk_idx:03d}" / f"file-{ep_idx:03d}.mp4")
            create_video_from_frames(
                ep_data[f"rgb_{cam}"], video_path, args.fps,
                image_width, image_height)
            sample = ep_data[f"rgb_{cam}"][:: max(1, length // 20)]
            pix = (sample.astype(np.float32) / 255.0).reshape(-1, 3)
            for c in range(3):
                cam_mom[cam][c].add(pix[:, c])

        actions = ep_data["actions"].astype(np.float64)
        for i in range(action_dim):
            a_mom[i].add(actions[:, i])
        state = None
        if state_dim and "robot_state" in ep_data:
            state = ep_data["robot_state"].astype(np.float64)
            for i in range(state_dim):
                s_mom[i].add(state[:, i])
        ts_mom.add(df["timestamp"].values)
        fi_mom.add(df["frame_index"].values)
        ei_mom.add(df["episode_index"].values)
        ix_mom.add(df["index"].values)
        ti_mom.add(df["task_index"].values)

        episode_lengths.append(length)
        dfs.append(df)
        episode_states.append({
            "actions": {"min": actions.min(0).tolist(), "max": actions.max(0).tolist(),
                        "mean": actions.mean(0).tolist(), "std": actions.std(0).tolist(),
                        "count": [length]},
            "state": ({"min": state.min(0).tolist(), "max": state.max(0).tolist(),
                       "mean": state.mean(0).tolist(), "std": state.std(0).tolist(),
                       "count": [length]} if state is not None else None),
        })

    # pre-create directories for the full episode count
    n_probe = sum(1 for _ in h5py.File(input_path, "r").keys())
    num_chunks = (n_probe + args.chunks_size - 1) // args.chunks_size
    (base_path / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    for c in range(num_chunks):
        (base_path / "data" / f"chunk-{c:03d}").mkdir(parents=True, exist_ok=True)
        for cam in rgb_cameras:
            (base_path / "videos" / f"observation.images.{cam}" /
             f"chunk-{c:03d}").mkdir(parents=True, exist_ok=True)

    # process the first episode, then the rest
    handle(first_ep)
    for ep_data, _cams, _sd in it:
        handle(ep_data)
        if (ep_idx + 1) % 50 == 0:
            logger.info(f"episodes processed: {ep_idx + 1}/{n_probe}")

    logger.info(f"Processed {n_probe} episodes; writing parquets/meta")

    for chunk_idx in range(num_chunks):
        start, end = chunk_idx * args.chunks_size, min(
            (chunk_idx + 1) * args.chunks_size, ep_idx + 1)
        combined = pd.concat(dfs[start:end], ignore_index=True)
        combined["task"] = combined["task"].astype("string")
        fields = []
        for col in combined.columns:
            if col == "task":
                fields.append(pa.field("task", pa.string()))
            elif col in ("action", "observation.state"):
                fields.append(pa.field(col, pa.list_(pa.float32())))
            elif col == "timestamp":
                fields.append(pa.field(col, pa.float32()))
            elif col in ("frame_index", "episode_index", "index", "task_index"):
                fields.append(pa.field(col, pa.int64()))
        pq.write_table(
            pa.Table.from_pandas(combined, schema=pa.schema(fields)),
            base_path / "data" / f"chunk-{chunk_idx:03d}" / "file-000.parquet")

    ep_rows = []
    for i, st in enumerate(episode_states):
        chunk_idx = chunk_of_i(i, args.chunks_size)
        em = {
            "episode_index": i, "data/chunk_index": chunk_idx, "data/file_index": 0,
            "dataset_from_index": sum(episode_lengths[:i]),
            "dataset_to_index": sum(episode_lengths[: i + 1]),
            "tasks": [args.task_name], "length": episode_lengths[i],
            "meta/episodes/chunk_index": chunk_idx, "meta/episodes/file_index": 0,
            "stats/action/min": st["actions"]["min"],
            "stats/action/max": st["actions"]["max"],
            "stats/action/mean": st["actions"]["mean"],
            "stats/action/std": st["actions"]["std"],
            "stats/action/count": st["actions"]["count"],
        }
        if st["state"]:
            em.update({
                "stats/observation.state/min": st["state"]["min"],
                "stats/observation.state/max": st["state"]["max"],
                "stats/observation.state/mean": st["state"]["mean"],
                "stats/observation.state/std": st["state"]["std"],
                "stats/observation.state/count": st["state"]["count"],
            })
        for cam in rgb_cameras:
            p = f"videos/observation.images.{cam}"
            em[f"{p}/chunk_index"] = chunk_idx
            em[f"{p}/file_index"] = i
        ep_rows.append(em)
    pd.DataFrame(ep_rows).to_parquet(
        base_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)

    pd.DataFrame({"task_index": [0]}, index=[args.task_name]).to_parquet(
        base_path / "meta" / "tasks.parquet", index=True)

    features = {
        "action": {"dtype": "float32", "shape": [action_dim],
                   "names": [f"action_{i}" for i in range(action_dim)],
                   "fps": float(args.fps)},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None, "fps": float(args.fps)},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None, "fps": float(args.fps)},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None, "fps": float(args.fps)},
        "index": {"dtype": "int64", "shape": [1], "names": None, "fps": float(args.fps)},
        "task_index": {"dtype": "int64", "shape": [1], "names": None, "fps": float(args.fps)},
        "task": {"dtype": "string", "shape": [1], "names": None, "fps": float(args.fps)},
    }
    if state_dim:
        features["observation.state"] = {
            "dtype": "float32", "shape": [state_dim],
            "names": [f"joint_{i}" for i in range(state_dim)], "fps": float(args.fps)}
    for cam in rgb_cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": [image_height, image_width, 3],
            "names": ["height", "width", "channels"],
            "info": {"video.fps": float(args.fps), "video.height": image_height,
                     "video.width": image_width, "video.channels": 3,
                     "video.codec": "mp4v", "video.pix_fmt": "yuv420p",
                     "video.is_depth_map": False, "has_audio": False},
        }

    data_mb = int(sum(f.stat().st_size for f in (base_path / "data").rglob("*.parquet")) / 1048576)
    info = {
        "codebase_version": "v3.0", "robot_type": args.robot_type,
        "total_episodes": ep_idx + 1, "total_frames": total_frames,
        "total_tasks": 1, "total_videos": (ep_idx + 1) * len(rgb_cameras),
        "total_chunks": num_chunks, "chunks_size": args.chunks_size,
        "fps": args.fps, "data_files_size_in_mb": data_mb,
        "splits": {"train": f"0:{ep_idx + 1}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    (base_path / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    stats = {"action": vec_stats(a_mom, action_dim)}
    if state_dim:
        stats["observation.state"] = vec_stats(s_mom, state_dim)
    for cam, chs in cam_mom.items():
        stats[f"observation.images.{cam}"] = {
            "mean": [[ch.summary()["mean"]] for ch in chs],
            "std": [[ch.summary()["std"]] for ch in chs],
            "max": [[ch.summary()["max"]] for ch in chs],
            "min": [[ch.summary()["min"]] for ch in chs],
            "count": [[total_frames]],
        }
    stats["timestamp"] = scalar_stats_float(ts_mom.summary(), total_frames)
    stats["frame_index"] = scalar_stats_int(fi_mom.summary(), ep_idx + 1)
    stats["episode_index"] = scalar_stats_int(ei_mom.summary(), ep_idx + 1)
    stats["index"] = scalar_stats_int(ix_mom.summary(), total_frames)
    stats["task_index"] = scalar_stats_int(ti_mom.summary(), ep_idx + 1)
    (base_path / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))

    logger.info(f"Conversion completed: {ep_idx + 1} episodes, {total_frames} frames, "
                f"{num_chunks} chunks -> {base_path}")
    return 0


def chunk_of_i(i, chunks_size):
    return i // chunks_size


if __name__ == "__main__":
    import sys
    sys.exit(main(tyro.cli(Args)))
