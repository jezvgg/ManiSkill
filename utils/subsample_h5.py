"""Stride subsample of a ManiSkill RecordEpisode trajectory.h5 (post-pass).

Writes a new trajectory.h5 keeping every K-th frame of actions/obs/rewards/
flags, so a 100 Hz episode becomes a 100/K Hz dataset without touching the
planner or the env. The original file is replaced atomically.

Usage:
    uv run python -m utils.subsample_h5 <run_dir>/trajectory.h5 --stride 10
"""

import argparse
import json
from pathlib import Path

import h5py


def _copy_strided(src: h5py.Group, dst: h5py.Group, stride: int) -> None:
    for key in src.keys():
        obj = src[key]
        if isinstance(obj, h5py.Group):
            sub = dst.create_group(key, track_order=True)
            _copy_strided(obj, sub, stride)
            continue
        if obj.ndim == 0 or obj.shape[0] <= 1 or key == "env_episode_ptr":
            dst.create_dataset(key, data=obj[...], dtype=obj.dtype)
            continue
        dst.create_dataset(
            key,
            data=obj[::stride],
            dtype=obj.dtype,
            compression="gzip" if key == "rgb" else None,
            compression_opts=5 if key == "rgb" else None,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traj_path", type=Path)
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()

    tmp_path = args.traj_path.with_suffix(".h5.strided")
    # actions/observations use a leading dummy frame; slicing from index 0 with
    # the same stride keeps the (obs_t, action_t) pairing rules intact.
    with h5py.File(args.traj_path, "r") as src, h5py.File(tmp_path, "w") as dst:
        for key in src.keys():
            obj = src[key]
            if isinstance(obj, h5py.Group):
                sub = dst.create_group(key, track_order=True)
                _copy_strided(obj, sub, args.stride)
            else:
                dst.create_dataset(key, data=obj[...], dtype=obj.dtype)
    tmp_path.replace(args.traj_path)

    json_path = args.traj_path.with_suffix(".json")
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        for ep in data.get("episodes", []):
            if "elapsed_steps" in ep:
                ep["elapsed_steps"] = (ep["elapsed_steps"] - 1) // args.stride + 1
            ep["stride"] = args.stride
        json_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
