"""Merge per-seed replay h5 files into one RecordEpisode-style trajectory.

Reads tmp_test/replay_dataset/seed*/trajectory.{h5,json} and writes
tmp_test/lerobot_src/trajectory.h5 with groups traj_0..traj_N plus a merged
trajectory.json (episode_id remapped, reset_kwargs/seeds preserved).

Usage:
    uv run python -m utils.merge_replays --src tmp_test/replay_dataset \
        --out tmp_test/lerobot_src
"""

import argparse
import json
from pathlib import Path

import h5py


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("tmp_test/replay_dataset"))
    parser.add_argument("--out", type=Path, default=Path("tmp_test/lerobot_src"))
    args = parser.parse_args()

    seeds = sorted(
        (int(p.name.replace("seed", "")) for p in args.src.iterdir() if p.is_dir())
    )
    args.out.mkdir(parents=True, exist_ok=True)
    out_h5 = args.out / "trajectory.h5"
    episodes = []
    env_info = None
    commit_info = None
    with h5py.File(out_h5, "w") as dst:
        for ep_id, seed in enumerate(seeds):
            src_dir = args.src / f"seed{seed}"
            with h5py.File(src_dir / "trajectory.h5", "r") as src:
                if env_info is None:
                    meta = json.loads((src_dir / "trajectory.json").read_text())
                    env_info = meta.get("env_info")
                    commit_info = meta.get("commit_info")
                src.copy(src[f"traj_0"], dst, name=f"traj_{ep_id}")
            meta = json.loads((src_dir / "trajectory.json").read_text())
            ep = meta["episodes"][0]
            ep["episode_id"] = ep_id
            episodes.append(ep)
    meta = dict(env_info=env_info, commit_info=commit_info, episodes=episodes)
    # merged metadata comes from the seed file that produced it; keep global
    # source_type/desc from the first seed's json
    first = json.loads(
        (args.src / f"seed{seeds[0]}" / "trajectory.json").read_text()
    )
    for key in ("source_type", "source_desc", "commit_info"):
        if key in first:
            meta[key] = first[key]
    (args.out / "trajectory.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"merged {len(seeds)} episodes -> {out_h5}")


if __name__ == "__main__":
    main()
