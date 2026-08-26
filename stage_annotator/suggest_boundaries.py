#!/usr/bin/env python3
"""Semi-automatic stage-boundary candidates for one episode (plans.md #2:
"semi-automatic from gripper open/close events + arm-velocity minima").
Read-only against the dataset -- never writes anything there. Output is a
plain list of candidate frame indices for annotate.py to pre-populate; you
confirm/adjust/name them interactively, this just saves the initial pass.

    python suggest_boundaries.py --dataset_root <lerobot3.0 root> --episode 0

State layout (robot_io_spec.md, same 65-dim order everywhere in this repo):
  0:7 left arm | 7:29 left hand | 29:36 right arm | 36:58 right hand | 58:65 head/torso
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pyarrow.parquet as pq

LEFT_ARM, LEFT_HAND = slice(0, 7), slice(7, 29)
RIGHT_ARM, RIGHT_HAND = slice(29, 36), slice(36, 58)


def episode_data(dataset_root: pathlib.Path, episode: int) -> np.ndarray:
    """observation.state for one episode, [T,65], via the episodes index (same
    chunk/file/from-to-index pattern used in keypoint_expt for video lookup)."""
    ep_table = pq.read_table(dataset_root / "meta/episodes/chunk-000/file-000.parquet",
                              columns=["episode_index", "data/chunk_index", "data/file_index",
                                       "dataset_from_index", "dataset_to_index"])
    d = ep_table.to_pydict()
    i = d["episode_index"].index(episode)
    chunk, file_ = d["data/chunk_index"][i], d["data/file_index"][i]
    lo, hi = d["dataset_from_index"][i], d["dataset_to_index"][i]

    data_path = dataset_root / "data" / f"chunk-{chunk:03d}" / f"file-{file_:03d}.parquet"
    table = pq.read_table(data_path, columns=["observation.state", "index"])
    idx = np.array(table["index"].to_pylist())
    rows = np.where((idx >= lo) & (idx < hi))[0]
    state = np.stack(table["observation.state"].to_pylist())[rows]
    return state.astype(np.float32)  # [T,65]


def suggest(state: np.ndarray, fps: int = 30, min_gap_s: float = 1.0) -> list[dict]:
    """Two signal families, merged and de-duplicated within min_gap_s:
    - arm velocity local minima (pauses -- likely stage transitions)
    - hand aggregate value threshold crossings (grasp/release events)
    """
    arm = np.concatenate([state[:, LEFT_ARM], state[:, RIGHT_ARM]], axis=1)
    vel = np.linalg.norm(np.diff(arm, axis=0), axis=1)  # [T-1]
    vel = np.concatenate([[vel[0]], vel])  # align length to T

    # local minima: below the 20th percentile AND lower than both neighbors over a small window
    w = max(1, fps // 6)
    is_min = np.zeros(len(vel), dtype=bool)
    thresh = np.percentile(vel, 20)
    for t in range(w, len(vel) - w):
        if vel[t] <= thresh and vel[t] == vel[t - w:t + w + 1].min():
            is_min[t] = True
    pause_frames = np.where(is_min)[0]

    hand = np.concatenate([state[:, LEFT_HAND], state[:, RIGHT_HAND]], axis=1).mean(axis=1)
    hand_smooth = np.convolve(hand, np.ones(w) / w, mode="same")
    crossings = np.where(np.diff(np.sign(hand_smooth - hand_smooth.mean())))[0]

    candidates = sorted(set(pause_frames.tolist()) | set(crossings.tolist()))
    merged, min_gap = [], min_gap_s * fps
    for f in candidates:
        if not merged or f - merged[-1] >= min_gap:
            merged.append(f)

    return [{"frame": int(f), "time_s": round(f / fps, 2), "stage": None,
             "source": "arm_velocity_min" if f in pause_frames else "hand_state_crossing"}
            for f in merged]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=pathlib.Path, required=True)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    state = episode_data(args.dataset_root, args.episode)
    print(f"episode {args.episode}: {len(state)} frames")
    candidates = suggest(state, fps=args.fps)
    print(f"{len(candidates)} candidate boundaries suggested")

    out_dir = pathlib.Path(__file__).parent / "annotations"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"ep{args.episode}_suggestions.json"
    out_path.write_text(json.dumps({"episode": args.episode, "num_frames": len(state),
                                     "fps": args.fps, "boundaries": candidates}, indent=2))
    print(f"-> {out_path}  (edit stage names / confirm in annotate.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
