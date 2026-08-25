#!/usr/bin/env python3
"""Go/no-go gate for the paper-keypoint-tracker plan (vitacformer++/plans.md #3):
run CoTracker3 on real head_left/wrist_left episode clips and overlay the
tracks so tracking quality can be judged visually before any training code
is touched. Dense grid over the whole frame (no paper mask yet -- that's a
later step, not needed to answer "do tracks survive occlusion").

    .venv/bin/python3 run_cotracker.py --episode 0 --seconds 10
"""
from __future__ import annotations

import argparse
import pathlib

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch

DATASET_ROOT = pathlib.Path(
    "/media/sai/CRUZER_BLA/ori/dataset/season_POC22061_2026_07_09_16_23_46_train/"
    "lerobot3.0_shortgop15_224"
)
CAMERAS = ["observation.images.head_left", "observation.images.wrist_left"]
FPS = 30
OUT_DIR = pathlib.Path(__file__).parent / "results"


def episode_row(episode: int) -> dict:
    cols = ["episode_index"] + [
        f"videos/{cam}/{field}"
        for cam in CAMERAS
        for field in ("chunk_index", "file_index", "from_timestamp")
    ]
    table = pq.read_table(DATASET_ROOT / "meta/episodes/chunk-000/file-000.parquet", columns=cols)
    d = table.to_pydict()
    i = d["episode_index"].index(episode)
    return {k: d[k][i] for k in d}


def read_clip(cam: str, row: dict, seconds: float) -> np.ndarray:
    chunk = row[f"videos/{cam}/chunk_index"]
    file_ = row[f"videos/{cam}/file_index"]
    start_frame = round(row[f"videos/{cam}/from_timestamp"] * FPS)
    path = DATASET_ROOT / "videos" / cam / f"chunk-{chunk:03d}" / f"file-{file_:03d}.mp4"
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    n = round(seconds * FPS)
    frames = []
    for _ in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)  # [T,H,W,3] uint8


def overlay_and_save(frames: np.ndarray, tracks: np.ndarray, visible: np.ndarray, out_path: pathlib.Path) -> None:
    h, w = frames.shape[1:3]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for t in range(frames.shape[0]):
        frame = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR).copy()
        for p in range(tracks.shape[1]):
            x, y = tracks[t, p]
            color = (0, 220, 0) if visible[t, p] else (0, 0, 220)  # BGR: green=visible, red=occluded
            cv2.circle(frame, (int(x), int(y)), 2, color, -1)
        writer.write(frame)
    writer.release()


def read_full_video(path: pathlib.Path, start_seconds: float = 0.0, seconds: float | None = None) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if start_seconds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(start_seconds * FPS))
    n = round(seconds * FPS) if seconds is not None else None
    frames = []
    while n is None or len(frames) < n:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


def track_online(model, frames: np.ndarray, grid_size: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window streaming pass -- bounds active compute to model.step*2
    frames regardless of total video length (facebookresearch/co-tracker
    online_demo.py's own pattern). Raw uint8 frames still accumulate in
    `window_frames`, but that's ~150KB/frame, cheap compared to activations."""
    window_frames: list[np.ndarray] = []
    is_first_step = True
    pred_tracks = pred_visibility = None
    for i, frame in enumerate(frames):
        if i % model.step == 0 and i != 0:
            chunk = torch.tensor(np.stack(window_frames[-model.step * 2:]), device=device).float().permute(0, 3, 1, 2)[None]
            with torch.no_grad():
                pred_tracks, pred_visibility = model(chunk, is_first_step=is_first_step, grid_size=grid_size)
            is_first_step = False
            print(f"    step {i}/{len(frames)}", flush=True)
        window_frames.append(frame)
    chunk = torch.tensor(np.stack(window_frames[-(i % model.step) - model.step - 1:]), device=device).float().permute(0, 3, 1, 2)[None]
    with torch.no_grad():
        pred_tracks, pred_visibility = model(chunk, is_first_step=is_first_step, grid_size=grid_size)
    return pred_tracks[0].cpu().numpy(), pred_visibility[0].cpu().numpy()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--grid_size", type=int, default=15)
    p.add_argument("--online", action="store_true",
                    help="cotracker3_online: streams in windows, far lower peak memory than offline")
    p.add_argument("--video", type=pathlib.Path, default=None,
                    help="run on a raw video file directly, bypassing episode slicing (implies --online)")
    p.add_argument("--start_seconds", type=float, default=0.0, help="seek offset within --video")
    p.add_argument("--out_name", default=None, help="output basename (default derived from source)")
    args = p.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    if args.video is not None:
        args.online = True
        name = args.out_name or args.video.stem
        print(f"loading CoTracker3 (online) via torch.hub...")
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(device)
        model.eval()
        clip_seconds = args.seconds if args.seconds != 10.0 or args.start_seconds else None
        print(f"reading {args.video} from t={args.start_seconds}s"
              + (f" for {clip_seconds}s" if clip_seconds else " (full file)") + " ...")
        frames = read_full_video(args.video, start_seconds=args.start_seconds, seconds=clip_seconds)
        print(f"tracking {args.grid_size**2} grid points over {frames.shape[0]} frames, streaming window={model.step*2} ({device})...")
        tracks, visible = track_online(model, frames, args.grid_size, device)
        pct_visible = visible.mean() * 100
        out_path = OUT_DIR / f"{name}_overlay.mp4"
        overlay_and_save(frames, tracks, visible, out_path)
        np.savez_compressed(OUT_DIR / f"{name}_tracks.npz", tracks=tracks, visible=visible)
        print(f"mean visibility={pct_visible:.1f}% -> {out_path}")
        return 0

    row = episode_row(args.episode)
    print(f"episode {args.episode}: length={row.get('length', '?')}")
    print("loading CoTracker3 (offline) via torch.hub -- first run downloads weights...")
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    model.eval()

    for cam in CAMERAS:
        print(f"[{cam}] reading {args.seconds}s clip...")
        frames = read_clip(cam, row, args.seconds)
        video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(device)  # [1,T,3,H,W]

        print(f"[{cam}] tracking {args.grid_size**2} grid points over {frames.shape[0]} frames ({device})...")
        with torch.no_grad():
            pred_tracks, pred_visibility = model(video, grid_size=args.grid_size)
        tracks = pred_tracks[0].cpu().numpy()       # [T,N,2]
        visible = pred_visibility[0].cpu().numpy()  # [T,N]

        pct_visible = visible.mean() * 100
        out_path = OUT_DIR / f"ep{args.episode}_{cam.split('.')[-1]}_overlay.mp4"
        overlay_and_save(frames, tracks, visible, out_path)
        np.savez_compressed(OUT_DIR / f"ep{args.episode}_{cam.split('.')[-1]}_tracks.npz",
                             tracks=tracks, visible=visible)
        print(f"[{cam}] mean visibility={pct_visible:.1f}% -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
