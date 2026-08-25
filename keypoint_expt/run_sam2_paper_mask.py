#!/usr/bin/env python3
"""Paper segmentation via SAM2 video mask propagation: prompt once (a point on
the paper in frame 0), propagate across the whole clip, save a boolean mask
per frame. Built to feed CoTracker's query grid -- restrict points to
mask==True instead of a naive whole-frame grid (today's run put ~400 points
across the frame and got ~1 on the paper, since paper is 0.14-1.05% of frame
per plans.md's own measurement).

GPU-only in practice (Colab): SAM2's video predictor keeps a per-frame memory
bank for propagation, architecturally the same class of memory cost as
CoTracker's online mode -- not worth fighting on this box's 15GB CPU RAM
given today's lesson on that exact failure mode.

Setup (once, e.g. in a Colab cell):
    !git clone https://github.com/facebookresearch/sam2.git
    %cd sam2 && pip install -q -e . && cd checkpoints && ./download_ckpts.sh && cd ..

Usage:
    python run_sam2_paper_mask.py --video <path> --point X Y \
        [--seconds 90] [--start_seconds 60] [--out_name ep0_head_left]

--point is REQUIRED and must be a pixel coordinate (x,y) on the paper in the
FIRST frame of the clip -- SAM2 has no built-in notion of "paper", it only
propagates a mask from a manual prompt. Pick it by eye from an extracted
first frame (see --dump_first_frame).
"""
from __future__ import annotations

import argparse
import pathlib

import cv2
import numpy as np
import torch

FPS = 30
OUT_DIR = pathlib.Path(__file__).parent / "results"

SAM2_CHECKPOINT = "sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def read_clip(path: pathlib.Path, start_seconds: float, seconds: float | None) -> np.ndarray:
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
        if len(frames) % 500 == 0:
            print(f"    read {len(frames)}{f'/{n}' if n else ''} frames", flush=True)
    cap.release()
    print(f"    read {len(frames)} frames total", flush=True)
    return np.stack(frames)


def overlay_mask_video(frames: np.ndarray, masks: np.ndarray, out_path: pathlib.Path) -> None:
    """masks: [T,H,W] bool. Paper region tinted green, semi-transparent."""
    h, w = frames.shape[1:3]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for t in range(frames.shape[0]):
        frame = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR).copy()
        overlay = frame.copy()
        overlay[masks[t]] = (0, 255, 0)
        blended = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
        writer.write(blended)
    writer.release()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=pathlib.Path, required=True)
    p.add_argument("--point", type=int, nargs=2, metavar=("X", "Y"),
                    help="pixel coords on the paper in the clip's first frame")
    p.add_argument("--start_seconds", type=float, default=0.0)
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--out_name", default=None)
    p.add_argument("--dump_first_frame", action="store_true",
                    help="save results/<out_name>_frame0.png and exit -- use this first to pick --point")
    args = p.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    name = args.out_name or args.video.stem
    frames = read_clip(args.video, args.start_seconds, args.seconds if args.dump_first_frame is False else 1)

    if args.dump_first_frame:
        out = OUT_DIR / f"{name}_frame0.png"
        cv2.imwrite(str(out), cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
        print(f"first frame -> {out}; pick a paper pixel (x,y) from it and pass --point X Y")
        return 0

    if args.point is None:
        raise SystemExit("--point X Y is required (or run --dump_first_frame first to pick one)")

    from sam2.build_sam import build_sam2_video_predictor  # local import: only needed past this point

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16 if device == "cuda" else torch.float32):
        state = predictor.init_state(video_path=frames)  # accepts an array of frames directly
        predictor.add_new_points_or_box(
            state, frame_idx=0, obj_id=1,
            points=np.array([args.point], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),  # 1 = foreground click
        )
        masks = np.zeros(frames.shape[:3], dtype=bool)  # [T,H,W]
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            masks[frame_idx] = (mask_logits[0] > 0).cpu().numpy().squeeze()
            if frame_idx % 500 == 0:
                print(f"    propagated {frame_idx}/{frames.shape[0]}", flush=True)

    pct = masks.mean(axis=(1, 2))
    print(f"paper mask covers {pct.mean()*100:.2f}% of frame on average "
          f"(min {pct.min()*100:.2f}%, max {pct.max()*100:.2f}%)")

    out_path = OUT_DIR / f"{name}_papermask_overlay.mp4"
    overlay_mask_video(frames, masks, out_path)
    np.savez_compressed(OUT_DIR / f"{name}_papermask.npz", masks=masks)
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
