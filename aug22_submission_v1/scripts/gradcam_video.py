"""Grad-CAM over a whole episode, stitched straight to an mp4.

Uses the INFERENCE forward path (policy.py: `self.model(qpos, image, env_state,
tactile, epoch=999, tactile_next=...)` -- no ground-truth actions, sampled from
the prior) and backprops sum(|a_hat|), so the maps show what the backbone
attended to while producing the prediction it would actually emit on the robot.
my_utils/gradcam.py's save_gradcam_grid instead backprops the L1 error against
ground truth, which is the right question during training but not here.

Frames are piped to ffmpeg as rawvideo, so nothing but the mp4 hits disk.

  PYTHONPATH=.:vitacformer:vitacformer/detr python scripts/gradcam_video.py \
      --dataset_root .../lerobot3.0_shortgop15_224 \
      --ckpt_dir .../assets/<run> --episode 0 --stride 5 --out gradcam.mp4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_smoothing import Runner, build_normalizer, build_policy  # noqa: E402
from episode_io import frame_inputs, open_episode, to_model_inputs  # noqa: E402
from my_utils.gradcam import _colorize, _denorm_for_display  # noqa: E402

TACTILE_H = 18


@torch.enable_grad()
def overlays_for_frame(policy, qpos_flat, image, tactile_flat, tactile_next, n_cams):
    """Returns [n_cams] of [H,W,3] uint8 Grad-CAM overlays for a batch of 1."""
    # Hook the shared vision module's output. Same reasoning as
    # my_utils/gradcam.py: retain_grad() per captured tensor rather than a
    # backward hook, because the module is invoked once per camera inside a
    # single forward and backward hooks under-fire in that case.
    acts = []

    def _hook(_m, _i, out):
        feat = out["0"] if isinstance(out, dict) else out
        feat.retain_grad()
        acts.append(feat)

    h = policy.model.backbones[0][0].register_forward_hook(_hook)
    orig = [p.requires_grad for p in policy.parameters()]
    try:
        for p in policy.parameters():
            p.requires_grad_(True)
        a_hat, _, (_, _), _ = policy.model(
            qpos_flat, image, None, tactile_flat, epoch=999, tactile_next=tactile_next
        )
        policy.zero_grad(set_to_none=True)
        a_hat.abs().sum().backward()
    finally:
        h.remove()
        for p, o in zip(policy.parameters(), orig):
            p.requires_grad_(o)

    if len(acts) != n_cams or any(a.grad is None for a in acts):
        return None

    H, W = image.shape[-2:]
    was_normed = image.min().item() < -0.01 or image.max().item() > 1.01
    out = []
    for cam in range(n_cams):
        act, grad = acts[cam], acts[cam].grad
        w = grad.mean(dim=(2, 3), keepdim=True)
        cam_map = F.relu((w * act).sum(dim=1, keepdim=True))
        cam_map = F.interpolate(cam_map, size=(H, W), mode="bilinear", align_corners=False)
        c = cam_map[0, 0].detach().cpu().numpy()
        c = c / (c.max() + 1e-8)
        base = _denorm_for_display(image[0, cam], was_normed).numpy()
        out.append((0.55 * base + 0.45 * _colorize(c)).astype(np.uint8))
    return out


def tile(overlays, names, frame_idx, t_s, label):
    """2x2 grid with per-camera captions and a footer."""
    h, w, _ = overlays[0].shape
    pad, foot = 18, 22
    canvas = Image.new("RGB", (w * 2, h * 2 + pad * 2 + foot), (14, 14, 16))
    d = ImageDraw.Draw(canvas)
    for i, (ov, nm) in enumerate(zip(overlays, names)):
        x, y = (i % 2) * w, (i // 2) * (h + pad) + pad
        canvas.paste(Image.fromarray(ov), (x, y))
        d.text((x + 4, y - 13), nm.rsplit(".", 1)[-1], fill=(235, 235, 235))
    d.text((4, h * 2 + pad * 2 + 4),
           f"{label}   frame {frame_idx:5d}   t={t_s:7.2f}s", fill=(180, 180, 190))
    return np.asarray(canvas)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=Path, required=True)
    p.add_argument("--ckpt_dir", type=Path, required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    p.add_argument("--max_frames", type=int, default=None, help="cap frames (debug/timing)")
    p.add_argument("--fps", type=int, default=30, help="output video fps")
    p.add_argument("--label", type=str, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, pc, ckpt_name = build_policy(args.ckpt_dir, device)
    normalizer, use_image_norm = build_normalizer(args.ckpt_dir, device)
    policy.eval()
    label = args.label or args.ckpt_dir.name
    cams = pc["camera_names"]
    print(f"[gradcam] {ckpt_name} device={device} cams={len(cams)}")

    images_fn, window_fn, ep_len, _ = open_episode(args.dataset_root, args.episode)
    idxs = list(range(0, ep_len, args.stride))
    if args.max_frames:
        idxs = idxs[: args.max_frames]
    print(f"[gradcam] episode {args.episode}: {ep_len} frames, rendering {len(idxs)} "
          f"(stride {args.stride}) -> {args.out}")

    runner = Runner(policy, pc, normalizer, use_image_norm, device)
    runner.reset()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    proc, t0, done = None, time.time(), 0
    try:
        for n, i in enumerate(idxs):
            img, state, tac19 = frame_inputs(i, cams, pc, normalizer, device,
                                             images_fn, window_fn, runner._img)
            qpos, im, tac, tac_next = to_model_inputs(img, state, tac19, device)

            ov = overlays_for_frame(policy, qpos, im, tac, tac_next, len(cams))
            if ov is None:
                print(f"[gradcam] frame {i}: no gradient captured, aborting")
                return 1
            rgb = tile(ov, cams, i, i / 30.0, label)

            if proc is None:
                H, W, _ = rgb.shape
                proc = subprocess.Popen(
                    ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                     "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-",
                     "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                     "-pix_fmt", "yuv420p", "-g", "15", str(args.out)],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            proc.stdin.write(rgb.tobytes())
            done += 1

            if n and n % 25 == 0:
                el = time.time() - t0
                print(f"[gradcam] {n}/{len(idxs)}  {el/n:.2f}s/frame  "
                      f"eta {(len(idxs)-n)*el/n/60:.1f} min", flush=True)
    finally:
        if proc is not None:
            proc.stdin.close()
            proc.wait()

    print(f"[gradcam] wrote {done} frames in {(time.time()-t0)/60:.1f} min -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
