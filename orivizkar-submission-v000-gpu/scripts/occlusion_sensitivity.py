"""Which image regions does the policy actually rely on?

Grad-CAM answers "what correlated with the output magnitude" through a 7x7
gradient average -- a weak, well-known-to-be-noisy attribution. This instead
measures the thing directly: occlude one 32x32 patch of one camera, re-run the
real inference path, and record how far the predicted action chunk moved.

Produces a 7x7 sensitivity map per camera in the same geometry as the Grad-CAM
grid, so the centre-mass numbers are directly comparable to cam_mass.py -- but
grounded in "the action changed" rather than "the gradient was large".

All 49 occlusions for a camera go through as one batch, so this costs one
forward per camera per frame, not 49.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_smoothing import Runner, build_normalizer, build_policy  # noqa: E402
from episode_io import frame_inputs, open_episode, to_model_inputs  # noqa: E402

TACTILE_H = 18
GRID = 7
UNIFORM = 9 / 49


@torch.inference_mode()
def sensitivity(policy, qpos, image, tac, tac_next, cam_id, grid=GRID):
    """[grid,grid] map of ||a_hat(occluded) - a_hat(clean)|| for one camera."""
    base = policy.model(qpos, image, None, tac, epoch=999, tactile_next=tac_next)[0]

    H, W = image.shape[-2:]
    ph, pw = H // grid, W // grid
    # One batch entry per occluded cell; occlusion value is the per-image mean,
    # so the patch carries no edge energy of its own.
    fill = image[0, cam_id].mean()
    batch = image.repeat(grid * grid, 1, 1, 1, 1)
    for r in range(grid):
        for c in range(grid):
            batch[r * grid + c, cam_id, :, r * ph:(r + 1) * ph, c * pw:(c + 1) * pw] = fill

    out = policy.model(
        qpos.repeat(grid * grid, 1), batch, None, tac.repeat(grid * grid, 1),
        epoch=999, tactile_next=tac_next.repeat(grid * grid, 1, 1),
    )[0]
    return (out - base).flatten(1).norm(dim=1).reshape(grid, grid).cpu().numpy()


def centre_frac(m):
    h, w = m.shape
    tot = m.sum()
    if tot <= 1e-12:
        return float("nan")
    return float(m[h // 2 - 1:h // 2 + 2, w // 2 - 1:w // 2 + 2].sum() / tot)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=Path, required=True)
    p.add_argument("--ckpt_dir", type=Path, action="append", required=True)
    p.add_argument("--label", type=str, action="append", default=None)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--stride", type=int, default=500)
    p.add_argument("--max_frames", type=int, default=8)
    args = p.parse_args()

    labels = args.label or [d.name[:22] for d in args.ckpt_dir]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    for ck, lab in zip(args.ckpt_dir, labels):
        policy, pc, name = build_policy(ck, device)
        normalizer, use_img_norm = build_normalizer(ck, device)
        policy.eval()
        cams = pc["camera_names"]
        images_fn, window_fn, ep_len, _ = open_episode(args.dataset_root, args.episode)
        idxs = list(range(0, ep_len, args.stride))[: args.max_frames]

        runner = Runner(policy, pc, normalizer, use_img_norm, device)
        per_cam_centre = [[] for _ in cams]
        per_cam_total = [[] for _ in cams]
        for i in idxs:
            img, state, tac19 = frame_inputs(i, cams, pc, normalizer, device,
                                             images_fn, window_fn, runner._img)
            q, im, tac, tn = to_model_inputs(img, state, tac19, device)

            for c in range(len(cams)):
                m = sensitivity(policy, q, im, tac, tn, c)
                per_cam_centre[c].append(centre_frac(m))
                per_cam_total[c].append(float(m.mean()))

        results[lab] = (cams,
                        [float(np.nanmean(v)) for v in per_cam_centre],
                        [float(np.nanmean(v)) for v in per_cam_total])
        print(f"[occl] {lab}: {name} ({len(idxs)} frames)")

    w = max(max(len(l) for l in results), 10) + 2
    cams = next(iter(results.values()))[0]

    print(f"\nA) Share of action-change caused by occluding the centre 3x3")
    print(f"   (uniform = {UNIFORM:.3f}; higher = model genuinely relies on the centre)\n")
    print("camera".ljust(16) + "".join(l.ljust(w) for l in results))
    print("-" * (16 + w * len(results)))
    for ci, cam in enumerate(cams):
        row = cam.rsplit(".", 1)[-1].ljust(16)
        for lab in results:
            row += f"{results[lab][1][ci]:.3f}".ljust(w)
        print(row)
    print("-" * (16 + w * len(results)))
    row = "MEAN".ljust(16)
    for lab in results:
        row += f"{np.mean(results[lab][1]):.3f}".ljust(w)
    print(row)

    print(f"\nB) Mean |delta a_hat| per occluded patch (radians) -- how much the")
    print(f"   camera matters at all; near zero = the policy barely uses it\n")
    print("camera".ljust(16) + "".join(l.ljust(w) for l in results))
    print("-" * (16 + w * len(results)))
    for ci, cam in enumerate(cams):
        row = cam.rsplit(".", 1)[-1].ljust(16)
        for lab in results:
            row += f"{results[lab][2][ci]:.4f}".ljust(w)
        print(row)


if __name__ == "__main__":
    sys.exit(main())
