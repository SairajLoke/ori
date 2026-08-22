"""Is Grad-CAM mass actually centre- or border-biased?

The CAM is a 7x7 grid upsampled to 224x224, so eyeballing the overlay is
misleading: bilinear upsampling smears edge cells outward and makes them look
larger than they are. Measure on the native grid instead.

Reports, per camera, the fraction of CAM mass falling in the centre 3x3 cells.
The centre is 9/49 = 18.4% of the grid, so that is the uniform-attention
baseline: above it means centre-biased, below means border-biased.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_smoothing import Runner, build_normalizer, build_policy  # noqa: E402
from episode_io import frame_inputs, open_episode, to_model_inputs  # noqa: E402

TACTILE_H = 18
UNIFORM = 9 / 49  # centre 3x3 share of a 7x7 grid


@torch.enable_grad()
def cam_grids(policy, qpos, image, tac, tac_next, n_cams):
    """Native-resolution [h,w] CAM per camera, no upsampling."""
    acts = []

    def _hook(_m, _i, out):
        f = out["0"] if isinstance(out, dict) else out
        f.retain_grad()
        acts.append(f)

    h = policy.model.backbones[0][0].register_forward_hook(_hook)
    orig = [p.requires_grad for p in policy.parameters()]
    try:
        for p in policy.parameters():
            p.requires_grad_(True)
        a_hat, _, (_, _), _ = policy.model(qpos, image, None, tac, epoch=999, tactile_next=tac_next)
        policy.zero_grad(set_to_none=True)
        a_hat.abs().sum().backward()
    finally:
        h.remove()
        for p, o in zip(policy.parameters(), orig):
            p.requires_grad_(o)

    if len(acts) != n_cams or any(a.grad is None for a in acts):
        return None
    out = []
    for c in range(n_cams):
        w = acts[c].grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * acts[c]).sum(dim=1, keepdim=True))[0, 0]
        out.append(cam.detach().cpu().numpy())
    return out


def centre_frac(cam):
    """Share of total CAM mass inside the centre 3x3 of the grid."""
    h, w = cam.shape
    r0, r1 = h // 2 - 1, h // 2 + 2
    c0, c1 = w // 2 - 1, w // 2 + 2
    tot = cam.sum()
    return float(cam[r0:r1, c0:c1].sum() / tot) if tot > 1e-12 else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=Path, required=True)
    p.add_argument("--ckpt_dir", type=Path, action="append", required=True,
                    help="repeatable; one column per checkpoint")
    p.add_argument("--label", type=str, action="append", default=None)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--stride", type=int, default=200)
    p.add_argument("--max_frames", type=int, default=25)
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
        per_cam = [[] for _ in cams]
        for i in idxs:
            img, state, tac19 = frame_inputs(i, cams, pc, normalizer, device,
                                             images_fn, window_fn, runner._img)
            q, im, tac, tn = to_model_inputs(img, state, tac19, device)

            grids = cam_grids(policy, q, im, tac, tn, len(cams))
            if grids is None:
                continue
            for c, g in enumerate(grids):
                per_cam[c].append(centre_frac(g))

        results[lab] = (cams, [float(np.nanmean(v)) for v in per_cam], len(idxs))
        print(f"[cam_mass] {lab}: {name}  ({len(idxs)} frames)")

    print(f"\nShare of Grad-CAM mass in the centre 3x3 of the 7x7 grid")
    print(f"(uniform attention = {UNIFORM:.3f}; higher = more centred)\n")
    cams = next(iter(results.values()))[0]
    w = max(len(l) for l in results) + 2
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
    print(f"\nuniform baseline = {UNIFORM:.3f}")


if __name__ == "__main__":
    sys.exit(main())
