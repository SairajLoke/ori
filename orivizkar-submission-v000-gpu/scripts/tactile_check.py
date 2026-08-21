"""Is the tactile branch doing anything?

Two questions, both over a full episode:

A) Is `tactile_hat` predictive? The auxiliary head predicts the next 18 tactile
   readings. Compare it against two baselines that require no learning:
   "repeat the last observed reading" and "predict the dataset centre" (0 in
   normalized space). If it does not beat those, the head learned nothing.

B) Does the `tactile_pred` token affect the action? At inference detr_vae uses
   its OWN tactile_hat (epoch>=75), projects it, and feeds it as an encoder
   token -- so passing a different `tactile_next` argument does nothing. The
   real ablation is to zero the tactile_head OUTPUT, which makes the
   tactile_pred token a constant (projection bias only), and see how far the
   action moves.

Matters because the keypoint proposal would build a second instance of exactly
this pattern (aux head -> dedicated token -> cross-attn).
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


def gt_tactile_next(window_fn, norm, i, device):
    """The [18,120] target detr_vae is trained against: normalized tactile at
    offsets +1..+18 concatenated with its deltas, built exactly as convert_batch
    does (normalize first, then diff)."""
    fut = window_fn("observation.tactile", i + TACTILE_H, TACTILE_H + 1)  # offsets 0..+18
    t = torch.from_numpy(fut).to(device)
    if norm is not None:
        t = norm.normalize("observation.tactile", t)
    return torch.cat([t[1:], torch.diff(t, dim=0)], dim=-1)  # [18,120]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=Path, required=True)
    p.add_argument("--ckpt_dir", type=Path, required=True)
    p.add_argument("--label", type=str, default=None)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--horizon", type=int, default=25)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, pc, ckpt = build_policy(args.ckpt_dir, device)
    norm, use_img_norm = build_normalizer(args.ckpt_dir, device)
    policy.eval()
    cams = pc["camera_names"]
    label = args.label or args.ckpt_dir.name[:24]
    images_fn, window_fn, n, actions = open_episode(args.dataset_root, args.episode)
    runner = Runner(policy, pc, norm, use_img_norm, device)

    idxs = [i for i in range(0, n, args.stride)
            if i + TACTILE_H + 1 < n and i + args.horizon <= len(actions)]
    print(f"[tactile] {label}: {ckpt}")
    print(f"[tactile] episode {args.episode}: {n} frames, evaluating {len(idxs)} "
          f"(stride {args.stride}, spans the whole episode)\n")

    # zeroing tactile_head's output makes the tactile_pred token constant
    zero_head = {"on": False}
    policy.model.tactile_head.register_forward_hook(
        lambda m, i_, o: torch.zeros_like(o) if zero_head["on"] else None)

    se_hat, se_hold, se_mean, dact, dmse_clean, dmse_abl = [], [], [], [], [], []
    for k, i in enumerate(idxs):
        img, state, tac19 = frame_inputs(i, cams, pc, norm, device, images_fn, window_fn, runner._img)
        q, im, tac, tn = to_model_inputs(img, state, tac19, device)
        gt = gt_tactile_next(window_fn, norm, i, device)                     # [18,120]

        with torch.inference_mode():
            a_clean, _, _, tac_hat = policy.model(q, im, None, tac, epoch=999, tactile_next=tn)
            zero_head["on"] = True
            a_abl = policy.model(q, im, None, tac, epoch=999, tactile_next=tn)[0]
            zero_head["on"] = False

        th = tac_hat[0]                                                       # [18,120]
        se_hat.append(float(((th - gt) ** 2).mean()))
        hold = tac19[-1].repeat(TACTILE_H, 1)                                 # last obs, value half
        hold = torch.cat([hold, torch.zeros_like(hold)], dim=-1)[:, :gt.shape[-1]]
        se_hold.append(float(((hold - gt) ** 2).mean()))
        se_mean.append(float((gt ** 2).mean()))                               # predict 0 (= centre)

        ac = a_clean if norm is None else norm.denormalize("action", a_clean.float())
        ab = a_abl if norm is None else norm.denormalize("action", a_abl.float())
        ac, ab = ac[0, :args.horizon].cpu().numpy(), ab[0, :args.horizon].cpu().numpy()
        g = actions[i:i + args.horizon]
        dact.append(float(np.linalg.norm(ab - ac)))
        dmse_clean.append(float(((ac - g) ** 2).mean()))
        dmse_abl.append(float(((ab - g) ** 2).mean()))
        if (k + 1) % 100 == 0:
            print(f"[tactile] {k+1}/{len(idxs)}", flush=True)

    m = lambda x: float(np.mean(x))
    print("\nA) tactile_hat vs ground-truth future tactile (normalized units, MSE)\n")
    print(f"{'predictor':38s}{'MSE':>10s}{'vs hat':>10s}")
    print("-" * 58)
    print(f"{'tactile_hat (the aux head)':38s}{m(se_hat):10.5f}{'--':>10s}")
    print(f"{'repeat last observed reading':38s}{m(se_hold):10.5f}{m(se_hold)/m(se_hat):9.2f}x")
    print(f"{'predict dataset centre (zeros)':38s}{m(se_mean):10.5f}{m(se_mean)/m(se_hat):9.2f}x")
    print("-" * 58)
    print("  >1.0x means the head beats that baseline\n")

    print("B) zeroing tactile_head -> tactile_pred token becomes constant\n")
    print(f"  ||delta a_hat||        = {m(dact):.4f} rad  "
          f"({m(dact)/np.sqrt(args.horizon*65):.5f} per joint-step)")
    print(f"  action MSE clean       = {m(dmse_clean):.5f}")
    print(f"  action MSE ablated     = {m(dmse_abl):.5f}  "
          f"({(m(dmse_abl)/m(dmse_clean)-1)*100:+.1f}%)")


if __name__ == "__main__":
    sys.exit(main())
