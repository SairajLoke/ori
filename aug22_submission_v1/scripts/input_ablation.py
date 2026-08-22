"""Which inputs does the policy actually rely on?

Perturbs one modality at a time at inference and measures what happens to the
predicted action chunk. Two perturbations per target:

  noise k    x <- x + k*N(0,1)   in NORMALIZED space, so k is "k standard
                                 deviations of the training distribution" and
                                 is comparable across modalities
  replace    x <- 0              also normalized space, i.e. the centre of the
                                 training distribution: a valid but
                                 UNINFORMATIVE input (grey frame / average
                                 pose). Answers "what if this sensor told me
                                 nothing", and unlike noise it is immune to the
                                 token-count imbalance below.

Token-count caveat: the encoder sequence is [tactile, tactile_pred, latent,
proprio] + 196 image tokens, so perturbing "images" hits 196 tokens while
"proprio" hits 1. Noise rankings are biased by that; replacement is not.

Tactile noise is injected on the [19,60] window BEFORE torch.diff, so the delta
channel amplifies it the way real sensor noise would.

Nothing in the training path is touched -- this only reuses the eval loaders.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_smoothing import Runner, build_normalizer, build_policy  # noqa: E402

TACTILE_H = 18


def open_episode(dataset_root: Path, episode: int):
    """Images on demand (decode); state/tactile/action columns read once from the
    parquet (no decode) so true history windows are cheap.

    The history must match training: base's DELTA_TIMESTAMPS asks for
    observation.state at [-0.167 .. 0] and tactile at [-0.6 .. 0], i.e. 6 and 19
    CONSECUTIVE 30 fps frames. Rebuilding history from sparsely sampled frames --
    the way the server's deque does when infer() is called rarely -- would feed a
    window spanning tens of seconds and put every input off-distribution, which
    is not what this test is trying to measure.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=None, root=dataset_root, episodes=[episode],
                        video_backend="pyav", tolerance_s=0.02)
    cols = {k: np.asarray(ds.hf_dataset[k], dtype=np.float32)
            for k in ("action", "observation.state", "observation.tactile")}

    def images(i, camera_names):
        s = ds[i]
        return {c: s[c] for c in camera_names}

    def window(key, i, n):
        """n consecutive frames ending at i, clamped at the episode start
        (replicating frame 0, which is what the server does on a cold start)."""
        idx = np.clip(np.arange(i - n + 1, i + 1), 0, len(ds) - 1)
        return cols[key][idx]

    return images, window, len(ds), cols["action"]


def build_variants(img, state, tac19, cam_names, levels, gen, image_norm, targets=None):
    """(name, img, state, tac19) for clean + every ablation."""
    short = [c.rsplit(".", 1)[-1] for c in cam_names]
    cam_targets = {n: [i] for i, n in enumerate(short)}
    cam_targets["cameras_all"] = list(range(len(short)))
    grey = 0.0 if image_norm else 0.5   # normalized-space centre of the pixel distribution

    keep = (lambda t: True) if targets is None else (lambda t: t in targets)
    out = [("clean", img, state, tac19)]

    def noisy(x, k):
        return x + k * torch.randn(x.shape, generator=gen, dtype=x.dtype, device=x.device)

    for name, idxs in cam_targets.items():
        if not keep(name):
            continue
        for k in levels:
            v = img.clone()
            for i in idxs:
                v[i] = noisy(v[i], k)
            out.append((f"{name}|noise{k}", v, state, tac19))
        v = img.clone()
        for i in idxs:
            v[i] = grey
        out.append((f"{name}|replace", v, state, tac19))

    if keep("proprio"):
        for k in levels:
            out.append((f"proprio|noise{k}", img, noisy(state, k), tac19))
        out.append(("proprio|replace", img, torch.zeros_like(state), tac19))

    if keep("tactile"):
        for k in levels:
            out.append((f"tactile|noise{k}", img, state, noisy(tac19, k)))
        out.append(("tactile|replace", img, state, torch.zeros_like(tac19)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=Path, required=True)
    p.add_argument("--ckpt_dir", type=Path, required=True)
    p.add_argument("--label", type=str, default=None)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--stride", type=int, default=300)
    p.add_argument("--max_frames", type=int, default=20)
    p.add_argument("--horizon", type=int, default=25, help="chunk depth scored against GT")
    p.add_argument("--levels", type=str, default="0.25,0.5,1.0")
    p.add_argument("--targets", type=str, default=None,
                    help="comma-separated subset to ablate (default: all). Fewer targets "
                         "means a smaller batch per frame and a proportionally faster run.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    levels = [float(x) for x in args.levels.split(",")]
    targets = set(args.targets.split(",")) if args.targets else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, pc, ckpt_name = build_policy(args.ckpt_dir, device)
    normalizer, use_image_norm = build_normalizer(args.ckpt_dir, device)
    policy.eval()
    cams = pc["camera_names"]
    label = args.label or args.ckpt_dir.name[:24]
    print(f"[ablate] {label}: {ckpt_name}  device={device}  image_norm={use_image_norm}")

    get_images, window, ep_len, gt_actions = open_episode(args.dataset_root, args.episode)
    idxs = [i for i in range(0, ep_len, args.stride)][: args.max_frames]
    idxs = [i for i in idxs if i + args.horizon <= len(gt_actions)]
    print(f"[ablate] episode {args.episode}: {ep_len} frames, scoring {len(idxs)} "
          f"at horizon {args.horizon}, noise levels {levels}")
    print(f"[ablate] history: {pc['proprioceptive_temporal_horizon']} consecutive state frames, "
          f"{TACTILE_H + 1} consecutive tactile frames (matches training delta_timestamps)")

    runner = Runner(policy, pc, normalizer, use_image_norm, device)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    acc_mse, acc_delta, names = {}, {}, None

    for n, i in enumerate(idxs):
        imgs_raw = get_images(i, cams)
        img = torch.stack([runner._img(imgs_raw[c])[0] for c in cams]).to(device)  # [4,3,H,W]

        state = torch.from_numpy(
            window("observation.state", i, pc["proprioceptive_temporal_horizon"])).to(device)
        if normalizer is not None:
            state = normalizer.normalize("observation.state", state)

        tac19 = torch.from_numpy(
            window("observation.tactile", i, TACTILE_H + 1)).to(device)
        if normalizer is not None:
            tac19 = normalizer.normalize("observation.tactile", tac19)

        variants = build_variants(img, state, tac19, cams, levels, gen, use_image_norm, targets)
        if names is None:
            names = [v[0] for v in variants]

        imgs = torch.stack([v[1] for v in variants])                        # [V,4,3,H,W]
        qpos = torch.stack([v[2].reshape(-1) for v in variants])            # [V,390]
        tacs = torch.stack([torch.cat([v[3][1:], torch.diff(v[3], dim=0)], dim=-1).reshape(-1)
                            for v in variants])                             # [V,2160]
        tn = torch.zeros((len(variants), TACTILE_H, 120), device=device)

        with torch.inference_mode():
            a = policy.model(qpos, imgs, None, tacs, epoch=999, tactile_next=tn)[0]
        if normalizer is not None:
            a = normalizer.denormalize("action", a.float())
        a = a[:, : args.horizon].cpu().numpy()

        gt = gt_actions[i:i + args.horizon][None]
        mse = ((a - gt) ** 2).mean(axis=(1, 2))
        delta = np.linalg.norm((a - a[0:1]).reshape(len(a), -1), axis=1)
        for nm, m, d in zip(names, mse, delta):
            acc_mse.setdefault(nm, []).append(float(m))
            acc_delta.setdefault(nm, []).append(float(d))
        if (n + 1) % 5 == 0:
            print(f"[ablate] {n+1}/{len(idxs)} frames", flush=True)

    clean = float(np.mean(acc_mse["clean"]))
    print(f"\nclean MSE vs ground truth = {clean:.5f}   (horizon {args.horizon}, "
          f"{len(idxs)} frames, ckpt {ckpt_name})\n")

    rows = []
    for nm in names:
        if nm == "clean":
            continue
        tgt, kind = nm.split("|")
        m = float(np.mean(acc_mse[nm]))
        rows.append((tgt, kind, m, m - clean, (m / clean - 1) * 100, float(np.mean(acc_delta[nm]))))

    hdr = f"{'input':14s}{'perturb':10s}{'MSE':>9s}{'dMSE':>9s}{'  %worse':>10s}{'||da_hat||':>12s}"
    print(hdr); print("-" * len(hdr))
    order = [t for t in ([c.rsplit('.', 1)[-1] for c in cams] + ["cameras_all", "proprio", "tactile"])
             if any(r[0] == t for r in rows)]
    for tgt in order:
        for r in [r for r in rows if r[0] == tgt]:
            print(f"{r[0]:14s}{r[1]:10s}{r[2]:9.5f}{r[3]:+9.5f}{r[4]:+9.1f}%{r[5]:12.4f}")
        print("-" * len(hdr))

    print("\nRanking by REPLACE (removes the modality; unaffected by token count):")
    rep = sorted([r for r in rows if r[1] == "replace"], key=lambda r: -r[3])
    for rank, r in enumerate(rep, 1):
        print(f"  {rank}. {r[0]:14s} dMSE={r[3]:+.5f}  ({r[4]:+.1f}%)   ||da_hat||={r[5]:.4f}")

    if args.out:
        import json
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "ckpt": ckpt_name, "label": label, "episode": args.episode,
            "horizon": args.horizon, "frames": len(idxs), "levels": levels,
            "clean_mse": clean,
            "mse": {k: float(np.mean(v)) for k, v in acc_mse.items()},
            "delta": {k: float(np.mean(v)) for k, v in acc_delta.items()},
        }, indent=1))
        print(f"\n[ablate] wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
