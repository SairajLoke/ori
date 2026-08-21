"""Offline receding-horizon replay to compare chunk-seam smoothing modes.

Replays a held-out episode through the real checkpoint the way the organizer
would: one observation per infer() call, history rebuilt across calls exactly as
TeamPolicy does, chunks consumed only `stride` rows deep before re-querying.
Then assembles the executed trajectory and measures smoothness vs fidelity.

Reported per (stride, mode):
  seam_jump  -- |x[t+1]-x[t]| at chunk boundaries (the artifact being fixed)
  step_jump  -- same, at chunk-interior steps (the natural motion baseline)
  max_step   -- worst adjacent-step delta anywhere (evaluator jump-check proxy)
  mse_gt     -- MSE of the assembled trajectory vs dataset ground-truth actions

Also reports raw open-loop error by horizon depth (mode-independent), which is
what bounds how deep a chunk is worth consuming.

Usage:
  PYTHONPATH=<submission>:<submission>/vitacformer:<submission>/vitacformer/detr \\
  python eval_smoothing.py --dataset_root ... --ckpt_dir ... --episode 0
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from smoothing import MODES, smooth_chunk

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TACTILE_H = 18  # detr_vae.py: architecturally fixed


def build_policy(ckpt_dir: Path, device):
    """Rebuild the architecture from whichever record the run left behind.

    Newer runs (and make_untrained_ckpt.py) put a policy_config dict straight
    into training_configs.json. The aug20 runs wrote a flat config instead, so
    fall back to parsing the info log's "Policy Config" block.
    """
    from policy import ACTPolicy

    cfg_path = ckpt_dir / "training_configs.json"
    pc = None
    if cfg_path.exists():
        pc = json.loads(cfg_path.read_text()).get("policy_config")

    if pc is None:
        info = next(ckpt_dir.glob("info_*.log"))
        cfg, in_block = {}, False
        for line in info.read_text().splitlines():
            if line.startswith("--- Policy Config ---"):
                in_block = True
                continue
            if in_block:
                if not line.strip():
                    break
                k, _, v = line.strip().partition(": ")
                cfg[k] = v
        pc = {
            "lr": float(cfg["lr"]), "num_queries": int(cfg["num_queries"]),
            "kl_weight": int(cfg["kl_weight"]), "hidden_dim": int(cfg["hidden_dim"]),
            "dim_feedforward": int(cfg["dim_feedforward"]), "lr_backbone": float(cfg["lr_backbone"]),
            "backbone": cfg["backbone"], "enc_layers": int(cfg["enc_layers"]),
            "dec_layers": int(cfg["dec_layers"]), "nheads": int(cfg["nheads"]),
            "camera_names": eval(cfg["camera_names"]), "use_tactile": cfg["use_tactile"] == "True",
            "state_dim": int(cfg["state_dim"]),
            "proprioceptive_temporal_horizon": int(cfg["proprioceptive_temporal_horizon"]),
        }
    pc = dict(pc)
    pc["backbone_weights"] = None   # random init here; load_state_dict overwrites anyway
    ckpt_path = next(ckpt_dir.glob("*.ckpt"))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy = ACTPolicy(pc)
    policy.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    policy.eval().to(device)
    return policy, pc, ckpt_path.name


def build_normalizer(ckpt_dir: Path, device):
    from my_utils.normalizer import OriNormalizer

    cfg = json.loads((ckpt_dir / "normalizer_config.json").read_text())
    if cfg.get("disable_normalization", False):
        return None, cfg.get("image_norm", True)
    n = OriNormalizer(stats=cfg["stats"], feature_modes=cfg["feature_modes"], device=device,
                      degenerate_spread=cfg.get("degenerate_spread", 1e-3), clip=cfg.get("clip"))
    return n, cfg.get("image_norm", True)


def load_episode(dataset_root: Path, episode: int, camera_names, n_frames: int):
    """Frame-indexed access with no delta_timestamps: the server gets exactly one
    current observation per call and rebuilds history itself."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=None, root=dataset_root, episodes=[episode],
                        video_backend="pyav", tolerance_s=0.02)
    total = len(ds)
    n = min(n_frames, total)
    frames = []
    for i in range(n):
        s = ds[i]
        frames.append({
            "state": s["observation.state"].numpy().astype(np.float32),
            "tactile": s["observation.tactile"].numpy().astype(np.float32),
            "action": s["action"].numpy().astype(np.float32),
            "images": {c: s[c] for c in camera_names},
        })
    return frames, total


class Runner:
    """Mirrors TeamPolicy history reconstruction + preprocessing."""

    def __init__(self, policy, pc, normalizer, use_image_norm, device):
        self.policy, self.pc, self.normalizer = policy, pc, normalizer
        self.use_image_norm, self.device = use_image_norm, device
        self.proprio_h = pc["proprioceptive_temporal_horizon"]
        self.cams = pc["camera_names"]
        self.reset()

    def reset(self):
        self.state_hist = collections.deque(maxlen=self.proprio_h)
        self.tactile_hist = collections.deque(maxlen=TACTILE_H + 1)

    def _img(self, t):
        x = t.unsqueeze(0).float()
        if x.max() > 1.5:
            x = x / 255.0
        if self.use_image_norm:
            mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
            std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
            x = (x - mean) / std
        return x

    @torch.inference_mode()
    def infer(self, frame):
        img = torch.stack([self._img(frame["images"][c]) for c in self.cams], dim=1).to(self.device)

        st = frame["state"]
        if not self.state_hist:
            for _ in range(self.proprio_h):
                self.state_hist.append(st)
        else:
            self.state_hist.append(st)
        q = torch.from_numpy(np.stack(list(self.state_hist))).to(self.device)
        if self.normalizer is not None:
            q = self.normalizer.normalize("observation.state", q)
        qpos = q.reshape(1, -1)

        tac = frame["tactile"]
        if not self.tactile_hist:
            for _ in range(TACTILE_H + 1):
                self.tactile_hist.append(tac)
        else:
            self.tactile_hist.append(tac)
        t = torch.from_numpy(np.stack(list(self.tactile_hist))).to(self.device)
        if self.normalizer is not None:
            t = self.normalizer.normalize("observation.tactile", t)
        tac_in = torch.cat([t[1:], torch.diff(t, dim=0)], dim=-1).reshape(1, -1)
        tac_next = torch.zeros((1, TACTILE_H, 120), device=self.device)

        a = self.policy(qpos, img, device=self.device, tactile=tac_in, tactile_next=tac_next)
        if self.normalizer is not None:
            a = self.normalizer.denormalize("action", a.float())
        return a[0].cpu().numpy().astype(np.float32)


def overlap_disagreement(raw, stride):
    """How much do consecutive chunks disagree about the SAME absolute timesteps?
    Pure model-level measure of the artifact -- independent of execution regime
    and of any smoothing, so it isolates what the seam fix is up against."""
    vals = []
    for a, b in zip(raw, raw[1:]):
        n = min(len(a) - stride, len(b))
        if n > 0:
            vals.append(np.abs(a[stride:stride + n] - b[:n]))
    if not vals:
        return {}
    v = np.concatenate([x.reshape(-1) for x in vals])
    first = np.concatenate([x[0] for x in vals])  # disagreement at the seam row itself
    return {"overlap_mean": float(v.mean()), "overlap_max": float(v.max()),
            "seam_row_mean": float(first.mean()), "seam_row_max": float(first.max())}


def metrics(traj, gt, stride):
    d = np.abs(np.diff(traj, axis=0))                       # [N-1, 65]
    seam_idx = [i for i in range(len(d)) if (i + 1) % stride == 0]
    step_idx = [i for i in range(len(d)) if (i + 1) % stride != 0]
    f = lambda idx: (float(d[idx].mean()), float(d[idx].max())) if idx else (float("nan"),) * 2
    seam_mean, seam_max = f(seam_idx)
    step_mean, step_max = f(step_idx)
    return {
        "seam_jump_mean": seam_mean, "seam_jump_max": seam_max,
        "step_jump_mean": step_mean, "step_jump_max": step_max,
        "max_step": float(d.max()),
        "seam_over_step": (seam_mean / step_mean) if step_mean and step_mean == step_mean else float("nan"),
        "mse_gt": float(((traj - gt) ** 2).mean()),
        # Guards against "smooth because frozen": a mode that stops commanding
        # motion scores perfectly on every jump metric. Compare against gt_path.
        "path_len": float(d.sum()),
        "gt_path_len": float(np.abs(np.diff(gt, axis=0)).sum()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=Path, required=True)
    p.add_argument("--ckpt_dir", type=Path, required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--horizon", type=int, default=25, help="deployed action_horizon")
    p.add_argument("--total_steps", type=int, default=100)
    p.add_argument("--strides", type=str, default="1,5,10,25")
    p.add_argument("--regime", choices=["dataset", "shadow"], default="dataset",
                    help="dataset: obs from the episode each call (organizer re-observes). "
                         "shadow: images/tactile frozen at the first frame and state fed from "
                         "our own last executed row -- the open-loop rollout "
                         "remote_participant_development.md describes when horizon < 100.")
    p.add_argument("--depth_points", type=int, default=16,
                    help="query points spread across the loaded window for the depth curve")
    p.add_argument("--blend_steps", type=int, default=4)
    p.add_argument("--ensemble_decay", type=float, default=0.35)
    p.add_argument("--max_step_rad", type=float, default=0.10)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, pc, ckpt_name = build_policy(args.ckpt_dir, device)
    normalizer, use_image_norm = build_normalizer(args.ckpt_dir, device)
    print(f"[eval] ckpt={ckpt_name} device={device} num_queries={pc['num_queries']}")

    need = args.total_steps + pc["num_queries"] + 1
    frames, ep_len = load_episode(args.dataset_root, args.episode, pc["camera_names"], need)
    gt = np.stack([f["action"] for f in frames])
    print(f"[eval] episode {args.episode}: {ep_len} frames, loaded {len(frames)}")

    strides = [int(s) for s in args.strides.split(",")]
    runner = Runner(policy, pc, normalizer, use_image_norm, device)
    results, disagree = {}, {}

    # --- depth curve: raw open-loop error vs horizon depth, from query points
    # spread across the window (mode- and stride-independent). Bounds how deep a
    # chunk is worth consuming before the prediction stops being trustworthy.
    nq = pc["num_queries"]
    pts = np.linspace(0, max(0, len(frames) - nq - 1), args.depth_points).astype(int)
    errs = np.zeros(nq)
    runner.reset()
    for f0 in pts:
        ch = runner.infer(frames[int(f0)])
        errs += ((ch - gt[f0:f0 + nq]) ** 2).mean(axis=1)
    depth_mse = (errs / len(pts)).tolist()
    print(f"[eval] depth curve from {len(pts)} query points")

    for stride in strides:
        q_at = list(range(0, args.total_steps, stride))
        t0 = time.time()

        if args.regime == "dataset":
            # Model output is independent of smoothing here (obs come from the
            # episode), so run the model once and post-process per mode.
            runner.reset()
            raw = [runner.infer(frames[f0]) for f0 in q_at]
            states = [frames[f0]["state"] for f0 in q_at]
            per_mode_raw = {m: (raw, states) for m in MODES}
            disagree[stride] = overlap_disagreement(raw, stride)
        else:
            # Shadow: state feeds back from our own output, so each mode needs
            # its own rollout. Images/tactile stay frozen at the first frame.
            per_mode_raw = {}
            for mode in MODES:
                runner.reset()
                raw, states, st, prev = [], [], frames[0]["state"], None
                for _ in q_at:
                    f = dict(frames[0]); f["state"] = st
                    ch = runner.infer(f)
                    raw.append(ch); states.append(st)
                    out = ch[:args.horizon]
                    if mode != "none":
                        out = smooth_chunk(out, st, prev, mode, args.blend_steps,
                                           args.ensemble_decay, args.max_step_rad)
                    prev = out
                    st = out[:stride][-1]   # organizer's local state update
                per_mode_raw[mode] = (raw, states)
            disagree[stride] = overlap_disagreement(per_mode_raw["none"][0], stride)

        dt = (time.time() - t0) / max(1, len(q_at))
        print(f"[eval] stride={stride}: {len(q_at)} queries/rollout, {dt*1000:.0f} ms/query")

        for mode in MODES:
            raw, states = per_mode_raw[mode]
            prev, traj = None, []
            for ch, st in zip(raw, states):
                out = ch[:args.horizon]
                if mode != "none":
                    out = smooth_chunk(out, st, prev, mode, args.blend_steps,
                                       args.ensemble_decay, args.max_step_rad)
                prev = out
                traj.append(out[:stride])
            traj = np.concatenate(traj)[:args.total_steps]
            results[f"stride{stride}/{mode}"] = metrics(traj, gt[:len(traj)], stride)

    print(f"\n{'config':28s} {'seam':>10s} {'max_step':>9s} {'seam/step':>10s} "
          f"{'mse_gt':>9s} {'path':>8s} {'gt_path':>8s}")
    print("-" * 92)
    for k, m in results.items():
        print(f"{k:28s} {m['seam_jump_mean']:10.5f} {m['max_step']:9.4f} "
              f"{m['seam_over_step']:10.2f} {m['mse_gt']:9.5f} "
              f"{m['path_len']:8.2f} {m['gt_path_len']:8.2f}")

    print(f"\nconsecutive-chunk disagreement on shared timesteps (raw model, no smoothing):")
    for stride, d in disagree.items():
        if d:
            print(f"  stride={stride:3d}: seam_row_mean={d['seam_row_mean']:.5f} "
                  f"seam_row_max={d['seam_row_max']:.4f} overlap_mean={d['overlap_mean']:.5f}")

    print(f"\nraw open-loop MSE by horizon depth (ckpt={ckpt_name}, {len(pts)} query points):")
    row = "  ".join(f"d{d}={depth_mse[d-1]:.4f}" for d in (1, 5, 10, 25, 50, 100) if d <= len(depth_mse))
    print(f"  {row}")

    if args.out:
        args.out.write_text(json.dumps({
            "ckpt": ckpt_name, "ckpt_dir": str(args.ckpt_dir), "episode": args.episode,
            "regime": args.regime, "horizon": args.horizon, "total_steps": args.total_steps,
            "params": {"blend_steps": args.blend_steps, "ensemble_decay": args.ensemble_decay,
                       "max_step_rad": args.max_step_rad},
            "results": results, "depth_mse": depth_mse, "disagreement": disagree,
        }, indent=1))
        print(f"\n[eval] wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
