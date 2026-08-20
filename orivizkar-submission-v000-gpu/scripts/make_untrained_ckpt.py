#!/usr/bin/env python3
"""Generate a randomly-initialised checkpoint set for integration testing.

Produces exactly what a real training run drops into its ckpt_dir --
a loadable .ckpt plus the training_configs.json and normalizer_config.json
sidecars vitac_policy_server.py rebuilds the deployed model from -- but with
untrained weights. That exercises the entire deploy path (architecture
reconstruction, normalization, image preprocessing, the Zenoh contract) while
a real checkpoint is still training.

The actions it predicts are meaningless. This validates plumbing, not policy.

Usage:
    python scripts/make_untrained_ckpt.py --out-dir checkpoints
    python scripts/make_untrained_ckpt.py --out-dir checkpoints --backbone vit_b_16
    python scripts/make_untrained_ckpt.py --out-dir checkpoints --no-normalization
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
SUBMISSION_DIR = SCRIPT_DIR.parent
VITACFORMER_ROOT = SUBMISSION_DIR / "vitacformer"

# Match how the Dockerfile wires PYTHONPATH, so this builds the same model the
# container will.
sys.path.insert(0, str(VITACFORMER_ROOT))
sys.path.insert(0, str(VITACFORMER_ROOT / "detr"))

# configs.py raises at import time when DATASET_ROOT is unset. Nothing here
# needs a dataset, but policy.py -> detr.main pulls configs in transitively.
os.environ.setdefault("DATASET_ROOT", "/nonexistent-not-used-by-this-script")

from policy import ACTPolicy  # noqa: E402

# Keys the training-time normalizer covers, and the mode each gets. Mirrors
# my_utils/normalizer.py::recommended_modes() -- kept in sync by hand because
# importing it would drag in the training-side config surface.
FEATURE_MODES = {
    "observation.state": "quantile",
    "action": "quantile",
    "observation.images.head_left": None,
    "observation.images.head_right": None,
    "observation.images.wrist_left": None,
    "observation.images.wrist_right": None,
    "observation.state.joint_torque": "quantile",
    "observation.tactile": "quantile",
    "observation.tactile_next": "quantile",
}
# normalizer_config.json stores stats under the DATASET's key names;
# OriNormalizer maps model keys onto them via key_aliases.
STATS_KEYS = {
    "observation.state": "observation.state",
    "action": "action",
    "observation.state.joint_torque": "observation.state.joint_torque",
    "observation.tactile": "observation.tactile",
    "observation.tactile_next": "observation.tactile",
}
STAT_FIELDS = ("mean", "std", "min", "max", "q01", "q99")


def build_policy_config(args) -> dict:
    """The ACTPolicy args_override dict, identical in shape to the one
    origami_imitate_episodes.py assembles and saves."""
    return {
        "lr": 3e-4,
        "num_queries": args.chunk_size,
        "kl_weight": 10,
        "hidden_dim": args.hidden_dim,
        "dim_feedforward": args.dim_feedforward,
        "lr_backbone": 1e-5,
        "backbone": args.backbone,
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "camera_names": [
            "observation.images.head_left",
            "observation.images.head_right",
            "observation.images.wrist_right",
            "observation.images.wrist_left",
        ],
        "use_tactile": True,
        "state_dim": args.state_dim,
        "proprioceptive_temporal_horizon": args.proprio_horizon,
        # Resolved locally at load time by vitac_policy_server.py, so the value
        # baked here is irrelevant -- null keeps it honest rather than
        # recording a path that only existed on some training machine.
        "backbone_weights": None,
        "vit_unfrozen_layers": None,
    }


def load_dataset_stats(dataset_root: Path) -> dict | None:
    """Pull real per-feature stats out of a LeRobot meta/stats.json.

    Using genuine statistics means the normalization path is exercised with
    realistic scales -- degenerate dims, shared tactile grouping and all --
    instead of the identity-ish behaviour synthetic stats would produce.
    """
    stats_path = dataset_root / "meta" / "stats.json"
    if not stats_path.exists():
        return None
    with open(stats_path) as f:
        raw = json.load(f)
    out = {}
    for key in set(STATS_KEYS.values()):
        src = raw.get(key)
        if src is None:
            continue
        out[key] = {
            field: np.asarray(src[field], dtype=np.float64).ravel().tolist()
            for field in STAT_FIELDS
            if field in src
        }
    return out or None


def synth_stats() -> dict:
    """Fallback stats when no dataset is reachable: unit-ish ranges, correct
    widths. Enough to keep normalize/denormalize well-defined."""
    widths = {
        "observation.state": 65,
        "action": 65,
        "observation.state.joint_torque": 65,
        "observation.tactile": 60,
    }
    out = {}
    for key, d in widths.items():
        out[key] = {
            "mean": [0.0] * d,
            "std": [1.0] * d,
            "min": [-1.0] * d,
            "max": [1.0] * d,
            "q01": [-0.5] * d,
            "q99": [0.5] * d,
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=str(SUBMISSION_DIR / "checkpoints"),
                   help="where to write the checkpoint set (default: checkpoints/)")
    p.add_argument("--name", default="policy_untrained.ckpt",
                   help="checkpoint filename")
    p.add_argument("--backbone", default="resnet18",
                   help="resnet18/34/50 or vit_b_16 etc -- must match what you deploy")
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--dim-feedforward", type=int, default=3200)
    p.add_argument("--state-dim", type=int, default=65)
    p.add_argument("--chunk-size", type=int, default=100)
    p.add_argument("--proprio-horizon", type=int, default=6)
    p.add_argument("--dataset-root", default=None,
                   help="LeRobot dataset root to lift real stats.json from")
    p.add_argument("--no-normalization", action="store_true",
                   help="write a disable_normalization=true sidecar (identity path)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("  Untrained checkpoint set for integration testing")
    print("=" * 68)
    print(f"  out-dir  : {out_dir}")
    print(f"  backbone : {args.backbone}")

    # --- model -------------------------------------------------------------
    policy_config = build_policy_config(args)
    print("\n[model] building ACTPolicy (random init)...")
    policy = ACTPolicy(policy_config)
    n_params = sum(t.numel() for t in policy.parameters())
    print(f"[model] {n_params:,} parameters")

    ckpt_path = out_dir / args.name
    # Same envelope the periodic training checkpoints use, so the server's
    # dict-with-'model' branch is the one under test.
    torch.save(
        {"model": policy.state_dict(), "epoch": 0, "global_step": 0,
         "untrained": True},
        ckpt_path,
    )
    print(f"[ckpt ] wrote {ckpt_path}  ({ckpt_path.stat().st_size / 1e6:.1f} MB)")

    # --- training_configs.json --------------------------------------------
    train_cfg = {
        "state_dim": args.state_dim,
        "policy_class": "ACT",
        "policy_config": policy_config,
        "camera_names": policy_config["camera_names"],
        "use_tactile": True,
        "mask_fingers": False,
        "hand_mask": [1] * 5 + [1] * 4 + [1] * 4 + [0] * 4 + [0] * 5,
        "image_hw": [224, 224],
        "disable_normalization": bool(args.no_normalization),
        "_note": "UNTRAINED integration-test checkpoint -- predictions are meaningless",
    }
    cfg_path = out_dir / "training_configs.json"
    with open(cfg_path, "w") as f:
        json.dump(train_cfg, f, indent=2)
    print(f"[cfg  ] wrote {cfg_path}")

    # --- normalizer_config.json -------------------------------------------
    if args.no_normalization:
        norm_cfg = {
            "feature_modes": {k: None for k in FEATURE_MODES},
            "stats": {},
            "degenerate_spread": 1e-3,
            "clip": {},
            "disable_normalization": True,
            "norm_disable_keys": [],
            "image_norm": True,
        }
        print("[norm ] identity (disable_normalization=true)")
    else:
        stats = None
        if args.dataset_root:
            stats = load_dataset_stats(Path(args.dataset_root))
            if stats:
                print(f"[norm ] real stats from {args.dataset_root}: {sorted(stats)}")
            else:
                print(f"[norm ] no meta/stats.json under {args.dataset_root}")
        if stats is None:
            stats = synth_stats()
            print("[norm ] synthetic unit-range stats (no dataset stats available)")
        # Drop modes whose stats are missing, otherwise OriNormalizer would
        # fail resolving them at load time on the inference side.
        modes = {}
        for key, mode in FEATURE_MODES.items():
            if mode is None:
                modes[key] = None
            elif STATS_KEYS.get(key) in stats:
                modes[key] = mode
        norm_cfg = {
            "feature_modes": modes,
            "stats": stats,
            "degenerate_spread": 1e-3,
            "clip": {},
            "disable_normalization": False,
            "norm_disable_keys": [],
            "image_norm": True,
        }
    norm_path = out_dir / "normalizer_config.json"
    with open(norm_path, "w") as f:
        json.dump(norm_cfg, f, indent=2)
    print(f"[norm ] wrote {norm_path}")

    # --- self-check: reload exactly the way the server does ----------------
    print("\n[check] reloading via the server's own code path...")
    reloaded = torch.load(ckpt_path, map_location="cpu")
    fresh = ACTPolicy(policy_config)
    fresh.load_state_dict(reloaded["model"])
    fresh.eval()

    qpos = torch.zeros(1, args.proprio_horizon * args.state_dim)
    image = torch.zeros(1, len(policy_config["camera_names"]), 3, 224, 224)
    tactile = torch.zeros(1, 18 * 120)
    tactile_next = torch.zeros(1, 18, 120)
    with torch.inference_mode():
        a_hat = fresh(qpos, image, device=torch.device("cpu"),
                      tactile=tactile, tactile_next=tactile_next)
    expected = (1, args.chunk_size, args.state_dim)
    assert tuple(a_hat.shape) == expected, f"{tuple(a_hat.shape)} != {expected}"
    assert torch.isfinite(a_hat).all(), "forward pass produced non-finite actions"
    print(f"[check] OK -- load_state_dict + forward -> {tuple(a_hat.shape)}, all finite")

    print("\nDone. Build the image with:")
    print(f"  ./scripts/local_contract_test.sh   # auto-detects {args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
