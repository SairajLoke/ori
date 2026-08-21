"""Derive per-dimension action-loss weights from the dataset's torque signal.

Motion cannot tell a working finger from a passenger -- every finger moves about
the same amount (0.0047-0.0071 rad/step measured over 14 episodes). Torque can:
thumb/index/middle carry ~10x the load of ring/pinky and are far more consistent
episode to episode. So finger weights come from

    w = mean|torque| / (1 + CV)          CV = std/mean of per-episode mean|torque|

dividing by (1+CV) discounts joints whose force varies wildly between episodes
(ring/pinky sit at CV 0.25-0.82 vs 0.12-0.19 for the working fingers).

Arms are left uniform on purpose: their torque is dominated by gravity, not task
load -- shoulder j1 reads 18.8 while barely moving, wrist j5 reads 1.2 while
moving most. Torque there measures limb mass, so it is not an importance signal.

Also records dims whose action is constant across the whole dataset, with the
value to emit for them.

    python tools/compute_action_weights.py --dataset_root <lerobot3.0 root> \
        --out action_weights.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

# per-hand joint layout, 22 dof (see robot_io_spec.md section 2)
FINGERS = {"thumb": (0, 5), "index": (5, 9), "middle": (9, 13),
           "ring": (13, 17), "pinky": (17, 22)}
HANDS = {"left": 7, "right": 36}
CONSTANT_SPREAD = 1e-3   # dims with (max-min) below this are treated as constant


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("action_weights.json"))
    p.add_argument("--episodes", type=int, default=None, help="cap episode count")
    p.add_argument("--arm_weight", type=float, default=1.0)
    p.add_argument("--motor_weight", type=float, default=1.0)
    args = p.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    n_ep = args.episodes or json.loads(
        (args.dataset_root / "meta" / "info.json").read_text())["total_episodes"]

    torque, action, frames = [], [], 0
    for e in range(n_ep):
        ds = LeRobotDataset(repo_id=None, root=args.dataset_root, episodes=[e],
                            video_backend="pyav", tolerance_s=0.02)
        torque.append(np.asarray(ds.hf_dataset["observation.state.joint_torque"], dtype=np.float32))
        action.append(np.asarray(ds.hf_dataset["action"], dtype=np.float32))
        frames += len(torque[-1])
    print(f"[weights] {n_ep} episodes, {frames} frames")

    # finger weights: mean|torque| discounted by cross-episode inconsistency,
    # then symmetrised (one season's handedness should not enter the loss)
    raw = {}
    for f, (a, b) in FINGERS.items():
        vals = []
        for off in HANDS.values():
            per_ep = [np.abs(t[:, off + a:off + b]).mean() for t in torque]
            mt, cv = float(np.mean(per_ep)), float(np.std(per_ep) / (np.mean(per_ep) + 1e-12))
            vals.append(mt / (1.0 + cv))
        raw[f] = float(np.mean(vals))
    scale = float(np.mean(list(raw.values())))
    fingers = {f: round(v / scale, 4) for f, v in raw.items()}

    # constant action dims: nothing to learn, and emitting a fixed value is safer
    # than leaving an unsupervised dim free to drift into the evaluator's
    # jump/velocity check
    A = np.concatenate(action)
    spread = A.max(0) - A.min(0)
    constant = {int(i): round(float(A[:, i].mean()), 6)
                for i in np.where(spread < CONSTANT_SPREAD)[0]}

    out = {
        "fingers": fingers,
        "groups": {"left_arm": args.arm_weight, "right_arm": args.arm_weight,
                   "motor": args.motor_weight},
        "constant_dims": {str(k): v for k, v in constant.items()},
        "meta": {
            "source": str(args.dataset_root), "episodes": n_ep, "frames": frames,
            "generated": datetime.now().isoformat(timespec="seconds"),
            "method": "finger w = mean|torque|/(1+CV), symmetrised, mean-1 over fingers; "
                      "arms uniform (torque there is gravity-dominated)",
            "constant_spread_threshold": CONSTANT_SPREAD,
        },
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[weights] fingers: {fingers}")
    print(f"[weights] constant dims: {constant}")
    print(f"[weights] wrote {args.out}")


if __name__ == "__main__":
    main()
