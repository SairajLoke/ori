#!/usr/bin/env python3
"""Launch participant_local_evaluator with observation capture wired in.

Monkeypatches RemoteObservationClient.get_observation so every frame the
Shadow evaluator pulls (both "Connect and Read One Frame" and each Shadow
step) is also saved to --capture-dir -- same live connection, no second
poller competing for the remote interface's single-concurrency slot.

    uv run --no-sync python run_evaluator_with_capture.py \
        --robot-assets-dir ... --no-gpu [--capture-dir captures/slot1]
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys
import time

import numpy as np

from participant_local_evaluator import __main__ as evaluator_main
from participant_local_evaluator.remote_client import RemoteObservationClient

# Exact key order from docs/robot_io_spec.md §1 -- tactile_raw is the only
# optional field, so its absence in a given frame is expected, not an error.
DOC_KEYS = (
    "observation/image/head_left",
    "observation/image/head_right",
    "observation/image/wrist_left",
    "observation/image/wrist_right",
    "observation/state",
    "observation/state/joint_torque",
    "observation/tactile",
    "observation/image/tactile_deform",
    "observation/image/tactile_raw",
    "prompt",
)


def _save_frame(capture_dir: pathlib.Path, i: int, obs: dict) -> list[str]:
    """One file per key, key name embedded in the filename (docs order)."""
    saved = []
    for key in DOC_KEYS:
        if key not in obs:
            continue
        stem = f"obs_{i:06d}__{key.replace('/', '__')}"
        if key == "prompt":
            (capture_dir / f"{stem}.txt").write_text(obs[key])
        else:
            np.save(capture_dir / f"{stem}.npy", obs[key])
        saved.append(stem)
    return saved


def _install_capture(capture_dir: pathlib.Path) -> None:
    capture_dir.mkdir(parents=True, exist_ok=True)
    log_path = capture_dir / "capture_log.jsonl"
    state = {"i": 0, "prev_t": None}
    original = RemoteObservationClient.get_observation

    def patched(self):
        t0 = time.time()
        obs = original(self)
        i = state["i"]
        gap_s = None if state["prev_t"] is None else t0 - state["prev_t"]
        state["prev_t"] = t0
        state["i"] = i + 1

        saved = _save_frame(capture_dir, i, obs)
        record = {
            "index": i,
            "local_request_time": t0,
            "latency_s": time.time() - t0,
            "server_observation_timestamp": self.last_observation_timestamp,
            "gap_since_prev_request_s": gap_s,
            "prompt": obs["prompt"],
            "files": saved,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        hz = f"{1.0 / gap_s:.2f} Hz" if gap_s else "n/a"
        print(f"[capture {i}] freq={hz} prompt={obs['prompt']!r}", file=sys.stderr)
        return obs

    RemoteObservationClient.get_observation = patched


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--capture-dir", type=pathlib.Path, default=pathlib.Path("captures/session"))
    known, rest = pre.parse_known_args()
    # MdayHMin suffix (e.g. slot1 -> slot1_08232112) so re-runs never overwrite a prior capture.
    stamp = datetime.datetime.now().strftime("%m%d%H%M")
    capture_dir = known.capture_dir.parent / f"{known.capture_dir.name}_{stamp}"
    _install_capture(capture_dir)
    print(f"[capture] logging every evaluator frame to {capture_dir}", file=sys.stderr)
    return evaluator_main.main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
