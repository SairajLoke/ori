#!/usr/bin/env python3
"""Client half of the dummy network test -- pulls N frames from
mock_remote_server.py over a real Zenoh/TCP hop and captures them exactly
like run_evaluator_with_capture.py does (one file per key, docs order).

    uv run --no-sync python test_client.py \
        --session-id test-session --token test-token --port 17448 --count 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SDK_EXAMPLES = Path(__file__).resolve().parents[1] / (
    "origami-inference-kit-participant/sharpa_north_ces_lite_sdk-main/examples"
)
sys.path.insert(0, str(SDK_EXAMPLES))
from remote_observation_client import RemoteObservationClient, RemoteObservationError  # noqa: E402

# Same order as robot_io_spec.md §1 / run_evaluator_with_capture.py's DOC_KEYS.
DOC_KEYS = (
    "observation/image/head_left", "observation/image/head_right",
    "observation/image/wrist_left", "observation/image/wrist_right",
    "observation/state", "observation/state/joint_torque", "observation/tactile",
    "observation/image/tactile_deform", "observation/image/tactile_raw", "prompt",
)


def save_frame(out_dir: Path, i: int, obs: dict) -> None:
    for key in DOC_KEYS:
        if key not in obs:
            continue
        stem = f"obs_{i:06d}__{key.replace('/', '__')}"
        if key == "prompt":
            (out_dir / f"{stem}.txt").write_text(obs[key])
        else:
            np.save(out_dir / f"{stem}.npy", obs[key])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=17448)
    p.add_argument("--session-id", required=True)
    p.add_argument("--token", required=True)
    p.add_argument("--count", type=int, default=5)
    p.add_argument("--out-dir", type=Path, default=Path("dummy_captures"))
    args = p.parse_args()

    out_dir = args.out_dir / time.strftime("%m%d%H%M")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "capture_log.jsonl"

    ok = 0
    with RemoteObservationClient(
        f"tcp/127.0.0.1:{args.port}", session_id=args.session_id, token=args.token,
    ) as client, open(log_path, "a") as log:
        prev_t = None
        for i in range(args.count):
            t0 = time.time()
            try:
                obs = client.get_observation()
            except (RemoteObservationError, TimeoutError) as error:
                print(f"[{i}] FAIL: {error}", file=sys.stderr)
                continue
            gap_s = None if prev_t is None else t0 - prev_t
            prev_t = t0
            save_frame(out_dir, i, obs)
            log.write(json.dumps({
                "index": i, "local_request_time": t0, "latency_s": time.time() - t0,
                "server_observation_timestamp": client.last_observation_timestamp,
                "gap_since_prev_request_s": gap_s, "prompt": obs["prompt"],
            }) + "\n")
            print(f"[{i}] PASS state={obs['observation/state'].shape} "
                  f"prompt={obs['prompt']!r} latency={(time.time() - t0) * 1000:.0f}ms")
            ok += 1

    print(f"\n{ok}/{args.count} frames captured -> {out_dir}")
    return 0 if ok == args.count else 1


if __name__ == "__main__":
    raise SystemExit(main())
