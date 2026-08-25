#!/usr/bin/env python3
"""Pull remote observations in a loop and persist them for later reference.

Records, per frame: local request time, round-trip latency, the organizer's
own observation_timestamp, and the inter-arrival gap since the previous frame
(the actual observed frequency). Each observation's arrays are saved to an
.npz; one running JSONL line per frame carries the timing/metadata so the
whole session can be reviewed without reopening every .npz.

    uv run --no-sync python examples/capture_remote_observations.py \
        --out-dir captures/slot1 --count 200
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from examples.remote_observation_client import (  # noqa: E402
    RemoteObservationClient,
    RemoteObservationError,
)

RETRYABLE_BACKOFF_S = 1.0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint", default=__import__("os").environ.get("ORIGAMI_REMOTE_ENDPOINT"))
    p.add_argument("--session-id", default=__import__("os").environ.get("ORIGAMI_REMOTE_SESSION_ID"))
    p.add_argument("--token", default=__import__("os").environ.get("ORIGAMI_REMOTE_TOKEN"))
    p.add_argument("--tls-root-ca-certificate", default=__import__("os").environ.get("ORIGAMI_REMOTE_TLS_CA"))
    p.add_argument("--out-dir", required=True, type=pathlib.Path)
    p.add_argument("--count", type=int, default=0, help="0 = run until Ctrl-C")
    p.add_argument("--min-interval", type=float, default=0.0, help="floor between requests, seconds")
    args = p.parse_args(argv)
    if not (args.endpoint and args.session_id and args.token):
        p.error("--endpoint/--session-id/--token (or the ORIGAMI_REMOTE_* env vars) are required")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "capture_log.jsonl"

    with RemoteObservationClient(
        args.endpoint, session_id=args.session_id, token=args.token,
        tls_root_ca_certificate=args.tls_root_ca_certificate,
    ) as client, open(log_path, "a") as log:
        prev_request_time = None
        i = 0
        try:
            while args.count == 0 or i < args.count:
                t0 = time.time()
                try:
                    obs = client.get_observation()
                except RemoteObservationError as error:
                    print(f"[{i}] retryable error: {error}", file=sys.stderr)
                    time.sleep(RETRYABLE_BACKOFF_S)
                    continue
                latency_s = time.time() - t0
                gap_s = None if prev_request_time is None else t0 - prev_request_time
                prev_request_time = t0

                npz_path = args.out_dir / f"obs_{i:06d}.npz"
                np.savez_compressed(npz_path, **obs)  # arrays only; prompt handled below

                record = {
                    "index": i,
                    "local_request_time": t0,
                    "latency_s": latency_s,
                    "server_observation_timestamp": client.last_observation_timestamp,
                    "gap_since_prev_request_s": gap_s,
                    "prompt": obs["prompt"],
                    "npz": npz_path.name,
                }
                log.write(json.dumps(record) + "\n")
                log.flush()
                hz = f"{1.0 / gap_s:.2f} Hz" if gap_s else "n/a"
                print(f"[{i}] latency={latency_s * 1000:.0f}ms  freq={hz}  prompt={obs['prompt']!r}")

                i += 1
                if args.min_interval:
                    remaining = args.min_interval - (time.time() - t0)
                    if remaining > 0:
                        time.sleep(remaining)
        except KeyboardInterrupt:
            pass
    print(f"\ncaptured {i} frames -> {args.out_dir}  (log: {log_path.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
