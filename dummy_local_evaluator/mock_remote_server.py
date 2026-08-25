#!/usr/bin/env python3
"""Mock origami-remote-v1 relay backed by real dataset frames.

Serves observations from a real LeRobot v3.0 episode over a genuine local
Zenoh network hop (tcp/127.0.0.1:<port>) -- exercises the real wire codec,
schema validation, and network path that RemoteObservationClient uses in
production, just pointed at localhost instead of the organizer's relay.
Does not touch anything under the inference/submission code -- reads the
dataset and reuses the SDK's own RealObservationSource read-only.

    uv run --no-sync python mock_remote_server.py \
        --dataset-root /media/sai/CRUZER_BLA/ori/dataset/.../lerobot3.0_shortgop15 \
        --session-id test-session --token test-token --port 17448
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import zenoh

SDK_EXAMPLES = Path(__file__).resolve().parents[1] / (
    "origami-inference-kit-participant/sharpa_north_ces_lite_sdk-main/examples"
)
sys.path.insert(0, str(SDK_EXAMPLES))
from real_observation_source import RealObservationSource  # noqa: E402
from remote_observation_client import (  # noqa: E402
    JOINT_NAMES,
    OBSERVATION_FIELD_METADATA,
    REMOTE_PROTOCOL_VERSION,
    SEMANTIC_PROTOCOL_VERSION,
    pack_payload,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--episode-index", type=int, default=0)
    p.add_argument("--session-id", required=True)
    p.add_argument("--token", required=True)
    p.add_argument("--port", type=int, default=17448)
    args = p.parse_args()

    source = RealObservationSource(
        dataset_root=args.dataset_root,
        drop_tactile_raw_every_n=0,  # never drop -- exercise the full schema
        episode_index=args.episode_index,
    )

    config = zenoh.Config()
    config.insert_json5("mode", json.dumps("peer"))
    config.insert_json5("listen/endpoints", json.dumps([f"tcp/0.0.0.0:{args.port}"]))
    config.insert_json5("scouting/multicast/enabled", "false")
    session = zenoh.open(config)

    def handle(query) -> None:
        try:
            import msgpack
            req = msgpack.unpackb(bytes(query.payload.to_bytes()), raw=False, strict_map_key=False)
        except Exception as error:
            print(f"malformed request: {error}", file=sys.stderr)
            return
        if req.get("token") != args.token or req.get("session_id") != args.session_id:
            reply = {
                "protocol_version": REMOTE_PROTOCOL_VERSION, "operation": "observation",
                "request_id": req.get("request_id"), "session_id": req.get("session_id"),
                "error": {"code": "UNAUTHORIZED", "message": "bad session/token", "retryable": False},
            }
            query.reply(str(query.key_expr), pack_payload(reply), encoding="application/msgpack")
            return
        if not source.has_next():
            source.reset_episode()
        observation = source.next_observation(frame_stride=1)
        reply = {
            "protocol_version": REMOTE_PROTOCOL_VERSION,
            "operation": "observation",
            "request_id": req["request_id"],
            "session_id": args.session_id,
            "observation_timestamp": time.time(),
            "metadata": {
                "protocol_version": SEMANTIC_PROTOCOL_VERSION,
                "observation_schema": "policy-infer-input",
                "observation_fields": OBSERVATION_FIELD_METADATA,
                "joint_names": list(JOINT_NAMES),
            },
            "observation": observation,
        }
        query.reply(str(query.key_expr), pack_payload(reply), encoding="application/msgpack")

    key = f"{REMOTE_PROTOCOL_VERSION}/{args.session_id}/observation"
    queryable = session.declare_queryable(key, handle, complete=True)
    print(f"mock relay READY key={key} endpoint=tcp/127.0.0.1:{args.port} "
          f"episode={args.episode_index} frames={len(source._frame_indices)}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    queryable.undeclare()
    session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
