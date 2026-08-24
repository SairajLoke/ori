#!/usr/bin/env bash
# Single entry point for the origami-remote-v1 workflow -- consolidates the
# scattered commands from docs/remote_participant_development.md and
# competition_participant_complete_guide.md §3.
#
#   ./remote_dev.sh check                           # one-shot connectivity test
#   ./remote_dev.sh evaluator [--capture-dir DIR]    # Shadow UI + obs capture, localhost:7861
#
# evaluator logs every frame it pulls (capture_log.jsonl + one .npz each) to
# --capture-dir (default captures/session) via a monkeypatch, not a second
# poller -- the remote interface only allows one concurrent request per
# session, so capture must ride the evaluator's own connection, not compete
# with it.
#
# Credentials come from ORIGAMI_REMOTE_* env vars. If set_remote_vars.sh
# exists next to this script, it is sourced automatically (see the security
# note printed below -- that file has real credentials and is git-tracked).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ -f set_remote_vars.sh ]; then
    # shellcheck source=/dev/null
    source set_remote_vars.sh
    echo "[warn] set_remote_vars.sh is git-tracked with a real token in it -- rotate/gitignore it." >&2
fi

: "${ORIGAMI_REMOTE_ENDPOINT:?export ORIGAMI_REMOTE_ENDPOINT or add it to set_remote_vars.sh}"
: "${ORIGAMI_REMOTE_SESSION_ID:?export ORIGAMI_REMOTE_SESSION_ID}"
: "${ORIGAMI_REMOTE_TOKEN:?export ORIGAMI_REMOTE_TOKEN}"

uv sync --frozen --no-install-project >/dev/null

case "${1:-}" in
    check)
        uv run --no-sync python examples/remote_observation_client.py
        ;;
    evaluator)
        shift
        ASSETS="${ORIGAMI_ROBOT_ASSETS_DIR:-/home/sai/Desktop/ORI/ori/north_poc2_2_urdf_usd}"
        uv run --no-sync python run_evaluator_with_capture.py \
            --robot-assets-dir "$ASSETS" --no-gpu "$@"
        ;;
    *)
        echo "usage: $0 {check|evaluator} [args...]" >&2
        exit 1
        ;;
esac
