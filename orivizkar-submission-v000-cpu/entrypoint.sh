#!/usr/bin/env bash

set -euo pipefail

# =============================================================================
# Validate required configuration -- fail fast, before paying the cost of
# importing torch and loading the checkpoint, if something is obviously wrong.
# =============================================================================

if [ -z "${ORIGAMI_ZENOH_ENDPOINT:-}" ]; then
    echo "ERROR: ORIGAMI_ZENOH_ENDPOINT is required" >&2
    echo "  Example: ORIGAMI_ZENOH_ENDPOINT=tcp/origami-router:7447" >&2
    exit 1
fi

if [ -z "${ORIGAMI_SESSION_ID:-}" ]; then
    echo "ERROR: ORIGAMI_SESSION_ID is required" >&2
    exit 1
fi

: "${VITAC_CKPT_PATH:=/app/checkpoints/policy_best.ckpt}"
: "${ORIGAMI_ACTION_HORIZON:=25}"

# vitacformer/ is resolved via PYTHONPATH (set in the Dockerfile), not an
# app-level env var -- check the fixed location the image places it at.
if [ ! -f "/app/vitacformer/policy.py" ]; then
    echo "ERROR: /app/vitacformer/policy.py not found" >&2
    echo "  Check the COPY paths and PYTHONPATH in the Dockerfile." >&2
    exit 1
fi

if [ ! -s "${VITAC_CKPT_PATH}" ]; then
    echo "ERROR: checkpoint not found (or empty) at VITAC_CKPT_PATH=${VITAC_CKPT_PATH}" >&2
    echo "  Place your trained .ckpt at this path before building the image." >&2
    exit 1
fi

# =============================================================================
# Writable cache/home directories (matches the Dockerfile's pre-created dirs;
# re-asserted here so the image also behaves correctly if HOME/XDG_CACHE_HOME
# are overridden at `docker run` time).
# =============================================================================

export HOME="${HOME:-/tmp/origami-home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/origami-cache}"
export IPYTHONDIR="${IPYTHONDIR:-/tmp/origami-ipython}"
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$IPYTHONDIR"

# =============================================================================
# Logging
# =============================================================================

echo "=============================================="
echo "ViTacFormer Policy Server"
echo "=============================================="
echo "  Endpoint:        ${ORIGAMI_ZENOH_ENDPOINT}"
echo "  Session ID:      ${ORIGAMI_SESSION_ID}"
echo "  Action horizon:  ${ORIGAMI_ACTION_HORIZON}"
echo "  Checkpoint:      ${VITAC_CKPT_PATH}"
echo "  BACKBONE_PATH    ${BACKBONE_PATH}" 
echo "  PYTHONPATH:      ${PYTHONPATH:-<unset>}"
echo "=============================================="

# =============================================================================
# Start Policy Server
# =============================================================================

exec python3 /app/vitac_policy_server.py
