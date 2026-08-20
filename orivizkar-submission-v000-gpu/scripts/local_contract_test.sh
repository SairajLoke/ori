#!/usr/bin/env bash
# =============================================================================
# Local contract test -- one-shot driver for docs/competition_participant_
# complete_guide.md section 13 ("Start the Router, Image, and Black-Box
# Validator Locally").
#
# Brings up the Zenoh router + policy container on a private docker network,
# waits for the policy server to report READY, opens separate terminals for
# the router/policy logs, then runs examples/check_zenoh_policy.py against it.
#
# Every container is force-removed by name before being (re)created, so a
# crashed or half-cleaned previous run never blocks this one.
#
#   ./scripts/local_contract_test.sh                 # synthetic + dataset checks
#   ./scripts/local_contract_test.sh --obs-type synthetic
#   ./scripts/local_contract_test.sh --build         # docker build first
#   ./scripts/local_contract_test.sh --no-terminals  # logs inline, no GUI windows
#   ./scripts/local_contract_test.sh --keep          # leave containers up afterwards
#   ./scripts/local_contract_test.sh down            # just tear everything down
#
# Every variable below can be overridden from the environment, e.g.
#   IMAGE=orvizkar/origami-policy:submission-vitac000 ./scripts/local_contract_test.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMISSION_DIR="$(dirname "$SCRIPT_DIR")"
ORI_ROOT="$(dirname "$SUBMISSION_DIR")"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Router image pinned by digest in the public docs -- do not float this tag.
ROUTER_IMAGE="${ROUTER_IMAGE:-eclipse/zenoh@sha256:157965d71e0bfd0a044d76a985ff0e5c306ad3968929168fb9678cd2a7fec23f}"
IMAGE="${IMAGE:-vitac-policy:dev}"
SESSION="${SESSION:-local-contract-test}"

NETWORK="${NETWORK:-origami-contract-test}"
ROUTER_NAME="${ROUTER_NAME:-origami-contract-router}"
POLICY_NAME="${POLICY_NAME:-origami-contract-policy}"

ROUTER_PORT="${ROUTER_PORT:-7447}"
HOST_PORT="${HOST_PORT:-17447}"

# Must equal the image's real fixed horizon (Dockerfile ORIGAMI_ACTION_HORIZON).
EXPECTED_HORIZON="${EXPECTED_HORIZON:-25}"

SDK_DIR="${SDK_DIR:-${ORI_ROOT}/origami-inference-kit-participant/sharpa_north_ces_lite_sdk-main}"
VALIDATOR="${VALIDATOR:-${SDK_DIR}/examples/check_zenoh_policy.py}"

# Validator behaviour. OBS_TYPE is synthetic | dataset | both -- 'dataset' and
# 'both' replay real frames through infer() (the local addition to
# check_zenoh_policy.py), so they need DATASET_ROOT to exist.
OBS_TYPE="${OBS_TYPE:-both}"
REQUESTS="${REQUESTS:-3}"
DATASET_ROOT="${DATASET_ROOT:-/media/sai/CRUZER_BLA/ori/dataset/season_POC22061_2026_07_09_16_23_46_train/lerobot3.0}"
EPISODE_INDEX="${EPISODE_INDEX:-0}"
DATASET_REQUESTS="${DATASET_REQUESTS:-3}"
# Stride == horizon gives non-overlapping action chunks.
FRAME_STRIDE="${FRAME_STRIDE:-$EXPECTED_HORIZON}"
OUT_DIR="${OUT_DIR:-dataset_checks}"
QUERY_TIMEOUT="${QUERY_TIMEOUT:-180}"

# How long to wait for the policy container to log READY. A cold start that
# builds a torch.compile graph is slow; 300s is deliberately generous.
READY_TIMEOUT="${READY_TIMEOUT:-300}"

# Production-parity sandbox limits (docs section 13 / 15).
MEMORY="${MEMORY:-32g}"
CPUS="${CPUS:-8}"
SHM_SIZE="${SHM_SIZE:-8g}"

DO_BUILD=0
USE_TERMINALS=1
KEEP_UP=0
USE_GPU="auto"
RELAX_SANDBOX=0
SUBCOMMAND="up"

# ---------------------------------------------------------------------------
# Pretty output
# ---------------------------------------------------------------------------

if [ -t 1 ]; then
    RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'
    BLUE=$'\033[0;34m'; BOLD=$'\033[1m'; NC=$'\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; BLUE=''; BOLD=''; NC=''
fi

log()      { echo "${BLUE}[info]${NC} $*"; }
ok()       { echo "${GREEN}[ ok ]${NC} $*"; }
warn()     { echo "${YELLOW}[warn]${NC} $*"; }
err()      { echo "${RED}[fail]${NC} $*" >&2; }
section()  { echo; echo "${BOLD}=== $* ===${NC}"; }

die() { err "$*"; exit 1; }

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

usage() {
    sed -n '3,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        up|down)          SUBCOMMAND="$1"; shift ;;
        --build)          DO_BUILD=1; shift ;;
        --no-terminals)   USE_TERMINALS=0; shift ;;
        --keep)           KEEP_UP=1; shift ;;
        --gpu)            USE_GPU="yes"; shift ;;
        --no-gpu)         USE_GPU="no"; shift ;;
        --relax-sandbox)  RELAX_SANDBOX=1; shift ;;
        --obs-type)       OBS_TYPE="$2"; shift 2 ;;
        --image)          IMAGE="$2"; shift 2 ;;
        --dataset-root)   DATASET_ROOT="$2"; shift 2 ;;
        --episode-index)  EPISODE_INDEX="$2"; shift 2 ;;
        --requests)       REQUESTS="$2"; shift 2 ;;
        --dataset-requests) DATASET_REQUESTS="$2"; shift 2 ;;
        --horizon)        EXPECTED_HORIZON="$2"; FRAME_STRIDE="$2"; shift 2 ;;
        -h|--help)        usage ;;
        *)                die "unknown argument: $1  (try --help)" ;;
    esac
done

case "$OBS_TYPE" in
    synthetic|dataset|both) ;;
    *) die "--obs-type must be one of: synthetic, dataset, both (got '$OBS_TYPE')" ;;
esac

# ---------------------------------------------------------------------------
# Teardown -- always force-remove by name so a stale container never blocks us
# ---------------------------------------------------------------------------

teardown() {
    log "removing containers/network (ignoring absent ones)"
    docker rm -f "$POLICY_NAME" >/dev/null 2>&1 || true
    docker rm -f "$ROUTER_NAME" >/dev/null 2>&1 || true
    docker network rm "$NETWORK" >/dev/null 2>&1 || true
}

if [ "$SUBCOMMAND" = "down" ]; then
    section "Teardown"
    teardown
    ok "torn down"
    exit 0
fi

on_exit() {
    local rc=$?
    if [ "$KEEP_UP" = "1" ]; then
        echo
        log "--keep set: leaving containers running."
        log "  policy logs : docker logs -f $POLICY_NAME"
        log "  tear down   : $0 down"
    else
        section "Cleanup"
        teardown
    fi
    exit $rc
}
trap on_exit EXIT

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

section "Preflight"

command -v docker >/dev/null 2>&1 || die "docker not found on PATH"
docker info >/dev/null 2>&1 || die "cannot talk to the docker daemon (is it running? are you in the docker group?)"
ok "docker reachable"

# GPU: this only works if the nvidia container runtime is actually registered.
# Passing --gpus all without it fails the container at start with an opaque
# error, so detect rather than assume.
if [ "$USE_GPU" = "auto" ]; then
    if docker info 2>/dev/null | grep -qi 'runtimes:.*nvidia'; then
        USE_GPU="yes"
    else
        USE_GPU="no"
    fi
fi
if [ "$USE_GPU" = "yes" ]; then
    ok "nvidia container runtime present -- passing --gpus all"
    GPU_ARGS=(--gpus all)
else
    warn "no nvidia container runtime -- running CPU-only (--gpus all omitted)"
    warn "  interface/contract checks are still valid; latency numbers are not"
    GPU_ARGS=()
fi

# Router image: pinned by digest, so a missing one needs a network pull.
if ! docker image inspect "$ROUTER_IMAGE" >/dev/null 2>&1; then
    warn "router image not present locally, pulling: $ROUTER_IMAGE"
    docker pull "$ROUTER_IMAGE" || die "could not pull the router image"
fi
ok "router image present"

# Optional build. Do it before the image-exists check so --build can create it.
if [ "$DO_BUILD" = "1" ]; then
    section "Build"
    # The Dockerfile COPYs the checkpoint set from checkpoints/ -- all three
    # files, since vitac_policy_server.py rebuilds architecture and
    # normalization from the two sidecars rather than hardcoded constants.
    missing=()
    for f in training_configs.json normalizer_config.json; do
        [ -s "${SUBMISSION_DIR}/checkpoints/${f}" ] || missing+=("$f")
    done
    if ! ls "${SUBMISSION_DIR}"/checkpoints/*.ckpt >/dev/null 2>&1; then
        missing+=("<policy>.ckpt")
    fi
    if [ ${#missing[@]} -gt 0 ]; then
        err "checkpoints/ is missing: ${missing[*]}"
        err "  Copy the .ckpt plus the training_configs.json and"
        err "  normalizer_config.json written beside it during training into"
        err "  ${SUBMISSION_DIR}/checkpoints/ before building."
        exit 1
    fi
    log "docker build -t $IMAGE $SUBMISSION_DIR"
    docker build -t "$IMAGE" "$SUBMISSION_DIR"
    ok "built $IMAGE"
fi

docker image inspect "$IMAGE" >/dev/null 2>&1 \
    || die "policy image not found: $IMAGE  (build it, or pass --build / --image <name>)"
ok "policy image present: $IMAGE"

[ -f "$VALIDATOR" ] || die "validator not found: $VALIDATOR"
command -v uv >/dev/null 2>&1 || die "uv not found on PATH (needed to run the validator)"
ok "validator present"

if [ "$OBS_TYPE" != "synthetic" ]; then
    if [ ! -d "$DATASET_ROOT" ]; then
        err "dataset root not found: $DATASET_ROOT"
        err "  Pass --dataset-root <path-to-.../lerobot3.0>, or use --obs-type synthetic."
        exit 1
    fi
    ok "dataset root present: $DATASET_ROOT"
fi

if ss -ltn 2>/dev/null | grep -q ":${HOST_PORT}\b"; then
    warn "host port ${HOST_PORT} looks busy -- if the router fails to bind, free it or set HOST_PORT"
fi

# ---------------------------------------------------------------------------
# Bring up router + policy
# ---------------------------------------------------------------------------

section "Reset"
teardown

section "Network"
docker network create "$NETWORK" >/dev/null
ok "created network $NETWORK"

section "Router"
docker run -d --name "$ROUTER_NAME" \
    --network "$NETWORK" \
    -p "127.0.0.1:${HOST_PORT}:${ROUTER_PORT}" \
    "$ROUTER_IMAGE" \
    -l "tcp/0.0.0.0:${ROUTER_PORT}" \
    --no-multicast-scouting \
    --cfg 'transport/shared_memory/enabled:false' >/dev/null
ok "router up on 127.0.0.1:${HOST_PORT} (container port ${ROUTER_PORT})"

section "Policy"
# Sandbox flags mirror the production evaluation sandbox from docs section 13:
# read-only root, no capabilities, non-root uid, tmpfs for the writable paths
# the entrypoint needs. --relax-sandbox drops them when debugging a container
# that only misbehaves under those constraints.
SANDBOX_ARGS=(
    --read-only
    --cap-drop ALL
    --security-opt no-new-privileges=true
    --user 65532:65532
    --tmpfs /tmp:rw,noexec,nosuid,nodev,size=4g
    --tmpfs /run:rw,noexec,nosuid,nodev,size=64m
    --pids-limit 512
)
if [ "$RELAX_SANDBOX" = "1" ]; then
    warn "--relax-sandbox: dropping read-only/cap-drop/non-root constraints"
    warn "  a pass here does NOT prove the image passes the real sandbox"
    SANDBOX_ARGS=()
fi

docker run -d --name "$POLICY_NAME" \
    --network "$NETWORK" \
    "${GPU_ARGS[@]}" \
    "${SANDBOX_ARGS[@]}" \
    --shm-size "$SHM_SIZE" \
    --memory "$MEMORY" \
    --cpus "$CPUS" \
    -e ORIGAMI_ZENOH_ENDPOINT="tcp/${ROUTER_NAME}:${ROUTER_PORT}" \
    -e ORIGAMI_SESSION_ID="$SESSION" \
    "$IMAGE" >/dev/null
ok "policy container started"

# ---------------------------------------------------------------------------
# Log terminals
# ---------------------------------------------------------------------------

open_term() {
    # open_term <title> <command>  -- best-effort; never fatal.
    local title="$1" cmd="$2"
    if [ "$USE_TERMINALS" != "1" ] || [ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
        return 1
    fi
    if command -v tilix >/dev/null 2>&1; then
        tilix --new-process -t "$title" -e bash -c "$cmd" >/dev/null 2>&1 &
    elif command -v gnome-terminal >/dev/null 2>&1; then
        gnome-terminal --title="$title" -- bash -c "$cmd" >/dev/null 2>&1 &
    elif command -v xterm >/dev/null 2>&1; then
        xterm -T "$title" -e bash -c "$cmd" >/dev/null 2>&1 &
    else
        return 1
    fi
    return 0
}

TERMS_OPENED=0
if [ "$USE_TERMINALS" = "1" ]; then
    section "Log terminals"
    # `exec bash` keeps the window alive after the stream ends, so a container
    # that dies leaves its final traceback on screen instead of vanishing.
    if open_term "origami: policy logs" \
        "echo '=== docker logs -f ${POLICY_NAME} ==='; docker logs -f '${POLICY_NAME}'; echo; echo '[stream ended]'; exec bash"; then
        TERMS_OPENED=1
        ok "opened policy log window"
    fi
    if open_term "origami: router logs" \
        "echo '=== docker logs -f ${ROUTER_NAME} ==='; docker logs -f '${ROUTER_NAME}'; echo; echo '[stream ended]'; exec bash"; then
        ok "opened router log window"
    fi
    if [ "$TERMS_OPENED" = "0" ]; then
        warn "no GUI terminal available -- falling back to inline log tail"
    fi
fi

# ---------------------------------------------------------------------------
# Wait for READY
# ---------------------------------------------------------------------------

section "Waiting for policy READY (timeout ${READY_TIMEOUT}s)"
log "first start is slow: torch import, checkpoint load, optional torch.compile"

deadline=$(( SECONDS + READY_TIMEOUT ))
ready=0
while [ $SECONDS -lt $deadline ]; do
    # A dead container will never print READY -- bail out immediately with its
    # logs rather than burning the whole timeout.
    state="$(docker inspect -f '{{.State.Status}}' "$POLICY_NAME" 2>/dev/null || echo missing)"
    if [ "$state" != "running" ]; then
        err "policy container is '$state' -- it exited during startup"
        echo "${BOLD}---- last 60 log lines ----${NC}"
        docker logs --tail 60 "$POLICY_NAME" 2>&1 || true
        echo "${BOLD}---------------------------${NC}"
        exit 1
    fi
    if docker logs "$POLICY_NAME" 2>&1 | grep -q 'READY transport='; then
        ready=1
        break
    fi
    sleep 2
done

if [ "$ready" != "1" ]; then
    err "policy did not report READY within ${READY_TIMEOUT}s"
    echo "${BOLD}---- last 60 log lines ----${NC}"
    docker logs --tail 60 "$POLICY_NAME" 2>&1 || true
    echo "${BOLD}---------------------------${NC}"
    exit 1
fi
ok "policy READY"
docker logs "$POLICY_NAME" 2>&1 | grep 'READY transport=' | tail -1

if [ "$TERMS_OPENED" = "0" ]; then
    echo "${BOLD}---- policy startup log ----${NC}"
    docker logs --tail 30 "$POLICY_NAME" 2>&1 || true
    echo "${BOLD}----------------------------${NC}"
fi

# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

section "Validator (obs-type=${OBS_TYPE})"

VAL_ARGS=(
    --endpoint "tcp/127.0.0.1:${HOST_PORT}"
    --session-id "$SESSION"
    --timeout "$QUERY_TIMEOUT"
    --expected-horizon "$EXPECTED_HORIZON"
    --obs-type "$OBS_TYPE"
)
# --requests drives the synthetic queries; the dataset flags drive the real
# replay. 'both' needs each set, so add them per obs-type rather than always.
if [ "$OBS_TYPE" = "synthetic" ] || [ "$OBS_TYPE" = "both" ]; then
    VAL_ARGS+=(--requests "$REQUESTS")
fi
if [ "$OBS_TYPE" = "dataset" ] || [ "$OBS_TYPE" = "both" ]; then
    VAL_ARGS+=(
        --dataset-root "$DATASET_ROOT"
        --episode-index "$EPISODE_INDEX"
        --dataset-requests "$DATASET_REQUESTS"
        --frame-stride "$FRAME_STRIDE"
        --out-dir "$OUT_DIR"
    )
fi

log "cd $SDK_DIR"
log "uv run --no-sync python examples/check_zenoh_policy.py ${VAL_ARGS[*]}"
echo

set +e
# VIRTUAL_ENV is unset so uv resolves the SDK's own .venv rather than warning
# about (and ignoring) whatever venv happens to be active in the calling shell.
( cd "$SDK_DIR" && env -u VIRTUAL_ENV uv run --no-sync python "$VALIDATOR" "${VAL_ARGS[@]}" )
VAL_RC=$?
set -e

section "Result"
if [ $VAL_RC -eq 0 ]; then
    ok "validator PASSED (exit 0)"
    if [ "$OBS_TYPE" != "synthetic" ]; then
        log "dataset artifacts under: ${SDK_DIR}/${OUT_DIR}"
    fi
    warn "passing this proves interface compatibility only -- not task quality or action safety"
else
    err "validator FAILED (exit ${VAL_RC})"
    echo "${BOLD}---- last 40 policy log lines ----${NC}"
    docker logs --tail 40 "$POLICY_NAME" 2>&1 || true
    echo "${BOLD}----------------------------------${NC}"
fi

exit $VAL_RC
