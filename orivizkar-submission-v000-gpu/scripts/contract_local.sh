#!/usr/bin/env bash
# =============================================================================
# One-shot local contract run: build -> router -> policy -> validator, and leave
# the policy container up with its log streaming so you can watch it.
#
#   ./scripts/contract_local.sh              # build (cached) + run both checks
#   ./scripts/contract_local.sh logs         # just follow the policy log
#   ./scripts/contract_local.sh down         # tear everything down
#   ./scripts/contract_local.sh --no-build   # reuse the existing image as-is
#   ./scripts/contract_local.sh --prune-cache  # drop docker build cache first
#
# Everything is local: a bridge network, the Zenoh router published only on
# 127.0.0.1, and the policy container reaching it by container name. No traffic
# leaves the host and no GPU is required.
#
# torch.compile is OFF by default here (VITAC_OPTIMIZATION=none): on a CPU-only
# box the inductor cache is slow and can exhaust RAM, which kills the container
# during startup. The shipped image default is still "compile".
#
# Image hygiene: one fixed tag, force-removed containers by name, and the
# previous image ID is deleted after a rebuild if it was left dangling -- so
# repeated runs never accumulate copies. Layer cache is always used (the
# Dockerfile puts apt/torch/pip above the first source COPY, so editing
# vitac_policy_server.py or smoothing.py only re-runs cheap file-copy layers).
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMISSION_DIR="$(dirname "$SCRIPT_DIR")"
ORI_ROOT="$(dirname "$SUBMISSION_DIR")"

IMAGE="${IMAGE:-vitac-policy:dev}"
SESSION="${SESSION:-local-contract-test}"
NETWORK="${NETWORK:-origami-contract-test}"
ROUTER_NAME="${ROUTER_NAME:-origami-contract-router}"
POLICY_NAME="${POLICY_NAME:-origami-contract-policy}"
ROUTER_IMAGE="${ROUTER_IMAGE:-eclipse/zenoh@sha256:157965d71e0bfd0a044d76a985ff0e5c306ad3968929168fb9678cd2a7fec23f}"
ROUTER_PORT="${ROUTER_PORT:-7447}"
HOST_PORT="${HOST_PORT:-17447}"
HORIZON="${EXPECTED_HORIZON:-25}"

# Policy behaviour knobs, forwarded into the container.
VITAC_OPTIMIZATION="${VITAC_OPTIMIZATION:-none}"
VITAC_SMOOTHING="${VITAC_SMOOTHING:-auto}"

# Validator inputs. NOTE: --obs-type dataset needs a root that still carries
# observation.images.tactile_deform (robot_io_spec.md marks it required); the
# *_224 roots were built without the tactile streams. Guarded below.
DATASET_ROOT="${DATASET_ROOT:-/media/sai/CRUZER_BLA/ori/dataset/season_POC22061_2026_07_09_16_23_46_train/lerobot3.0_shortgop15}"
EPISODE_INDEX="${EPISODE_INDEX:-0}"
DATASET_REQUESTS="${DATASET_REQUESTS:-6}"
REQUESTS="${REQUESTS:-6}"
QUERY_TIMEOUT="${QUERY_TIMEOUT:-180}"
READY_TIMEOUT="${READY_TIMEOUT:-300}"
MEMORY="${MEMORY:-12g}"
CPUS="${CPUS:-6}"
SHM_SIZE="${SHM_SIZE:-8g}"

SDK_DIR="${SDK_DIR:-${ORI_ROOT}/origami-inference-kit-participant/sharpa_north_ces_lite_sdk-main}"
VALIDATOR="${VALIDATOR:-${SDK_DIR}/examples/check_zenoh_policy.py}"

DO_BUILD=1
PRUNE_CACHE=0
MIN_FREE_GB="${MIN_FREE_GB:-15}"   # image is ~14GB; warn below this before building
SUBCOMMAND="up"
while [ $# -gt 0 ]; do
    case "$1" in
        up|down|logs) SUBCOMMAND="$1"; shift ;;
        --no-build)   DO_BUILD=0; shift ;;
        --prune-cache) PRUNE_CACHE=1; shift ;;
        --smoothing)  VITAC_SMOOTHING="$2"; shift 2 ;;
        --compile)    VITAC_OPTIMIZATION="compile"; shift ;;
        -h|--help)    sed -n '3,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1 (try --help)" >&2; exit 1 ;;
    esac
done

if [ -t 1 ]; then
    GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; RED=$'\033[0;31m'; BLUE=$'\033[0;34m'; BOLD=$'\033[1m'; NC=$'\033[0m'
else
    GREEN=''; YELLOW=''; RED=''; BLUE=''; BOLD=''; NC=''
fi
log()  { echo "${BLUE}[info]${NC} $*"; }
ok()   { echo "${GREEN}[ ok ]${NC} $*"; }
warn() { echo "${YELLOW}[warn]${NC} $*"; }
die()  { echo "${RED}[fail]${NC} $*" >&2; exit 1; }
sec()  { echo; echo "${BOLD}=== $* ===${NC}"; }

teardown() {
    docker rm -f "$POLICY_NAME" "$ROUTER_NAME" >/dev/null 2>&1 || true
    docker network rm "$NETWORK" >/dev/null 2>&1 || true
}

case "$SUBCOMMAND" in
    down) sec "Teardown"; teardown; ok "torn down"; exit 0 ;;
    logs) exec docker logs -f "$POLICY_NAME" ;;
esac

# ---------------------------------------------------------------------------
sec "Preflight"
command -v docker >/dev/null 2>&1 || die "docker not on PATH"
docker info >/dev/null 2>&1 || die "cannot reach the docker daemon"
ok "docker reachable"
[ -f "$VALIDATOR" ] || die "validator not found: $VALIDATOR"
command -v uv >/dev/null 2>&1 || die "uv not found on PATH (needed to run the validator)"
ok "validator present"

docker image inspect "$ROUTER_IMAGE" >/dev/null 2>&1 || {
    warn "pulling router image"; docker pull "$ROUTER_IMAGE" >/dev/null || die "router pull failed"
}
ok "router image present"

# Dataset replay needs the tactile stream; fall back to synthetic-only if absent
# rather than failing halfway through with a schema error.
RUN_DATASET=1
if [ ! -d "$DATASET_ROOT" ]; then
    warn "dataset root not found, running synthetic checks only: $DATASET_ROOT"
    RUN_DATASET=0
elif ! grep -q "observation.images.tactile_deform" "$DATASET_ROOT/meta/info.json" 2>/dev/null; then
    warn "dataset root has no observation.images.tactile_deform -- skipping --obs-type dataset"
    warn "  robot_io_spec.md marks it required, so infer() would fail schema validation."
    warn "  Point DATASET_ROOT at a root that still has the tactile videos."
    RUN_DATASET=0
else
    ok "dataset root usable: $DATASET_ROOT"
fi

# ---------------------------------------------------------------------------
if [ "$DO_BUILD" = "1" ]; then
    sec "Build (layer cache reused; one tag, no copies)"
    # The image is ~14GB and every rebuild adds build-cache layers, which is a
    # realistic way to run the disk out. Warn, never prune behind your back.
    if [ "$PRUNE_CACHE" = "1" ]; then
        log "--prune-cache: dropping docker build cache first"
        docker builder prune -af >/dev/null 2>&1 || true
    fi
    FREE_GB="$(df -BG --output=avail / | tail -1 | tr -dc '0-9')"
    if [ -n "$FREE_GB" ] && [ "$FREE_GB" -lt "$MIN_FREE_GB" ]; then
        warn "only ${FREE_GB}GB free on / (image is ~14GB, build cache grows per rebuild)"
        warn "  reclaim with: docker builder prune -af     (cache only, no images/containers)"
        warn "  or re-run this script with --prune-cache"
    fi
    # A run directory is the unit: checkpoints/<run>/{*.ckpt,training_configs.json,
    # normalizer_config.json}. CKPT_DIR picks the run, CKPT_FILE picks the epoch.
    if [ -z "${CKPT_DIR:-}" ]; then
        mapfile -t runs < <(find "${SUBMISSION_DIR}/checkpoints" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
        [ ${#runs[@]} -gt 0 ] || die "no run directory under checkpoints/ (expected checkpoints/<run>/ with the .ckpt and both JSON sidecars)"
        [ ${#runs[@]} -eq 1 ] || die "checkpoints/ holds ${#runs[@]} run dirs: ${runs[*]}
  Leave one, or set CKPT_DIR=<run> to pick explicitly."
        CKPT_DIR="${runs[0]}"
    fi
    RUN_PATH="${SUBMISSION_DIR}/checkpoints/${CKPT_DIR}"
    [ -d "$RUN_PATH" ] || die "checkpoints/${CKPT_DIR} is not a directory"
    for f in training_configs.json normalizer_config.json; do
        [ -s "${RUN_PATH}/${f}" ] || die "checkpoints/${CKPT_DIR}/${f} missing"
    done
    if [ -z "${CKPT_FILE:-}" ]; then
        mapfile -t ckpts < <(find "$RUN_PATH" -maxdepth 1 -name '*.ckpt' -printf '%f\n' | sort)
        [ ${#ckpts[@]} -gt 0 ] || die "no .ckpt in checkpoints/${CKPT_DIR}/"
        [ ${#ckpts[@]} -eq 1 ] || die "checkpoints/${CKPT_DIR}/ holds ${#ckpts[@]} checkpoints: ${ckpts[*]}
  Set CKPT_FILE=<name>.ckpt to pick which epoch to deploy."
        CKPT_FILE="${ckpts[0]}"
    fi
    [ -s "${RUN_PATH}/${CKPT_FILE}" ] || die "checkpoints/${CKPT_DIR}/${CKPT_FILE} not found"
    log "run       : $CKPT_DIR"
    log "checkpoint: $CKPT_FILE"
    # The server needs policy_config to rebuild the architecture; catch its
    # absence here rather than at container startup.
    grep -q '"policy_config"' "${RUN_PATH}/training_configs.json" \
        || warn "training_configs.json has no \"policy_config\" key -- vitac_policy_server.py
         reads train_cfg[\"policy_config\"] and will fail at startup. Older runs
         wrote a flat config; regenerate it or add the key before deploying."

    OLD_ID="$(docker image inspect "$IMAGE" --format '{{.Id}}' 2>/dev/null || true)"
    docker build --build-arg "CKPT_DIR=${CKPT_DIR}" --build-arg "CKPT_FILE=${CKPT_FILE}" \
        -t "$IMAGE" "$SUBMISSION_DIR"
    NEW_ID="$(docker image inspect "$IMAGE" --format '{{.Id}}')"
    if [ -n "$OLD_ID" ] && [ "$OLD_ID" != "$NEW_ID" ]; then
        # Only removes the previous build of THIS tag, now untagged. Never a
        # blanket prune, so unrelated images are untouched.
        docker rmi "$OLD_ID" >/dev/null 2>&1 && ok "removed superseded image ${OLD_ID:7:12}" \
            || warn "previous image ${OLD_ID:7:12} still referenced, left in place"
    fi
    ok "built $IMAGE"
else
    docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE not found (drop --no-build)"
    warn "--no-build: using existing $IMAGE (may be stale)"
fi

# ---------------------------------------------------------------------------
sec "Bring up (local bridge network, router on 127.0.0.1:${HOST_PORT})"
teardown
docker network create "$NETWORK" >/dev/null
docker run -d --name "$ROUTER_NAME" --network "$NETWORK" \
    -p "127.0.0.1:${HOST_PORT}:${ROUTER_PORT}" "$ROUTER_IMAGE" \
    -l "tcp/0.0.0.0:${ROUTER_PORT}" --no-multicast-scouting \
    --cfg 'transport/shared_memory/enabled:false' >/dev/null
ok "router up"

GPU_ARGS=()
if docker info 2>/dev/null | grep -qi 'runtimes:.*nvidia'; then
    GPU_ARGS=(--gpus all); ok "nvidia runtime present"
else
    warn "no nvidia runtime -- CPU only (contract checks valid, latencies are not)"
fi

docker run -d --name "$POLICY_NAME" --network "$NETWORK" "${GPU_ARGS[@]}" \
    --read-only --cap-drop ALL --security-opt no-new-privileges=true \
    --user 65532:65532 \
    --tmpfs /tmp:rw,noexec,nosuid,nodev,size=4g \
    --tmpfs /run:rw,noexec,nosuid,nodev,size=64m \
    --pids-limit 512 --shm-size "$SHM_SIZE" --memory "$MEMORY" --cpus "$CPUS" \
    -e ORIGAMI_ZENOH_ENDPOINT="tcp/${ROUTER_NAME}:${ROUTER_PORT}" \
    -e ORIGAMI_SESSION_ID="$SESSION" \
    -e VITAC_OPTIMIZATION="$VITAC_OPTIMIZATION" \
    -e VITAC_SMOOTHING="$VITAC_SMOOTHING" \
    "$IMAGE" >/dev/null
ok "policy up (optimization=$VITAC_OPTIMIZATION smoothing=$VITAC_SMOOTHING)"

# Open a log window if a GUI terminal exists; the command is always printed so
# you can run it yourself in another terminal either way.
LOG_CMD="docker logs -f $POLICY_NAME"
if [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
    for term in tilix gnome-terminal xterm konsole; do
        if command -v "$term" >/dev/null 2>&1; then
            case "$term" in
                tilix)          tilix --new-process -t "origami: policy" -e bash -c "$LOG_CMD; exec bash" >/dev/null 2>&1 & ;;
                gnome-terminal) gnome-terminal --title="origami: policy" -- bash -c "$LOG_CMD; exec bash" >/dev/null 2>&1 & ;;
                konsole)        konsole -e bash -c "$LOG_CMD; exec bash" >/dev/null 2>&1 & ;;
                xterm)          xterm -T "origami: policy" -e bash -c "$LOG_CMD; exec bash" >/dev/null 2>&1 & ;;
            esac
            ok "opened policy log window ($term)"; break
        fi
    done
fi
echo "${BOLD}    follow the policy log with:${NC}  $LOG_CMD"
echo "${BOLD}    or:${NC}  $0 logs"

sec "Waiting for READY (timeout ${READY_TIMEOUT}s)"
deadline=$(( SECONDS + READY_TIMEOUT ))
until docker logs "$POLICY_NAME" 2>&1 | grep -q 'READY transport='; do
    [ "$(docker inspect -f '{{.State.Status}}' "$POLICY_NAME" 2>/dev/null || echo missing)" = "running" ] || {
        echo "${BOLD}---- last 40 log lines ----${NC}"; docker logs --tail 40 "$POLICY_NAME" 2>&1 || true
        die "policy container exited during startup"
    }
    [ $SECONDS -lt $deadline ] || { docker logs --tail 40 "$POLICY_NAME" 2>&1 || true; die "no READY within ${READY_TIMEOUT}s"; }
    sleep 2
done
ok "policy READY"
docker logs "$POLICY_NAME" 2>&1 | grep -E 'smoothing=|Optimization:|READY transport=' | tail -3

# ---------------------------------------------------------------------------
run_validator() {  # run_validator <obs-type> <extra args...>
    local obs="$1"; shift
    sec "Validator (obs-type=${obs})"
    ( cd "$SDK_DIR" && env -u VIRTUAL_ENV uv run --no-sync python "$VALIDATOR" \
        --endpoint "tcp/127.0.0.1:${HOST_PORT}" --session-id "$SESSION" \
        --timeout "$QUERY_TIMEOUT" --expected-horizon "$HORIZON" \
        --obs-type "$obs" "$@" )
}

RC=0
run_validator synthetic --requests "$REQUESTS" || RC=$?
if [ "$RUN_DATASET" = "1" ] && [ $RC -eq 0 ]; then
    run_validator dataset --dataset-root "$DATASET_ROOT" --episode-index "$EPISODE_INDEX" \
        --dataset-requests "$DATASET_REQUESTS" --frame-stride "$HORIZON" \
        --out-dir dataset_checks || RC=$?
fi

sec "Result"
if [ $RC -eq 0 ]; then
    ok "validator PASSED"
    warn "this proves interface compatibility only -- not task quality or action safety"
else
    echo "${RED}[fail]${NC} validator exited $RC"
    echo "${BOLD}---- last 40 policy log lines ----${NC}"; docker logs --tail 40 "$POLICY_NAME" 2>&1 || true
fi

echo
log "containers left running so the log keeps streaming:"
log "  follow : $LOG_CMD"
log "  stop   : $0 down"
exit $RC
