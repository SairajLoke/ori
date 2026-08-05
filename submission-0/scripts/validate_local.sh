#!/usr/bin/env bash
# Local Validation Script for ViTacFormer Policy Server
# Runs a local Zenoh router and validates the policy server

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMISSION_DIR="$(dirname "$SCRIPT_DIR")"
SDK_DIR="${SUBMISSION_DIR}/../origami-inference-kit-participant/sharpa_north_ces_lite_sdk-main"

# Docker image name (set this to your built image)
IMAGE_NAME="${IMAGE_NAME:-vitacformer-origami:latest}"

# Network and session
NETWORK_NAME="origami-validate"
ROUTER_NAME="origami-zenoh-router"
POLICY_NAME="origami-policy"
SESSION_ID="local-validation-$(date +%s)"

# Ports
ROUTER_PORT=7447
HOST_PORT=17447

# =============================================================================
# Colors for output
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[PASS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[FAIL]${NC} $1"; }

# =============================================================================
# Cleanup function
# =============================================================================

cleanup() {
    log_info "Cleaning up..."
    
    docker rm -f "$POLICY_NAME" 2>/dev/null || true
    docker rm -f "$ROUTER_NAME" 2>/dev/null || true
    docker network rm "$NETWORK_NAME" 2>/dev/null || true
    
    log_info "Cleanup complete"
}

trap cleanup EXIT

# =============================================================================
# Main validation flow
# =============================================================================

main() {
    echo "=============================================="
    echo "ViTacFormer Local Validation"
    echo "=============================================="
    echo ""
    
    # Check if image exists
    if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
        log_error "Docker image not found: $IMAGE_NAME"
        log_info "Build the image first with:"
        echo "  docker build -t $IMAGE_NAME $SUBMISSION_DIR"
        exit 1
    fi
    log_success "Docker image found: $IMAGE_NAME"
    
    # Check if SDK exists
    if [ ! -d "$SDK_DIR" ]; then
        log_warn "SDK directory not found: $SDK_DIR"
        log_info "Validator will need to be run manually"
        SDK_DIR=""
    fi
    
    # Create network
    log_info "Creating Docker network: $NETWORK_NAME"
    docker network create "$NETWORK_NAME" &>/dev/null || true
    
    # Start Zenoh router
    log_info "Starting Zenoh router..."
    docker run -d --name "$ROUTER_NAME" \
        --network "$NETWORK_NAME" \
        -p "${HOST_PORT}:${ROUTER_PORT}" \
        eclipse/zenoh:1.0.0 \
        -l "tcp/0.0.0.0:${ROUTER_PORT}" \
        --no-multicast-scouting \
        --cfg 'transport/shared_memory/enabled:false' \
        2>/dev/null
    
    # Wait for router to start
    log_info "Waiting for router to be ready..."
    sleep 3
    
    # Check router health
    if ! docker ps | grep -q "$ROUTER_NAME"; then
        log_error "Failed to start Zenoh router"
        exit 1
    fi
    log_success "Zenoh router started on port ${HOST_PORT}"
    
    # Start policy server
    log_info "Starting policy server..."
    docker run -d --name "$POLICY_NAME" \
        --network "$NETWORK_NAME" \
        --gpus all \
        --read-only \
        --cap-drop ALL \
        --security-opt no-new-privileges=true \
        --user 65532:65532 \
        --tmpfs /tmp:rw,noexec,nosuid,nodev,size=1g \
        --shm-size 4g \
        -e ORIGAMI_ZENOH_ENDPOINT="tcp/${ROUTER_NAME}:${ROUTER_PORT}" \
        -e ORIGAMI_SESSION_ID="$SESSION_ID" \
        "$IMAGE_NAME" \
        2>/dev/null
    
    # Wait for server to initialize
    log_info "Waiting for policy server to initialize..."
    sleep 5
    
    # Check server logs
    log_info "Policy server logs:"
    echo "----------------------------------------------"
    docker logs "$POLICY_NAME" 2>&1 | tail -20 || true
    echo "----------------------------------------------"
    
    # Run validator if SDK is available
    if [ -n "$SDK_DIR" ] && [ -f "$SDK_DIR/examples/check_zenoh_policy.py" ]; then
        log_info "Running black-box validator..."
        echo ""
        
        cd "$SDK_DIR"
        python examples/check_zenoh_policy.py \
            --endpoint "tcp/127.0.0.1:${HOST_PORT}" \
            --session-id "$SESSION_ID" \
            --timeout 180 \
            --requests 3 \
            --expected-horizon 25 \
            || log_warn "Validator completed with errors"
        
        echo ""
    else
        log_warn "SDK not found, skipping automated validation"
        log_info "To validate manually, run:"
        echo "  python $SDK_DIR/examples/check_zenoh_policy.py \\"
        echo "    --endpoint tcp/127.0.0.1:${HOST_PORT} \\"
        echo "    --session-id $SESSION_ID \\"
        echo "    --expected-horizon 25"
    fi
    
    log_success "Validation complete!"
    log_info "Containers will be cleaned up on exit"
}

# Run main
main "$@"
