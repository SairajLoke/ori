# Origami Inference Kit - Complete Guide

This document explains the complete Origami competition inference system, including all workflows, Docker images, and commands.

---

## 📊 System Architecture Flowchart

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ORIGAMI COMPETITION SYSTEM OVERVIEW                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  DEVELOPMENT PHASE (Before Submission)                                      │
│  ================================                                           │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ 1. REMOTE OBSERVATION CLIENT (origami-remote-v1)                     │  │
│  │    - Reads REAL robot observations via public Zenoh                  │  │
│  │    - Read-only: NO action/control capability                         │  │
│  │    - Uses: tls/<endpoint>, session_id, token, TLS CA                 │  │
│  │                                                                      │  │
│  │    [Robot] → [Organizer Relay] → [Your Policy] → [Local Viz]        │  │
│  │         (read-only, no feedback)                                     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ 2. PARTICIPANT LOCAL EVALUATOR (Web UI @ localhost:7861)            │  │
│  │    - Connects remote observations to your submission image           │  │
│  │    - Runs local Zenoh router + your container                        │  │
│  │    - Shows URDF visualization (Shadow mode)                          │  │
│  │    - Validates: shape, dtype, finite values, velocity limits         │  │
│  │                                                                      │  │
│  │    [Remote Obs] → [Local Router] → [Your Docker] → [URDF Viz]       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ 3. SYNTHETIC VALIDATOR (check_zenoh_policy.py)                       │  │
│  │    - Black-box protocol validator                                    │  │
│  │    - Sends synthetic observations, validates responses               │  │
│  │    - Checks: metadata, reset, infer, 65 joint names                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SUBMISSION PHASE                                                           │
│  ================                                                           │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ 4. PARTICIPANT SUBMISSION IMAGE (origami-zenoh-v1)                   │  │
│  │    - Self-contained Docker with model + Zenoh server                 │  │
│  │    - Declares 3 queryables: metadata, reset, infer                   │  │
│  │    - Receives: observations via Zenoh                                │  │
│  │    - Returns: actions (T, 65) float32 radians                        │  │
│  │                                                                      │  │
│  │    [Organizer Router] ↔ [Your Container]                            │  │
│  │         (query/reply only, no publish)                               │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  EVALUATION PHASE (By Organizer)                                            │
│  ======================                                                     │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ 5. SHADOW EVALUATION (URDF Visualization)                            │  │
│  │    - Organizer runs your image with real observations                │  │
│  │    - Displays predicted trajectory on URDF                           │  │
│  │    - NO real robot actions                                           │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                     │                                       │
│                                     ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ 6. LIVE EVALUATION (Real Robot Execution)                            │  │
│  │    - ONLY after organizer authorization                              │  │
│  │    - Your actions executed on real robot                             │  │
│  │    - Full safety checks applied                                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🐳 Docker Images Reference

### SDK Images (sharpa_north_ces_lite_sdk-main/scripts/docker/)

| Image | Dockerfile | Purpose | Command |
|-------|------------|---------|---------|
| **Policy Template** | `policy-template.Dockerfile` | Basic origami-zenoh-v1 server skeleton | `docker build -f scripts/docker/policy-template.Dockerfile -t origami-policy-template:dev .` |
| **Validator** | `validator.Dockerfile` | Black-box protocol tester | `docker build -f scripts/docker/validator.Dockerfile -t origami-policy-validator:dev .` |
| **Remote Client** | `remote-client.Dockerfile` | Read real robot observations | `docker build -f scripts/docker/remote-client.Dockerfile -t origami-remote-observation-client:dev .` |

### OpenPI Images (openpi-base-main/scripts/docker/)

| Image | Dockerfile | Purpose |
|-------|------------|---------|
| **Runtime** | (internal) | OpenPI runtime with JAX/Flax |
| **Submission Bundled** | `submission-zenoh-bundled.Dockerfile` | Full OpenPI + Zenoh submission image |
| **Serve Policy** | `serve_policy.Dockerfile` | Development inference server |

### ViTacFormer Images (submission-0/)

| Image | Dockerfile | Purpose |
|-------|------------|---------|
| **ViTacFormer Submission** | `Dockerfile` | ViTacFormer ACT + Zenoh server |

---

## 📜 Docker Commands Explained

### 1. Build Your Submission Image

```bash
docker build \
  --build-context checkpoint=/path/to/checkpoints \
  -t vitacformer-origami:latest \
  ori/submission-0
```

| Flag | Meaning |
|------|---------|
| `--build-context checkpoint=...` | Named context for large checkpoint files (keeps them out of git context) |
| `-t vitacformer-origami:latest` | Tag the image |
| `ori/submission-0` | Build context directory |

---

### 2. Run Local Zenoh Router

```bash
docker run -d --name zenoh-router \
  -p 7447:7447 \
  eclipse/zenoh:1.0.0 \
  -l tcp/0.0.0.0:7447
```

| Flag | Meaning |
|------|---------|
| `-d` | Detached mode (background) |
| `--name zenoh-router` | Container name |
| `-p 7447:7447` | Port mapping (host:container) |
| `eclipse/zenoh:1.0.0` | Official Zenoh router image |
| `-l tcp/0.0.0.0:7447` | Listen on TCP port 7447 |

---

### 3. Run Your Policy Server

```bash
docker run --rm \
  --gpus all \
  --network host \
  -e ORIGAMI_ZENOH_ENDPOINT=tcp/127.0.0.1:7447 \
  -e ORIGAMI_SESSION_ID=test-session-001 \
  -e VITACFORMER_CHECKPOINT=/opt/vitacformer/checkpoints/policy.ckpt \
  vitacformer-origami:latest
```

| Flag | Meaning |
|------|---------|
| `--rm` | Auto-remove container on exit |
| `--gpus all` | Enable all GPUs |
| `--network host` | Use host network (simpler for local testing) |
| `-e ORIGAMI_ZENOH_ENDPOINT=...` | Zenoh router address |
| `-e ORIGAMI_SESSION_ID=...` | Session identifier |
| `-e VITACFORMER_CHECKPOINT=...` | Path to model weights |

---

### 4. Production-Style Run (Matches Organizer's Setup)

```bash
docker run --rm \
  --gpus all \
  --network origami-eval \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=1g \
  --shm-size 8g \
  --memory 32g \
  --cpus 8 \
  -e ORIGAMI_ZENOH_ENDPOINT=tcp/origami-router:7447 \
  -e ORIGAMI_SESSION_ID=team-episode-worker-0 \
  vitacformer-origami:latest
```

| Flag | Meaning |
|------|---------|
| `--network origami-eval` | Isolated Docker network |
| `--read-only` | Read-only root filesystem |
| `--tmpfs /tmp:...` | Writable temp directory (no exec) |
| `--shm-size 8g` | Shared memory for GPU |
| `--memory 32g` | RAM limit |
| `--cpus 8` | CPU limit |

---

### 5. Run Validator

```bash
python examples/check_zenoh_policy.py \
  --endpoint tcp/127.0.0.1:7447 \
  --session-id test-session \
  --timeout 180 \
  --requests 3 \
  --expected-horizon 25
```

| Argument | Meaning |
|----------|---------|
| `--endpoint` | Zenoh router address |
| `--session-id` | Session ID (must match server) |
| `--timeout` | Seconds per query |
| `--requests` | Number of infer queries to send |
| `--expected-horizon` | Expected action horizon (T) |

---

### 6. Run Local Evaluator (Web UI)

```bash
cd sharpa_north_ces_lite_sdk-main
python -m participant_local_evaluator \
  --robot-assets-dir /path/to/north-assets
```

Then open `http://127.0.0.1:7861` in browser.

**Features:**
- Upload your `.tar.zst` submission or provide local image path
- Connect to remote observations (requires organizer credentials)
- Run Shadow mode with URDF visualization
- View action timeline and validation results

---

### 7. Full Validation Script (From Your submission-0)

```bash
bash scripts/validate_local.sh
```

This script:
1. Creates isolated Docker network (`origami-validate`)
2. Starts Zenoh router on port 17447
3. Starts your policy server in the network
4. Runs the black-box validator
5. Cleans up all containers on exit

---

## 🔍 Key Docker Concepts

| Concept | Why It Matters |
|---------|----------------|
| **Named Contexts** | `--build-context checkpoint=/path` keeps large files out of git context |
| **Multi-stage Build** | Reduces final image size (builder stage → runtime stage) |
| **Non-root User** | `USER 65532:65532` for security (matches organizer's constraints) |
| **Read-only Rootfs** | `--read-only` required by organizer's security model |
| **tmpfs Mounts** | `--tmpfs /tmp` for writable temp storage |
| **Resource Limits** | `--memory`, `--cpus`, `--shm-size` match competition constraints |
| **Isolated Network** | `--network origami-eval` for container-to-container communication |

---

## 📋 Complete Workflow Summary

### Phase 1: Development

```bash
# 1. Build policy template (optional, for learning)
docker build -f scripts/docker/policy-template.Dockerfile \
  -t origami-policy-template:dev \
  sharpa_north_ces_lite_sdk-main

# 2. Run remote observation client (requires organizer credentials)
export ORIGAMI_REMOTE_ENDPOINT='tls/<public-host>:<port>'
export ORIGAMI_REMOTE_SESSION_ID='<assigned-team-session>'
export ORIGAMI_REMOTE_TOKEN='<secret>'
python examples/remote_observation_client.py

# 3. Test with synthetic validator
docker run -d --name zenoh-router -p 7447:7447 eclipse/zenoh:1.0.0 -l tcp/0.0.0.0:7447
python examples/check_zenoh_policy.py --endpoint tcp/127.0.0.1:7447
```

### Phase 2: Build Submission

```bash
# 1. Copy your checkpoint
cp /path/to/policy_globalstep_4200_loss_0.0836.ckpt \
   ori/submission-0/checkpoints/policy.ckpt

# 2. Build the image
cd ori/submission-0
docker build --build-context checkpoint=./checkpoints \
             -t vitacformer-origami:latest .

# 3. Run local validation
bash scripts/validate_local.sh
```

### Phase 3: Local Shadow Evaluation

```bash
# 1. Start the web-based evaluator
cd sharpa_north_ces_lite_sdk-main
python -m participant_local_evaluator \
  --robot-assets-dir /path/to/north-assets

# 2. Open browser to http://127.0.0.1:7861
# 3. Enter remote endpoint credentials
# 4. Load your Docker image
# 5. Run Shadow mode
```

### Phase 4: Submission

```bash
# 1. Export image to archive
docker save vitacformer-origami:latest | zstd -o vitacformer-submission.tar.zst

# 2. Compute SHA-256 checksum
sha256sum vitacformer-submission.tar.zst

# 3. Submit archive + checksum to organizer
```

### Phase 5: Organizer Evaluation

```
┌─────────────────────────────────────────────────────────────┐
│ Organizer receives:                                         │
│ - vitacformer-submission.tar.zst                           │
│ - SHA-256 checksum                                         │
├─────────────────────────────────────────────────────────────┤
│ Organizer runs:                                             │
│ 1. Load image, verify checksum                             │
│ 2. Shadow evaluation (URDF visualization)                  │
│ 3. If authorized: Live evaluation (real robot)             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Protocol Summary

### origami-remote-v1 (Development)

| Property | Value |
|----------|-------|
| **Purpose** | Read real robot observations |
| **Direction** | Organizer → Participant (read-only) |
| **Transport** | TLS (public network) |
| **Queryable** | `origami-remote-v1/{session_id}/observation` |
| **Auth** | Session ID + Token |

### origami-zenoh-v1 (Submission)

| Property | Value |
|----------|-------|
| **Purpose** | Production inference interface |
| **Direction** | Bidirectional (query/reply) |
| **Transport** | TCP (isolated network) |
| **Queryables** | `metadata`, `reset`, `infer` |
| **Auth** | Session ID only |

---

## 📌 Important Notes

1. **Two Independent Workflows**: Remote development (`origami-remote-v1`) and production submission (`origami-zenoh-v1`) use the same observation schema but different transports and authentication.

2. **No Action Publishing**: Your submission image only replies to queries. It never publishes actions directly to the robot.

3. **History Management**: Your policy must maintain temporal history (6-step state, 18-step tactile for ViTacFormer) and clear it on `reset()`.

4. **Action Horizon**: Fixed for the lifetime of your container. Must match `metadata.action_horizon` and every `infer` reply.

5. **Joint Order**: All 65 joint names must exactly match `robot_io_spec.md` order. No sorting, reordering, or padding allowed.

---

## 🔗 Related Documentation

- `docs/robot_io_spec.md` - Observation/action tensor contract
- `docs/participant_zenoh_submission.md` - Production protocol specification
- `docs/container_submission.md` - Container runtime requirements
- `docs/remote_participant_development.md` - Development interface guide
- `docs/competition_participant_complete_guide.md` - End-to-end workflow

---

## Authors

ViTacFormer Team - Origami Robotics Challenge 2026
