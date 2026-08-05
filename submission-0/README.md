# ViTacFormer Policy Server for Origami Competition

Self-contained Docker image for running ViTacFormer policy inference via the `origami-zenoh-v1` protocol.

## Overview

This submission implements a ViTacFormer-based policy server for the Origami robotics challenge. It uses:

- **Model**: ViTacFormer ACT (Action Chunking with Transformers) with ResNet18 backbone
- **Protocol**: `origami-zenoh-v1` (Zenoh query/reply)
- **History**: 6-step proprioceptive + 18-step tactile temporal buffers
- **Output**: 25-step action chunks, 65-DOF absolute joint positions (radians)

## Quick Start

### Build the Docker Image

```bash
cd ori/submission-0

# Build with checkpoints as named context (keeps large files out of git context)
docker build \
  --build-context checkpoint=/path/to/your/checkpoints \
  -t vitacformer-origami:latest \
  .
```

### Run Locally for Testing

```bash
# Start a local Zenoh router
docker run -d --name zenoh-router \
  -p 7447:7447 \
  eclipse/zenoh:1.0.0 \
  -l tcp/0.0.0.0:7447

# Run the policy server
docker run --rm \
  --gpus all \
  --network host \
  -e ORIGAMI_ZENOH_ENDPOINT=tcp/127.0.0.1:7447 \
  -e ORIGAMI_SESSION_ID=test-session-001 \
  -e VITACFORMER_CHECKPOINT=/opt/vitacformer/checkpoints/policy.ckpt \
  vitacformer-origami:latest
```

### Validate with Black-Box Tester

```bash
cd ../origami-inference-kit-participant/sharpa_north_ces_lite_sdk-main

python examples/check_zenoh_policy.py \
  --endpoint tcp/127.0.0.1:7447 \
  --session-id test-session-001 \
  --timeout 180 \
  --requests 3 \
  --expected-horizon 25
```

## Environment Variables

### Required (injected by organizer at runtime)

| Variable | Description | Example |
|----------|-------------|---------|
| `ORIGAMI_ZENOH_ENDPOINT` | Zenoh router endpoint | `tcp/origami-router:7447` |
| `ORIGAMI_SESSION_ID` | Opaque session identifier | `episode-001-worker-0` |

### Optional (configure model behavior)

| Variable | Default | Description |
|----------|---------|-------------|
| `VITACFORMER_CHECKPOINT` | `/opt/vitacformer/checkpoints/policy.ckpt` | Path to model checkpoint |
| `VITACFORMER_USE_TACTILE` | `true` | Enable tactile input |
| `VITACFORMER_ACTION_HORIZON` | `25` | Number of action steps to predict |
| `VITACFORMER_DEVICE` | `auto` | Target device (`cuda`, `cpu`, or `auto`) |

## File Structure

```
submission-0/
├── Dockerfile              # Multi-stage build (builder + runtime)
├── entrypoint.sh           # Validates env vars, starts server
├── requirements.lock       # Pinned Python dependencies
├── policy_server.py        # Main Zenoh server implementation
├── .dockerignore           # Build exclusions
├── README.md               # This file
├── vitac_policy/           # ViTacFormer adapter module
│   ├── __init__.py
│   ├── history_buffer.py   # Temporal history management
│   ├── model_loader.py     # Checkpoint loading
│   └── adapter.py          # Observation/action conversion
├── assets/                 # Normalization stats (if needed)
│   └── .gitkeep
├── checkpoints/            # Model weights (add your checkpoint here)
│   └── .gitkeep
└── scripts/
    └── validate_local.sh   # Local validation helper
```

## Protocol Specification

### Metadata Reply

```python
{
    "protocol_version": "origami-zenoh-v1",
    "operation": "metadata",
    "request_id": "<unique-id>",
    "session_id": "<session-id>",
    "metadata": {
        "protocol_version": "origami-v1",
        "action_dim": 65,
        "action_horizon": 25,
        "action_type": "absolute_joint_position",
        "action_units": "radians",
        "joint_names": [...],  # 65 names per robot_io_spec.md
    },
}
```

### Observation Input

The server receives the full observation contract from `robot_io_spec.md`:

- `observation/image/head_left`: uint8[224, 224, 3] RGB
- `observation/image/head_right`: uint8[224, 224, 3] RGB
- `observation/image/wrist_left`: uint8[224, 224, 3] RGB
- `observation/image/wrist_right`: uint8[224, 224, 3] RGB
- `observation/state`: float32[65] joint angles (radians)
- `observation/state/joint_torque`: float32[65] (may be zero-filled)
- `observation/tactile`: float32[60] 10 fingertips × 6-DoF wrench
- `observation/image/tactile_deform`: uint8[480, 1200, 3]
- `observation/image/tactile_raw`: uint8[480, 1600, 3] (optional)
- `prompt`: str

### Action Output

```python
{
    "protocol_version": "origami-zenoh-v1",
    "operation": "infer",
    "request_id": "<unique-id>",
    "session_id": "<session-id>",
    "actions": float32[25, 65],  # (horizon, 65-DOF)
}
```

## Model Architecture

ViTacFormer uses an ACT (Action Chunking with Transformers) architecture:

| Component | Configuration |
|-----------|---------------|
| Backbone | ResNet18 (4 cameras) |
| Encoder | 4 layers, 8 heads |
| Decoder | 7 layers |
| Hidden Dim | 512 |
| FFN Dim | 3200 |
| Action Queries | 100 |
| State History | 6 steps |
| Tactile History | 18 steps |

## Checkpoint Format

Expected checkpoint structure:

```python
{
    "model": state_dict,      # Model weights
    "global_step": int,       # Training step
    "epoch": int,             # Training epoch
}
```

Place your checkpoint in `checkpoints/` before building:

```bash
cp /path/to/policy_globalstep_4200_loss_0.0836.ckpt \
   ori/submission-0/checkpoints/policy.ckpt
```

## Resource Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | NVIDIA 16GB | NVIDIA 40GB+ |
| RAM | 16GB | 32GB |
| CPU | 4 cores | 8 cores |
| Shared Memory | 4GB | 8GB |

Example Docker run with resource limits:

```bash
docker run --rm \
  --gpus all \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=2g \
  --shm-size 8g \
  --memory 32g \
  --cpus 8 \
  -e ORIGAMI_ZENOH_ENDPOINT=tcp/router:7447 \
  -e ORIGAMI_SESSION_ID=eval-001 \
  vitacformer-origami:latest
```

## License

- ViTacFormer model code: [Check your license]
- Zenoh: Apache 2.0
- MessagePack: ISC

## Troubleshooting

### "Checkpoint not found"

Ensure the checkpoint is copied to `checkpoints/` before building:

```bash
ls -la checkpoints/
# Should show: policy.ckpt
```

### "Zenoh connection failed"

Verify the router is accessible:

```bash
# Test connectivity
nc -zv <router-host> 7447

# Check endpoint format
echo $ORIGAMI_ZENOH_ENDPOINT
# Should be: tcp/<host>:<port>
```

### "Actions contain NaN"

This indicates a model inference failure. Check:
1. Checkpoint loaded correctly (see startup logs)
2. Input observation has valid values
3. GPU memory is sufficient

## Authors

ViTacFormer submission for Origami Robotics Challenge.

## Citation

If using this code, please cite:

```bibtex
@misc{vitacformer2026,
  title={ViTacFormer: Vision-Tactile Fusion for Robotic Origami},
  author={Your Team},
  year={2026}
}
```
