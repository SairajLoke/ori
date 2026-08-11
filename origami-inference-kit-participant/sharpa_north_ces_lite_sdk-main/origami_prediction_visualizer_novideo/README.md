# North Model Prediction Visualizer

This is a stripped-down version of the participant local evaluator.

It intentionally removes:
- remote observation / serve sessions
- authentication and TLS
- Zenoh
- Docker / policy containers
- participant policy execution
- camera/tactile observation handling

It keeps:
- the official North 65-joint contract
- the official North URDF + STL Three.js viewer
- joint position limits
- step-jump limits
- velocity checks
- a timeline slider and play/step controls

## Prediction format

The loader accepts either:

```text
[T, 65]
```

or the original non-overlapping inference form:

```text
[T', 25, 65]
```

If the latter is supplied, it is flattened to:

```text
[T' * 25, 65]
```

The UI shows both:
- global step: `0 ... T-1`
- stage: `0 ... T'-1`
- horizon step: `0 ... 24`

## Run

```bash
python -m participant_local_evaluator \
    --predictions /absolute/path/predictions.npy \
    --robot-assets-dir /absolute/path/to/robot/assets \
    --action-horizon 25 \
    --control-hz 30
```

Then open:

```text
http://127.0.0.1:7861
```

If you have a real state immediately before the first prediction, optionally provide:

```bash
--initial-state /absolute/path/initial_state.npy
```

That enables the velocity/step-jump check for prediction step 0 as well.

Without an initial state, step 0 is treated as the first known state, so motion-limit checks begin between prediction 0 and prediction 1.

## Assumption

The 65 values are absolute joint positions in radians, in the public North joint order.
