# Origami Prediction Visualizer

Offline local viewer for Sharpa North model predictions against a Robotic Origami Challenge LeRobot episode.

It has no robot, policy, Docker, authentication, remote session, or network dependency. The only HTTP server is the local browser UI.

## What it visualizes

- predicted absolute joint-position actions from `.npy`
  - `[T, 65]`, or
  - `[T', 25, 65]` (flattened to `[T'*25, 65]`)
- dataset `action` (65-D absolute joint-position actions)
- two North robot models: prediction + translucent GT action
- synchronized `observation.images.head_left` video
- synchronized `observation.images.wrist_left` video
- joint error, URDF limits and velocity checks

## LeRobot v3.0 video handling

You do **not** pass MP4 paths for v3.0. Multiple episodes can share one MP4. The viewer reads:

`meta/episodes/**/file-*.parquet`

and uses the per-camera fields:

- `videos/<camera>/chunk_index`
- `videos/<camera>/file_index`
- `videos/<camera>/from_timestamp`
- `videos/<camera>/to_timestamp`

This resolves the correct shared MP4 and the exact segment belonging to the selected episode.

## Install

From the directory containing `pyproject.toml`:

```bash
pip install -e '.[dataset]'
```

## Run

```bash
origami-visualizer \
  --predictions /path/to/predictions.npy \
  --robot-assets-dir /path/to/north_poc2_2_urdf_usd \
  --dataset-root /media/sai/CRUZER_BLA/ori/dataset/season_POC22061_2026_07_09_16_23_46_train/lerobot3.0 \
  --episode-index 123 \
  --action-horizon 25 \
  --control-hz 30
```

The browser UI is at `http://127.0.0.1:7861`.

Explicit `--head-left-video` and `--wrist-left-video` options remain available as overrides, but they are not needed for normal LeRobot v3.0 datasets.
