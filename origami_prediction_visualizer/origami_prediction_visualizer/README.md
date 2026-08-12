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


### Recording and colors

Prediction and GT robots are rendered in faint blue and faint green. The interactive player uses native MP4 playback as the timing clock during normal playback; it only seeks the dataset videos when scrubbing/resetting, avoiding repeated `currentTime` seeks.

Enable camera presets from the CLI:

```bash
origami-visualizer \
  --predictions /path/to/preds.npy \
  --robot-assets-dir /path/to/north_poc2_2_urdf_usd \
  --dataset-root /path/to/season/lerobot3.0 \
  --episode-index 0 \
  --action-horizon 25 \
  --control-hz 30 \
  --record-views front top hands
```

The page then exposes checkboxes for the enabled views and records each selected view sequentially as a browser-downloaded `.webm` file. Presets are `front`, `top`, and `hands` (close-up manipulation view).

## Exact recording camera views

You can define exact Three.js world-space camera coordinates in a JSON file and select any named views from the CLI.

Starter template: `cameras.json` in the project root.

Example:

```bash
origami-visualizer \
  --predictions /path/to/preds.npy \
  --robot-assets-dir /path/to/north_poc2_2_urdf_usd \
  --camera-config /path/to/cameras.json \
  --record-views front top hands
```

Each view supports:

- `position`: `[x, y, z]` camera world coordinates
- `target`: `[x, y, z]` point the camera looks at
- `fov`: perspective field of view in degrees
- `near`, `far`: optional clipping planes
- `label`: optional UI/recording label

Coordinates are absolute Three.js scene/world coordinates; they are not scaled by the robot bounding box. This makes recordings reproducible and lets you tune the camera by editing only the JSON.

You can define additional views, for example:

```json
{
  "views": {
    "my_custom_view": {
      "label": "My custom view",
      "position": [1.8, 1.0, 2.2],
      "target": [0.0, 1.2, 0.0],
      "fov": 40
    }
  }
}
```

Then:

```bash
origami-visualizer ... --camera-config cameras.json --record-views my_custom_view
```

If `--camera-config` is omitted, the built-in `front`, `top`, and `hands` views remain available.
