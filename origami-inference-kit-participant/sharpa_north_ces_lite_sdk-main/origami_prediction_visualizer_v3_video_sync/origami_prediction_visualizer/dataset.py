"""Local reader for Robotic Origami Challenge LeRobot v3.0 exports.

LeRobot v3.0 stores multiple episodes in shared Parquet/MP4 files.  The
per-episode metadata under meta/episodes tells us which data/video shard and
which timestamp window belong to a particular episode.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np

try:
    import pyarrow.dataset as ds
except ImportError:  # pragma: no cover
    ds = None


CAMERAS = ("head_left", "wrist_left")


class DatasetEpisode:
    def __init__(self, root: pathlib.Path, episode_index: int):
        self.root = root.expanduser().resolve()
        self.episode_index = int(episode_index)
        self.export_root = self._find_export_root()
        self.info = self._load_json("meta/info.json")
        self.fps = float(self.info.get("fps", 30.0))
        self.actions: np.ndarray
        self.states: np.ndarray | None
        self.timestamps: np.ndarray
        self.frame_indices: np.ndarray
        self.video_segments: dict[str, dict[str, Any]] = {}
        self.actions, self.states, self.timestamps, self.frame_indices = self._load_episode()
        self._load_video_segments()

    def _find_export_root(self) -> pathlib.Path:
        candidates = [self.root]
        candidates += [p for p in self.root.rglob("lerobot3.0") if p.is_dir()]
        candidates += [p for p in self.root.rglob("lerobotv2.1") if p.is_dir()]
        for candidate in candidates:
            if (candidate / "meta" / "info.json").is_file():
                return candidate
        raise ValueError(
            f"Could not find a LeRobot export below {self.root}. "
            "Point --dataset-root at a season's lerobot3.0/lerobotv2.1 directory "
            "or at the dataset repository root."
        )

    def _load_json(self, relative: str) -> dict[str, Any]:
        with open(self.export_root / relative, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _number(value: Any, default: float | int = 0):
        if value is None:
            return default
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _load_episode(self):
        if ds is None:
            raise RuntimeError(
                "Dataset support requires pyarrow. Install it with: pip install pyarrow"
            )
        data_dir = self.export_root / "data"
        if not data_dir.is_dir():
            raise ValueError(f"Dataset data directory not found: {data_dir}")

        dataset = ds.dataset(str(data_dir), format="parquet", partitioning="hive")
        names = set(dataset.schema.names)
        required = {"episode_index", "action"}
        missing = required - names
        if missing:
            raise ValueError(f"Dataset parquet is missing columns: {sorted(missing)}")

        columns = ["episode_index", "action"]
        for name in ("observation.state", "timestamp", "frame_index"):
            if name in names:
                columns.append(name)

        table = dataset.to_table(
            columns=columns,
            filter=(ds.field("episode_index") == self.episode_index),
        )
        if table.num_rows == 0:
            raise ValueError(f"Episode {self.episode_index} was not found in {data_dir}")

        frame_col = (
            table.column("frame_index").to_numpy(zero_copy_only=False)
            if "frame_index" in names
            else np.arange(table.num_rows)
        )
        order = np.argsort(frame_col, kind="stable")

        action_col = table.column("action").to_pylist()
        actions = np.asarray([action_col[i] for i in order], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 65:
            raise ValueError(
                f"Dataset action for episode {self.episode_index} has shape {actions.shape}, expected [T,65]"
            )

        states = None
        if "observation.state" in names:
            state_col = table.column("observation.state").to_pylist()
            states = np.asarray([state_col[i] for i in order], dtype=np.float32)
            if states.ndim != 2 or states.shape[1] != 65:
                states = None

        if "timestamp" in names:
            ts_col = table.column("timestamp").to_numpy(zero_copy_only=False)
            timestamps = np.asarray(ts_col[order], dtype=np.float64)
        else:
            timestamps = np.arange(table.num_rows, dtype=np.float64) / self.fps

        frame_indices = np.asarray(frame_col[order], dtype=np.int64)
        return actions, states, timestamps, frame_indices

    def _load_video_segments(self) -> None:
        """Resolve each camera to its shared MP4 and episode time window.

        v3.0 episode metadata contains columns such as:
          videos/<camera>/chunk_index
          videos/<camera>/file_index
          videos/<camera>/from_timestamp
          videos/<camera>/to_timestamp
        """
        episodes_dir = self.export_root / "meta" / "episodes"
        if not episodes_dir.is_dir() or ds is None:
            return

        episode_ds = ds.dataset(str(episodes_dir), format="parquet")
        names = set(episode_ds.schema.names)
        if "episode_index" not in names:
            return

        camera_columns: dict[str, dict[str, str]] = {}
        for camera in CAMERAS:
            prefix = f"videos/observation.images.{camera}"
            needed = {
                "chunk": f"{prefix}/chunk_index",
                "file": f"{prefix}/file_index",
                "start": f"{prefix}/from_timestamp",
                "end": f"{prefix}/to_timestamp",
            }
            if all(v in names for v in needed.values()):
                camera_columns[camera] = needed

        if not camera_columns:
            return

        columns = ["episode_index"]
        for spec in camera_columns.values():
            columns.extend(spec.values())
        table = episode_ds.to_table(
            columns=columns,
            filter=(ds.field("episode_index") == self.episode_index),
        )
        if table.num_rows == 0:
            return

        row = table.slice(0, 1).to_pydict()
        video_template = self.info.get(
            "video_path",
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        )

        for camera, spec in camera_columns.items():
            chunk = int(self._number(row[spec["chunk"]][0]))
            file_index = int(self._number(row[spec["file"]][0]))
            start = float(self._number(row[spec["start"]][0]))
            end = float(self._number(row[spec["end"]][0], start + len(self.actions) / self.fps))
            key = f"observation.images.{camera}"
            relative = video_template.format(
                video_key=key,
                chunk_index=chunk,
                file_index=file_index,
            )
            path = (self.export_root / relative).resolve()
            self.video_segments[camera] = {
                "path": path,
                "relative_path": relative,
                "from_timestamp": start,
                "to_timestamp": end,
                "episode_duration": max(0.0, end - start),
                "fps": self.fps,
            }

    def video_candidates(self, camera_key: str) -> list[pathlib.Path]:
        video_root = self.export_root / "videos" / f"observation.images.{camera_key}"
        if not video_root.is_dir():
            return []
        return sorted(video_root.rglob("*.mp4"))

    def manifest(self) -> dict[str, Any]:
        return {
            "export_root": str(self.export_root),
            "episode_index": self.episode_index,
            "fps": self.fps,
            "num_action_frames": int(len(self.actions)),
            "num_state_frames": int(len(self.states)) if self.states is not None else None,
            "timestamp_start": float(self.timestamps[0]),
            "timestamp_end": float(self.timestamps[-1]),
            "frame_index_start": int(self.frame_indices[0]),
            "frame_index_end": int(self.frame_indices[-1]),
            "video_segments": {
                camera: {
                    "path": str(info["path"]),
                    "relative_path": info["relative_path"],
                    "from_timestamp": info["from_timestamp"],
                    "to_timestamp": info["to_timestamp"],
                    "episode_duration": info["episode_duration"],
                    "fps": info["fps"],
                }
                for camera, info in self.video_segments.items()
            },
        }
