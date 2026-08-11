"""Run the local North prediction/episode visualizer."""
from __future__ import annotations

import argparse
import pathlib

from .controller import PredictionController
from .dataset import DatasetEpisode
from .trajectory import DEFAULT_URDF_RELATIVE_PATH, TrajectoryValidator
from .web import create_server


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=pathlib.Path, required=True)
    parser.add_argument("--robot-assets-dir", type=pathlib.Path, required=True)
    parser.add_argument("--urdf-relative-path", default=DEFAULT_URDF_RELATIVE_PATH)
    parser.add_argument("--initial-state", type=pathlib.Path, default=None)
    parser.add_argument("--action-horizon", type=int, default=25)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--dataset-root", type=pathlib.Path, default=None,
                        help="Robotic_Origami_Challenge root or a lerobot3.0/lerobotv2.1 export")
    parser.add_argument("--episode-index", type=int, default=None,
                        help="Dataset episode_index to use for ground-truth action playback")
    parser.add_argument("--head-left-video", type=pathlib.Path, default=None,
                        help="Optional exact head-left MP4; served by the local visualizer")
    parser.add_argument("--wrist-left-video", type=pathlib.Path, default=None,
                        help="Optional exact wrist-left MP4; served by the local visualizer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args(argv)

    if (args.dataset_root is None) != (args.episode_index is None):
        parser.error("--dataset-root and --episode-index must be supplied together")

    dataset_episode = None
    if args.dataset_root is not None:
        dataset_episode = DatasetEpisode(args.dataset_root, args.episode_index)

    validator = TrajectoryValidator(
        args.robot_assets_dir,
        urdf_relative_path=args.urdf_relative_path,
    )
    video_files = {}
    video_segments = {}
    # Prefer the v3.0 per-episode metadata. It tells us which shared MP4
    # contains this episode and the exact [from_timestamp, to_timestamp]
    # segment inside that MP4. Explicit MP4 arguments remain available as a
    # fallback/override.
    if dataset_episode is not None:
        for key, segment in dataset_episode.video_segments.items():
            path = pathlib.Path(segment["path"])
            if path.is_file():
                video_files[key] = path
                video_segments[key] = {
                    "url": f"/video/{key}",
                    "from_timestamp": float(segment["from_timestamp"]),
                    "to_timestamp": float(segment["to_timestamp"]),
                    "episode_duration": float(segment["episode_duration"]),
                    "relative_path": segment["relative_path"],
                }
            else:
                print(f"WARNING: dataset video missing for {key}: {path}")

    for key, value in (("head_left", args.head_left_video), ("wrist_left", args.wrist_left_video)):
        if value is not None:
            path = value.expanduser().resolve()
            if not path.is_file():
                parser.error(f"Video not found: {path}")
            video_files[key] = path
            # Explicit videos are assumed to start at episode time zero.
            video_segments[key] = {
                "url": f"/video/{key}",
                "from_timestamp": 0.0,
                "to_timestamp": None,
                "episode_duration": None,
                "relative_path": str(path),
            }

    controller = PredictionController(
        args.predictions,
        validator,
        control_hz=args.control_hz,
        initial_state=args.initial_state,
        action_horizon=args.action_horizon,
        dataset_episode=dataset_episode,
        video_urls={key: f"/video/{key}" for key in video_files},
        video_segments=video_segments,
    )

    server = create_server(
        controller,
        host=args.host,
        port=args.port,
        static_root=pathlib.Path(__file__).resolve().parent / "static",
        robot_assets_root=validator.assets_root,
        video_files=video_files,
    )
    print(f"North episode visualizer: http://{args.host}:{args.port}")
    print(f"Predictions: {controller.predictions.shape}")
    if dataset_episode is not None:
        print(f"Dataset episode: {dataset_episode.episode_index} ({len(dataset_episode.actions)} GT actions)")
        print(f"Dataset export: {dataset_episode.export_root}")
    if video_files:
        for key, path in video_files.items():
            print(f"{key} video: {path}")
    print(f"Stages: {controller.status()['stage_count']}, horizon: {args.action_horizon}")
    if validator.load_error:
        print(f"WARNING: URDF checks unavailable: {validator.load_error}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
