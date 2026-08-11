"""Run the local North prediction visualizer."""

from __future__ import annotations

import argparse
import pathlib

from .controller import PredictionController
from .trajectory import DEFAULT_URDF_RELATIVE_PATH, TrajectoryValidator
from .web import create_server

def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=pathlib.Path, required=True,
                        help="NumPy .npy/.npz containing [T,65] or [T',25,65]")
    parser.add_argument("--robot-assets-dir", type=pathlib.Path, required=True)
    parser.add_argument("--urdf-relative-path", default=DEFAULT_URDF_RELATIVE_PATH)
    parser.add_argument("--initial-state", type=pathlib.Path, default=None,
                        help="Optional .npy containing the real/desired state before prediction step 0")
    parser.add_argument("--action-horizon", type=int, default=25)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args(argv)

    validator = TrajectoryValidator(
        args.robot_assets_dir,
        urdf_relative_path=args.urdf_relative_path,
    )
    controller = PredictionController(
        args.predictions,
        validator,
        control_hz=args.control_hz,
        initial_state=args.initial_state,
        action_horizon=args.action_horizon,
    )

    server = create_server(
        controller,
        host=args.host,
        port=args.port,
        static_root=pathlib.Path(__file__).resolve().parent / "static",
        robot_assets_root=validator.assets_root,
    )
    print(f"North prediction visualizer: http://{args.host}:{args.port}")
    print(f"Predictions: {controller.predictions.shape}")
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
