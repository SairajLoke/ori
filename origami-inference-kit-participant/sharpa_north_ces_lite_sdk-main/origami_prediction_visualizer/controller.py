"""Local prerecorded prediction controller: no robot, policy, Docker, or network."""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np

from .contract import ACTION_DIM, JOINT_NAMES
from .trajectory import TrajectoryValidator

class PredictionController:
    def __init__(
        self,
        predictions: np.ndarray | str | pathlib.Path,
        validator: TrajectoryValidator,
        *,
        control_hz: float = 30.0,
        initial_state: np.ndarray | str | pathlib.Path | None = None,
        action_horizon: int = 25,
    ) -> None:
        self.validator = validator
        self.control_hz = float(control_hz)
        self.action_horizon = int(action_horizon)
        self.predictions = self._load_predictions(predictions, self.action_horizon)
        self.initial_state = self._load_initial_state(initial_state)
        self.validation = validator.validate(
            self.predictions,
            control_hz=self.control_hz,
            initial_state=self.initial_state,
        )
        self._events = [
            f"Loaded predictions: shape={list(self.predictions.shape)}",
            f"Action horizon: {self.action_horizon}",
            f"Control frequency: {self.control_hz:g} Hz",
        ]
        if self.initial_state is None:
            self._events.append(
                "No initial state supplied: motion checks begin at prediction step 1."
            )
        else:
            self._events.append("Initial state supplied: motion checks include step 0.")

    @staticmethod
    def _load_predictions(value: np.ndarray | str | pathlib.Path, horizon: int) -> np.ndarray:
        if isinstance(value, (str, pathlib.Path)):
            path = pathlib.Path(value).expanduser().resolve()
            loaded = np.load(path, allow_pickle=False)
            if isinstance(loaded, np.lib.npyio.NpzFile):
                if "predictions" in loaded.files:
                    loaded = loaded["predictions"]
                elif len(loaded.files) == 1:
                    loaded = loaded[loaded.files[0]]
                else:
                    raise ValueError("NPZ must contain 'predictions' or exactly one array")
        else:
            loaded = np.asarray(value)

        arr = np.asarray(loaded)
        # Accept both flattened [T,65] and original [T',25,65].
        if arr.ndim == 3:
            if arr.shape[1] != horizon or arr.shape[2] != ACTION_DIM:
                raise ValueError(
                    f"3-D predictions must have shape [T', {horizon}, {ACTION_DIM}], got {arr.shape}"
                )
            arr = arr.reshape(-1, ACTION_DIM)
        elif arr.ndim != 2 or arr.shape[1] != ACTION_DIM:
            raise ValueError(
                f"predictions must have shape [T, {ACTION_DIM}] or "
                f"[T', {horizon}, {ACTION_DIM}], got {arr.shape}"
            )

        if arr.shape[0] == 0:
            raise ValueError("predictions cannot be empty")
        if not np.isfinite(arr).all():
            raise ValueError("predictions contain NaN or Inf")
        return np.ascontiguousarray(arr, dtype=np.float32)

    @staticmethod
    def _load_initial_state(value: Any | None) -> np.ndarray | None:
        if value is None:
            return None
        if isinstance(value, (str, pathlib.Path)):
            arr = np.asarray(np.load(pathlib.Path(value).expanduser(), allow_pickle=False))
        else:
            arr = np.asarray(value)
        if arr.shape != (ACTION_DIM,):
            raise ValueError(f"initial state must have shape ({ACTION_DIM},), got {arr.shape}")
        return np.ascontiguousarray(arr, dtype=np.float32)

    def status(self) -> dict[str, Any]:
        t = self.predictions.shape[0]
        return {
            "prediction_shape": [int(x) for x in self.predictions.shape],
            "total_steps": t,
            "action_dim": ACTION_DIM,
            "action_horizon": self.action_horizon,
            "stage_count": (t + self.action_horizon - 1) // self.action_horizon,
            "control_hz": self.control_hz,
            "initial_state_provided": self.initial_state is not None,
            "compatible": bool(self.validation["compatible"]),
            "urdf_limits_loaded": self.validator.has_urdf_limits,
            "urdf_error": self.validator.load_error,
        }

    def trajectory(self) -> dict[str, Any]:
        return {
            "prediction": self.predictions.tolist(),
            "current_state": (
                self.initial_state.tolist()
                if self.initial_state is not None
                else None
            ),
            "action_shape": list(self.predictions.shape),
            "action_min": float(self.predictions.min()),
            "action_max": float(self.predictions.max()),
            "chunk_count": (len(self.predictions) + self.action_horizon - 1) // self.action_horizon,
            "action_horizon": self.action_horizon,
            "stage_count": (len(self.predictions) + self.action_horizon - 1) // self.action_horizon,
            "control_hz": self.control_hz,
            "validation": self.validation,
            "compatible": bool(self.validation["compatible"]),
            "metadata": {
                "action_dim": ACTION_DIM,
                "action_type": "absolute_joint_position",
                "action_units": "radians",
                "joint_names": list(JOINT_NAMES),
            },
        }

    def observation(self) -> dict[str, Any]:
        state = (
            self.initial_state
            if self.initial_state is not None
            else self.predictions[0]
        )
        return {
            "state": state.tolist(),
            "joint_names": list(JOINT_NAMES),
            "prompt": "Prerecorded model predictions",
        }

    def robot_config(self) -> dict[str, Any]:
        return self.validator.robot_config()

    def logs(self) -> dict[str, Any]:
        return {"events": self._events}
