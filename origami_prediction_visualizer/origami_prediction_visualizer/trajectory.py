"""North URDF position/velocity validation for a prerecorded trajectory."""

from __future__ import annotations

import math
import pathlib
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import quote

import numpy as np

from .contract import ACTION_DIM, JOINT_GROUPS, JOINT_NAMES

DEFAULT_URDF_RELATIVE_PATH = "north_poc2_2_v3_1.urdf"

DEFAULT_GROUP_JUMPS = {
    "left_arm": 0.1,
    "left_hand": math.radians(60),
    "right_arm": 0.1,
    "right_hand": math.radians(60),
    "motor": 0.06,
}

class TrajectoryValidator:
    def __init__(
        self,
        robot_assets_dir: str | pathlib.Path,
        *,
        urdf_relative_path: str = DEFAULT_URDF_RELATIVE_PATH,
        position_tolerance_rad: float = math.radians(2),
    ) -> None:
        self.assets_root = pathlib.Path(robot_assets_dir).expanduser().resolve()
        relative = pathlib.PurePosixPath(urdf_relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("URDF path must stay below --robot-assets-dir")
        self.urdf_path = (self.assets_root / pathlib.Path(*relative.parts)).resolve()
        self.urdf_path.relative_to(self.assets_root)
        self.urdf_relative_path = relative.as_posix()
        self.position_tolerance_rad = float(position_tolerance_rad)
        self.load_error: str | None = None
        self._limits: np.ndarray | None = None
        try:
            self._limits = self._load_limits()
        except ValueError as error:
            self.load_error = str(error)

    @property
    def has_urdf_limits(self) -> bool:
        return self._limits is not None

    def robot_config(self) -> dict[str, Any]:
        limits: dict[str, Any] = {}
        if self._limits is not None:
            limits = {
                name: {
                    "lower": float(self._limits[i, 0]),
                    "upper": float(self._limits[i, 1]),
                    "velocity": float(self._limits[i, 2]),
                    "jump": self._jump_limit(i),
                }
                for i, name in enumerate(JOINT_NAMES)
            }

        layout = [
            {"index": i, "name": JOINT_NAMES[i], "group": group}
            for group, start, end in JOINT_GROUPS
            for i in range(start, end)
        ]
        return {
            "urdf_url": f"/robot-assets/{quote(self.urdf_relative_path, safe='/')}",
            "urdf_available": self.urdf_path.is_file(),
            "urdf_limits_loaded": self.has_urdf_limits,
            "urdf_error": self.load_error,
            "joint_map": layout,
            "limits": limits,
            "viewer": "local Three.js URDFLoader with STL meshes",
        }

    def validate(
        self,
        actions: Any,
        *,
        control_hz: float = 30.0,
        initial_state: Any | None = None,
    ) -> dict[str, Any]:
        trajectory = np.asarray(actions)
        input_violations: list[dict[str, Any]] = []

        if (
            trajectory.dtype != np.dtype(np.float32)
            or trajectory.ndim != 2
            or trajectory.shape[1:] != (ACTION_DIM,)
        ):
            input_violations.append({
                "type": "input", "field": "actions",
                "message": "must be float32[T,65]",
                "actual_dtype": str(trajectory.dtype),
                "actual_shape": list(trajectory.shape),
            })

        if trajectory.size and not np.isfinite(trajectory).all():
            input_violations.append({
                "type": "input", "field": "actions",
                "message": "contains NaN or Inf",
            })

        if not math.isfinite(float(control_hz)) or float(control_hz) <= 0:
            input_violations.append({
                "type": "input", "field": "control_hz",
                "message": "must be positive",
            })

        initial = None
        if initial_state is not None:
            initial = np.asarray(initial_state)
            if initial.dtype != np.dtype(np.float32) or initial.shape != (ACTION_DIM,):
                input_violations.append({
                    "type": "input", "field": "initial_state",
                    "message": "must be float32[65]",
                    "actual_dtype": str(initial.dtype),
                    "actual_shape": list(initial.shape),
                })
            elif not np.isfinite(initial).all():
                input_violations.append({
                    "type": "input", "field": "initial_state",
                    "message": "contains NaN or Inf",
                })

        if input_violations:
            return {
                "compatible": False,
                "validation_level": "shape-finite",
                "input_violations": input_violations,
                "violations": [],
                "step_reports": [],
                "urdf_error": self.load_error,
            }

        # If no real initial state is supplied, the first prediction has no
        # preceding state. We therefore skip motion-limit checks for step 0.
        previous = initial
        step_reports = []
        violations: list[dict[str, Any]] = []

        for step, target in enumerate(trajectory):
            report = {
                "step": step,
                "source": "predictions",
                "compatible": True,
                "violations": [],
                "joint_violations": {},
            }

            if self._limits is not None:
                for index, value in enumerate(target):
                    lower, upper, velocity_limit = self._limits[index]

                    if value < lower - self.position_tolerance_rad:
                        self._add_issue(
                            report, self._issue("lower_limit", step, index, value, lower)
                        )
                    elif value > upper + self.position_tolerance_rad:
                        self._add_issue(
                            report, self._issue("upper_limit", step, index, value, upper)
                        )

                    if previous is not None:
                        delta = abs(float(value) - float(previous[index]))
                        jump_limit = self._jump_limit(index)
                        if delta > jump_limit:
                            self._add_issue(
                                report, self._issue("step_jump", step, index, delta, jump_limit)
                            )

                        velocity = delta * float(control_hz)
                        if velocity > velocity_limit:
                            self._add_issue(
                                report, self._issue("velocity", step, index, velocity, velocity_limit)
                            )

            report["compatible"] = not report["violations"]
            violations.extend(report["violations"])
            step_reports.append(report)
            previous = target

        return {
            "compatible": not violations,
            "validation_level": (
                "shape-finite-urdf-position-velocity"
                if self._limits is not None else "shape-finite"
            ),
            "input_violations": [],
            "violations": violations,
            "step_reports": step_reports,
            "steps": step_reports,
            "steps_with_violations": sum(bool(r["violations"]) for r in step_reports),
            "urdf_error": self.load_error,
            "initial_state_provided": initial is not None,
        }

    def _load_limits(self) -> np.ndarray:
        if not self.urdf_path.is_file():
            raise ValueError(f"North URDF not found: {self.urdf_path}")
        try:
            root = ET.parse(self.urdf_path).getroot()
        except (OSError, ET.ParseError) as error:
            raise ValueError(f"cannot parse North URDF: {error}") from error

        joints = {
            joint.get("name"): joint
            for joint in root.findall(".//joint")
            if joint.get("name")
        }
        rows, errors = [], []
        for name in JOINT_NAMES:
            joint = joints.get(name)
            limit = None if joint is None else joint.find("limit")
            try:
                lower = float(limit.attrib["lower"])  # type: ignore[union-attr]
                upper = float(limit.attrib["upper"])  # type: ignore[union-attr]
                velocity = float(limit.attrib["velocity"])  # type: ignore[union-attr]
            except (AttributeError, KeyError, TypeError, ValueError):
                errors.append(f"{name}: missing numeric lower/upper/velocity")
                continue
            if not all(math.isfinite(v) for v in (lower, upper, velocity)):
                errors.append(f"{name}: non-finite limit")
            elif lower >= upper or velocity <= 0:
                errors.append(f"{name}: invalid lower/upper/velocity")
            else:
                rows.append((lower, upper, velocity))
        if errors:
            raise ValueError("invalid North URDF contract joints: " + "; ".join(errors))
        return np.asarray(rows, dtype=np.float64)

    @staticmethod
    def _issue(kind: str, step: int, index: int, value: float, limit: float) -> dict[str, Any]:
        return {
            "type": kind, "step": step, "joint_index": index,
            "joint_name": JOINT_NAMES[index],
            "value": float(value), "limit": float(limit),
        }

    @staticmethod
    def _add_issue(report: dict[str, Any], issue: dict[str, Any]) -> None:
        report["violations"].append(issue)
        report["joint_violations"].setdefault(issue["joint_name"], []).append(issue)

    @staticmethod
    def _jump_limit(index: int) -> float:
        for group, start, end in JOINT_GROUPS:
            if start <= index < end:
                return DEFAULT_GROUP_JUMPS[group]
        raise IndexError(index)
