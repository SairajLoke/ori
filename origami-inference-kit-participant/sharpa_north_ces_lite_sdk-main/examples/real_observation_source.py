"""
real_observation_source.py

Drop-in replacement data source for `make_synthetic_observation()` in the
origami-zenoh-v1 black-box validator. Reads actual frames from a LeRobot
v3.0 dataset instead of generating synthetic pixels, so real captured
episodes can be replayed through the same protocol path used against a live
policy server -- useful for checking policy accuracy against real data, not
just protocol conformance.

Nothing about the protocol, validator control flow, or message schema
changes. This module only produces observation dicts with EXACTLY the same
keys / shapes / dtypes that make_synthetic_observation() already produced --
see REQUIRED_IMAGE_SPECS / OPTIONAL_IMAGE_SPECS / VECTOR_SPECS in the policy
server file, which the validator's synthetic observation already matches:

    observation/image/head_left      uint8 (224, 224, 3)
    observation/image/head_right     uint8 (224, 224, 3)
    observation/image/wrist_left     uint8 (224, 224, 3)
    observation/image/wrist_right    uint8 (224, 224, 3)
    observation/image/tactile_deform uint8 (480, 1200, 3)
    observation/image/tactile_raw    uint8 (480, 1600, 3)   [optional]
    observation/state                float32 (65,)
    observation/state/joint_torque   float32 (65,)
    observation/tactile              float32 (60,)
    prompt                           str

Note: this is single-frame replay, NOT a windowed/delta_timestamps dataset
like origami_dataset.py's get_origami_full_dataset() used at training time --
deliberately so, since the server rebuilds its own temporal window and
tactile past/delta history internally, one raw frame per infer() call, and
that's what real deployment traffic looks like too.

--------------------------------------------------------------------------
Minimal integration into the validator (nothing else changes):

    from real_observation_source import RealObservationSource

    source = RealObservationSource(
        dataset_root=args.dataset_root,   # .../lerobot3.0
        episode_index=args.episode_index,
    )

    # in run_validation(), replace:
    #     observation = make_synthetic_observation()
    #     for index in range(requests):
    #         request_observation = dict(observation)
    #         if index == requests - 1:
    #             request_observation.pop("observation/image/tactile_raw")
    #         ...
    # with:
    #     for index in range(requests):
    #         request_observation = source.next_observation()
    #         ...
    #
    # and call source.reset_episode() right after the validator's own
    # "reset" query, so replay position and the server's internal history
    # buffer restart together:
    #
    #     validate_reset(query_once(session, "reset", session_id, timeout))
    #     source.reset_episode()
--------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "real_observation_source.py needs the `lerobot` package installed "
        "in the validator's environment (same one origami_dataset.py uses)."
    ) from exc


# Must exactly match the server's / validator's wire contract.
IMAGE_SHAPE = (224, 224, 3)  # head_left/right, wrist_left/right
TACTILE_DEFORM_SHAPE = (480, 1200, 3)
TACTILE_RAW_SHAPE = (480, 1600, 3)
STATE_DIM = 65
TACTILE_DIM = 60

# dataset feature key -> wire observation key, for the square cams that need
# a resize (native 480x480 -> wire 224x224 -- a plain square resize, matching
# "a square 224x224 stretch of the native frame" per TeamPolicy's own docs;
# NOT the 224x320 stretch training used, that's a separate preprocessing path).
_RESIZED_CAM_KEYS = {
    "observation.images.head_left": "observation/image/head_left",
    "observation.images.head_right": "observation/image/head_right",
    "observation.images.wrist_left": "observation/image/wrist_left",
    "observation.images.wrist_right": "observation/image/wrist_right",
}

# dataset feature key -> wire observation key, for images already stored at
# (or near) the exact wire resolution.
_PASSTHROUGH_IMAGE_KEYS = {
    "observation.images.tactile_deform": "observation/image/tactile_deform",
    "observation.images.tactile_raw": "observation/image/tactile_raw",
}


def _float_chw_to_uint8_hwc(frame) -> np.ndarray:
    """LeRobotDataset video frames decode as float32 CHW in [0, 1]. The wire
    protocol wants uint8 HWC in [0, 255] -- the exact inverse of what
    TeamPolicy._preprocess_image() does on the way in (tensor/255.0)."""
    arr = frame.detach().cpu().numpy()  # [C, H, W], float32, ~[0, 1]
    arr = np.transpose(arr, (1, 2, 0))  # [H, W, C]
    arr = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return arr


def _resize_uint8_hwc(image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Plain resize (no letterboxing/cropping), matching the competition's
    'square stretch' framing. Uses cv2 if available, falls back to PIL."""
    try:
        import cv2

        return cv2.resize(
            image, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR
        )
    except ImportError:
        from PIL import Image

        pil = Image.fromarray(image)
        pil = pil.resize((target_hw[1], target_hw[0]), Image.BILINEAR)
        return np.array(pil)


class RealObservationSource:
    """
    Wraps a LeRobot v3.0 dataset and hands out one real observation dict per
    call, matching make_synthetic_observation()'s exact wire contract.
    Advances frame-by-frame through one episode; call reset_episode()
    whenever you also send the server's "reset" operation.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        drop_tactile_raw_every_n: int,
        episode_index: int = 0,
        prompt_override: Optional[str] = None,
    ):
        """
        dataset_root: path to the `lerobot3.0/` directory (the one
            containing meta/info.json), e.g.
            "Robotic_Origami_Challenge/season_.../lerobot3.0"
        episode_index: which episode to replay observations from.
        prompt_override: if set, always use this string for "prompt"
            instead of looking up the episode's task text from
            meta/tasks.jsonl.
        drop_tactile_raw_every_n: if set (e.g. 5), every Nth observation
            omits "observation/image/tactile_raw" -- exercises the same
            optional-field code path the original synthetic loop tested
            (there, only on the very last request), just periodically here.
        """
        self.dataset_root = Path(dataset_root)
        self.ds = LeRobotDataset(
            repo_id=None,
            root=self.dataset_root,
            video_backend="pyav",
        )
        self.prompt_override = prompt_override
        self.drop_tactile_raw_every_n = drop_tactile_raw_every_n
        self._call_count = 0

        self.episode_index = episode_index
        self._frame_indices = self._episode_frame_indices(episode_index)
        if not self._frame_indices:
            raise ValueError(f"episode {episode_index} has no frames in this dataset")
        self._cursor = 0
        self._episode_prompt = self.prompt_override or self._lookup_task_text(
            self._frame_indices[0]
        )

    # ------------------------------------------------------------------ #

    def _episode_frame_indices(self, episode_index: int) -> list[int]:
        """All dataset row indices belonging to one episode, in order."""
        table = self.ds.hf_dataset  # underlying memory-mapped HF dataset
        return [
            i
            for i in range(len(table))
            if int(table[i]["episode_index"]) == episode_index
        ]

    def _lookup_task_text(self, frame_idx: int) -> str:
        row = self.ds.hf_dataset[frame_idx]
        task_index = int(row["task_index"])
        try:
            # Most lerobot versions expose task text keyed by task_index on
            # dataset.meta; fall back to reading tasks.jsonl directly if the
            # attribute differs in yours.
            return self.ds.meta.tasks[task_index]
        except Exception:
            tasks_path = self.dataset_root / "meta" / "tasks.jsonl"
            if tasks_path.exists():
                with open(tasks_path) as f:
                    for line in f:
                        entry = json.loads(line)
                        if entry.get("task_index") == task_index:
                            return entry.get("task", "origami task")
            return "origami task"

    # ------------------------------------------------------------------ #

    def reset_episode(self, episode_index: Optional[int] = None) -> None:
        """Call alongside the server's 'reset' operation so replay position
        and the server's internal temporal-history buffer restart together."""
        if episode_index is not None:
            self.episode_index = episode_index
            self._frame_indices = self._episode_frame_indices(episode_index)
            if not self._frame_indices:
                raise ValueError(f"episode {episode_index} has no frames")
            self._episode_prompt = self.prompt_override or self._lookup_task_text(
                self._frame_indices[0]
            )
        self._cursor = 0

    def has_next(self) -> bool:
        return self._cursor < len(self._frame_indices)

    def assert_requests_validity(self, requests: int, frame_stride: int = 1) -> None:
        """
        Raise a clear, upfront ValueError if `requests` calls to
        next_observation(frame_stride=frame_stride) would run past the end
        of this episode -- instead of failing with a StopIteration mid-run
        after already spending requests-1 real server queries.

        frame_stride must match whatever stride you're about to actually
        replay with (see next_observation()'s frame_stride param / the
        module docstring section on overlapping vs non-overlapping chunks):
          - frame_stride=1              -> every call advances one real frame
          - frame_stride=action_horizon -> matches real deployment cadence
                                            (only re-query after the previous
                                            chunk would have been fully
                                            executed) -> non-overlapping chunks
        """
        if requests < 1:
            raise ValueError(f"requests must be >= 1, got {requests}")
        if frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
        frames_needed = (requests - 1) * frame_stride + 1
        frames_available = len(self._frame_indices) - self._cursor
        if frames_needed > frames_available:
            raise ValueError(
                f"requests={requests} at frame_stride={frame_stride} needs "
                f"{frames_needed} more frame(s), but episode "
                f"{self.episode_index} only has {frames_available} remaining "
                f"from the current cursor (episode length="
                f"{len(self._frame_indices)}). Reduce --requests, reduce "
                f"--frame-stride, or point --episode-index at a longer episode."
            )
     
    # def get_last_frame_info(self) -> dict[str, Any]:
    #     """Metadata about whichever frame next_observation() most recently
    #     returned -- call this right after next_observation(), not instead
    #     of it. Not part of the wire observation dict (see the note in
    #     __init__): the server validates observations against a closed key
    #     set, so this rides alongside the reply for logging/saving instead."""
    #     return {
    #         "episode_index": self.episode_index,
    #         "dataset_row_index": self.last_dataset_row_index,
    #         "frame_index": self.last_frame_index,
    #         "timestamp_s": self.last_timestamp_s,         
    #         # "future gt actions? "   ,
            
    #     }
 


    def next_observation(self, frame_stride) -> dict[str, Any]:
        """Returns one observation dict matching
        make_synthetic_observation()'s exact contract.

        frame_stride: how many dataset frames to advance the cursor by after
        this call. Defaults to 1 (dense replay, produces overlapping
        predicted action chunks if the server returns a multi-step horizon
        per call). Pass frame_stride=action_horizon to instead replay at the
        cadence a real deployed robot would use -- only requesting a new
        chunk once the previous one would have finished executing -- which
        gives non-overlapping chunks."""
        stride = frame_stride 
        assert stride is not None and stride >= 0, "frame_stride must be >= 0"
        if not self.has_next():
            raise StopIteration(
                f"episode {self.episode_index} exhausted "
                f"({len(self._frame_indices)} frames replayed) -- call "
                f"reset_episode(next_episode_index) or reduce --requests"
            )
        frame_idx = self._frame_indices[self._cursor]
        self._cursor += stride
        self._call_count += 1

        sample = self.ds[frame_idx]
        observation: dict[str, Any] = {}

        for ds_key, wire_key in _RESIZED_CAM_KEYS.items():
            img = _float_chw_to_uint8_hwc(sample[ds_key])
            img = _resize_uint8_hwc(img, IMAGE_SHAPE[:2])
            observation[wire_key] = np.ascontiguousarray(img)

        for ds_key, wire_key in _PASSTHROUGH_IMAGE_KEYS.items():
            if ds_key not in sample:
                continue  # e.g. tactile_raw absent in this dataset variant
            img = _float_chw_to_uint8_hwc(sample[ds_key])
            expected_shape = (
                TACTILE_DEFORM_SHAPE if "deform" in ds_key else TACTILE_RAW_SHAPE
            )
            if img.shape != expected_shape:
                img = _resize_uint8_hwc(img, expected_shape[:2])
            observation[wire_key] = np.ascontiguousarray(img)

        state = sample["observation.state"].detach().cpu().numpy().astype(np.float32)
        observation["observation/state"] = np.ascontiguousarray(state[:STATE_DIM])

        if "observation.state.joint_torque" in sample:
            torque = (
                sample["observation.state.joint_torque"]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        else:
            torque = np.zeros(STATE_DIM, dtype=np.float32)
        observation["observation/state/joint_torque"] = np.ascontiguousarray(
            torque[:STATE_DIM]
        )

        tactile = sample["observation.tactile"].detach().cpu().numpy().astype(np.float32)
        observation["observation/tactile"] = np.ascontiguousarray(tactile[:TACTILE_DIM])

        observation["prompt"] = self._episode_prompt

        if (
            self.drop_tactile_raw_every_n
            and self._call_count % self.drop_tactile_raw_every_n == 0
        ):
            observation.pop("observation/image/tactile_raw", None)

        return observation