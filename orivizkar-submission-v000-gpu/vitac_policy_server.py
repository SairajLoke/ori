#!/usr/bin/env python3
"""Framework-neutral origami-zenoh-v1 policy server template.

Replace ``TeamPolicy`` with the team's model adapter. The public server boundary
must remain unchanged.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import os
import signal
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Mapping
from typing import Any

import msgpack
import numpy as np
import zenoh

# ---------------------------------------------------------------------------
# VITAC / vitacformer wiring
# ---------------------------------------------------------------------------
# VITACFORMER_ROOT must point at the directory containing policy.py/configs.py
# inside the image (e.g. /app/vitacformer). The vendored source there needs
# policy.py, detr/ (incl. its own util/ subpackage), configs.py, train_utils.py,
# my_utils/normalizer.py (zero lerobot/dataset dependency -- only os, warnings,
# typing, numpy, torch -- safe to vendor standalone) -- nothing else from
# dataset/ or the lerobot-dependent training scripts is required for inference.
VITACFORMER_ROOT = os.environ.get("VITACFORMER_ROOT", "/app/vitacformer")
if VITACFORMER_ROOT not in sys.path:
    sys.path.insert(0, VITACFORMER_ROOT)
import os
# os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/app/.cache/torchinductor"

import torch  # noqa: E402  (import after sys.path wiring)
import torch

# Configure valid PyTorch SDPA backends
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(True)


# policy.py does `from detr.main import build_ACT_model_and_optimizer` and
# `from train_utils import _stats`, both resolved relative to VITACFORMER_ROOT.
from policy import ACTPolicy  # noqa: E402
from my_utils.normalizer import OriNormalizer  # noqa: E402
from smoothing import MODES as SMOOTHING_MODES, smooth_chunk  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

TRANSPORT_VERSION = "origami-zenoh-v1"
SEMANTIC_VERSION = "origami-v1"
ACTION_DIM = 65
IMAGE_SHAPE = (224, 224, 3)
REQUIRED_IMAGE_SPECS = {
    "observation/image/head_left": IMAGE_SHAPE,
    "observation/image/head_right": IMAGE_SHAPE,
    "observation/image/wrist_left": IMAGE_SHAPE,
    "observation/image/wrist_right": IMAGE_SHAPE,
    "observation/image/tactile_deform": (480, 1200, 3),
}
OPTIONAL_IMAGE_SPECS = {
    "observation/image/tactile_raw": (480, 1600, 3),
}
VECTOR_SPECS = {
    "observation/state": (ACTION_DIM,),
    "observation/state/joint_torque": (ACTION_DIM,),
    "observation/tactile": (60,),
}
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024


def _hand_joint_names(side: str) -> tuple[str, ...]:
    return (
        f"{side}_thumb_CMC_FE",
        f"{side}_thumb_CMC_AA",
        f"{side}_thumb_MCP_FE",
        f"{side}_thumb_MCP_AA",
        f"{side}_thumb_IP",
        f"{side}_index_MCP_FE",
        f"{side}_index_MCP_AA",
        f"{side}_index_PIP",
        f"{side}_index_DIP",
        f"{side}_middle_MCP_FE",
        f"{side}_middle_MCP_AA",
        f"{side}_middle_PIP",
        f"{side}_middle_DIP",
        f"{side}_ring_MCP_FE",
        f"{side}_ring_MCP_AA",
        f"{side}_ring_PIP",
        f"{side}_ring_DIP",
        f"{side}_pinky_CMC",
        f"{side}_pinky_MCP_FE",
        f"{side}_pinky_MCP_AA",
        f"{side}_pinky_PIP",
        f"{side}_pinky_DIP",
    )


JOINT_NAMES = (
    tuple(f"left_arm_joint_{index}" for index in range(1, 8))
    + _hand_joint_names("left")
    + tuple(f"right_arm_joint_{index}" for index in range(1, 8))
    + _hand_joint_names("right")
    + (
        "lower_body_joint_1",
        "lower_body_joint_2",
        "lower_body_joint_3",
        "lower_body_joint_4",
        "lower_body_joint_5",
        "neck_joint_1",
        "neck_joint_2",
    )
)

if len(JOINT_NAMES) != ACTION_DIM or len(set(JOINT_NAMES)) != ACTION_DIM:
    raise RuntimeError("joint contract must contain 65 unique names")


def _pack_numpy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"O", "V", "c"}:
            raise ValueError(f"unsupported numpy dtype: {value.dtype}")
        array = np.ascontiguousarray(value)
        return {
            b"__ndarray__": True,
            b"data": array.tobytes(),
            b"dtype": array.dtype.str,
            b"shape": array.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _mapping_value(value: Mapping[Any, Any], key: str) -> Any:
    return value[key] if key in value else value.get(key.encode())


def _unpack_numpy(value: dict[Any, Any]) -> Any:
    if _mapping_value(value, "__ndarray__") is True:
        data = _mapping_value(value, "data")
        shape = _mapping_value(value, "shape")
        dtype = np.dtype(_mapping_value(value, "dtype"))
        if (
            not isinstance(data, bytes)
            or not isinstance(shape, (list, tuple))
            or len(shape) > 8
            or dtype.kind in {"O", "V", "c"}
            or dtype.hasobject
        ):
            raise ValueError("invalid numpy array payload")
        normalized_shape = tuple(int(dimension) for dimension in shape)
        if any(dimension < 0 for dimension in normalized_shape):
            raise ValueError("invalid numpy array shape")
        expected_size = math.prod(normalized_shape) * dtype.itemsize
        if expected_size > MAX_PAYLOAD_BYTES or len(data) != expected_size:
            raise ValueError("numpy array payload size does not match shape")
        return np.frombuffer(data, dtype=dtype).reshape(normalized_shape)
    if _mapping_value(value, "__npgeneric__") is True:
        return np.dtype(_mapping_value(value, "dtype")).type(
            _mapping_value(value, "data")
        )
    return value


def pack_payload(value: Any) -> bytes:
    payload = msgpack.packb(value, default=_pack_numpy, use_bin_type=True)
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("response exceeds 64 MiB")
    return payload


def unpack_payload(value: Any) -> Any:
    payload = value.to_bytes() if hasattr(value, "to_bytes") else bytes(value)
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("request exceeds 64 MiB")
    return msgpack.unpackb(
        payload,
        object_hook=_unpack_numpy,
        raw=False,
        strict_map_key=False,
        max_bin_len=MAX_PAYLOAD_BYTES,
        max_array_len=1_000_000,
        max_map_len=10_000,
        max_str_len=1_000_000,
    )


class TeamPolicy:
    """VITAC (vitacformer ACT) adapter for the origami-zenoh-v1 protocol.

    - qpos:          2D  [B, 390]         (6-step state history, flattened)
    - image:         5D  [B, 4, 3, 224, 224]  (per-camera resize, ImageNet-normalized
                                             if the checkpoint trained with it)
    - tactile (past): 2D  [B, 2160]        (18 steps x [value(60), delta(60)], flattened --
                                             detr_vae.py: `tactile_dim_all = 18 * 120`,
                                             `input_proj_tactile = nn.Linear(tactile_dim_all, ...)`)
    - tactile_next:   3D  [B, 18, 120]      (future-tactile training target; at inference
                                             the model uses its own prediction instead --
                                             only .shape is read, content is irrelevant)
    - a_hat:          3D  [B, 100, 65]      (num_queries is architecturally hardcoded to
                                             100 -- `assert num_queries == 100` in
                                             DETRVAE.__init__ -- not tunable)

    Architecture (hidden_dim, backbone, enc/dec_layers, nheads, camera order,
    mask_fingers, image_hw, ...) is NOT hardcoded here -- none of it is
    recoverable from the .ckpt file itself, so it is loaded from
    training_configs.json, which origami_imitate_episodes.py writes next to
    the checkpoint. Normalization is loaded the same way, from
    normalizer_config.json. See _load_training_config() / _load_normalizer().
    """

    TACTILE_TEMPORAL_HORIZON = 18  # detr_vae.py: architecturally fixed, not saved anywhere
    TACTILE_RAW_DIM = 60           # raw observation/tactile width (per robot_io_spec.md)
    TACTILE_FEATURE_DIM = 120      # [value(60), delta(60)] per step -- detr_vae.py tactile_dim

    # Get checkpoint path from environment or use default
    checkpoint_path = os.environ.get("VITAC_CKPT_PATH", None)
    # "/app/checkpoints/policy_best.ckpt"

    def _load_training_config(self) -> dict:
        """Load training_configs.json from next to the checkpoint -- the
        ACTPolicy args_override dict (policy_config) plus run settings
        (mask_fingers, hand_mask, image_hw, ...) origami_imitate_episodes.py
        writes verbatim. The single source of truth for architecture, since
        none of it is recoverable from the .ckpt file itself: guessing wrong
        means load_state_dict fails outright, or worse, silently succeeds
        with a mismatched architecture if shapes happen to coincide.
        """
        ckpt_dir = os.path.dirname(os.path.abspath(self.checkpoint_path))
        path = os.path.join(ckpt_dir, "training_configs.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Checkpoints from before this file was written "
                f"cannot be deployed safely -- there is no other record of the "
                f"training run's architecture."
            )
        with open(path) as f:
            cfg = json.load(f)
        logging.info("training config loaded from %s", path)
        return cfg

    def _load_normalizer(self) -> tuple[OriNormalizer | None, bool]:
        """Rebuild the normalizer the checkpoint was trained with, from the
        normalizer_config.json sidecar origami_imitate_episodes.py writes next
        to it -- mirrors origami_inference.py's load_training_normalizer().
        Every checkpoint trained with USE_NORMALIZATION=1 needs this or its
        inputs/outputs are silently in the wrong units. Set
        VITAC_ASSUME_UNNORMALIZED=1 only for a checkpoint you are certain
        trained with --disable_normalization and predates this sidecar.
        """
        if os.environ.get("VITAC_ASSUME_UNNORMALIZED", "").lower() in ("1", "true"):
            logging.warning("VITAC_ASSUME_UNNORMALIZED=1 -- running without normalization")
            return None, os.environ.get("VITAC_IMAGE_NORM", "1").lower() not in ("0", "false")

        sidecar = os.path.join(os.path.dirname(os.path.abspath(self.checkpoint_path)),
                                "normalizer_config.json")
        if not os.path.exists(sidecar):
            raise FileNotFoundError(
                f"{sidecar} not found. Inference must match the checkpoint's training "
                f"normalization, or actions come out in the wrong units. Set "
                f"VITAC_ASSUME_UNNORMALIZED=1 only if certain this checkpoint was "
                f"trained with --disable_normalization."
            )
        with open(sidecar) as f:
            cfg = json.load(f)
        use_image_norm = cfg.get("image_norm", True)
        if cfg.get("disable_normalization", False):
            logging.info("checkpoint trained WITHOUT normalization (identity)")
            return None, use_image_norm
        normalizer = OriNormalizer(
            stats=cfg["stats"], feature_modes=cfg["feature_modes"], device=self.device,
            degenerate_spread=cfg.get("degenerate_spread", 1e-3), clip=cfg.get("clip"),
        )
        logging.info("normalizer rebuilt from %s", sidecar)
        return normalizer, use_image_norm

    def _load_policy(self, policy_config, optimization_type) :
        logging.info("loading VITAC checkpoint from %s", self.checkpoint_path)
        logging.info("Optimization: %s", optimization_type)
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.policy = ACTPolicy(policy_config)
        # origami_inference.py's load_policy(): handles both {'model': state_dict, ...}
        # (periodic checkpoints) and a raw state_dict (the 'best' checkpoint format).
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            self.policy.load_state_dict(checkpoint["model"])
        else:
            self.policy.load_state_dict(checkpoint)
        self.policy.eval().to(self.device)
        
        
        if optimization_type == "compile":
            logging.info("Compiling policy with torch.compile...")
            # Use 'reduce-overhead' for CUDA, 'default' for CPU
            compile_mode = "reduce-overhead" if self.device.type == "cuda" else "default"
            self.policy = torch.compile(self.policy, mode=compile_mode)

            logging.info("Running dummy forward pass to warm up compiled graph...")
            dummy_qpos = torch.zeros((1, self.proprioceptive_temporal_horizon * self.state_dim), device=self.device)
            dummy_image = torch.zeros((1, len(self.camera_order), 3, *self.train_image_hw), device=self.device)
            dummy_tactile = torch.zeros((1, self.TACTILE_TEMPORAL_HORIZON * self.TACTILE_FEATURE_DIM), device=self.device)
            dummy_tactile_next = torch.zeros((1, self.TACTILE_TEMPORAL_HORIZON, self.TACTILE_FEATURE_DIM), device=self.device)

            with torch.no_grad():
                _ = self.policy(
                    dummy_qpos,
                    dummy_image,
                    device=self.device,
                    tactile=dummy_tactile,
                    tactile_next=dummy_tactile_next,
                )
            logging.info("Model warm-up and compilation complete.")
            
        return self.policy        
        
        
    def __init__(self, action_horizon: int) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_cfg = self._load_training_config()
        policy_config = dict(train_cfg["policy_config"])  # ACTPolicy args_override, copy before mutating

        self.chunk_size = policy_config["num_queries"]
        if action_horizon > self.chunk_size:
            raise ValueError(
                f"action_horizon ({action_horizon}) cannot exceed this checkpoint's "
                f"num_queries ({self.chunk_size}); the model only ever predicts "
                f"num_queries actions per call."
            )
        self.action_horizon = action_horizon

        self.state_dim = policy_config["state_dim"]
        self.proprioceptive_temporal_horizon = policy_config["proprioceptive_temporal_horizon"]
        self.mask_fingers = train_cfg.get("mask_fingers", False)
        self.hand_mask = train_cfg.get("hand_mask", [1] * 5 + [1] * 4 + [1] * 4 + [0] * 4 + [0] * 5)
        self.train_image_hw = tuple(train_cfg.get("image_hw", (224, 224)))
        # Dims the dataset holds constant: trained with zero loss weight, so the
        # model never learned them and its output there is unconstrained. The
        # reply is still a fixed 65 columns and the organizer checks jumps and
        # velocity, so emit the recorded constant instead of whatever drifted out.
        self.constant_action_dims = {int(k): float(v) for k, v in
                                     (train_cfg.get("constant_action_dims") or {}).items()}
        # When the model predicts a subset of the 65 contract columns, its output
        # position i maps to column predicted_action_dims[i]. Everything else has
        # to be filled: recorded constants where we have them, otherwise the
        # joint's current measured pose (hold it where it is). Order matters --
        # robot_io_spec.md fixes the column order and forbids reordering.
        self.predicted_action_dims = train_cfg.get("predicted_action_dims")
        if self.predicted_action_dims is not None:
            self.predicted_action_dims = [int(i) for i in self.predicted_action_dims]
            held = sorted(set(range(ACTION_DIM)) - set(self.predicted_action_dims)
                          - set(self.constant_action_dims))
            logging.info("model predicts %d/%d action dims; %d held at observed pose: %s",
                         len(self.predicted_action_dims), ACTION_DIM, len(held), held)

        # camera_names in policy_config is dataset-format ("observation.images.X");
        # CAMERA_ORDER is the public protocol's dict keys ("observation/image/X").
        # Deriving one from the other (instead of a second hand-maintained tuple)
        # is what guarantees the backbone-slot order always matches training.
        self.camera_order = tuple(
            name.replace("observation.images.", "observation/image/")
            for name in policy_config["camera_names"]
        )

        # backbone_weights in policy_config is a training-machine-absolute path and
        # will not exist in this image; re-resolve locally. load_state_dict below
        # overwrites these weights entirely regardless, so this only matters for
        # avoiding a network call (which would just fail offline) at construction.
        backbone = os.environ.get("VITAC_BACKBONE") or policy_config["backbone"]
        local_weights = os.path.join(VITACFORMER_ROOT, "assets", "backbones", f"{backbone}_imagenet.pth")
        policy_config["backbone"] = backbone
        policy_config["backbone_weights"] = (
            os.environ.get("VITAC_BACKBONE_WEIGHTS")
            or (local_weights if os.path.exists(local_weights) else None)
        )

        self.normalizer, self.use_image_norm = self._load_normalizer()

        optimizations = ['compile', 'tflite', 'none']
        OPTIMIZATION_IDX = 0
        # Env override: torch.compile is unusable on a CPU-only box (very slow,
        # and the inductor cache can exhaust RAM), so allow disabling it there.
        optimization = os.environ.get("VITAC_OPTIMIZATION") or optimizations[OPTIMIZATION_IDX]
        if optimization not in optimizations:
            raise ValueError(f"VITAC_OPTIMIZATION must be one of {optimizations}, got {optimization!r}")

        # Chunk-seam smoothing. robot_io_spec.md §6 lists history and temporal
        # ensembling as participant-internal, so any of these is contract-legal:
        # the reply stays finite float32[T,65] absolute radians and every buffer
        # below is cleared in reset(). See SMOOTHING_MODES for what each does.
        SMOOTHING_IDX = 6  # 'auto' -- index into smoothing.MODES; VITAC_SMOOTHING overrides
        self.smoothing = os.environ.get("VITAC_SMOOTHING") or SMOOTHING_MODES[SMOOTHING_IDX]
        if self.smoothing not in SMOOTHING_MODES:
            raise ValueError(f"VITAC_SMOOTHING must be one of {SMOOTHING_MODES}, got {self.smoothing!r}")
        self.blend_steps = int(os.environ.get("VITAC_BLEND_STEPS", "4"))
        self.ensemble_decay = float(os.environ.get("VITAC_ENSEMBLE_DECAY", "0.35"))
        self.max_step_rad = float(os.environ.get("VITAC_MAX_STEP_RAD", "0.10"))
        logging.info("smoothing=%s blend_steps=%d max_step_rad=%.3f",
                     self.smoothing, self.blend_steps, self.max_step_rad)

        logging.info("using device: %s", self.device)

        self.policy = self._load_policy(policy_config=policy_config,
                                        optimization_type=optimization)


        # Episode-scoped rolling history. The public protocol only ever sends the
        # *current* frame per infer() call, but the model was trained on short
        # windows of state/tactile history, so the server must reconstruct that
        # window itself across successive infer() calls within one episode, and
        # clear it on reset() (start of a new episode).
        self._state_history: collections.deque[np.ndarray] = collections.deque(
            maxlen=self.proprioceptive_temporal_horizon
        )
        # Need TACTILE_TEMPORAL_HORIZON + 1 raw readings to compute
        # TACTILE_TEMPORAL_HORIZON deltas (each delta needs a previous reading).
        self._tactile_history: collections.deque[np.ndarray] = collections.deque(
            maxlen=self.TACTILE_TEMPORAL_HORIZON + 1
        )
        self._prev_chunk: np.ndarray | None = None  # last reply, for 'ensemble'

        logging.info("Done Initializing the TeamPolicy ++++++++++")

        #MAYBE WE SHOULD DO A DUMMY FWD PASS HERE JUST SO THE MODELS COME IN CACHE?


    def reset(self) -> None:
        """Called at the start of every new episode -- must fully clear temporal
        state, or the first steps of a new episode would be conditioned on the
        previous episode's history."""
        
        logging.info(" TeamPolicy Reset called ++++++++")
        self._state_history.clear()
        self._tactile_history.clear()
        self._prev_chunk = None

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        #permute | to tensor | resize | [0,1] | ImageNet norm (if the checkpoint trained with it)
        tensor = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=self.train_image_hw,
            mode="bilinear",
            align_corners=False,
        )
        tensor = tensor / 255.0
        if self.use_image_norm:
            mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(1, 3, 1, 1)
            std = torch.tensor(IMAGENET_STD, device=tensor.device).view(1, 3, 1, 1)
            tensor = (tensor - mean) / std
        return tensor
            
        

    @torch.inference_mode()
    def infer(self, observation: dict[str, Any]) -> np.ndarray:
        
        
        logging.info(" TeamPolicy Infer called ++++++++")
        # --- images: HWC uint8 [0,255] -> resized to train_image_hw -> [0,1] ->
        # ImageNet-normalized if the checkpoint trained with it (_preprocess_image).
        cams = []
        for key in self.camera_order:
            tensor = self._preprocess_image(observation[key])
            cams.append(tensor)
        image_tensor = torch.stack(cams, dim=1).to(self.device)  # [1, n_cams, 3, H, W]
        logging.debug(
            "+++++= img obs min=%s max=%s shape=%s dtype=%s",
            torch.min(image_tensor),
            torch.max(image_tensor),
            image_tensor.shape,
            image_tensor.dtype,
        )

        # --- state history: 6-step window, oldest -> newest, flattened to [1, 390] ---
        state = np.asarray(observation["observation/state"], dtype=np.float32)
        logging.debug(
            "+++++= state obs min=%s max=%s shape=%s dtype=%s",
            np.min(state),
            np.max(state),
            state.shape,
            state.dtype,
        )
        
        if not self._state_history:
            # Cold start at episode begin: no history yet, so backfill with the
            # first reading. Not something the model was explicitly trained for --
            # verify it doesn't cause a visible transient in the first action chunk.
            for _ in range(self.proprioceptive_temporal_horizon):
                self._state_history.append(state)
        else:
            self._state_history.append(state)
        qpos = np.stack(list(self._state_history), axis=0)  # [6, 65] raw physical units
        qpos_t = torch.from_numpy(qpos).to(self.device)
        if self.normalizer is not None:
            qpos_t = self.normalizer.normalize("observation.state", qpos_t)

        if self.mask_fingers:
            mask = torch.as_tensor(self.hand_mask, dtype=qpos_t.dtype, device=qpos_t.device)
            qpos_t[:, 7:7 + len(mask)] *= mask
            qpos_t[:, 7 + 22 + 7:7 + 22 + 7 + len(mask)] *= mask
        qpos_flat = qpos_t.reshape(1, -1)  # [1, 390]

        # --- tactile history: 19 raw readings -> 18 x [value, delta] -> flattened ---
        tactile = np.asarray(observation["observation/tactile"], dtype=np.float32)
        logging.debug(
            "+++++= tactile obs min=%s max=%s shape=%s dtype=%s",
            np.min(tactile),
            np.max(tactile),
            tactile.shape,
            tactile.dtype,
        )
        
        if not self._tactile_history:
            for _ in range(self.TACTILE_TEMPORAL_HORIZON + 1):
                self._tactile_history.append(tactile)
        else:
            self._tactile_history.append(tactile)
        tactile_raw = np.stack(list(self._tactile_history), axis=0)  # [19, 60] raw physical units
        tactile_t = torch.from_numpy(tactile_raw).to(self.device)
        if self.normalizer is not None:
            # Normalize BEFORE the diff, same as convert_batch: diff(x/s) == diff(x)/s
            # only holds if both halves see the same per-dim scale, which this ensures.
            tactile_t = self.normalizer.normalize("observation.tactile", tactile_t)
        tactile_deltas = torch.diff(tactile_t, dim=0)                # [18, 60]
        tactile_features = torch.cat([tactile_t[1:], tactile_deltas], dim=-1)  # [18, 120]
        # Flattened to [1, 2160]: detr_vae.py's input_proj_tactile is
        # nn.Linear(tactile_dim_all=18*120, hidden_dim) -- a single projection over
        # the whole flattened window, NOT a per-timestep sequence input. Confirmed
        # directly from the model source; keeping this 3D (an earlier, less-verified
        # version of this adapter did) silently feeds the wrong tensor rank in.
        tactile_tensor = tactile_features.reshape(1, -1)  # [1, 2160]

        # tactile_next is the model's auxiliary *future*-tactile training target.
        # At inference (epoch>=75 internally) the model uses its own predicted
        # tactile_hat instead of this value -- only its shape is read (detr_vae.py
        # does `B, T, D = tactile_hat.shape` then `tactile_next.view(B, T*D)`), so
        # a zero placeholder of the right 3D shape is safe.
        tactile_next_tensor = torch.zeros(
            (1, self.TACTILE_TEMPORAL_HORIZON, self.TACTILE_FEATURE_DIM),
            dtype=torch.float32,
            device=self.device,
        )

        a_hat = self.policy(
            qpos_flat,
            image_tensor,
            device=self.device,
            tactile=tactile_tensor,
            tactile_next=tactile_next_tensor,
        )  # [1, 100, 65] -- normalized model-space units if self.normalizer is set,
           # denormalized to absolute joint-position radians below.
        # Scatter a subset-predicting model back into the 65 contract columns
        # BEFORE denormalizing, so each column meets its own per-dim stats.
        # Unpredicted columns are filled after denormalization, in raw radians.
        if self.predicted_action_dims is not None:
            full = a_hat.new_zeros(a_hat.shape[:-1] + (ACTION_DIM,))
            full[..., self.predicted_action_dims] = a_hat
            a_hat = full
        if self.normalizer is not None:
            a_hat = self.normalizer.denormalize("action", a_hat.float())
        # after denormalization: the recorded constants are raw radians
        for _d, _v in self.constant_action_dims.items():
            a_hat[..., _d] = _v
        if self.predicted_action_dims is not None:
            _held = [i for i in range(ACTION_DIM)
                     if i not in self.predicted_action_dims and i not in self.constant_action_dims]
            if _held:
                # hold at the measured pose; `state` is this call's authoritative reading
                _cur = torch.as_tensor(state, dtype=a_hat.dtype, device=a_hat.device)
                a_hat[..., _held] = _cur[_held]
        logging.debug(
            "+++++= a_hat min=%s max=%s shape=%s dtype=%s",
            torch.min(a_hat),
            torch.max(a_hat),
            a_hat.shape,
            a_hat.dtype,
        )

        actions = a_hat[0, : self.action_horizon].detach().to("cpu").numpy().astype(np.float32)
        if self.smoothing != 'none':
            actions = smooth_chunk(
                actions, state, self._prev_chunk, self.smoothing,
                blend_steps=self.blend_steps, ensemble_decay=self.ensemble_decay,
                max_step_rad=self.max_step_rad,
            ).astype(np.float32)
        self._prev_chunk = actions
        return np.ascontiguousarray(actions)


class OrigamiZenohServer:
    def __init__(
        self,
        policy: TeamPolicy,
        *,
        endpoint: str,
        session_id: str,
        action_horizon: int,
    ) -> None:
        if action_horizon < 1 or action_horizon > 1024:
            raise ValueError("action_horizon must be in [1, 1024]")
        self.policy = policy
        self.endpoint = endpoint
        self.session_id = session_id
        self.action_horizon = action_horizon
        self._policy_lock = threading.Lock()
        self._stop = threading.Event()
        self._session: Any | None = None
        self._queryables: list[Any] = []
        self.metadata = {
            "protocol_version": SEMANTIC_VERSION,
            "action_dim": ACTION_DIM,
            "action_horizon": action_horizon,
            "action_type": "absolute_joint_position",
            "action_units": "radians",
            "joint_names": JOINT_NAMES,
        }

    def serve_forever(self) -> None:
        config = zenoh.Config()
        config.insert_json5("mode", json.dumps("client"))
        config.insert_json5("connect/endpoints", json.dumps([self.endpoint]))
        config.insert_json5("scouting/multicast/enabled", "false")
        config.insert_json5("transport/shared_memory/enabled", "false")
        self._session = zenoh.open(config)
        self._queryables = [
            self._session.declare_queryable(
                f"{TRANSPORT_VERSION}/{operation}",
                self._handle_query,
                complete=True,
            )
            for operation in ("metadata", "reset", "infer")
        ]
        signal.signal(signal.SIGTERM, lambda *_: self._stop.set())
        signal.signal(signal.SIGINT, lambda *_: self._stop.set())
        logging.info(
            "READY transport=%s endpoint=%s horizon=%d",
            TRANSPORT_VERSION,
            self.endpoint,
            self.action_horizon,
        )
        self._stop.wait()
        for queryable in self._queryables:
            queryable.undeclare()
        self._session.close()

    def _handle_query(self, query: Any) -> None:
        operation = str(query.key_expr).rsplit("/", 1)[-1]
        request: Any = None
        try:
            request = unpack_payload(query.payload)
            response = self.process(operation, request)
        except Exception as exc:  # noqa: BLE001 - sanitized protocol error
            error_id = uuid.uuid4().hex
            logging.error(
                "request failed operation=%s error_id=%s type=%s\n%s",
                operation,
                error_id,
                type(exc).__name__,
                traceback.format_exc(),
            )
            public_message = f"request failed; error_id={error_id}"
            if not isinstance(request, Mapping):
                query.reply_err(
                    pack_payload(
                        {
                            "error": {
                                "code": "INVALID_REQUEST",
                                "message": public_message,
                                "retryable": False,
                            }
                        }
                    ),
                    encoding="application/msgpack",
                )
                return
            response = self._envelope(operation, request)
            response["error"] = {
                "code": (
                    "INFERENCE_FAILED" if operation == "infer" else "INVALID_REQUEST"
                ),
                "message": public_message,
                "retryable": False,
            }
        query.reply(
            str(query.key_expr),
            pack_payload(response),
            encoding="application/msgpack",
        )

    def process(self, operation: str, request: Any) -> dict[str, Any]:
        if not isinstance(request, Mapping):
            raise ValueError("request must be a MessagePack map")
        response = self._envelope(operation, request)
        if request.get("protocol_version") != TRANSPORT_VERSION:
            raise ValueError("invalid protocol_version")
        if request.get("operation") != operation:
            raise ValueError("operation does not match queryable key")
        if request.get("session_id") != self.session_id:
            raise ValueError("session_id does not match assigned session")
        if not isinstance(request.get("request_id"), str) or not request["request_id"]:
            raise ValueError("request_id must be a non-empty string")

        if operation == "metadata":
            response["metadata"] = self.metadata
            return response
        if operation == "reset":
            with self._policy_lock:
                self.policy.reset()
            response["ok"] = True
            return response
        if operation != "infer":
            raise ValueError(f"unsupported operation: {operation}")

        observation = request.get("observation")
        self._validate_observation(observation)
        started = time.monotonic()
        with self._policy_lock:
            actions = self.policy.infer(dict(observation))
        actions = np.asarray(actions)
        expected_shape = (self.action_horizon, ACTION_DIM)
        if actions.dtype != np.float32 or actions.shape != expected_shape:
            raise ValueError(
                f"policy actions must be float32{expected_shape}, "
                f"got {actions.dtype}{actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("policy actions contain NaN or Inf")
        response["actions"] = np.ascontiguousarray(actions)
        response["server_timing"] = {
            "infer_ms": (time.monotonic() - started) * 1000.0
        }
        return response

    def _envelope(
        self,
        operation: str,
        request: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "protocol_version": TRANSPORT_VERSION,
            "operation": operation,
            "request_id": request.get("request_id"),
            "session_id": self.session_id,
        }

    @staticmethod
    def _validate_observation(observation: Any) -> None:
        if not isinstance(observation, Mapping):
            raise ValueError("infer request must contain an observation map")
        required = {*REQUIRED_IMAGE_SPECS, *VECTOR_SPECS, "prompt"}
        allowed = required | set(OPTIONAL_IMAGE_SPECS)
        if not required.issubset(observation) or not set(observation).issubset(allowed):
            raise ValueError("observation keys do not match the public full schema")
        for key, shape in {**REQUIRED_IMAGE_SPECS, **OPTIONAL_IMAGE_SPECS}.items():
            if key not in observation:
                continue
            image = observation.get(key)
            if (
                not isinstance(image, np.ndarray)
                or image.dtype != np.uint8
                or image.shape != shape
            ):
                raise ValueError(f"{key} must be uint8{shape}")
        for key, shape in VECTOR_SPECS.items():
            vector = observation.get(key)
            if (
                not isinstance(vector, np.ndarray)
                or vector.dtype != np.float32
                or vector.shape != shape
                or not np.isfinite(vector).all()
            ):
                raise ValueError(f"{key} must be finite float32{shape}")
        if not isinstance(observation.get("prompt"), str):
            raise ValueError("prompt must be a string")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("ORIGAMI_ZENOH_ENDPOINT"),
    )
    parser.add_argument(
        "--session-id",
        default=os.environ.get("ORIGAMI_SESSION_ID"),
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=int(os.environ.get("ORIGAMI_ACTION_HORIZON", "25")),
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    if not args.endpoint:
        raise SystemExit("--endpoint or ORIGAMI_ZENOH_ENDPOINT is required")
    if not args.session_id:
        raise SystemExit("--session-id or ORIGAMI_SESSION_ID is required")
    policy = TeamPolicy(args.action_horizon)
    server = OrigamiZenohServer(
        policy,
        endpoint=args.endpoint,
        session_id=args.session_id,
        action_horizon=args.action_horizon,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raise SystemExit(main())