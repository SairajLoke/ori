"""
lerobot_normalizer.py

A real, working OriNormalizer for the origami pipeline.

STATUS: NOT currently wired into origami_imitate_episodes.py.
    - The current normalizer in that script is built with every feature's
      mode set to None, and the normalize() calls in origami_forward_pass()
      are commented out — qpos/action/tactile are passed to the policy raw.
    - This file exists so that when you DO want normalization, you don't
      have to re-derive which mode fits which feature from stats.json again.
      Just swap the import and flip the modes on (see "HOW TO ENABLE" below).

Design is based on inspecting your actual stats.json (14 episodes /
91338 frames, robot_type="north_ces"):

    feature                          -> recommended mode  -> why
    -----------------------------------------------------------------------
    observation.state                -> "quantile"          per-dim std spans
                                                              1e-6 .. 0.42, some
                                                              joints ~frozen ->
                                                              mean/std would
                                                              blow those up
    action                           -> "quantile"           same as state
    observation.state.joint_torque   -> "quantile"           heavy-tailed,
                                                              contact spikes
    observation.tactile              -> "quantile"           cross-axis scale
                                                              mismatch (force
                                                              vs torque axes)
                                                              + contact spikes
    observation.tactile_next         -> "quantile"           same signal as
                                                              observation.tactile,
                                                              just next-frame ->
                                                              aliased to that
                                                              key's stats
    observation.state.tcp            -> EXCLUDED             all-zero in stats
                                                              (mean=std=min=max=0)
                                                              -> feature is
                                                              unpopulated in
                                                              this dataset;
                                                              normalizing it
                                                              divides by ~0
    observation.images.*             -> None (identity)      already float
                                                              [0,1]; let the
                                                              vision backbone
                                                              do its own
                                                              preprocessing
    observation.images.tactile_raw/
    observation.images.tactile_deform-> None (identity)      image-shaped
                                                              tactile maps, not
                                                              vector features
    timestamp/frame_index/episode_index/
    index/task_index                 -> NOT A MODEL INPUT     bookkeeping only,
                                                              never pass to
                                                              normalizer

HOW TO ENABLE (when you're ready):

    from lerobot_normalizer import OriNormalizer, recommended_modes

    normalizer = OriNormalizer(
        train_dataset.meta.stats,
        recommended_modes(use_tactile=use_tactile),
        device=device,
    )

    # in origami_forward_pass():
    qpos_data_norm    = normalizer.normalize("observation.state", qpos_data)
    action_data_norm  = normalizer.normalize("action", action_data)
    tactile_norm      = normalizer.normalize("observation.tactile", tactile)
    tactile_next_norm = normalizer.normalize("observation.tactile_next", tactile_next)

    # and to convert a predicted action chunk back to physical units, e.g.
    # for eval / visualization:
    action_pred = normalizer.denormalize("action", action_pred_norm)

Nothing here talks to the dataloader or the training loop directly — this
module only does tensor <-> tensor normalization given a `stats` dict in
OriDataset's `meta.stats` shape (mean/std/min/max/q01/q99/... per
feature, as lists/np arrays).
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, Optional, Union

import numpy as np
import torch

EPS = 1e-6

# Minimum spread a dimension must have before we are willing to divide by it.
#
# Measured on this dataset (meta/stats.json):
#   action[58]           q99-q01 = 1.3e-05   std = 1.0e-06   -> effectively constant
#   action[59]           q99-q01 = 2.9e-05   std = 1.6e-06   -> effectively constant
#   observation.state[58] q99-q01 = 5.3e-04
#   observation.state[59] q99-q01 = 3.3e-04
#   observation.tactile   14 dims below 1e-2, 4 of them below 1e-3
#
# Dividing by 1.3e-05 multiplies sensor noise by ~150,000x. The old workaround
# was to copy raw values back over action[:, :, 58] and [:, :, 59] in
# convert_batch -- but observation.state[58]/[59] got no such treatment, so the
# SAME physical quantity reached the policy normalized on the input side and
# unnormalized on the output side. Clamping the denominator here fixes both
# sides identically and needs no per-dim hardcoding.
#
# 1e-3 is chosen to sit above state[58]/[59] (5.3e-4 / 3.3e-4) and well below
# the median spread (~0.53 for state/action, ~0.12 for tactile), so genuinely
# informative low-range dims -- e.g. tactile[16], spread 8.2e-3, std 2.2e-3,
# which normalizes to a healthy std of ~0.53 -- are left alone.
DEGENERATE_SPREAD = 1e-3

# ---------------------------------------------------------------------------
# Shared-scale groups
# ---------------------------------------------------------------------------
# observation.tactile is 60 dims = 10 sensors x [fx, fy, fz, tx, ty, tz]
# (left/right x thumb, index, middle, ring, little -- confirmed from
# meta/info.json "names").
#
# Normalizing each of those 60 dims by ITS OWN q01/q99 is wrong for this signal.
# Measured on the real frames (91338 of them):
#
#   sensor         mean(q99-q01)   raw |max|   max |normalized|
#   left_thumb          8.56          30.0            3.1
#   left_index          6.15          34.6            3.9
#   left_ring           0.011          0.78         135.0   <-- idle finger
#   left_little         0.214         13.6           63.3   <-- mostly idle
#
# The thumb and index fingers do the work, so their percentile band describes a
# real operating range. The ring and little fingers are idle in most episodes,
# so THEIR band describes sensor noise -- per-dim scaling blows that noise up to
# unit variance, and then a genuine press reads as 135. The 135 is not a glitch:
# it is a smooth 10-frame ramp to -0.78 N in episode 4, i.e. a real contact seen
# through a scale calibrated on an idle sensor.
#
# Sharing one scale across all 10 sensors per axis type fixes both directions at
# once: idle-finger noise stays small, real contacts stay on-scale, and the
# relative loading between fingers (a thumb press really is ~100x a ring press)
# is preserved. Measured: max |normalized| drops 135 -> 3.2 with ZERO values
# above 10, while the working fingers keep the same resolution (std 0.66 vs
# 0.68). The 6.9-second sustained left_little contact in episode 4 -- real task
# signal that a clip would have flattened -- peaks at 2.2 and survives intact.
#
# Group ids are per-dim: dims sharing an id share one scale. For tactile that is
# `i % 6`, i.e. group by axis type.
_DEFAULT_STAT_GROUPS = {
    "observation.tactile": lambda d: [i % 6 for i in range(d)],
    "observation.tactile_next": lambda d: [i % 6 for i in range(d)],
}

# Post-normalization clipping, per feature key. With grouped scaling nothing
# reaches these limits (max |normalized| is ~3.2 for tactile, ~6.1 for
# state/action), so this is a safety net against an unseen season with a wilder
# tail rather than something the current data needs.
# Override with ORI_NORM_CLIP (float, 0 disables) or clip={...}.
_DEFAULT_CLIP = {
    "observation.tactile": 25.0,
    "observation.tactile_next": 25.0,
}

# Modes supported per feature. `None` / "identity" = pass-through.
_VALID_MODES = {None, "identity", "mean_std", "gaussian", "quantile", "min_max"}

# Keys that are pure bookkeeping / indexing — never valid normalizer targets.
_NON_MODEL_KEYS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}

# Keys whose stats.json entry is known-degenerate (all-zero) for this dataset.
# Attempting to normalize these will divide by ~0 and blow up. We refuse and
# force identity instead, with a loud warning, rather than fail silently.
_KNOWN_DEGENERATE_KEYS = {"observation.state.tcp"}

# Keys that don't exist in stats.json but should reuse another key's stats
# because they're the same underlying signal (e.g. a "next frame" version).
_DEFAULT_KEY_ALIASES = {
    "observation.tactile_next": "observation.tactile",
}


def recommended_modes(use_tactile: bool = True) -> Dict[str, Optional[str]]:
    """
    Returns the feature -> mode mapping recommended for this dataset,
    based on the distributional analysis of stats.json (see module docstring).

    Pass this straight into OriNormalizer(..., feature_modes=...) once
    you're ready to turn normalization on.
    """
    modes: Dict[str, Optional[str]] = {
        "observation.state": "quantile",
        "action": "quantile",
        "observation.images.head_left": None,
        "observation.images.head_right": None,
        "observation.images.wrist_left": None,
        "observation.images.wrist_right": None,

        "observation.state.joint_torque": "quantile", ############
    }
    if use_tactile:
        modes["observation.tactile"] = "quantile"
        modes["observation.tactile_next"] = "quantile"
    # observation.state.tcp intentionally omitted: all-zero in stats.json,
    # exclude it from the model's inputs entirely rather than normalize it.
    # observation.state.joint_torque intentionally omitted: only include it
    # if you actually feed torque into the policy; add
    # modes["observation.state.joint_torque"] = "quantile" yourself if so.
    return modes


def _to_tensor(x, device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


class _FeatureStats:
    """Holds one feature's stats as flat 1-D torch tensors, ready to broadcast."""

    __slots__ = ("mean", "std", "min", "max", "q01", "q99", "dim",
                 "degenerate_dims", "shared_groups")

    def __init__(self, raw_stats: dict, device):
        def flat(key):
            if key not in raw_stats or raw_stats[key] is None:
                return None
            arr = np.asarray(raw_stats[key], dtype=np.float32)
            return torch.as_tensor(arr.reshape(-1), device=device)

        self.mean = flat("mean")
        self.std = flat("std")
        self.min = flat("min")
        self.max = flat("max")
        self.q01 = flat("q01")
        self.q99 = flat("q99")
        # dim = length of the flattened feature vector (e.g. 65 for
        # observation.state, or 3 for an image's per-channel stats).
        self.dim = self.mean.numel() if self.mean is not None else None
        self.degenerate_dims = []
        self.shared_groups = None

    def share_scale(self, group_ids):
        """Collapse q01/q99 (and min/max) so every dim in a group shares a scale.

        Uses min(q01) / max(q99) over each group. That is computable from
        stats.json alone -- no raw frames needed -- and was verified against the
        true pooled 1st/99th percentile over all 91338 frames: it gives a
        slightly WIDER (more conservative) band, max |normalized| 3.2 vs 5.1,
        with zero values above 10 either way. Aggregating with median() or
        mean() instead does NOT work (median gives max |normalized| 105).
        """
        if group_ids is None:
            return
        d = self.q01.numel() if self.q01 is not None else (
            self.min.numel() if self.min is not None else 0)
        if d == 0 or len(group_ids) != d:
            return
        for lo_name, hi_name in (("q01", "q99"), ("min", "max")):
            lo = getattr(self, lo_name)
            hi = getattr(self, hi_name)
            if lo is None or hi is None:
                continue
            lo_new, hi_new = lo.clone(), hi.clone()
            for g in sorted(set(group_ids)):
                cols = [i for i, gi in enumerate(group_ids) if gi == g]
                idx = torch.as_tensor(cols, device=lo.device, dtype=torch.long)
                lo_new[idx] = lo[idx].min()
                hi_new[idx] = hi[idx].max()
            setattr(self, lo_name, lo_new)
            setattr(self, hi_name, hi_new)

    def guard_spread(self, mode: str, threshold: float = DEGENERATE_SPREAD):
        """Neutralise dimensions whose spread is too small to divide by.

        For each such dim the scale is forced to 1 and the offset to the dim's
        own centre, so the dim passes through as (x - centre): unit scale, zero
        amplification, and still centred like its neighbours. Records the
        affected indices in `degenerate_dims` for logging.
        """
        if mode in ("mean_std", "gaussian"):
            bad = self.std < threshold
            if bool(bad.any()):
                self.degenerate_dims = torch.nonzero(bad).flatten().tolist()
                self.std = torch.where(bad, torch.ones_like(self.std), self.std)
        elif mode == "quantile":
            spread = self.q99 - self.q01
            bad = spread < threshold
            if bool(bad.any()):
                self.degenerate_dims = torch.nonzero(bad).flatten().tolist()
                # centre the pass-through interval on the dim's own value so
                # (x - q01)/1 * 2 - 1 does not push it far from the other dims
                self.q01 = torch.where(bad, self.q01 - 0.5, self.q01)
                self.q99 = torch.where(bad, self.q01 + 1.0, self.q99)
        elif mode == "min_max":
            spread = self.max - self.min
            bad = spread < threshold
            if bool(bad.any()):
                self.degenerate_dims = torch.nonzero(bad).flatten().tolist()
                self.min = torch.where(bad, self.min - 0.5, self.min)
                self.max = torch.where(bad, self.min + 1.0, self.max)


class OriNormalizer:
    """
    Per-feature normalizer built directly from a OriDataset's
    `dataset.meta.stats` dict (same shape as the stats.json format).

    Args:
        stats: dict like train_dataset.meta.stats — {feature_key: {"mean":...,
            "std":..., "min":..., "max":..., "q01":..., "q99":..., ...}}.
        feature_modes: dict {feature_key: mode}. mode is one of:
            - None / "identity"  -> pass-through, no-op (safe default)
            - "mean_std" / "gaussian" -> (x - mean) / (std + eps)
            - "quantile"          -> (x - q01) / (q99 - q01 + eps) * 2 - 1
            - "min_max"           -> (x - min) / (max - min + eps) * 2 - 1
        device: torch device stats tensors live on.
        key_aliases: optional override of which stats a feature key should
            borrow (default handles observation.tactile_next -> observation.tactile).
        strict: if True, raise instead of warn when a requested feature/mode
            can't be satisfied (e.g. quantile mode requested but stats.json
            has no q01/q99 for that key).
    """

    def __init__(
        self,
        stats: dict,
        feature_modes: Dict[str, Optional[str]],
        device: Union[str, torch.device] = "cpu",
        key_aliases: Optional[Dict[str, str]] = None,
        strict: bool = False,
        degenerate_spread: float = DEGENERATE_SPREAD,
        clip: Optional[Dict[str, float]] = None,
        stat_groups: Optional[Dict[str, object]] = None,
    ):
        self.device = torch.device(device)
        self.strict = strict
        self.degenerate_spread = degenerate_spread
        self.key_aliases = {**_DEFAULT_KEY_ALIASES, **(key_aliases or {})}
        self.stat_groups = ({**_DEFAULT_STAT_GROUPS, **(stat_groups or {})}
                            if stat_groups is not False else {})

        if clip is None:
            _env = os.environ.get("ORI_NORM_CLIP")
            if _env is None:
                clip = dict(_DEFAULT_CLIP)
            elif float(_env) <= 0:
                clip = {}
            else:
                clip = {k: float(_env) for k in _DEFAULT_CLIP}
        self.clip = dict(clip)

        bad_modes = {k: m for k, m in feature_modes.items() if m not in _VALID_MODES}
        if bad_modes:
            raise ValueError(f"Unknown normalization mode(s): {bad_modes}. Valid: {_VALID_MODES}")

        for k in feature_modes:
            if k in _NON_MODEL_KEYS:
                raise ValueError(
                    f"'{k}' is a bookkeeping/index field (timestamp, frame_index, "
                    f"episode_index, index, task_index), not a model input. "
                    f"Remove it from feature_modes."
                )

        # self.transforms mirrors the shape the training script already logs
        # via `normalizer.transforms` in info.log — keep that attribute name.
        self.transforms: Dict[str, Optional[str]] = dict(feature_modes)

        self._stats: Dict[str, _FeatureStats] = {}
        for key, mode in feature_modes.items():
            if mode in (None, "identity"):
                continue  # no stats needed for pass-through

            stats_key = self.key_aliases.get(key, key)
            if stats_key in _KNOWN_DEGENERATE_KEYS:
                self._warn_or_raise(
                    f"'{stats_key}' is a known-degenerate feature in this dataset "
                    f"(mean=std=min=max=0 in stats.json — looks unpopulated/broken "
                    f"upstream). Forcing mode to identity for '{key}' instead of "
                    f"'{mode}' to avoid dividing by ~0."
                )
                self.transforms[key] = None
                continue

            if stats_key not in stats:
                self._warn_or_raise(
                    f"Feature '{key}' (stats key '{stats_key}') requested mode "
                    f"'{mode}' but has no entry in the provided stats dict. "
                    f"Forcing identity for '{key}'."
                )
                self.transforms[key] = None
                continue

            fstats = _FeatureStats(stats[stats_key], self.device)

            if mode in ("mean_std", "gaussian") and (fstats.mean is None or fstats.std is None):
                self._warn_or_raise(f"'{stats_key}' missing mean/std for mode '{mode}'.")
                self.transforms[key] = None
                continue
            if mode == "quantile" and (fstats.q01 is None or fstats.q99 is None):
                self._warn_or_raise(
                    f"'{stats_key}' missing q01/q99 for mode 'quantile' "
                    f"(older stats.json without quantiles?). Forcing identity for '{key}'."
                )
                self.transforms[key] = None
                continue
            if mode == "min_max" and (fstats.min is None or fstats.max is None):
                self._warn_or_raise(f"'{stats_key}' missing min/max for mode 'min_max'.")
                self.transforms[key] = None
                continue

            # Degenerate-but-not-in-our-known-list guard: if std is ~0
            # everywhere for a mean_std feature, or q99==q01 everywhere for
            # a quantile feature, normalizing would blow up regardless of
            # whether we happened to hardcode it above.
            if mode in ("mean_std", "gaussian") and torch.all(fstats.std < EPS):
                self._warn_or_raise(
                    f"'{stats_key}' has ~zero std across all dims — normalizing "
                    f"would blow up. Forcing identity for '{key}'."
                )
                self.transforms[key] = None
                continue
            if mode == "quantile" and torch.all((fstats.q99 - fstats.q01) < EPS):
                self._warn_or_raise(
                    f"'{stats_key}' has ~zero (q99-q01) across all dims — "
                    f"normalizing would blow up. Forcing identity for '{key}'."
                )
                self.transforms[key] = None
                continue

            # Share one scale across grouped dims BEFORE the degenerate guard:
            # grouping is what stops idle sensors from being scaled by their own
            # noise floor, and it also lifts most of their spreads above the
            # degenerate threshold so fewer dims need pinning at all.
            group_ids = self.stat_groups.get(key)
            if callable(group_ids):
                _d = fstats.q01.numel() if fstats.q01 is not None else fstats.min.numel()
                group_ids = group_ids(_d)
            if group_ids is not None:
                fstats.share_scale(group_ids)
                fstats.shared_groups = len(set(group_ids))

            # Per-dimension guard: neutralise individual near-constant dims
            # instead of either amplifying them ~1e5x or disabling the whole
            # feature. See DEGENERATE_SPREAD for the measured numbers.
            fstats.guard_spread(mode, self.degenerate_spread)
            if fstats.degenerate_dims:
                warnings.warn(
                    f"[OriNormalizer] '{key}': dims {fstats.degenerate_dims} have spread < "
                    f"{self.degenerate_spread:g} and are passed through with unit scale "
                    f"(centred, not amplified)."
                )

            self._stats[key] = fstats

    def _warn_or_raise(self, msg: str):
        if self.strict:
            raise ValueError(msg)
        warnings.warn(f"[OriNormalizer] {msg}")

    # ------------------------------------------------------------------ #
    # core normalize / denormalize
    # ------------------------------------------------------------------ #

    def _mode(self, key: str) -> Optional[str]:
        if key not in self.transforms:
            raise KeyError(
                f"'{key}' was never registered with this normalizer. "
                f"Known keys: {list(self.transforms.keys())}"
            )
        return self.transforms[key]

    def _aligned(self, stat_vec: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Slice + reshape a flat (D,) stats vector so it broadcasts correctly
        against x's last dimension, regardless of how many leading
        (batch/time) dims x has. Mirrors the `stats[..., :x.shape[-1]]`
        pattern used by openpi's normalize.py, generalized to arbitrary
        leading dims via a reshape instead of relying on numpy's `...`.
        """
        d = x.shape[-1]
        if stat_vec.numel() < d:
            raise ValueError(
                f"Stats vector has {stat_vec.numel()} dims but input has "
                f"{d} — can't normalize (stats vector must be >= feature dim)."
            )
        sliced = stat_vec[:d]
        # reshape to (1, 1, ..., 1, d) matching x.dim()
        view_shape = (1,) * (x.dim() - 1) + (d,)
        return sliced.view(view_shape)

    def normalize(self, key: str, x: torch.Tensor) -> torch.Tensor:
        mode = self._mode(key)
        if mode in (None, "identity"):
            return x

        x = x.to(self.device)
        s = self._stats[key]

        if mode in ("mean_std", "gaussian"):
            mean = self._aligned(s.mean, x)
            std = self._aligned(s.std, x)
            out = (x - mean) / (std + EPS)
        elif mode == "quantile":
            q01 = self._aligned(s.q01, x)
            q99 = self._aligned(s.q99, x)
            out = (x - q01) / (q99 - q01 + EPS) * 2.0 - 1.0
        elif mode == "min_max":
            xmin = self._aligned(s.min, x)
            xmax = self._aligned(s.max, x)
            out = (x - xmin) / (xmax - xmin + EPS) * 2.0 - 1.0
        else:
            raise RuntimeError(f"unreachable: mode={mode}")

        # Tail clipping (inputs only -- denormalize() never clips, so a
        # round-trip of an in-range value is still exact).
        limit = self.clip.get(key)
        if limit:
            out = out.clamp(-limit, limit)
        return out

    def denormalize(self, key: str, x: torch.Tensor) -> torch.Tensor:
        """Inverse of normalize() — e.g. to turn a predicted action chunk
        back into physical joint-space units for eval/logging/deployment."""
        mode = self._mode(key)
        if mode in (None, "identity"):
            return x

        x = x.to(self.device)
        s = self._stats[key]

        if mode in ("mean_std", "gaussian"):
            mean = self._aligned(s.mean, x)
            std = self._aligned(s.std, x)
            return x * (std + EPS) + mean

        if mode == "quantile":
            q01 = self._aligned(s.q01, x)
            q99 = self._aligned(s.q99, x)
            return (x + 1.0) / 2.0 * (q99 - q01 + EPS) + q01

        if mode == "min_max":
            xmin = self._aligned(s.min, x)
            xmax = self._aligned(s.max, x)
            return (x + 1.0) / 2.0 * (xmax - xmin + EPS) + xmin

        raise RuntimeError(f"unreachable: mode={mode}")

    # ------------------------------------------------------------------ #
    # introspection / logging helpers
    # ------------------------------------------------------------------ #

    def describe(self) -> str:
        """Human-readable summary, handy for info.log (mirrors how the
        training script already logs `normalizer.transforms`)."""
        lines = ["OriNormalizer feature modes:"]
        for key, mode in self.transforms.items():
            stats_key = self.key_aliases.get(key, key)
            note = f" (stats: {stats_key})" if stats_key != key else ""
            clip = self.clip.get(key)
            note += f"  [clip +/-{clip:g}]" if clip else ""
            lines.append(f"  {key:40s} -> {mode!s:10s}{note}")
            fs = self._stats.get(key)
            if fs is not None and fs.shared_groups:
                lines.append(f"  {'':40s}    scale shared across {fs.shared_groups} group(s) "
                             f"(per-axis for tactile)")
            if fs is not None and fs.degenerate_dims:
                lines.append(f"  {'':40s}    degenerate dims (unit scale, spread < "
                             f"{self.degenerate_spread:g}): {fs.degenerate_dims}")
        return "\n".join(lines)

    def __repr__(self):
        return f"OriNormalizer(features={list(self.transforms.keys())})"
