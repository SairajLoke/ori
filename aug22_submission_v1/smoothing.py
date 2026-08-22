"""Chunk-seam smoothing for the origami-zenoh-v1 action reply.

Consecutive infer() calls are independent rollouts that need not agree where one
chunk ends and the next begins, so the assembled trajectory can step
discontinuously at every seam even though each chunk is individually smooth.
Every mode here re-ties the new chunk to something continuous with what the
robot actually did.

robot_io_spec.md §6 lists history and temporal ensembling as participant-internal,
so all of this is contract-legal: the reply stays finite float32[T,65] absolute
radians, and the caller clears prev_chunk on reset().

Pure numpy and side-effect free so the offline evaluator exercises the same code
the server ships.
"""

from __future__ import annotations

import numpy as np

MODES = ('none', 'anchor', 'ensemble', 'clamp', 'anchor_clamp', 'ensemble_clamp', 'auto')


def rate_limit(chunk: np.ndarray, state: np.ndarray, max_step_rad: float) -> np.ndarray:
    """Cap every per-step joint delta, walking forward from the measured pose.
    Directly targets the evaluator's adjacent-step jump/velocity check."""
    out = np.empty_like(chunk)
    prev = state
    for i, row in enumerate(chunk):
        prev = prev + np.clip(row - prev, -max_step_rad, max_step_rad)
        out[i] = prev
    return out


def anchor(chunk: np.ndarray, state: np.ndarray, blend_steps: int) -> np.ndarray:
    """Re-tie the chunk to the measured pose without stalling the motion.

    chunk[0] is the model's estimate of *now*, so the first row we can actually
    command is chunk[1]; the whole plan is advanced one step (last row held).
    The residual offset (state - chunk[0]) is then added back and decayed to
    zero over blend_steps, which removes the seam discontinuity.

    Row 0 therefore lands on state + (chunk[1]-chunk[0]): the measured pose plus
    the model's own first-step velocity. Emitting `state` itself instead would
    be a zero-jump but zero-progress command -- and under the Shadow rule
    (next state := final row of the previous chunk) a stride-1 caller would feed
    that same pose straight back, freezing the robot.
    """
    k = min(blend_steps, len(chunk))
    if k <= 0:
        return chunk
    out = np.vstack([chunk[1:], chunk[-1:]])          # advance one step, hold last
    w = np.linspace(1.0, 0.0, k + 1)[:-1, None]
    out[:k] += (state - chunk[0])[None, :] * w
    return out


def _overlap(chunk: np.ndarray, state: np.ndarray,
             prev_chunk: np.ndarray | None) -> tuple[int, int]:
    """(start, count) of the previous chunk's rows that still overlap this one.
    count == 0 means the caller consumed the whole previous chunk, leaving
    nothing to average against."""
    if prev_chunk is None:
        return 0, 0
    s = int(np.argmin(np.linalg.norm(prev_chunk - state[None, :], axis=1))) + 1
    return s, max(0, min(len(prev_chunk) - s, len(chunk)))


def ensemble(chunk: np.ndarray, state: np.ndarray, prev_chunk: np.ndarray | None,
             decay: float) -> np.ndarray:
    """ACT-style overlap averaging. We are never told how many rows were executed
    (participant_zenoh_submission.md: "may consume only a prefix"), so recover it:
    the previous chunk's row closest to the measured pose is where the robot got to.

    That row is *now*, so the old plan's prediction for the next timestep is
    prev[s+1] -- aligning new row 0 against prev[s] instead would blend a
    "stay here" command in, and since w[0]==0 it would emit exactly `state`,
    freezing a stride-1 caller under the Shadow state-update rule.
    """
    s, n = _overlap(chunk, state, prev_chunk)
    if n <= 0:
        return chunk
    w = (1.0 - np.exp(-decay * np.arange(n)))[:, None]
    out = chunk.copy()
    out[:n] = w * out[:n] + (1.0 - w) * prev_chunk[s:s + n]
    return out


def smooth_chunk(chunk: np.ndarray, state: np.ndarray, prev_chunk: np.ndarray | None,
                 mode: str, blend_steps: int = 4, ensemble_decay: float = 0.35,
                 max_step_rad: float = 0.10) -> np.ndarray:
    """chunk [T,65] raw model output, state [65] authoritative measured pose."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    out = chunk
    if mode == 'auto':
        # Cadence is organizer-controlled and unknown, and no fixed mode wins at
        # every consumption depth: ensemble needs leftover overlap, which does not
        # exist once a caller consumes the whole horizon. Use it when there is
        # overlap to average, fall back to anchor when there is not.
        if _overlap(chunk, state, prev_chunk)[1] > 0:
            out = ensemble(out, state, prev_chunk, ensemble_decay)
        else:
            out = anchor(out, state, blend_steps)
        return rate_limit(out, state, max_step_rad)
    if mode in ('anchor', 'anchor_clamp'):
        out = anchor(out, state, blend_steps)
    elif mode in ('ensemble', 'ensemble_clamp'):
        out = ensemble(out, state, prev_chunk, ensemble_decay)
    if mode in ('clamp', 'anchor_clamp', 'ensemble_clamp'):
        out = rate_limit(out, state, max_step_rad)
    return out
