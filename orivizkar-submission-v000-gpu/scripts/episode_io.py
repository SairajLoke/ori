"""Episode access with TRAINING-FAITHFUL history windows.

The analysis scripts (ablation, Grad-CAM, occlusion, CAM mass) all ask "what is
this model doing on a given frame", which only means something if the model is
fed the inputs it was trained on. base's DELTA_TIMESTAMPS asks for
observation.state at [-0.167 .. 0] and observation.tactile at [-0.6 .. 0]:
6 and 19 CONSECUTIVE 30 fps frames.

Rebuilding history from a deque that appends once per sampled frame -- what
TeamPolicy does, because at deployment that is all it gets -- spans
stride/30 * 6 seconds instead. At stride 300 that is a 50-second "0.167 second"
window, and every input goes off-distribution.

That deque behaviour is correct for eval_smoothing.py, which is *simulating*
deployment at a given query cadence. It is wrong for these analyses, which want
the model in its trained regime. Hence this module.

state/tactile/action come from the parquet columns (no video decode), so a full
history window costs nothing; only the sampled frame's images are decoded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

TACTILE_H = 18


def open_episode(dataset_root: Path, episode: int, video_backend: str = "pyav"):
    """-> (images_fn, window_fn, n_frames, actions[n,65])

    images_fn(i, camera_names) -> {cam: CHW tensor}   (decodes)
    window_fn(key, i, n)       -> [n, D] of the n consecutive frames ending at i
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=None, root=dataset_root, episodes=[episode],
                        video_backend=video_backend, tolerance_s=0.02)
    cols = {k: np.asarray(ds.hf_dataset[k], dtype=np.float32)
            for k in ("action", "observation.state", "observation.tactile")}
    n = len(ds)

    def images(i, camera_names):
        s = ds[i]
        return {c: s[c] for c in camera_names}

    def window(key, i, count):
        # Clamped at the episode start, replicating frame 0 -- the same cold-start
        # backfill TeamPolicy does on its first infer() of an episode.
        idx = np.clip(np.arange(i - count + 1, i + 1), 0, n - 1)
        return cols[key][idx]

    return images, window, n, cols["action"]


def frame_inputs(i, cams, pc, normalizer, device, images_fn, window_fn, img_preproc):
    """Normalized model inputs for frame i, with training-faithful history.

    Returns (img[4,3,H,W], state[T1,65], tac19[19,60]) all normalized; callers
    flatten/diff them as the model expects.
    """
    raw = images_fn(i, cams)
    img = torch.stack([img_preproc(raw[c])[0] for c in cams]).to(device)

    state = torch.from_numpy(
        window_fn("observation.state", i, pc["proprioceptive_temporal_horizon"])).to(device)
    if normalizer is not None:
        state = normalizer.normalize("observation.state", state)

    tac19 = torch.from_numpy(window_fn("observation.tactile", i, TACTILE_H + 1)).to(device)
    if normalizer is not None:
        tac19 = normalizer.normalize("observation.tactile", tac19)

    return img, state, tac19


def to_model_inputs(img, state, tac19, device, batch=1):
    """(img,state,tac19) -> the exact tensors DETRVAE.forward wants."""
    qpos = state.reshape(1, -1)
    tac = torch.cat([tac19[1:], torch.diff(tac19, dim=0)], dim=-1).reshape(1, -1)
    tac_next = torch.zeros((1, TACTILE_H, 120), device=device)
    im = img.unsqueeze(0)
    if batch > 1:
        qpos, tac = qpos.repeat(batch, 1), tac.repeat(batch, 1)
        tac_next, im = tac_next.repeat(batch, 1, 1), im.repeat(batch, 1, 1, 1, 1)
    return qpos, im, tac, tac_next
