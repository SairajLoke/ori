

import json
import logging
import os

import torch
from enum import Enum

import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot.datasets.video_utils import _default_decoder_cache
from configs import (TACTILE_TEMPORAL_HORIZON, MAX_EPISODES, IMAGE_HW,
                      PROPRIOCEPTIVE_TEMPORAL_HORIZON, JITTER_HISTORY,
                      JITTER_MAX_GAP_FRAMES, JITTER_BASE_GAP_FRAMES,
                      TACTILE_PAST_POOL_LEN)

import time

from my_utils.ori_logging import get_logger, log_tensor, log_tensors, TRACE, StepGate

log = get_logger("data")
# Structural per-batch logging is throttled: first 3 batches of every dataloader
# iterator, then one in every 500. Tensor stats (TRACE) are gated on top of that.
_convert_gate = StepGate(first_n=3, every=500)

from torchvision.transforms import v2
torchvision_transforms = v2.Compose([
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
])

# ImageNet statistics. The backbone is an ImageNet-pretrained ResNet, so its
# first conv expects inputs standardised with these -- upstream ViTacFormer does
# exactly this in dataset/pipelines_v2/transform.py::ImageProcess
# (`cur_img = cur_img / 255.0; cur_img = (cur_img - self.mean) / self.std`,
# configured from data_tactile.py with these same constants). The LeRobot
# origami path skipped it and fed raw [0,1], which is why the old logs show
# image ranges of [0,1] where the original pipeline shows [-2.12, 2.64].
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Set ORI_IMAGE_NORM=0 to reproduce the old un-normalized behaviour.
USE_IMAGENET_NORM = os.environ.get("ORI_IMAGE_NORM", "1") not in ("0", "false", "False")

_imagenet_mean_t = None
_imagenet_std_t = None


def _apply_imagenet_norm(img):
    """Standardise a [B, 3, H, W] float tensor in [0,1] with ImageNet stats."""
    global _imagenet_mean_t, _imagenet_std_t
    if _imagenet_mean_t is None or _imagenet_mean_t.device != img.device or _imagenet_mean_t.dtype != img.dtype:
        _imagenet_mean_t = torch.tensor(IMAGENET_MEAN, device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
        _imagenet_std_t = torch.tensor(IMAGENET_STD, device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
    return (img - _imagenet_mean_t) / _imagenet_std_t
#can also create custom trasnsforms : https://huggingface.co/docs/lerobot/lerobot-dataset-v3

def _jitter_gather(pool, n_out, max_gap_frames):
    """
    Randomly subsample a dense native-FPS pool into an n_out-length window,
    with per-step gaps drawn i.i.d. (independently per batch row) in
    [1, max_gap_frames] native frames. Used to emulate the irregular history
    spacing real inference cadence would produce -- see configs.JITTER_HISTORY.

    pool: [B, L, D], pool[:, -1, :] must be "now" (offset 0). L must be >=
          (n_out-1)*max_gap_frames + 1, which is exactly how configs.py sizes
          STATE_POOL_LEN / TACTILE_PAST_POOL_LEN.

    Returns:
        gathered: [B, n_out, D], gathered[:, -1, :] == pool[:, -1, :] ("now").
        gaps: [B, n_out-1] int64, the realized native-frame gap between each
              consecutive pair of gathered steps, in the same order as
              torch.diff(gathered, dim=1) -- gaps[:, k] is the gap between
              gathered[:, k, :] and gathered[:, k+1, :].
    """
    B, L, D = pool.shape
    n_gaps = n_out - 1
    gaps = torch.randint(1, max_gap_frames + 1, (B, n_gaps), device=pool.device)
    # idx_back[:, k] = native frames from gathered step k back to "now".
    # idx_back[:, k] = sum(gaps[:, k:]); idx_back[:, -1] = 0 ("now" itself).
    cum_from_now = torch.cumsum(gaps.flip(-1), dim=-1).flip(-1)
    idx_back = torch.cat([cum_from_now, torch.zeros((B, 1), dtype=gaps.dtype, device=pool.device)], dim=-1)
    idx = (L - 1 - idx_back).clamp(min=0)
    gathered = torch.gather(pool, 1, idx.unsqueeze(-1).expand(-1, -1, D).long())
    return gathered, gaps


def log_before_after(name, data_raw, data_norm):
    """Log shape, device, dtype, range before/after normalization."""
    print(f"\n[{name}]")
    print(f"  Before: shape={data_raw.shape}, device={data_raw.device}, dtype={data_raw.dtype}")
    # Convert to float for stats if uint8 (return_uint8=True path)
    _raw_for_stats = data_raw.float() / 255.0 if data_raw.dtype == torch.uint8 else data_raw
    print(f"    Range: [{_raw_for_stats.min():.4f}, {_raw_for_stats.max():.4f}], μ={_raw_for_stats.mean():.4f}, σ={_raw_for_stats.std():.4f}")
    print(f"  After:  shape={data_norm.shape}, device={data_norm.device}, dtype={data_norm.dtype}")
    print(f"    Range: [{data_norm.min():.4f}, {data_norm.max():.4f}], μ={data_norm.mean():.4f}, σ={data_norm.std():.4f}")


def convert_batch(batch, use_tactile, delta_timestamps, epoch=0, batch_idx=0, normalizer=None,
                  predict_deltas=False, camera_names=None, image_crop=None,
                  training=True, qpos_mask_prob=0.0, qpos_mask_mode="fixed",
                  qpos_static_velocity_threshold=0.01):

    """
    Convert a LeRobot batch into the format expected by the old ACT code.

    qpos_mask_*: training-only proprioception dropout, to reduce state->action
    shortcutting. "static_adaptive" masks harder when qpos barely moved (where
    copying it forward would trivially minimize loss). training=False disables
    masking regardless of qpos_mask_prob.

    Returns:
        output: dict with keys "image", "lowdim", "action", "action_mask", "qpos_mask",
                and optionally "tactile", "tactile_next".
        output["_timing"]: dict of per-sub-step timings in seconds.
    """

    timing = {}
    _t0 = time.time()
    if camera_names is None:
        from configs import CAMERA_NAMES as camera_names

    # `_verbose` gates the *structural* (DEBUG) logging. Tensor statistics are
    # additionally gated on the TRACE level, since they force CUDA syncs.
    _verbose = (((epoch == 0 and batch_idx < 3) or _convert_gate())
                and log.isEnabledFor(logging.DEBUG))

    if _verbose:
        log.debug("convert_batch: epoch=%d batch_idx=%d use_tactile=%s", epoch, batch_idx, use_tactile)
        log.debug("  raw batch keys: %s", sorted(batch.keys()))
        # LeRobot emits <key>_is_pad for every key in delta_timestamps. These
        # currently go unused downstream -- log them so the gap is visible.
        _pad_keys = [k for k in batch if k.endswith("_is_pad")]
        if _pad_keys:
            for k in _pad_keys:
                _n = int(batch[k].sum().item())
                log.debug("  %-34s padded entries=%d / %d", k, _n, batch[k].numel())
        log_tensors(log, TRACE, "  raw/", batch)



    # # Normalize  ------------------------------------
    # lowdim        = normalizer.normalize("observation.state", batch["observation.state"]) if normalizer else batch["observation.state"]
    # joint_torque  = normalizer.normalize("observation.state.joint_torque", batch["observation.state.joint_torque"]) if normalizer else batch["observation.state.joint_torque"]

    # action        = normalizer.normalize("action", batch["action"]) if normalizer else batch["action"]

    # imgHeadLeft   = normalizer.normalize("observation.images.head_left", batch["observation.images.head_left"]) if normalizer else batch["observation.images.head_left"]
    # imgHeadRight  = normalizer.normalize("observation.images.head_right", batch["observation.images.head_right"]) if normalizer else batch["observation.images.head_right"]
    # imgWristRight = normalizer.normalize("observation.images.wrist_right", batch["observation.images.wrist_right"]) if normalizer else batch["observation.images.wrist_right"]
    # imgWristLeft  = normalizer.normalize("observation.images.wrist_left", batch["observation.images.wrist_left"]) if normalizer else batch["observation.images.wrist_left"]

    # tactile_raw 
    # tactile_deformation 
    # -------------------------------------------------
    # ============ NORMALIZE + LOG ============
    def log_before_after(name, data_raw, data_norm):
        """TRACE the shape/device/dtype/range of a feature either side of the
        normalizer. Skipped entirely unless ori.data is at TRACE."""
        if not log.isEnabledFor(TRACE):
            return
        # uint8 (return_uint8=True path) is rescaled so before/after are comparable
        _raw = data_raw.float() / 255.0 if data_raw.dtype == torch.uint8 else data_raw
        log_tensor(log, TRACE, f"  norm/{name}  before", _raw, stats=True)
        log_tensor(log, TRACE, f"  norm/{name}  after ", data_norm, stats=True)

    # ── Timing: normalization ──
    _t_norm_start = time.time()

    # Observation state
    lowdim_raw = batch["observation.state"]
    lowdim = normalizer.normalize("observation.state", lowdim_raw) if normalizer else lowdim_raw
    if _verbose:
        log_before_after("observation.state (lowdim)", lowdim_raw, lowdim)

    if JITTER_HISTORY:
        _pool_shape = tuple(lowdim.shape)
        lowdim, _state_gaps = _jitter_gather(lowdim, PROPRIOCEPTIVE_TEMPORAL_HORIZON, JITTER_MAX_GAP_FRAMES)
        # state carries no derived/scale-sensitive quantity (no diff is taken
        # of it), so jitter only changes WHICH raw pool samples are selected --
        # no delta rescaling needed here, unlike the tactile case below.
        if _verbose:
            log.debug("  jitter/observation.state: pool %s -> gathered %s, "
                      "row0 gaps(frames)=%s", _pool_shape, tuple(lowdim.shape),
                      _state_gaps[0].tolist())

    B_lowdim = lowdim.shape[0]
    qpos_mask = torch.zeros(B_lowdim, dtype=torch.bool, device=lowdim.device)
    if training and qpos_mask_prob > 0:
        if qpos_mask_mode == "fixed":
            qpos_mask = torch.rand(B_lowdim, device=lowdim.device) < qpos_mask_prob
        elif qpos_mask_mode == "static_adaptive":
            velocity = (lowdim[:, 1:, :] - lowdim[:, :-1, :]).norm(dim=-1).mean(dim=-1)
            is_static = velocity < qpos_static_velocity_threshold
            qpos_mask = is_static & (torch.rand(B_lowdim, device=lowdim.device) < qpos_mask_prob)
        else:
            raise ValueError(f"unknown qpos_mask_mode: {qpos_mask_mode!r}")
        if _verbose:
            log.debug("  qpos_mask: %d/%d masked (mode=%s)",
                      int(qpos_mask.sum().item()), B_lowdim, qpos_mask_mode)

    # Action. With predict_deltas the target is the residual against the CURRENT
    # pose -- observation.state at offset 0, which is the LAST entry of the
    # ascending past window (_state_past_offsets ends at -0.0). Computed in raw
    # radians, then normalized with action_delta's own stats: action and
    # observation.state have separate quantile stats, so the subtraction is only
    # meaningful before normalization.
    action_raw = batch["action"]
    if predict_deltas:
        _cur = batch["observation.state"][:, -1:, :]                 # [B,1,65] raw
        action_raw = action_raw - _cur
        action = normalizer.normalize("action_delta", action_raw) if normalizer else action_raw
    else:
        action = normalizer.normalize("action", action_raw) if normalizer else action_raw
    # NOTE: action dims 58/59 used to be copied back raw here to dodge their
    # tiny q99-q01 (1.3e-5 / 2.9e-5). observation.state[58]/[59] got no such
    # treatment, so the same physical quantity arrived normalized on the input
    # side and unnormalized on the output side. OriNormalizer now clamps
    # degenerate spreads for every feature symmetrically -- see
    # my_utils/normalizer.py:DEGENERATE_SPREAD.

    if _verbose:
        log_before_after("action", action_raw, action)

    # observation.state.joint_torque is NOT a model input. It used to be
    # normalized, moved to GPU and mutated here every single batch, then
    # dropped -- pure overhead on the critical path. Trace it if you want to
    # inspect it, but do no work by default.
    if _verbose and "observation.state.joint_torque" in batch:
        log_tensor(log, TRACE, "  raw/joint_torque (unused by the model)",
                   batch["observation.state.joint_torque"])

    # Images. The camera list is driven by camera_names (which the training run
    # records in policy_config) rather than four hardcoded locals, so --cameras
    # can select a subset without this function knowing which.
    cams = []
    for _cam in camera_names:
        _raw = batch[_cam]
        if _raw.dtype == torch.uint8:
            _raw = _raw.float() / 255.0
        _img = normalizer.normalize(_cam, _raw) if normalizer else _raw
        if _verbose:
            log_before_after(_cam, _raw, _img)
        cams.append(_img)

    timing['norm'] = time.time() - _t_norm_start

    # ── Timing: image resize ──
    _t_resize_start = time.time()

    resized = []
    for img in cams:
        # Images already converted to float32 [0,1] above (uint8→float32 conversion)
        img = F.interpolate(
            img,
            size=IMAGE_HW,
            mode="bilinear",
            align_corners=False,
        )
        # Resize BEFORE standardising so the interpolation happens in [0,1]
        # space, matching the upstream ImageProcess ordering.
        if USE_IMAGENET_NORM:
            img = _apply_imagenet_norm(img)
        if image_crop:
            # explicit centre CROP, not a resize -- the point is to drop border
            # pixels (and the tokens they produce), which a resize would keep.
            _h, _w = img.shape[-2:]
            _t, _l = (_h - image_crop) // 2, (_w - image_crop) // 2
            img = img[..., _t:_t + image_crop, _l:_l + image_crop]
        resized.append(img)
    image = torch.stack(resized, dim=1)
    assert action.shape[0] == image.shape[0] and action.shape[1] == len(delta_timestamps["action"])

    timing['resize'] = time.time() - _t_resize_start

    if _verbose:
        log.debug("  cams %s -> resized %s -> imagenet_norm=%s -> stacked %s "
                  "(order: head_L, head_R, wrist_R, wrist_L)",
                  tuple(cams[0].shape), IMAGE_HW, USE_IMAGENET_NORM, tuple(image.shape))
        log_tensor(log, TRACE, "  image (model input)", image)

    B, T = action.shape[:2]
    # LeRobot clamps out-of-episode queries to the last real frame and reports
    # which entries it faked in "<key>_is_pad". Near an episode end that means
    # the final action is duplicated for up to CHUNK_SIZE steps -- training on
    # those as if they were real teaches the policy to freeze. Use the real mask.
    if "action_is_pad" in batch:
        action_mask = batch["action_is_pad"][:, :T].to(torch.bool)
    else:
        action_mask = torch.zeros((B, T), dtype=torch.bool, device=action.device)
        log.warning("batch has no 'action_is_pad' -- falling back to an all-False action mask")
    if _verbose:
        _npad = int(action_mask.sum().item())
        log.debug("  action_mask from action_is_pad [%d,%d]: %d/%d padded steps (%.2f%%)",
                  B, T, _npad, action_mask.numel(), 100.0 * _npad / max(action_mask.numel(), 1))

    output = {
        "image": image,
        "lowdim": lowdim,
        "action": action,
        "action_mask": action_mask,
        "qpos_mask": qpos_mask,
    }

    if use_tactile:
        _t_tac_start = time.time()
        # print(batch["observation.tactile"].shape) #torch.Size([8, 19, 60])
        tactile_raw = batch["observation.tactile"]
        tactile_shape = tactile_raw.shape
        assert tactile_shape[0] == image.shape[0] and tactile_shape[1] == len(delta_timestamps["observation.tactile"]), \
            f"tactile shape {tactile_shape} mismatch"

        tactile_pastNfuture = normalizer.normalize("observation.tactile", batch["observation.tactile"]) if normalizer else batch["observation.tactile"]
         
        if _verbose:
            log_before_after("observation.tactile", tactile_raw, tactile_pastNfuture)

        if JITTER_HISTORY:
            # Past half is a dense pool (TACTILE_PAST_POOL_LEN, native-FPS
            # spaced, ending at "now"/offset 0); future half is unchanged --
            # exactly TACTILE_TEMPORAL_HORIZON entries at offsets +1..+18,
            # since the prediction target is never jittered.
            past_pool = tactile_pastNfuture[:, :TACTILE_PAST_POOL_LEN, :]
            future = tactile_pastNfuture[:, TACTILE_PAST_POOL_LEN:, :]

            tactile_past, _tac_gaps = _jitter_gather(past_pool, TACTILE_TEMPORAL_HORIZON + 1, JITTER_MAX_GAP_FRAMES)
            # Raw diff / realized-gap-in-frames gives a per-frame rate; scale
            # back up by JITTER_BASE_GAP_FRAMES so the delta's units stay
            # "change per JITTER_BASE_GAP_FRAMES native frames" regardless of
            # which gap was actually drawn -- i.e. the same units the
            # unjittered delta (and the normalizer's tactile stats) already
            # use, rather than introducing a differently-scaled quantity.
            _raw_tac_deltas = torch.diff(tactile_past, dim=1)
            tactile_deltas = _raw_tac_deltas * (JITTER_BASE_GAP_FRAMES / _tac_gaps.unsqueeze(-1).to(_raw_tac_deltas.dtype))
            output["tactile"] = torch.concat((tactile_past[:, 1:, :], tactile_deltas), dim=-1)  # [B, 18, 120]

            # "now" (== past_pool[:, -1, :] == tactile_past[:, -1, :]) prepended
            # to the fixed future half reconstructs the exact unjittered
            # tactile_next window (offsets 0..+18) the non-jitter path builds.
            tactile_next = torch.cat((tactile_past[:, -1:, :], future), dim=1)  # [B, 19, 60]
            tactile_next_deltas = torch.diff(tactile_next, dim=1)
            output["tactile_next"] = torch.concat((tactile_next[:, 1:, :], tactile_next_deltas), dim=-1)  # [B, 18, 120]

            if "observation.tactile_is_pad" in batch:
                _tac_pad = batch["observation.tactile_is_pad"].to(torch.bool)              # [B, TACTILE_PAST_POOL_LEN+18]
                _next_pad = _tac_pad[:, TACTILE_PAST_POOL_LEN - 1:]                        # [B, 19] offsets 0..+18
                output["tactile_next_mask"] = _next_pad[:, 1:] | _next_pad[:, :-1]         # [B, 18]
            else:
                output["tactile_next_mask"] = torch.zeros(
                    output["tactile_next"].shape[:2], dtype=torch.bool, device=tactile_next.device)
                log.warning("batch has no 'observation.tactile_is_pad' -- tactile target mask is all-False")

            if _verbose:
                log.debug("  jitter/observation.tactile: past_pool %s -> gathered %s, "
                          "row0 gaps(frames)=%s, base_gap=%d",
                          tuple(past_pool.shape), tuple(tactile_past.shape),
                          _tac_gaps[0].tolist(), JITTER_BASE_GAP_FRAMES)
        else:
            tactile_past = tactile_pastNfuture[:, :TACTILE_TEMPORAL_HORIZON+1, :]  # -18, -17 ... 0
            tactile_deltas = torch.diff(tactile_past, dim=1)                             #idx#  0 ,  1,     18
            output["tactile"] = torch.concat((tactile_past[:, 1:, : ], tactile_deltas), dim=-1)    # [B, 18, 120]

            tactile_next = tactile_pastNfuture[:, TACTILE_TEMPORAL_HORIZON : , :]
            tactile_next_deltas = torch.diff(tactile_next, dim=1)
            output["tactile_next"] =  torch.concat((tactile_next[:, 1:, : ], tactile_next_deltas), dim=-1)    # [B, 18, 120]

            # Padding mask for the tactile *target*. Row j of tactile_next holds the
            # value at offset j+1 and the delta (j+1)-(j), so it is only a real
            # supervision target when BOTH endpoints are inside the episode.
            if "observation.tactile_is_pad" in batch:
                _tac_pad = batch["observation.tactile_is_pad"].to(torch.bool)          # [B, 37]
                _next_pad = _tac_pad[:, TACTILE_TEMPORAL_HORIZON:]                     # [B, 19] offsets 0..+18
                output["tactile_next_mask"] = _next_pad[:, 1:] | _next_pad[:, :-1]     # [B, 18]
            else:
                output["tactile_next_mask"] = torch.zeros(
                    output["tactile_next"].shape[:2], dtype=torch.bool, device=tactile_next.device)
                log.warning("batch has no 'observation.tactile_is_pad' -- tactile target mask is all-False")

        timing['tactile'] = time.time() - _t_tac_start

        if _verbose:
            # Spell out the actual time axis: delta_timestamps order is taken
            # verbatim by LeRobot, so this is what each slice really contains.
            _ts = delta_timestamps["observation.tactile"]
            _idx = [round(t * 30.0) for t in _ts]   # frame offsets, FPS=30
            if JITTER_HISTORY:
                log.debug("  tactile delta idx order: dense past pool[:%d] frame offsets %s..%s, "
                          "fixed future[%d:] offsets %s",
                          TACTILE_PAST_POOL_LEN, _idx[0], _idx[TACTILE_PAST_POOL_LEN - 1],
                          TACTILE_PAST_POOL_LEN, _idx[TACTILE_PAST_POOL_LEN:])
                log.debug("  tactile       = concat(gathered_past[1:], scaled diff) %s  (row0 gaps(frames)=%s)",
                          tuple(output["tactile"].shape), _tac_gaps[0].tolist())
                log.debug("  tactile_next  = concat(nxt[1:], diff(nxt)) %s  (unjittered, offsets 0..+%d)",
                          tuple(output["tactile_next"].shape), TACTILE_TEMPORAL_HORIZON)
            else:
                log.debug("  tactile delta idx order = %s", _idx)
                log.debug("  tactile_past  = idx[:%d] -> frame offsets %s",
                          TACTILE_TEMPORAL_HORIZON + 1, _idx[:TACTILE_TEMPORAL_HORIZON + 1])
                log.debug("  tactile       = concat(past[1:], diff(past)) %s   (past[1:] offsets %s)",
                          tuple(output["tactile"].shape), _idx[1:TACTILE_TEMPORAL_HORIZON + 1])
                log.debug("  tactile_next  = concat(nxt[1:], diff(nxt))  %s   (nxt offsets %s)",
                          tuple(output["tactile_next"].shape), _idx[TACTILE_TEMPORAL_HORIZON:])
                if _idx[0] == 0:
                    log.warning("  tactile window is NEWEST-FIRST (idx[0]==0): history is time-reversed, "
                                "current frame is dropped by past[1:], and diff(next)[0] spans "
                                "offset %d -> %d. See configs.TACTILE_TEMPORAL_TOTAL_TIMESTAMPS.",
                                _idx[TACTILE_TEMPORAL_HORIZON], _idx[TACTILE_TEMPORAL_HORIZON + 1])
            log.debug("  tactile_next_mask: %d/%d padded target steps",
                      int(output["tactile_next_mask"].sum().item()),
                      output["tactile_next_mask"].numel())
            # The value half and the delta half share one input_proj_tactile, so
            # their relative scale decides how much the delta half can influence
            # the projection.
            #
            # Report the MEDIAN PER-CHANNEL ratio, not the ratio of pooled stds.
            # A pooled std over one small batch is dominated by whichever
            # channels happen to be active in it and swings wildly (a 2-sample
            # batch produced 1400x where the whole-dataset figure is 39x).
            # The per-channel ratio is also invariant to the normalizer, since
            # a per-dim linear map gives diff(x/s) = diff(x)/s.
            _D = output["tactile"].shape[-1] // 2
            _vals = output["tactile"][..., :_D].reshape(-1, _D)
            _dels = output["tactile"][..., _D:].reshape(-1, _D)
            if _vals.shape[0] > 1:
                _ratio = (_vals.std(0) / _dels.std(0).clamp(min=1e-9)).median().item()
                log.debug("  tactile halves: median per-channel value/delta std ratio = %.1fx "
                          "(batch-local; whole-dataset reference ~39x normalized, ~16x raw)",
                          _ratio)
            log_tensor(log, TRACE, "  out/tactile", output["tactile"])
            log_tensor(log, TRACE, "  out/tactile_next", output["tactile_next"])

    if _verbose:
        log.debug("  convert_batch output:")
        log_tensors(log, logging.DEBUG, "    ", output, stats=False)
        log_tensors(log, TRACE, "    stats/", output)

    timing['total'] = time.time() - _t0
    output["_timing"] = timing

    return output





class LeRobotSubset(Subset):
    def __getattr__(self, name):
        return getattr(self.dataset, name)
    


def _filter_by_duration(ds, max_duration_sec):
    """
    Filter a LeRobot dataset to only frames whose timestamp (relative to
    episode start) is <= max_duration_sec.

    Returns a Subset of valid indices, or the original dataset if
    max_duration_sec is None.
    """
    if max_duration_sec is None:
        log.info("no max_duration_sec filter -- using all %d frames", len(ds))
        return ds

    valid_indices = []
    log.info("filtering by max_duration_sec=%s (original length %d)", max_duration_sec, len(ds))

    # for i in range(len(ds)):
    #     frame_info = ds.hf_dataset[i]
    #     frame_ts = frame_info["timestamp"]  # relative to start of its own episode
    #     if frame_ts <= max_duration_sec:
    #         valid_indices.append(i)

    timestamps = ds.hf_dataset["timestamp"][:]
    valid_indices = [i for i, ts in enumerate(timestamps) if ts <= max_duration_sec]


    filtered_ds = LeRobotSubset(ds, valid_indices)
    log.info("filtered length: %d (dropped %d)", len(filtered_ds), len(ds) - len(filtered_ds))
    return filtered_ds


class SPLIT_TYPE(str, Enum):
    FULL = "full"
    TRAIN = "train"
    TEST = "test"
    VAL = "val" 


def plan_train_val_episodes(dataset_root, max_episodes, val_episodes):
    """Split episode indices into train and held-out val sets.

    `val_episodes` is an EXPLICIT list of indices (configs.VAL_EPISODES,
    default [0, 1]). Explicit beats "last N" because the holdout then stays
    identical when MAX_EPISODES changes -- otherwise a 50-episode run and a
    500-episode run validate on different episodes and their numbers cannot be
    compared.

    Splitting by EPISODE (never by frame) is essential: consecutive frames are
    near-duplicates, so a frame-level split leaks the val set into training and
    makes the val loss meaningless.
    """
    meta_path = dataset_root / "meta" / "info.json"
    with open(meta_path, "r") as f:
        total_episodes = json.load(f)["total_episodes"]

    all_eps = list(range(max_episodes)) if max_episodes > 0 else list(range(total_episodes))
    if max_episodes > total_episodes:
        log.warning("MAX_EPISODES=%d exceeds total_episodes=%d; clamping", max_episodes, total_episodes)
        all_eps = list(range(total_episodes))

    val_eps = sorted({int(e) for e in (val_episodes or [])})

    out_of_range = [e for e in val_eps if e not in all_eps]
    if out_of_range:
        raise ValueError(
            f"VAL_EPISODES {out_of_range} are outside the loaded episode range "
            f"{all_eps[0]}..{all_eps[-1]} (MAX_EPISODES={max_episodes}, "
            f"total_episodes={total_episodes})."
        )

    train_eps = [e for e in all_eps if e not in set(val_eps)]
    if not train_eps:
        raise ValueError(f"VAL_EPISODES {val_eps} leaves no training episodes.")

    if not val_eps:
        log.warning("VAL_EPISODES is empty -- validation disabled, no val metrics "
                    "and no policy_best.ckpt")
    else:
        log.info("episode split: %d train (%s..%s) | %d val %s",
                 len(train_eps), train_eps[0], train_eps[-1], len(val_eps), val_eps)
    return train_eps, val_eps


def get_origami_full_dataset(dataset_root, split: SPLIT_TYPE, delta_timestamps, TOLERANCE, use_tactile,
                             max_duration_sec, doImageTransforms, episodes=None, tag=""):
    _t_load_start = time.time()

    if split == "full":
        data_root = dataset_root
    else:
        data_root = dataset_root / split

    # Quick metadata inspection per season
    log.info("=" * 60)
    log.info("building origami dataset  split=%s%s  root=%s", split, f" [{tag}]" if tag else "", data_root)
    log.info("  delta_timestamps keys: %s",
             {k: f"{len(v)} offsets [{round(v[0]*30)} .. {round(v[-1]*30)}]" for k, v in delta_timestamps.items()})
    log.info("  tolerance_s=%s  use_tactile=%s  doImageTransforms=%s  max_duration_sec=%s",
             TOLERANCE, use_tactile, doImageTransforms, max_duration_sec)

    meta_path = data_root / "meta" / "info.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            info = json.load(f)
        log.info("  meta/info.json: total_episodes=%s total_frames=%s fps=%s",
                 info.get('total_episodes'), info.get('total_frames'), info.get('fps'))
    else:
        raise ValueError(f"Warning: Metadata not found for {split}")

    # Episode filtering. An explicit `episodes` list (from plan_train_val_episodes)
    # wins; otherwise fall back to the legacy MAX_EPISODES prefix behaviour.
    if episodes is not None:
        _episodes = list(episodes)
        log.info("  loading %d explicit episode indices: %s%s",
                 len(_episodes), _episodes[:8], " ..." if len(_episodes) > 8 else "")
    elif MAX_EPISODES > 0:
        _episodes = list(range(MAX_EPISODES))
        log.info("  MAX_EPISODES=%d -> loading episode indices 0..%d only", MAX_EPISODES, MAX_EPISODES - 1)
    else:
        _episodes = None
        log.info("  MAX_EPISODES=0 -> loading all episodes")

    #only loads base vitac particular keys
    # torchcodec is the fast path and stays the default. Overridable because the
    # wheel refuses to load against some system ffmpeg versions, which otherwise
    # makes the dataset unconstructable on that machine (pyav still works).
    _backend = os.environ.get("ORI_VIDEO_BACKEND", "torchcodec")
    log.info("  creating LeRobotDataset (return_uint8=True, video_backend=%s) ...", _backend)
    _t_ds_start = time.time()
    ds = LeRobotDataset(
        repo_id=None, root=data_root,
        image_transforms= torchvision_transforms if doImageTransforms else None  ,
        delta_timestamps=delta_timestamps,
        video_backend=_backend,
        tolerance_s=TOLERANCE,
        episodes=_episodes,
        return_uint8=True,  # return raw uint8 frames — /255 done batched in convert_batch
        # transform=drop_unused_keys,   # ###
    )
    _t_ds_end = time.time()
    log.info("  LeRobotDataset created in %.2fs -> %d frames", _t_ds_end - _t_ds_start, len(ds))

    # Log the actual video backend being used (confirms torchcodec loaded, not pyav fallback)
    _actual_backend = getattr(ds, "_video_backend", "unknown")
    log.info("  video_backend requested=%s  actual=%s", _backend, _actual_backend)
    if _actual_backend not in (_backend, "unknown"):
        log.warning("  video backend fell back to %r -- decode will be much slower", _actual_backend)


# stats keys dict_keys(['observation.images.wrist_right', 'observation.images.tactile_raw', 
#                       'timestamp', 'observation.images.tactile_deform', 'observation.state.tcp', 
#                       'action', 'observation.state.joint_torque', 'index', 'task_index', 
#                       'observation.images.head_right', 'frame_index', 'observation.tactile', 
#                       'observation.state', 'episode_index', 'observation.images.wrist_left', 
# #                       'observation.images.head_left'])
#     ignore_keys = [
#         'observation.images.tactile_raw',
#         'observation.images.tactile_deform',
#         # 'observation.state.joint_torque',
#         'observation.state.tcp',

#     ]
    # features = ds.meta.features
    # output_features = {
    #     key: ft for key, ft in features.items() 
    #     if key not in ignore_keys  # Excludes specific modalities like an unnecessary top view
    # }
    # print(output_features)
    # print(ds.meta.video_keys) 
    log.info("  original v3 features : %s", list(ds.meta.features.keys()))
    log.info("  original video keys  : %s", list(ds.meta.video_keys))

    # 2. OPTIMIZATION FOR v3: Prune features from the metadata schema dictionary
    # # This prevents the dataset's internal _load_features() loop from requesting these keys
    # for key in ignore_keys:
    #     if key in ds.meta.features:
    #         del ds.meta.features[key]
            
    # # 3. OPTIMIZATION FOR v3: Explicitly drop them from the video stream tracker
    # # This directly kills the PyAV decoding loop for the ignored cameras!
    # if hasattr(ds.meta, "video_keys"):
    #     ds.meta.video_keys = [k for k in ds.meta.video_keys if k not in ignore_keys]

    # print("Active v3 Features:", list(ds.meta.features.keys()))
    # print("Active v3 Video Keys:", ds.meta.video_keys)

    # for key in ignore_keys:
    #     if key in ds.meta.features:
    #         del ds.meta.features[key]
    # 3. Dynamic Bypass for the Read-Only property: 
    # Use python's __dict__ mapping or patch the property to bypass the setter block.
    #NOTE:::::::::::::::::::::::
    # if "video_keys" in ds.meta.__class__.__dict__:
    #     # Overwrite the class property or force the internal backing variable
    #     # Depending on how the v3 class stores it, it usually calculates dynamically.
    #     # To completely override what the dataset references, we patch the instance property:
    #     type(ds.meta).video_keys = property(lambda self: [
    #         k for k in ['observation.images.head_left', 'observation.images.wrist_left', 
    #                     'observation.images.wrist_right', 'observation.images.head_right', 
    #                     'observation.images.tactile_deform', 'observation.images.tactile_raw']
    #         if k not in ignore_keys
    #     ])

    # 2. OPTIMIZATION: Prune unused video features (tactile_raw, tactile_deform)
    # This prevents the dataset from decoding these video streams — saves CPU + RAM
    ignore_keys = [
        'observation.images.tactile_raw',
        'observation.images.tactile_deform',
    ]
    for key in ignore_keys:
        if key in ds.meta.features:
            del ds.meta.features[key]
            log.info("  pruned feature %s (its video stream will not be decoded)", key)

    # 3. video_keys is a computed @property from features (dtype == "video")
    #    Deleting from features above already removes them from video_keys.
    #    No need to set video_keys directly (it's read-only).


    # Log active keys to the info log file (passed via module-level variable)
    import os as _os
    _info_log = _os.environ.get("ORI_INFO_LOG_PATH", None)
    if _info_log:
        with open(_info_log, 'a') as f:
            f.write(f"\n--- Video Backend ---\n")
            f.write(f"Requested backend: torchcodec\n")
            f.write(f"Actual backend: {_actual_backend}\n")
            f.write(f"\n--- Active Dataset Keys ---\n")
            f.write(f"Active v3 Features: {list(ds.meta.features.keys())}\n")
            f.write(f"Active v3 Video Keys: {list(ds.meta.video_keys)}\n")
    log.info("  active v3 features   : %s", list(ds.meta.features.keys()))
    log.info("  active v3 video keys : %s", list(ds.meta.video_keys))

    # Filter by max duration if specified
    ds = _filter_by_duration(ds, max_duration_sec) #max_duration_sec == None when no filtering by time

    # Quick sanity check: random access still works
    sample = ds[0]
    _default_decoder_cache.clear()
    log.info("  ds[0] sanity sample -- per-key shapes:")
    for k in sorted(sample):
        if hasattr(sample[k], "shape"):
            log.info("    %-42s %s %s", k, tuple(sample[k].shape), sample[k].dtype)
    log_tensors(log, TRACE, "  ds[0]/", {k: v for k, v in sample.items() if hasattr(v, "shape")})

    # Log actual episodes used from dataset object
    # int() matters: hf_dataset returns 0-d tensors, which hash by identity, so
    # set() over them does not deduplicate (this used to report one "episode"
    # per frame).
    episode_indices = sorted({int(e) for e in ds.hf_dataset["episode_index"][:]})
    log.info("  episodes used: %d  (range %s .. %s)",
             len(episode_indices), min(episode_indices), max(episode_indices))

    # Log to info log file if available
    _info_log = _os.environ.get("ORI_INFO_LOG_PATH", None)
    if _info_log:
        with open(_info_log, 'a') as f:
            f.write(f"\n--- Episodes Used ---\n")
            f.write(f"Requested MAX_EPISODES: {MAX_EPISODES}\n")
            f.write(f"Actual episodes count: {len(episode_indices)}\n")
            f.write(f"Actual episode range: {min(episode_indices)} to {max(episode_indices)}\n")
            # f.write(f"Actual episode indices: {episode_indices}\n")

    _t_load_end = time.time()
    log.info("  total dataset loading time: %.2fs", _t_load_end - _t_load_start)
    log.info("=" * 60)

    return ds



