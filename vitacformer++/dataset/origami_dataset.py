

import json 

import torch
from enum import Enum

import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot.datasets.video_utils import _default_decoder_cache
from configs import TACTILE_TEMPORAL_HORIZON, MAX_EPISODES

import time 

from torchvision.transforms import v2
torchvision_transforms = v2.Compose([
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
])
#can also create custom trasnsforms : https://huggingface.co/docs/lerobot/lerobot-dataset-v3

def log_before_after(name, data_raw, data_norm):
    """Log shape, device, dtype, range before/after normalization."""
    print(f"\n[{name}]")
    print(f"  Before: shape={data_raw.shape}, device={data_raw.device}, dtype={data_raw.dtype}")
    # Convert to float for stats if uint8 (return_uint8=True path)
    _raw_for_stats = data_raw.float() / 255.0 if data_raw.dtype == torch.uint8 else data_raw
    print(f"    Range: [{_raw_for_stats.min():.4f}, {_raw_for_stats.max():.4f}], μ={_raw_for_stats.mean():.4f}, σ={_raw_for_stats.std():.4f}")
    print(f"  After:  shape={data_norm.shape}, device={data_norm.device}, dtype={data_norm.dtype}")
    print(f"    Range: [{data_norm.min():.4f}, {data_norm.max():.4f}], μ={data_norm.mean():.4f}, σ={data_norm.std():.4f}")


def convert_batch(batch, use_tactile, delta_timestamps, epoch=0, batch_idx=0, normalizer=None):

    """
    Convert a LeRobot batch into the format expected by the old ACT code.

    Returns:
        output: dict with keys "image", "lowdim", "action", "action_mask",
                and optionally "tactile", "tactile_next".
        output["_timing"]: dict of per-sub-step timings in seconds.
    """

    timing = {}
    _t0 = time.time()


    if epoch == 0 and batch_idx < 3:
        print("==================== converting the batch ====================")
        print("batch['observation.images.head_left']      :", batch["observation.images.head_left"].shape, batch["observation.images.head_left"].device)
        print("lowdim     :", batch["observation.state"].shape)
        print("action     :", batch["action"].shape)
        if use_tactile:
            print("tactile    :", batch["observation.tactile"].shape, batch["observation.tactile"].device)



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
        """Log shape, device, dtype, range before/after normalization."""
        print(f"\n[{name}]")
        print(f"  Before: shape={data_raw.shape}, device={data_raw.device}, dtype={data_raw.dtype}")
        # Convert to float for stats if uint8 (return_uint8=True path)
        _raw_for_stats = data_raw.float() / 255.0 if data_raw.dtype == torch.uint8 else data_raw
        print(f"    Range: [{_raw_for_stats.min():.4f}, {_raw_for_stats.max():.4f}], μ={_raw_for_stats.mean():.4f}, σ={_raw_for_stats.std():.4f}")
        print(f"  After:  shape={data_norm.shape}, device={data_norm.device}, dtype={data_norm.dtype}")
        print(f"    Range: [{data_norm.min():.4f}, {data_norm.max():.4f}], μ={data_norm.mean():.4f}, σ={data_norm.std():.4f}")

    # ── Timing: normalization ──
    _t_norm_start = time.time()

    # Observation state
    lowdim_raw = batch["observation.state"]
    lowdim = normalizer.normalize("observation.state", lowdim_raw) if normalizer else lowdim_raw
    if epoch == 0 and batch_idx < 3:
        log_before_after("observation.state (lowdim)", lowdim_raw, lowdim)

    # Action
    action_raw = batch["action"]
    action = normalizer.normalize("action", action_raw) if normalizer else action_raw

    if normalizer:
        action[:, :, 58] = action_raw[:, :, 58] #NOTE::::::::::::::
        action[:, :, 59] = action_raw[:, :, 59] #NOTE::::::::::::::

    if epoch == 0 and batch_idx < 3:
        log_before_after("action", action_raw, action)

    # Joint torque (log even if not used, to debug)
    if "observation.state.joint_torque" in batch:
        joint_torque_raw = batch["observation.state.joint_torque"]
        joint_torque = normalizer.normalize("observation.state.joint_torque", joint_torque_raw) if normalizer else joint_torque_raw
        joint_torque[:, 58:65] = 0.0  #NOTE::::::::::::::

        if epoch == 0 and batch_idx < 3:
            log_before_after("observation.state.joint_torque", joint_torque_raw, joint_torque)

    # Images - convert uint8→float32 first (required for model input, logging)
    # return_uint8=True means images arrive as uint8 [0,255] - must convert to float32 [0,1]
    imgHeadLeft_raw = batch["observation.images.head_left"]
    if imgHeadLeft_raw.dtype == torch.uint8:
        imgHeadLeft_raw = imgHeadLeft_raw.float() / 255.0
    imgHeadLeft = normalizer.normalize("observation.images.head_left", imgHeadLeft_raw) if normalizer else imgHeadLeft_raw
    if epoch == 0 and batch_idx < 3:
        log_before_after("observation.images.head_left", imgHeadLeft_raw, imgHeadLeft)

    imgHeadRight_raw = batch["observation.images.head_right"]
    if imgHeadRight_raw.dtype == torch.uint8:
        imgHeadRight_raw = imgHeadRight_raw.float() / 255.0
    imgHeadRight = normalizer.normalize("observation.images.head_right", imgHeadRight_raw) if normalizer else imgHeadRight_raw
    
    imgWristRight_raw = batch["observation.images.wrist_right"]
    if imgWristRight_raw.dtype == torch.uint8:
        imgWristRight_raw = imgWristRight_raw.float() / 255.0
    imgWristRight = normalizer.normalize("observation.images.wrist_right", imgWristRight_raw) if normalizer else imgWristRight_raw
    
    imgWristLeft_raw = batch["observation.images.wrist_left"]
    if imgWristLeft_raw.dtype == torch.uint8:
        imgWristLeft_raw = imgWristLeft_raw.float() / 255.0
    imgWristLeft = normalizer.normalize("observation.images.wrist_left", imgWristLeft_raw) if normalizer else imgWristLeft_raw

    timing['norm'] = time.time() - _t_norm_start

    # ── Timing: image resize ──
    _t_resize_start = time.time()

    # combine - ----------------------
    cams = [
        imgHeadLeft,
        imgHeadRight,
        imgWristRight,
        imgWristLeft
    ]

    resized = []
    for img in cams:
        # Images already converted to float32 [0,1] above (uint8→float32 conversion)
        # Just resize - no need for /255 here anymore
        img = F.interpolate(
            img,
            size=(224, 320),
            mode="bilinear",
            align_corners=False,
        )
        resized.append(img)
    image = torch.stack(resized, dim=1)    
    assert action.shape[0] == image.shape[0] and action.shape[1] == len(delta_timestamps["action"])

    timing['resize'] = time.time() - _t_resize_start


    B, T = action.shape[:2]
    action_mask = torch.zeros(
        (B, T),
        dtype=torch.bool,
        device=action.device,
    )

    output = {
        "image": image,
        "lowdim": lowdim,
        "action": action,
        "action_mask": action_mask,
    }

    if use_tactile:
        _t_tac_start = time.time()
        # print(batch["observation.tactile"].shape) #torch.Size([8, 19, 60])
        tactile_raw = batch["observation.tactile"]
        tactile_shape = tactile_raw.shape
        assert tactile_shape[0] == image.shape[0] and tactile_shape[1] == len(delta_timestamps["observation.tactile"]), \
            f"tactile shape {tactile_shape} mismatch"

        tactile_pastNfuture = normalizer.normalize("observation.tactile", batch["observation.tactile"]) if normalizer else batch["observation.tactile"]
         
        if epoch == 0 and batch_idx < 3:
            log_before_after("observation.tactile", tactile_raw, tactile_pastNfuture)
        
        tactile_past = tactile_pastNfuture[:, :TACTILE_TEMPORAL_HORIZON+1, :]  # -18, -17 ... 0  
        tactile_deltas = torch.diff(tactile_past, dim=1)                             #idx#  0 ,  1,     18 
        output["tactile"] = torch.concat((tactile_past[:, 1:, : ], tactile_deltas), dim=-1)    # [B, 18, 120]
        
        tactile_next = tactile_pastNfuture[:, TACTILE_TEMPORAL_HORIZON : , :]
        tactile_next_deltas = torch.diff(tactile_next, dim=1)   
        output["tactile_next"] =  torch.concat((tactile_next[:, 1:, : ], tactile_next_deltas), dim=-1)    # [B, 18, 60] # [B, 18, 60]
        
        timing['tactile'] = time.time() - _t_tac_start
        
        if epoch == 0 and batch_idx < 3:
            print("tactile        :", output["tactile"].shape)
            print("tactile next   :", output["tactile_next"].shape)

        
    if epoch == 0 and batch_idx < 3:
        print("\n[FINAL OUTPUT]")
        print(f"  image: {output['image'].shape}, {output['image'].device}, {output['image'].dtype}")
        print(f"  lowdim: {output['lowdim'].shape}, {output['lowdim'].device}, {output['lowdim'].dtype}")
        print(f"  action: {output['action'].shape}, {output['action'].device}, {output['action'].dtype}")
        print(f"  action_mask: {output['action_mask'].shape}, {output['action_mask'].device}")
        if use_tactile:
            print(f"  tactile: {output['tactile'].shape}, {output['tactile'].device}, {output['tactile'].dtype}")
            print(f"  tactile_next: {output['tactile_next'].shape}, {output['tactile_next'].device}, {output['tactile_next'].dtype}")
        print("=" * 70)

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
        return ds

    valid_indices = []
    print(f"  Filtering by max_duration_sec={max_duration_sec} ...")
    print(f"  Original dataset length: {len(ds)}")

    # for i in range(len(ds)):
    #     frame_info = ds.hf_dataset[i]
    #     frame_ts = frame_info["timestamp"]  # relative to start of its own episode
    #     if frame_ts <= max_duration_sec:
    #         valid_indices.append(i)

    timestamps = ds.hf_dataset["timestamp"][:]
    valid_indices = [i for i, ts in enumerate(timestamps) if ts <= max_duration_sec]


    filtered_ds = LeRobotSubset(ds, valid_indices)
    print(f"  Filtered length: {len(filtered_ds)}")
    return filtered_ds


class SPLIT_TYPE(str, Enum):
    FULL = "full"
    TRAIN = "train"
    TEST = "test"
    VAL = "val" 


def get_origami_full_dataset(dataset_root, split: SPLIT_TYPE, delta_timestamps, TOLERANCE, use_tactile, max_duration_sec, doImageTransforms):
    _t_load_start = time.time()
    
    if split == "full":
        data_root = dataset_root
    else:
        data_root = dataset_root / split

    # Quick metadata inspection per season
    meta_path = data_root / "meta" / "info.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            info = json.load(f)
        print(f"--- {split} ---")
        print(f"Total Episodes: {info['total_episodes']}")
        print(f"Total Frames: {info['total_frames']}")
    else:
        raise ValueError(f"Warning: Metadata not found for {split}")

    # Episode filtering: if MAX_EPISODES > 0, only load first N episodes
    _episodes = list(range(MAX_EPISODES)) if MAX_EPISODES > 0 else None
    if _episodes is not None:
        print(f"[Dataset] Limiting to first {MAX_EPISODES} episodes")

    #only loads base vitac particular keys 
    print(f"[Dataset] Creating LeRobotDataset (return_uint8=True, video_backend=torchcodec)...")
    _t_ds_start = time.time()
    ds = LeRobotDataset(
        repo_id=None, root=data_root,
        image_transforms= torchvision_transforms if doImageTransforms else None  ,
        delta_timestamps=delta_timestamps,
        video_backend="torchcodec",
        tolerance_s=TOLERANCE,
        episodes=_episodes,
        return_uint8=True,  # return raw uint8 frames — /255 done batched in convert_batch
        # transform=drop_unused_keys,   # ###
    )
    _t_ds_end = time.time()
    print(f"[Dataset] LeRobotDataset created in {_t_ds_end - _t_ds_start:.2f}s")

    # Log the actual video backend being used (confirms torchcodec loaded, not pyav fallback)
    _actual_backend = getattr(ds, "_video_backend", "unknown")
    print(f"[Dataset] Actual video_backend in use: {_actual_backend}")


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
    print("Original v3 Features:", list(ds.meta.features.keys()))
    print("original video keys",  list(ds.meta.video_keys))

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
    else:
        print("Active v3 Features:", list(ds.meta.features.keys()))
        print("Active v3 Video Keys:", ds.meta.video_keys)



    # Filter by max duration if specified
    ds = _filter_by_duration(ds, max_duration_sec) #max_duration_sec == None when no filtering by time 

    # Quick sanity check: random access still works
    sample = ds[0]
    _default_decoder_cache.clear()
    # print(f"Sample head_left shape: {sample['observation.images.head_left'].shape}")
    for k in sample:
        if hasattr(sample[k], "shape"):
            print(sample[k].shape)

    # Log actual episodes used from dataset object
    episode_indices = sorted(set(ds.hf_dataset["episode_index"][:]))
    print(f"[Dataset] Actual episodes used: {len(episode_indices)} episodes")
    print(f"[Dataset] Episode range: {min(episode_indices)} to {max(episode_indices)}")
    # print(f"[Dataset] Episode indices: {episode_indices}")

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
    print(f"[Dataset] Total loading time: {_t_load_end - _t_load_start:.2f}s")

    return ds



