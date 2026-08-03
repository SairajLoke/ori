

import json 

import torch

import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from configs import TACTILE_TEMPORAL_HORIZON, IS_SINGLE_DATASET
import time 

from torchvision.transforms import v2
torchvision_transforms = v2.Compose([
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
])
#can also create custom trasnsforms : https://huggingface.co/docs/lerobot/lerobot-dataset-v3




def convert_batch(batch, use_tactile, delta_timestamps):
    """
    Convert a LeRobot batch into the format expected by the old ACT code.
    """

    # -------------------------------------------------
    # Images
    # -------------------------------------------------
    print("+++++++++=== restructuring the batch ==++++++++++++++++")
    print("batch['observation.images.head_left']      :", batch["observation.images.head_left"].shape)
    print("lowdim     :", batch["observation.state"].shape)
    print("action     :", batch["action"].shape)
    if use_tactile:
        print("tactile    :", batch["observation.tactile"].shape)
    
    

    cams = [
        batch["observation.images.head_left"],
        batch["observation.images.head_right"],
        batch["observation.images.wrist_right"],
        batch["observation.images.wrist_left"],
    ]

    resized = []

    for img in cams:
        img = F.interpolate(
            img.float(),
            size=(224, 320),
            mode="bilinear",
            align_corners=False,
        )
        resized.append(img)

    image = torch.stack(resized, dim=1)

    # -------------------------------------------------
    # Robot state
    # -------------------------------------------------

    lowdim = batch["observation.state"]

    # -------------------------------------------------
    # Actions
    # -------------------------------------------------

    action = batch["action"]
    print('\n actions', batch["action"].shape)
    assert action.shape[0] == image.shape[0], action.shape[1] == len(delta_timestamps["action"])
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
        # print(batch["observation.tactile"].shape) #torch.Size([8, 19, 60])
        tactile_shape = batch["observation.tactile"].shape
        assert tactile_shape[0] == image.shape[0] and tactile_shape[1] == len(delta_timestamps["observation.tactile"]), f"tactile shape {tactile_shape} mismatch"
        
        tactile_past = batch["observation.tactile"][:, :TACTILE_TEMPORAL_HORIZON+1, :]  # -18, -17 ... 0  
        tactile_deltas = torch.diff(tactile_past, dim=1)                             #idx#  0 ,  1,     18 
        output["tactile"] = torch.concat((tactile_past[:, 1:, : ], tactile_deltas), dim=-1)    # [B, 18, 120]
        
        tactile_next = batch["observation.tactile"][:, TACTILE_TEMPORAL_HORIZON : , :]
        tactile_next_deltas = torch.diff(tactile_next, dim=1)   
        output["tactile_next"] =  torch.concat((tactile_next[:, 1:, : ], tactile_next_deltas), dim=-1)    # [B, 18, 60] # [B, 18, 60]
        
        print("tactile    :", output["tactile"].shape)
        print("tactile next   :", output["tactile_next"].shape)
        

    print("image      :", image.shape, image.device)
    print("lowdim     :", lowdim.shape)
    print("action     :", action.shape)
    print("action_mask", action_mask.shape)
    
    return output



class LeRobotSubset(Subset):
    def __getattr__(self, name):
        try:
            dataset = object.__getattribute__(self, 'dataset')
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(dataset, name)
    


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
    for i in range(len(ds)):
        frame_info = ds.hf_dataset[i]
        frame_ts = frame_info["timestamp"]  # relative to start of its own episode
        if frame_ts <= max_duration_sec:
            valid_indices.append(i)

    filtered_ds = None 
    if IS_SINGLE_DATASET:
        filtered_ds = LeRobotSubset(ds, valid_indices)
    else:
        filtered_ds = LeRobotDataset(ds, valid_indices)


    print(f"  Filtered length: {len(filtered_ds)}")
    return filtered_ds

from enum import Enum

class SPLIT_TYPE(str, Enum):
    FULL = "full"
    TRAIN = "train"
    TEST = "test"
    VAL = "val" 


def get_origami_full_dataset(dataset_root, split: SPLIT_TYPE, delta_timestamps, TOLERANCE, use_tactile, max_duration_sec, doImageTransforms):
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

    # Instantiate the dataset for this specific season
    ds = LeRobotDataset(repo_id=None, root=data_root,
                        # episodes=,
                        image_transforms= torchvision_transforms if doImageTransforms else None  ,
                        delta_timestamps=delta_timestamps,
                        video_backend="pyav",
                        tolerance_s=TOLERANCE)

    # Filter by max duration if specified
    ds = _filter_by_duration(ds, max_duration_sec) #max_duration_sec == None when no filtering by time 

    # Quick sanity check: random access still works
    sample = ds[0]
    print(f"Sample head_left shape: {sample['observation.images.head_left'].shape}")

    return ds






class LeRobotNormalizer:

    def __init__(self, stats, cfg, device):

        self.transforms = {}

        for key, mode in cfg.items():

            if mode is None:
                continue

            s = stats[key]

            mean = torch.tensor(s["mean"]).to(device)
            std = torch.tensor(s["std"]).to(device)
            minimum = torch.tensor(s["min"]).to(device)
            maximum = torch.tensor(s["max"]).to(device)

            self.transforms[key] = dict(
                mode=mode,
                mean=mean,
                std=std,
                min=minimum,
                max=maximum,
            )

    def normalize(self, key, x):

        if key not in self.transforms:
            return x

        t = self.transforms[key]

        mean = t["mean"].to(x.device)
        std = t["std"].to(x.device)

        if t["mode"] == "gaussian":
            return (x - mean) / (std + 1e-6)

        minimum = t["min"].to(x.device)
        maximum = t["max"].to(x.device)

        return 2 * (x - minimum) / (maximum - minimum + 1e-6) - 1