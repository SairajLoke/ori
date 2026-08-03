

import json 

import torch
from torch.utils.data import DataLoader, ConcatDataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset



# def convert_batch(batch):
    
#     for key , val in batch.items():
#         print(key, val.shape)

#     return {
#         "image": torch.stack([
#             batch["observation.images.head_left"],
#             batch["observation.images.head_right"],
#             batch["observation.images.right_wrist"],
#             batch["observation.images.left_wrist"],
#         ], dim=1),

#         "lowdim": batch["observation.state"],

#         "action": batch["action"],

#         "action_mask": torch.zeros( #NOTE why are these 0
#             batch["action"].shape[:2],
#             dtype=torch.bool,
#             device=batch["action"].device,
#         )
#     }
    
import torch
import torch.nn.functional as F


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
        # print(batch["observation.tactile"].shape) #torch.Size([8, 2, 60])
        tactile_shape = batch["observation.tactile"].shape
        assert tactile_shape[0] == image.shape[0], tactile_shape[1] == len(delta_timestamps["observation.tactile"])
        
        tactile = batch["observation.tactile"]
        output["tactile"] = tactile[:, 0]
        output["tactile_next"] = tactile[:, 1]
        
        

    print("image      :", image.shape, image.device)
    print("lowdim     :", lowdim.shape)
    print("action     :", action.shape)
    print("action_mask", action_mask.shape)
    if use_tactile:
        print("tactile    :", output["tactile"].shape)
        print("tactile next   :", output["tactile_next"].shape)
        assert not torch.equal(output["tactile"] , output["tactile_next"])
        

    return output

def get_origami_single_season_dataset(dataset_root, season, delta_timestamps, TOLERANCE, use_tactile):
    episode_root = dataset_root / season / "lerobot3.0"
        
    # Quick metadata inspection per season
    meta_path = episode_root / "meta" / "info.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            info = json.load(f)
        print(f"--- {season} ---")
        print(f"Total Episodes: {info['total_episodes']}")
        print(f"Total Frames: {info['total_frames']}")
    else:
        raise ValueError(f"Warning: Metadata not found for {season}")

    # Instantiate the dataset for this specific season
    ds = LeRobotDataset(repo_id=None, root=episode_root, 
                        delta_timestamps=delta_timestamps,
                        video_backend="pyav",
                        tolerance_s=TOLERANCE)

    # 4. Random access still works (PyTorch handles the index mapping under the hood)
    sample = ds[100]
    print(f"Sample head_left shape: {sample['observation.images.head_left'].shape}")
    
    return ds

def get_origami_multi_season_dataset(dataset_root, seasons, delta_timestamps, TOLERANCE, use_tactile):
    #TODO : something to do with use_tactile 

    individual_datasets = []

    # 2. Loop through each season to inspect metadata and build individual datasets
    for season in seasons:
        episode_root = dataset_root / season / "lerobot3.0"
        
        # Quick metadata inspection per season
        meta_path = episode_root / "meta" / "info.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                info = json.load(f)
            print(f"--- {season} ---")
            print(f"Total Episodes: {info['total_episodes']}")
            print(f"Total Frames: {info['total_frames']}")
        else:
            print(f"Warning: Metadata not found for {season}")
            continue

        # Instantiate the dataset for this specific season
        ds = LeRobotDataset(repo_id=None, root=episode_root, 
                            delta_timestamps=delta_timestamps,
                            video_backend="pyav",
                            tolerance_s=TOLERANCE)
        individual_datasets.append(ds)

    print("\n-----------------------------------------")

    # 3. Combine all seasons into a single dataset
    # ConcatDataset seamlessly merges them, summing up their total lengths
    multi_season_dataset = ConcatDataset(individual_datasets)
    print(f"Combined total frames across all seasons: {len(multi_season_dataset)}")

    # 4. Random access still works (PyTorch handles the index mapping under the hood)
    sample = multi_season_dataset[100]
    print(f"Sample head_left shape: {sample['observation.images.head_left'].shape}")
    
    return multi_season_dataset





class LeRobotNormalizer:

    def __init__(self, stats, cfg):

        self.transforms = {}

        for key, mode in cfg.items():

            if mode is None:
                continue

            s = stats[key]

            mean = torch.tensor(s["mean"])
            std = torch.tensor(s["std"])
            minimum = torch.tensor(s["min"])
            maximum = torch.tensor(s["max"])

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