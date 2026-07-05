

import json 

import torch
from torch.utils.data import DataLoader, ConcatDataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def get_origami_multi_season_dataset(dataset_root, seasons, delta_timestamps, TOLERANCE):
    
    # Configure your delta timestamps
    # delta_timestamps = {
    #     "observation.images.head_left": [-0.2, -0.1, 0.0]
    # }

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
                            delta_timestamps=delta_timestamps,video_backend="pyav",
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