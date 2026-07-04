# from pathlib import Path
# import json
# import cv2  # or use torchcodec if preferred

# dataset_root = Path("/home/ubuntu/iros2026/Robotic_Origami_Challenge")
# seasons = [
#     "season_POC22032_2026_05_14_19_21_01_train",
#     "season_POC22032_2026_05_14_20_40_58_train"
# ]

# for season in seasons:
#     video_dir = dataset_root / season / "lerobot3.0" / "videos"
#     if not video_dir.exists():
#         print(f"❌ Missing video dir for {season}")
#         continue
        
#     print(f"Checking {season}...")
#     # Find all mp4 files inside this season
#     for video_path in video_dir.rglob("*.mp4"):
#         # Quick size check (LFS pointers are usually < 200 bytes)
#         if video_path.stat().st_size < 1000:
#             print(f"⚠️ {video_path.name} is too small ({video_path.stat().st_size} bytes). Likely an un-pulled LFS pointer!")
#             continue

#         # Test if the video can actually be opened and decoded
#         cap = cv2.VideoCapture(str(video_path))
#         if not cap.isOpened():
#             print(f"❌ Corrupted file (cannot open): {video_path}")
#         else:
#             ret, frame = cap.read()
#             if not ret:
#                 print(f"❌ Corrupted frame data (cannot read first frame): {video_path}")
#             cap.release()

# print("Verification complete.")

import json
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, ConcatDataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset


TOLERANCE = 0.001

dataset_root = Path("/home/ubuntu/iros2026/Robotic_Origami_Challenge")

# 1. Define the list of seasons you want to inspect
seasons = [
    "season_POC22032_2026_05_14_19_21_01_train",
    "season_POC22032_2026_05_14_20_40_58_train",
    # Add your new seasons here as you download them:
    # "season_POC22033_train", 
    # "season_POC22034_train",
]

# Configure your delta timestamps
delta_timestamps = {
    "observation.images.head_left": [-0.2, -0.1, 0.0]
}

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

# 5. Pass the combined dataset to your DataLoader
batch_size = 16
data_loader = DataLoader(
    multi_season_dataset, 
    batch_size=batch_size, 
    pin_memory=True, 
    shuffle=True, 
    num_workers=min(os.cpu_count(), batch_size),
    persistent_workers=True,
)

# 6. Iterate through the combined data
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Total batches in DataLoader: {len(data_loader)}")


for count, batch in enumerate(data_loader):
    observations = batch["observation.state"].to(device)
    actions = batch["action"].to(device)
    images = batch["observation.images.head_left"].to(device)
    
    # print('')
    # Your model training forward pass goes here
    if count % 10 == 0:
        print(f"\nBatch {count} -> Obs: {observations.shape}, Actions: {actions.shape}, Images: {images.shape}")
    
    # break # Uncomment to test just the first batch