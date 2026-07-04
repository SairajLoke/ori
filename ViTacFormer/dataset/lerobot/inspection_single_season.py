
import json
from pathlib import Path
import os

dataset_root = Path("/home/ubuntu/iros2026/Robotic_Origami_Challenge")
episode_root = dataset_root / "season_POC22032_2026_05_14_19_21_01_train" / "lerobot3.0"

with open(episode_root / "meta" / "info.json", "r") as f:
    info = json.load(f)

print(info["total_episodes"])
print(info["total_frames"])
print(info["features"].keys())


import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# repo_id = "SharpaIT/Robotic_Origami_Challenge"

# 1) Load from the Hub (cached locally)
dataset = LeRobotDataset(repo_id=None, root=episode_root)

# 2) Random access by index
sample = dataset[100]
print(sample)

delta_timestamps = {
    "observation.images.head_left": [-0.2, -0.1, 0.0]  # 0.2s and 0.1s before current frame
}

dataset = LeRobotDataset(repo_id=None,  root=episode_root, delta_timestamps=delta_timestamps)

# Accessing an index now returns a stack for the specified key(s)
sample = dataset[100]
print(sample["observation.images.head_left"].shape)  # [T, C, H, W], where T=3

# 4) Wrap with a DataLoader for training
batch_size = 16
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                          pin_memory=True, shuffle=True, 
                                          num_workers=min(os.cpu_count(), batch_size),
                                          persistent_workers=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
count = 0
print(len(data_loader))
for batch in data_loader:
    observations = batch["observation.state"].to(device)
    actions = batch["action"].to(device)
    images = batch["observation.images.head_left"].to(device)
    # model.forward(batch)
    
    print(observations.shape, actions.shape, images.shape)
    print(count, end='-')
    count+= 1
    # break