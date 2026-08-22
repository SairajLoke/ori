import torch
from torch.utils.data import DataLoader, TensorDataset
import time

# Dummy dataset: 1000 samples, 224×224 RGB images
num_samples = 1000
images = torch.randn(num_samples, 3, 224, 224)  # ~600 MB
labels = torch.randint(0, 10, (num_samples,))

dataset = TensorDataset(images, labels)

# Test 1: pin_memory=False
loader_no_pin = DataLoader(dataset, batch_size=128, num_workers=4, pin_memory=True, persistent_workers=False)
start = time.time()
for batch_idx, (img, lbl) in enumerate(loader_no_pin):
    img = img.cuda()  # Transfer to GPU
    if batch_idx == 10:  # Measure first 10 batches
        break
t_no_pin = time.time() - start
print(f"pin_memory=False: {t_no_pin:.3f}s")

# Test 2: pin_memory=True
loader_pin = DataLoader(dataset, batch_size=128, num_workers=4, pin_memory=True, persistent_workers=True)
start = time.time()
for batch_idx, (img, lbl) in enumerate(loader_pin):
    img = img.cuda()  # Transfer to GPU
    if batch_idx == 10:
        break
t_pin = time.time() - start
print(f"pin_memory=True: {t_pin:.3f}s")

print(f"Speedup: {t_no_pin/t_pin:.2f}×")