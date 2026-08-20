# import torch

# def _stats(name: str, tensor: torch.Tensor):
#     if not isinstance(tensor, torch.Tensor):
#         print(f"❌ [{name}] Not a tensor: {type(tensor)}")
#         return
    
#     # Extract values safely
#     t_min = tensor.min().item() if tensor.numel() > 0 else "N/A"
#     t_max = tensor.max().item() if tensor.numel() > 0 else "N/A"
    
#     # Check for invalid values in floating point tensors
#     status = ""
#     if tensor.dtype.is_floating_point and tensor.numel() > 0:
#         if torch.isnan(tensor).any() or torch.isinf(tensor).any():
#             status = " ⚠️ (NaN/Inf Detected!)"

#     print(f" [{name}] Shape: {list(tensor.shape)} | dtype: {tensor.dtype} | Device: {tensor.device} | Min: {t_min} | Max: {t_max}{status}")