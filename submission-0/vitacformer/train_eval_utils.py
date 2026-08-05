"""
Shared training & evaluation utilities for the Origami ViTacFormer project.

Contains:
  - Joint group definitions (indices into 65-dim action/state vectors)
  - _detailed_stats:   print shape/min/max/mean/std for a tensor
  - log_input_stats:   log per-batch input stats to console, TensorBoard, and info.log
"""

import torch


# ──────────────────────────────────────────────────────────────────────
# Joint group definitions (indices into the 65-dim action/state vectors)
# ──────────────────────────────────────────────────────────────────────
JOINT_GROUPS = {
    "left_arm":   list(range(0, 7)),
    "left_hand":  list(range(7, 29)),
    "right_arm":  list(range(29, 36)),
    "right_hand": list(range(36, 58)),
    "motor":      list(range(58, 65)),
}

JOINT_GROUP_NAMES = {
    "left_arm":   [f"left_arm_j{i}" for i in range(7)],
    "left_hand":  [f"left_hand_j{i}" for i in range(22)],
    "right_arm":  [f"right_arm_j{i}" for i in range(7)],
    "right_hand": [f"right_hand_j{i}" for i in range(22)],
    "motor":      [f"motor_j{i}" for i in range(7)],
}

JOINT_GROUP_COLORS = {
    "left_arm":   "tab:blue",
    "left_hand":  "tab:orange",
    "right_arm":  "tab:green",
    "right_hand": "tab:red",
    "motor":      "tab:purple",
}


# ──────────────────────────────────────────────────────────────────────
# Detailed stats printer
# ──────────────────────────────────────────────────────────────────────
def _detailed_stats(name, tensor):
    """Print detailed stats (shape, min, max, mean, std) for a tensor."""
    if not isinstance(tensor, torch.Tensor):
        print(f"  [{name}] Not a tensor: {type(tensor)}")
        return
    if tensor.numel() == 0:
        print(f"  [{name}] Empty tensor, shape={list(tensor.shape)}")
        return
    t_min = tensor.min().item()
    t_max = tensor.max().item()
    t_mean = tensor.float().mean().item()
    t_std = tensor.float().std().item() if tensor.numel() > 1 else 0.0
    status = ""
    if tensor.dtype.is_floating_point:
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            status = " ⚠️ (NaN/Inf Detected!)"
    print(f"  [{name}] Shape: {list(tensor.shape)} | dtype: {tensor.dtype} | "
          f"Min: {t_min:.4f} | Max: {t_max:.4f} | Mean: {t_mean:.4f} | Std: {t_std:.4f}{status}")


# ──────────────────────────────────────────────────────────────────────
# Input stats logger (first 3 batches of training)
# ──────────────────────────────────────────────────────────────────────
def log_input_stats(data, use_tactile, writer, global_step, log_file=None):
    """
    Log detailed input stats for a converted batch to console, TensorBoard, and optionally info.log.

    Called for the first 3 batches of training to verify data integrity.
    """
    separator = "=" * 70
    header = f"\n[Input Stats] global_step={global_step}"
    print(separator)
    print(header)
    print(separator)

    lines = [separator, header, separator]

    def _log(name, tensor):
        _detailed_stats(name, tensor)
        if isinstance(tensor, torch.Tensor) and tensor.numel() > 0 and writer is not None:
            try:
                writer.add_histogram(f"input_stats/{name}", tensor.cpu(), global_step)
            except Exception:
                pass
        if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
            t_min = tensor.min().item()
            t_max = tensor.max().item()
            t_mean = tensor.float().mean().item()
            t_std = tensor.float().std().item() if tensor.numel() > 1 else 0.0
            lines.append(f"  [{name}] Shape: {list(tensor.shape)} | "
                         f"Min: {t_min:.4f} | Max: {t_max:.4f} | "
                         f"Mean: {t_mean:.4f} | Std: {t_std:.4f}")
        else:
            lines.append(f"  [{name}] Not a tensor or empty: {type(tensor)}")

    _log("image", data["image"])
    _log("lowdim", data["lowdim"])
    _log("action", data["action"])
    _log("action_mask", data["action_mask"])
    if use_tactile:
        _log("tactile", data["tactile"])
        _log("tactile_next", data["tactile_next"])

    # Per-joint-group stats for lowdim and action
    for group_name, indices in JOINT_GROUPS.items():
        ld_group = data["lowdim"][..., indices]
        act_group = data["action"][..., indices]
        _log(f"lowdim/{group_name}", ld_group)
        _log(f"action/{group_name}", act_group)

    print(separator)
    lines.append(separator)

    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write("\n".join(lines) + "\n")