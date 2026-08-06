"""
Origami Inference & Open-Loop Evaluation
========================================
Loads a trained ACT checkpoint, runs the model on the origami dataset,
compares predicted vs ground-truth actions, and produces timing info,
metrics, and accuracy plots.

Follows the patterns from:
  - _inference.py        (checkpoint loading, policy config, inference forward)
  - origami_imitate_episodes.py  (origami_forward_pass, convert_batch, configs)
  - open_loop_eval.py    (metrics, per-joint-group MSE, plotting)

Usage:
    python origami_inference.py \
        --ckpt_path 20260707_030650origami_tactile/policy_epoch_10_loss_0.073.ckpt \
        --stats_path 20260707_030650origami_tactile/normalize.pkl \
        --use_tactile \
        --n_samples 50 \
        --output_dir inference_results
"""

import os
import sys
import json
import time
import pickle
import argparse
from pathlib import Path
from functools import wraps

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from policy import ACTPolicy
from utils import apply_joint_mask
from dataset.origami_dataset import (
    get_origami_full_dataset,
    convert_batch,
    LeRobotNormalizer,
)
from train_eval_utils import (
    JOINT_GROUPS, JOINT_GROUP_NAMES, JOINT_GROUP_COLORS,
    _detailed_stats, log_input_stats,
)

from configs import (
    EPISODE_LEN, TOLERANCE, CAMERA_NAMES, STATE_DIM, LR_BACKBONE, BACKBONE,
    IS_ORIGAMI_TASK, DELTA_TIMESTAMPS, CHUNK_SIZE,
    PROPRIOCEPTIVE_TEMPORAL_HORIZON, MASK_FINGERS, HAND_MASK, MAXDURATION_IN_EPISODE_SEC, FPS,INFERENCE_DATASET_ROOT
)



# ──────────────────────────────────────────────────────────────────────
# Timing decorator
# ──────────────────────────────────────────────────────────────────────
def timing_measure(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_time = time.perf_counter() - start_time
        return result, elapsed_time
    return wrapper


# ──────────────────────────────────────────────────────────────────────
# Build policy config (matching training)
# ──────────────────────────────────────────────────────────────────────
def build_policy_config(args):
    return {
        'num_queries': CHUNK_SIZE,
        'hidden_dim': args.hidden_dim,
        'dim_feedforward': args.dim_feedforward,
        'kl_weight': 10,           # unused at inference
        'lr': 1e-4,                # unused at inference
        'lr_backbone': LR_BACKBONE,
        'backbone': BACKBONE,
        'enc_layers': 4,
        'dec_layers': 7,
        'nheads': 8,
        'camera_names': CAMERA_NAMES,
        'use_tactile': args.use_tactile,
        'state_dim': args.state_dim,
        'proprioceptive_temporal_horizon': PROPRIOCEPTIVE_TEMPORAL_HORIZON,
    }


# ──────────────────────────────────────────────────────────────────────
# Load checkpoint into policy
# ──────────────────────────────────────────────────────────────────────
def load_policy(ckpt_path, policy_config, device):
    print(f"[Checkpoint] Loading: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    policy = ACTPolicy(policy_config)

    # Handle both dict-with-'model' and raw state_dict
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        policy.load_state_dict(checkpoint['model'])
        print(f"  Loaded from checkpoint['model']")
        print(f"  Checkpoint epoch: {checkpoint.get('epoch', '?')}")
        print(f"  Checkpoint global_step: {checkpoint.get('global_step', '?')}")
    else:
        policy.load_state_dict(checkpoint)
        print(f"  Loaded raw state_dict")

    policy.eval().to(device)
    print(f"  Policy loaded and set to eval mode on {device}")
    return policy


# ──────────────────────────────────────────────────────────────────────
# Inference forward pass (mirrors origami_forward_pass but no actions)
# ──────────────────────────────────────────────────────────────────────
def origami_inference_forward(data, policy, device, use_tactile):
    """
    Run the policy in inference mode (no actions passed → samples from prior).

    Mirrors origami_forward_pass from origami_imitate_episodes.py but without
    actions / is_pad (inference mode).

    Returns:
        a_hat: [B, T, D_action] predicted actions
    """
    image_data = data["image"]       # [B, N_cam, 3, H, W]
    qpos_data  = data["lowdim"]      # [B, T1, D1]

    # No normalization (same as training — normalization is commented out in
    # origami_forward_pass)
    qpos_data_norm = qpos_data

    # Apply hand joint masking (same as training)
    if MASK_FINGERS:
        qpos_data_norm = apply_joint_mask(qpos_data_norm, HAND_MASK, start_index=7)
        qpos_data_norm = apply_joint_mask(qpos_data_norm, HAND_MASK, start_index=7 + 22 + 7)

    # Flatten
    B, T1, D1 = qpos_data_norm.shape
    qpos_data_norm = qpos_data_norm.view(B, T1 * D1)

    # Move to device
    qpos_data_norm = qpos_data_norm.to(device)
    image_data      = image_data.to(device)

    if use_tactile:
        tactile = data["tactile"]                          # [B, T2, D2]
        tactile_norm = tactile
        B_t, T2, D2 = tactile_norm.shape
        tactile_norm = tactile_norm.view(B_t, T2 * D2)      # → [B, T2 * D2]
        tactile_norm = tactile_norm.to(device)

        tactile_next = data["tactile_next"]                 # [B, T2, D2]
        tactile_next_norm = tactile_next
        tactile_next_norm = tactile_next_norm.to(device)

        a_hat = policy(qpos_data_norm, image_data,
                        device=device, tactile=tactile_norm,
                        tactile_next=tactile_next_norm)
    else:
        a_hat = policy(qpos_data_norm, image_data, device=device)

    return a_hat  # [B, T, D_action]



# ──────────────────────────────────────────────────────────────────────
# Metrics computation
# ──────────────────────────────────────────────────────────────────────
def compute_mse(pred, gt):
    """Overall MSE between predicted and ground-truth actions."""
    return float(np.mean((pred - gt) ** 2))


def compute_mae(pred, gt):
    """Overall MAE (L1) between predicted and ground-truth actions."""
    return float(np.mean(np.abs(pred - gt)))


def compute_mse_per_group(pred, gt, group_indices):
    """MSE for a specific joint group."""
    pred_group = pred[:, group_indices]
    gt_group = gt[:, group_indices]
    return float(np.mean((pred_group - gt_group) ** 2))


def compute_mae_per_group(pred, gt, group_indices):
    """MAE for a specific joint group."""
    pred_group = pred[:, group_indices]
    gt_group = gt[:, group_indices]
    return float(np.mean(np.abs(pred_group - gt_group)))


def compute_per_dof_mse(pred, gt):
    """MSE for each DOF individually. Returns array of shape [D]."""
    return np.mean((pred - gt) ** 2, axis=0)


def compute_per_timestep_mse(pred, gt):
    """MSE for each timestep in the prediction horizon. Returns array of shape [T]."""
    return np.mean((pred - gt) ** 2, axis=1)


# ──────────────────────────────────────────────────────────────────────
# Plotting functions
# ──────────────────────────────────────────────────────────────────────
def plot_predictions(pred_actions, gt_actions, sample_idx, output_dir):
    """
    Plot predicted vs ground-truth actions for each joint group.
    Solid blue = GT, dashed red = predicted.
    """
    n_steps = min(len(pred_actions), len(gt_actions))
    pred = pred_actions[:n_steps]
    gt = gt_actions[:n_steps]

    fig, axes = plt.subplots(5, 1, figsize=(16, 20), sharex=True)
    fig.suptitle(f"Open-Loop Inference — Sample {sample_idx}\n"
                 f"Predicted vs Ground-Truth Actions", fontsize=14, fontweight="bold")

    for ax, (group_name, indices) in zip(axes, JOINT_GROUPS.items()):
        for i, idx in enumerate(indices):
            ax.plot(range(n_steps), gt[:, idx], color="blue", alpha=0.5, linewidth=1)
            ax.plot(range(n_steps), pred[:, idx], color="red", alpha=0.7,
                    linewidth=1, linestyle="--")

        ax.set_ylabel(f"{group_name}\n({len(indices)} DOF)")
        ax.set_title(f"{group_name} — Blue=Ground Truth, Red=Predicted")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Action Step (Timestep in chunk)")
    plt.tight_layout()
    plot_path = Path(output_dir) / f"pred_vs_gt_sample{sample_idx}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def plot_per_dof_mse(per_dof_mse, output_dir):
    """Bar chart of per-DOF MSE, color-coded by joint group."""
    fig, ax = plt.subplots(figsize=(18, 5))
    dof_indices = list(range(len(per_dof_mse)))
    colors = []
    for idx in dof_indices:
        for group_name, indices in JOINT_GROUPS.items():
            if idx in indices:
                colors.append(JOINT_GROUP_COLORS[group_name])
                break

    ax.bar(dof_indices, per_dof_mse, color=colors, alpha=0.8)
    ax.set_xlabel("DOF Index")
    ax.set_ylabel("MSE")
    ax.set_title("Per-DOF MSE: Predicted vs Ground-Truth Actions")

    for group_name, indices in JOINT_GROUPS.items():
        start = indices[0]
        ax.text(start + len(indices) / 2 - 0.5, max(per_dof_mse) * 1.05,
                group_name, ha="center", fontsize=9, fontweight="bold")

    ax.set_xlim(-1, len(per_dof_mse))
    plt.tight_layout()
    plot_path = Path(output_dir) / "mse_per_dof.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def plot_per_timestep_mse(per_ts_mse, output_dir):
    """Line plot of MSE per timestep in the prediction horizon."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(range(len(per_ts_mse)), per_ts_mse, 'b-o', markersize=3, linewidth=1.5)
    ax.set_xlabel("Timestep in Prediction Horizon")
    ax.set_ylabel("MSE")
    ax.set_title("Per-Timestep MSE (Error Growth Over Horizon)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = Path(output_dir) / "mse_per_timestep.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def plot_group_mse_bar(group_mses, output_dir):
    """Grouped bar chart comparing MSE across joint groups."""
    fig, ax = plt.subplots(figsize=(10, 5))
    groups = list(group_mses.keys())
    mses = list(group_mses.values())
    colors = [JOINT_GROUP_COLORS[g] for g in groups]
    ax.bar(groups, mses, color=colors, alpha=0.8)
    ax.set_xlabel("Joint Group")
    ax.set_ylabel("MSE")
    ax.set_title("Per-Joint-Group MSE")
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plot_path = Path(output_dir) / "mse_per_group.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def plot_scatter_pred_vs_gt(pred_actions, gt_actions, sample_idx, output_dir):
    """Scatter plot of predicted vs GT for each joint group (ideal = diagonal)."""
    n_steps = min(len(pred_actions), len(gt_actions))
    pred = pred_actions[:n_steps]
    gt = gt_actions[:n_steps]

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f"Scatter: Predicted vs GT — Sample {sample_idx}", fontsize=14, fontweight="bold")

    for ax, (group_name, indices) in zip(axes, JOINT_GROUPS.items()):
        for i, idx in enumerate(indices):
            ax.scatter(gt[:, idx], pred[:, idx], s=8, alpha=0.5,
                       color=JOINT_GROUP_COLORS[group_name])
        # diagonal line
        all_vals = np.concatenate([gt[:, indices].flatten(), pred[:, indices].flatten()])
        lim_min, lim_max = all_vals.min(), all_vals.max()
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.5, linewidth=1)
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{group_name}")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = Path(output_dir) / f"scatter_pred_vs_gt_sample{sample_idx}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def plot_timing(inference_times, output_dir):
    """Bar chart of per-sample inference times."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(inference_times)), inference_times, color='tab:green', alpha=0.7)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Inference Time (s)")
    ax.set_title("Per-Sample Inference Time")
    mean_t = np.mean(inference_times)
    ax.axhline(y=mean_t, color='r', linestyle='--', alpha=0.7, label=f'Mean: {mean_t:.4f}s')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plot_path = Path(output_dir) / "timing_per_sample.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


# ──────────────────────────────────────────────────────────────────────
# Main inference & evaluation
# ──────────────────────────────────────────────────────────────────────
def get_episode_frame_indices(dataset, episode_idx):
    """
    Return the list of dataset frame indices that belong to the given episode.
    """
    indices = []
    for i in range(len(dataset)):
        frame_info = dataset.hf_dataset[i]
        if frame_info["episode_index"] == episode_idx:
            indices.append(i)
    return indices


def run_episode_inference(dataset, policy, device, use_tactile, episode_idx,
                          pred_horizon, output_dir):
    """
    Run inference over a single complete episode in non-overlapping chunks.

    For each chunk start frame, the model predicts the next `pred_horizon` actions.
    We save the predicted pose of every joint for each 64-step segment.

    Saves:
      - episode_pred_poses.npy   : [n_segments, pred_horizon, D]
      - episode_gt_poses.npy     : [n_segments, pred_horizon, D]
      - episode_segment_meta.json : per-segment metadata (start frame, etc.)
    """
    print(f"\n[Episode Inference] Filtering dataset for episode {episode_idx}...")
    ep_indices = get_episode_frame_indices(dataset, episode_idx)
    n_frames = len(ep_indices)
    print(f"  Episode {episode_idx}: {n_frames} frames")

    if n_frames == 0:
        print(f"  ERROR: No frames found for episode {episode_idx}")
        return

    # Non-overlapping chunks of pred_horizon
    chunk_starts = list(range(0, n_frames, pred_horizon))
    n_segments = len(chunk_starts)
    print(f"  Non-overlapping segments: {n_segments} (pred_horizon={pred_horizon})")

    all_pred_poses = []
    all_gt_poses = []
    segment_meta = []

    for seg_idx, start in enumerate(chunk_starts):
        end = min(start + pred_horizon, n_frames)
        actual_len = end - start
        frame_idx = ep_indices[start]

        print(f"  Segment {seg_idx+1}/{n_segments}: frame_idx={frame_idx}, "
              f"steps {start}..{end-1} ({actual_len} frames)")

        # Get the raw sample at this frame (the model uses delta_timestamps to
        # gather history + future actions)
        raw_sample = dataset[frame_idx]

        # Wrap single sample into a batch of 1
        raw_batch = {}
        for k, v in raw_sample.items():
            if isinstance(v, torch.Tensor):
                raw_batch[k] = v.unsqueeze(0)
            else:
                raw_batch[k] = v

        # Convert using the same convert_batch
        data = convert_batch(raw_batch, use_tactile=use_tactile,
                             delta_timestamps=DELTA_TIMESTAMPS)

        gt_actions = data["action"].cpu().numpy()  # [1, T, D]

        # Run model forward
        with torch.inference_mode():
            a_hat = origami_inference_forward(data, policy, device, use_tactile)

        pred_actions = a_hat.cpu().numpy()  # [1, T, D]

        # Take only the first pred_horizon predictions
        pred = pred_actions[0][:actual_len]  # [actual_len, D]
        gt = gt_actions[0][:actual_len]      # [actual_len, D]

        all_pred_poses.append(pred)
        all_gt_poses.append(gt)

        segment_meta.append({
            "segment_idx": seg_idx,
            "start_frame": start,
            "end_frame": end - 1,
            "n_frames": actual_len,
            "dataset_frame_idx": frame_idx,
        })

    # Stack into arrays
    # Note: last segment may be shorter, so pad to pred_horizon for uniform array
    D = all_pred_poses[0].shape[1]
    pred_padded = np.zeros((n_segments, pred_horizon, D), dtype=np.float32)
    gt_padded = np.zeros((n_segments, pred_horizon, D), dtype=np.float32)
    for i, (p, g) in enumerate(zip(all_pred_poses, all_gt_poses)):
        pred_padded[i, :len(p)] = p
        gt_padded[i, :len(g)] = g

    # Save numpy arrays
    pred_path = output_dir / f"episode{episode_idx}_pred_poses.npy"
    gt_path = output_dir / f"episode{episode_idx}_gt_poses.npy"
    np.save(pred_path, pred_padded)
    np.save(gt_path, gt_padded)
    print(f"\n  Saved predicted poses: {pred_path}  shape={pred_padded.shape}")
    print(f"  Saved ground-truth poses: {gt_path}  shape={gt_padded.shape}")

    # Save segment metadata + joint group info
    meta_path = output_dir / f"episode{episode_idx}_segment_meta.json"
    with open(meta_path, 'w') as f:
        json.dump({
            "episode_idx": episode_idx,
            "n_frames": n_frames,
            "n_segments": n_segments,
            "pred_horizon": pred_horizon,
            "state_dim": D,
            "joint_groups": {k: v for k, v in JOINT_GROUPS.items()},
            "joint_group_names": JOINT_GROUP_NAMES,
            "segments": segment_meta,
        }, f, indent=2)
    print(f"  Saved segment metadata: {meta_path}")

    # Also save a per-segment, per-joint-group summary CSV
    import csv
    csv_path = output_dir / f"episode{episode_idx}_segment_summary.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["segment_idx", "start_frame", "n_frames"]
        for gname in JOINT_GROUPS.keys():
            header.append(f"{gname}_mse")
        writer.writerow(header)
        for seg_idx, (pred, gt, meta) in enumerate(zip(all_pred_poses, all_gt_poses, segment_meta)):
            row = [seg_idx, meta["start_frame"], meta["n_frames"]]
            for gname, gindices in JOINT_GROUPS.items():
                mse = compute_mse_per_group(pred, gt, gindices)
                row.append(f"{mse:.6f}")
            writer.writerow(row)
    print(f"  Saved segment summary CSV: {csv_path}")

    # Plot: predicted vs GT for each joint group, concatenated across all segments
    fig, axes = plt.subplots(len(JOINT_GROUPS), 1, figsize=(20, 4 * len(JOINT_GROUPS)),
                             sharex=True)
    if len(JOINT_GROUPS) == 1:
        axes = [axes]
    # Build concatenated arrays with segment boundary markers
    total_steps = sum(len(p) for p in all_pred_poses)
    pred_concat = np.concatenate(all_pred_poses, axis=0)  # [total_steps, D]
    gt_concat = np.concatenate(all_gt_poses, axis=0)      # [total_steps, D]

    for ax, (gname, gindices) in zip(axes, JOINT_GROUPS.items()):
        for i, idx in enumerate(gindices):
            ax.plot(range(total_steps), gt_concat[:, idx], color="blue",
                    alpha=0.4, linewidth=0.8)
            ax.plot(range(total_steps), pred_concat[:, idx], color="red",
                    alpha=0.7, linewidth=0.8, linestyle="--")
        # Mark segment boundaries
        cum = 0
        for seg in all_pred_poses:
            cum += len(seg)
            ax.axvline(x=cum - 1, color='green', linestyle=':', alpha=0.5)
        ax.set_ylabel(f"{gname}\n({len(gindices)} DOF)")
        ax.set_title(f"{gname} — Blue=GT, Red=Pred (green dots = segment boundaries)")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Step (across all segments)")
    plt.suptitle(f"Episode {episode_idx} — Predicted vs GT Actions\n"
                 f"({n_segments} non-overlapping segments of {pred_horizon})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = output_dir / f"episode{episode_idx}_pred_vs_gt.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Saved episode plot: {plot_path}")

    # Print summary
    print(f"\n  Episode {episode_idx} Summary:")
    print(f"  {'Group':<15}  {'MSE':>10}")
    print(f"  {'-'*15}  {'-'*10}")
    for gname, gindices in JOINT_GROUPS.items():
        mse = compute_mse_per_group(pred_concat, gt_concat, gindices)
        print(f"  {gname:<15}  {mse:>10.6f}")
    overall_mse = compute_mse(pred_concat, gt_concat)
    print(f"  {'Overall':<15}  {overall_mse:>10.6f}")


def main():
    parser = argparse.ArgumentParser(description="Origami Inference & Open-Loop Evaluation")
    parser.add_argument('--ckpt_path', type=str, required=True, help='path to policy_*.ckpt')
    parser.add_argument('--stats_path', type=str, required=True, help='path to normalize.pkl')
    parser.add_argument('--use_tactile', action='store_true')
    parser.add_argument('--chunk_size', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dim_feedforward', type=int, default=3200)
    parser.add_argument('--state_dim', type=int, default=65)
    parser.add_argument('--n_samples', type=int, default=50,
                        help='Number of samples to evaluate (default: 50)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference (default: 1)')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Output directory for results and plots')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu (default: cuda)')
    # Episode-level inference mode
    parser.add_argument('--episode_idx', type=int, default=None,
                        help='If set, run inference over a single complete episode '
                             '(non-overlapping chunks). Overrides --n_samples mode.')
    parser.add_argument('--pred_horizon', type=int, default=64,
                        help='Number of steps to predict per segment in episode mode (default: 64)')
    args = parser.parse_args()


    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Origami Inference & Open-Loop Evaluation")
    print("=" * 70)
    print(f"  Checkpoint:  {args.ckpt_path}")
    print(f"  Stats:       {args.stats_path}")
    print(f"  Use tactile: {args.use_tactile}")
    if args.episode_idx is not None:
        print(f"  Mode:        Episode (idx={args.episode_idx}, pred_horizon={args.pred_horizon})")
    else:
        print(f"  N samples:   {args.n_samples}")
    print(f"  Output dir:   {output_dir}")
    print(f"  Device:       {args.device}")


    # ── Device ────────────────────────────────────────────────────────
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("  WARNING: CUDA not available, falling back to CPU")
        device = torch.device('cpu')

    # ── Build policy config & load model ──────────────────────────────
    policy_config = build_policy_config(args)
    policy = load_policy(args.ckpt_path, policy_config, device)

    # ── Load normalizer stats (for reference, not used in forward since training doesn't normalize) ──
    print(f"\n[Stats] Loading normalizer: {args.stats_path}")
    with open(args.stats_path, 'rb') as f:
        normalizer = pickle.load(f)
    print(f"  Normalizer type: {type(normalizer).__name__}")

    # ── Load dataset ──────────────────────────────────────────────────
    print("\n[Dataset] Loading origami dataset...")
    
    dataset = get_origami_full_dataset(
        dataset_root=INFERENCE_DATASET_ROOT,
        split="full", #NOTE: till i don't have train/test /val splits
        TOLERANCE=TOLERANCE,
        delta_timestamps=DELTA_TIMESTAMPS,
        use_tactile=args.use_tactile,
        max_duration_sec=MAXDURATION_IN_EPISODE_SEC,
        doImageTransforms=False,
    )

    print(f"  Dataset length: {len(dataset)}")

    # ── Episode-level inference mode ──────────────────────────────────
    # If --episode_idx is set, run inference over a single complete episode
    # in non-overlapping chunks of pred_horizon and save per-segment poses.
    if args.episode_idx is not None:
        run_episode_inference(
            dataset=dataset,
            policy=policy,
            device=device,
            use_tactile=args.use_tactile,
            episode_idx=args.episode_idx,
            pred_horizon=args.pred_horizon,
            output_dir=output_dir,
        )
        print(f"\n{'=' * 70}")
        print("  Episode inference done!")
        print(f"{'=' * 70}")
        return

    # ── Create dataloader ─────────────────────────────────────────────
    n_samples = min(args.n_samples, len(dataset))

    # Use sequential indices for reproducibility
    from torch.utils.data import Subset
    indices = list(range(n_samples))
    eval_dataset = Subset(dataset, indices)
    dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )
    print(f"  Evaluating on {n_samples} samples, batch_size={args.batch_size}")

    # ── Run inference ─────────────────────────────────────────────────
    print("\n[Inference] Running...")
    all_pred_actions = []
    all_gt_actions = []
    all_per_dof_mse = []
    all_per_ts_mse = []
    all_results = []
    inference_times = []
    data_load_times = []
    total_start = time.perf_counter()

    sample_idx = 0
    for batch_idx, raw_batch in enumerate(dataloader):
        # Time data loading + conversion
        t_data_start = time.perf_counter()
        data = convert_batch(raw_batch, use_tactile=args.use_tactile,
                             delta_timestamps=DELTA_TIMESTAMPS)
        t_data = time.perf_counter() - t_data_start
        data_load_times.append(t_data)

        gt_actions = data["action"].cpu().numpy()  # [B, T, D]

        # Time model forward
        t_fwd_start = time.perf_counter()
        with torch.inference_mode():
            a_hat = origami_inference_forward(data, policy, device, args.use_tactile)
        t_fwd = time.perf_counter() - t_fwd_start
        inference_times.append(t_fwd)

        pred_actions = a_hat.cpu().numpy()  # [B, T, D]

        # Process each sample in the batch
        B = pred_actions.shape[0]
        for b in range(B):
            pred = pred_actions[b]  # [T, D]
            gt = gt_actions[b]      # [T, D]

            n_pred = min(len(pred), len(gt))
            pred_trimmed = pred[:n_pred]
            gt_trimmed = gt[:n_pred]

            overall_mse = compute_mse(pred_trimmed, gt_trimmed)
            overall_mae = compute_mae(pred_trimmed, gt_trimmed)
            per_dof_mse = compute_per_dof_mse(pred_trimmed, gt_trimmed)
            per_ts_mse = compute_per_timestep_mse(pred_trimmed, gt_trimmed)
            all_per_dof_mse.append(per_dof_mse)
            all_per_ts_mse.append(per_ts_mse)

            group_mse = {}
            group_mae = {}
            for gname, gindices in JOINT_GROUPS.items():
                group_mse[gname] = compute_mse_per_group(pred_trimmed, gt_trimmed, gindices)
                group_mae[gname] = compute_mae_per_group(pred_trimmed, gt_trimmed, gindices)

            result = {
                "sample_idx": sample_idx,
                "overall_mse": overall_mse,
                "overall_mae": overall_mae,
                "group_mse": group_mse,
                "group_mae": group_mae,
                "n_pred_steps": n_pred,
                "inference_time_s": t_fwd,
                "data_load_time_s": t_data,
            }
            all_results.append(result)

            all_pred_actions.append(pred_trimmed)
            all_gt_actions.append(gt_trimmed)

            if (sample_idx + 1) % 10 == 0:
                print(f"  Processed {sample_idx + 1}/{n_samples} samples... "
                      f"(MSE={overall_mse:.6f}, MAE={overall_mae:.6f}, "
                      f"fwd={t_fwd:.4f}s)")

            # Plot first 3 samples
            if sample_idx < 3:
                plot_predictions(pred_trimmed, gt_trimmed, sample_idx, plots_dir)
                plot_scatter_pred_vs_gt(pred_trimmed, gt_trimmed, sample_idx, plots_dir)

            sample_idx += 1

    total_time = time.perf_counter() - total_start

    # ── Aggregate Results ─────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  EVALUATION RESULTS")
    print(f"{'=' * 70}")

    overall_mses = [r["overall_mse"] for r in all_results]
    overall_maes = [r["overall_mae"] for r in all_results]
    mean_mse = float(np.mean(overall_mses))
    std_mse = float(np.std(overall_mses))
    mean_mae = float(np.mean(overall_maes))
    std_mae = float(np.std(overall_maes))

    print(f"\n  Overall MSE: {mean_mse:.6f} ± {std_mse:.6f}")
    print(f"  Overall MAE: {mean_mae:.6f} ± {std_mae:.6f}")
    print(f"  Min MSE:     {float(np.min(overall_mses)):.6f}")
    print(f"  Max MSE:     {float(np.max(overall_mses)):.6f}")

    print(f"\n  Per-Joint-Group MSE (mean across all samples):")
    print(f"  {'Group':<15}  {'MSE':>10}  {'MAE':>10}")
    print(f"  {'-'*15}  {'-'*10}  {'-'*10}")
    mean_group_mse = {}
    mean_group_mae = {}
    for gname in JOINT_GROUPS.keys():
        group_mses = [r["group_mse"][gname] for r in all_results]
        group_maes = [r["group_mae"][gname] for r in all_results]
        mean_group_mse[gname] = float(np.mean(group_mses))
        mean_group_mae[gname] = float(np.mean(group_maes))
        print(f"  {gname:<15}  {mean_group_mse[gname]:>10.6f}  {mean_group_mae[gname]:>10.6f}")

    # Per-DOF MSE
    mean_per_dof_mse = np.mean(all_per_dof_mse, axis=0)
    print(f"\n  Per-DOF MSE (top 5 worst):")
    sorted_dof = np.argsort(mean_per_dof_mse)[::-1]
    for dof_idx in sorted_dof[:5]:
        group_name = None
        for gn, indices in JOINT_GROUPS.items():
            if dof_idx in indices:
                group_name = gn
                break
        print(f"    DOF {dof_idx:>2} ({group_name}): MSE = {mean_per_dof_mse[dof_idx]:.6f}")

    # ── Timing Summary ────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  TIMING SUMMARY")
    print(f"{'=' * 70}")
    total_fwd_time = float(np.sum(inference_times))
    total_data_time = float(np.sum(data_load_times))
    mean_fwd = float(np.mean(inference_times))
    mean_data = float(np.mean(data_load_times))
    throughput = n_samples / total_time if total_time > 0 else 0

    print(f"  Total samples:          {n_samples}")
    print(f"  Total wall time:        {total_time:.4f}s")
    print(f"  Total forward time:     {total_fwd_time:.4f}s")
    print(f"  Total data load time:   {total_data_time:.4f}s")
    print(f"  Mean forward time:      {mean_fwd:.4f}s / sample")
    print(f"  Mean data load time:    {mean_data:.4f}s / sample")
    print(f"  Throughput:             {throughput:.2f} samples/s")
    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"  Peak GPU memory:        {max_mem:.2f} GB")

    # ── Generate Plots ────────────────────────────────────────────────
    print(f"\n[Plots] Generating plots in {plots_dir}...")
    plot_per_dof_mse(mean_per_dof_mse, plots_dir)
    mean_per_ts_mse = np.mean(all_per_ts_mse, axis=0)
    plot_per_timestep_mse(mean_per_ts_mse, plots_dir)
    plot_group_mse_bar(mean_group_mse, plots_dir)
    plot_timing(inference_times, plots_dir)
    print(f"  Plots saved to {plots_dir}")

    # ── Save Results JSON ─────────────────────────────────────────────
    results_path = output_dir / "eval_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            "checkpoint": args.ckpt_path,
            "stats_path": args.stats_path,
            "use_tactile": args.use_tactile,
            "n_samples": n_samples,
            "chunk_size": CHUNK_SIZE,
            "state_dim": args.state_dim,
            "overall_mse": {
                "mean": mean_mse,
                "std": std_mse,
                "min": float(np.min(overall_mses)),
                "max": float(np.max(overall_mses)),
            },
            "overall_mae": {
                "mean": mean_mae,
                "std": std_mae,
            },
            "per_group_mse": mean_group_mse,
            "per_group_mae": mean_group_mae,
            "per_dof_mse": mean_per_dof_mse.tolist(),
            "per_timestep_mse": mean_per_ts_mse.tolist(),
            "timing": {
                "total_wall_time_s": total_time,
                "total_forward_time_s": total_fwd_time,
                "total_data_load_time_s": total_data_time,
                "mean_forward_time_s": mean_fwd,
                "mean_data_load_time_s": mean_data,
                "throughput_samples_per_s": throughput,
            },
            "samples": all_results,
        }, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    # ── Save Timing JSON ──────────────────────────────────────────────
    timing_path = output_dir / "timing.json"
    with open(timing_path, 'w') as f:
        json.dump({
            "inference_times_s": inference_times,
            "data_load_times_s": data_load_times,
            "total_wall_time_s": total_time,
            "total_forward_time_s": total_fwd_time,
            "total_data_load_time_s": total_data_time,
            "mean_forward_time_s": mean_fwd,
            "mean_data_load_time_s": mean_data,
            "throughput_samples_per_s": throughput,
            "n_samples": n_samples,
        }, f, indent=2)
    print(f"  Timing saved to: {timing_path}")

    print(f"\n{'=' * 70}")
    print("  Done!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
