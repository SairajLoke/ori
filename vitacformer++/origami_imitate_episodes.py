import torch
import numpy as np
import time
import os
import logging
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
import cv2
import json
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, DataLoader
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data.distributed import DistributedSampler

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torchvision.utils as vutils
import os
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter

from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from utils import unnormalize_image, normalize_action, denormalize_action, normalize_obs_lowdim, denormalize_obs_lowdim, normalize_tactile, denormalize_tactile, normalize_tactile_next, denormalize_tactile_next, apply_joint_mask
#TOOD add normin dataset transforms?
from pathlib import Path
from policy import ACTPolicy
from dataset.ha_pipelinev2_dataset import HaPipelineV2DatasetD020
from dataset.data import data
from dataset.data_tactile import data_tactile

from my_utils.normalizer import OriNormalizer, recommended_modes

# from visualize_episodes import save_videos

from dataset.origami_dataset import (
    get_origami_full_dataset,
    plan_train_val_episodes,
    convert_batch )

from lerobot.processor import  PolicyProcessorPipeline
from lerobot.policies.factory import make_pre_post_processors
# from lerobot.common.datasets import transforms
# RobotProcessorPipeline for actual h/w inference (unbatched), policy processingis for batched : 
# ref: https://huggingface.co/docs/lerobot/introduction_processors

from train_eval_utils import (JOINT_GROUPS, JOINT_GROUP_COLORS, _detailed_stats,
                              log_input_stats, log_final_model_inputs, build_action_dim_weights)
from my_utils.log_features import log_problematic_features
from my_utils.ori_logging import (setup_logging, get_logger, log_tensor, log_tensors,
                                  TRACE, StepGate)

log = get_logger("train")
_step_gate = StepGate(first_n=3, every=200)

def my_function():
    # Only loaded if this specific line is executed during a debug session
    import IPython
    IPython.embed() 



from configs import ( EPISODE_LEN, TOLERANCE, CAMERA_NAMES, STATE_DIM, LR_BACKBONE, BACKBONE, IS_ORIGAMI_TASK,
    FULL_DATASET, DELTA_TIMESTAMPS, CHUNK_SIZE, PROPRIOCEPTIVE_TEMPORAL_HORIZON, MASK_FINGERS, HAND_MASK, FPS, MAXDURATION_IN_EPISODE_SEC,
    MAX_EPISODES, VAL_EPISODES, VAL_EVERY_N_EPOCHS, BACKBONE_WEIGHTS, NORM_DISABLE_KEYS )


def print_time(s, e, name):
    print("="*50)
    print(" ", name, " : ", e-s, " sec")
    print("="*50)

def get_gpu_stats(device):
    """Return GPU memory allocated, reserved, and utilization as a dict."""
    if not torch.cuda.is_available():
        return None

    gpu_id = device.index if device.index is not None else 0

    # Memory stats from PyTorch
    mem_allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB
    mem_reserved  = torch.cuda.memory_reserved(device) / (1024 ** 3)   # GB
    max_mem       = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB

    # GPU utilization from nvidia-smi (subprocess call)
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.total,memory.used',
             '--format=csv,noheader,nounits', f'--id={gpu_id}'],
            capture_output=True, text=True, timeout=5
        )
        parts = result.stdout.strip().split(', ')
        gpu_util  = float(parts[0])            # %
        mem_total = float(parts[1]) / 1024      # GB
        mem_used  = float(parts[2]) / 1024      # GB
    except Exception:
        gpu_util, mem_total, mem_used = -1.0, -1.0, -1.0

    return {
        'gpu_util_pct': gpu_util,
        'mem_allocated_gb': mem_allocated,

        'mem_reserved_gb': mem_reserved,
        'mem_used_gb': mem_used,
        'mem_total_gb': mem_total,
        'max_mem_allocated_gb': max_mem,
    }


# ============================================================================
# ── Data Loader Timing Instrumentation ──────────────────────────────────────
# ============================================================================

class TimingDataLoader:
    """Wraps a DataLoader and times each __next__() call (raw data load time).

    This captures the full worker-side time: parquet read + video decode +
    collation + pin_memory + IPC transfer back to the main process.
    """

    def __init__(self, dataloader):
        self._dl = dataloader
        self.last_batch_time = 0.0

    def __iter__(self):
        self._iter = iter(self._dl)
        return self

    def __next__(self):
        _t = time.time()
        try:
            batch = next(self._iter)
        except StopIteration:
            self.last_batch_time = 0.0
            raise
        self.last_batch_time = time.time() - _t
        return batch

    def __len__(self):
        return len(self._dl)

    # Delegate attribute access to the underlying dataloader
    def __getattr__(self, name):
        return getattr(self._dl, name)


def _get_cpu_util():
    """Sample CPU utilization from /proc/stat. Returns utilisation % or None."""
    try:
        with open('/proc/stat', 'r') as f:
            line1 = f.readline()
        parts1 = list(map(int, line1.split()[1:]))
        idle1 = parts1[3] + parts1[4]  # idle + iowait
        total1 = sum(parts1)
        time.sleep(0.1)
        with open('/proc/stat', 'r') as f:
            line2 = f.readline()
        parts2 = list(map(int, line2.split()[1:]))
        idle2 = parts2[3] + parts2[4]
        total2 = sum(parts2)
        dt = total2 - total1
        di = idle2 - idle1
        if dt == 0:
            return None
        return round(100.0 * (1.0 - di / dt), 1)
    except Exception:
        return None


class BatchTimingLogger:
    """Collects per-batch timings and writes them to a log file.

    Logs a compact per-batch line and a periodic summary with percentiles.
    All output goes to the log file, not the terminal.
    """

    def __init__(self, log_path, summary_interval=50, device=None):
        self.log_path = log_path
        self.summary_interval = summary_interval
        self.records = []  # list of dicts, one per batch
        self._batch_count = 0
        self._device = device


    def log_batch(self, timings: dict, gpu_stats=None):
        """timings: dict with keys like 'dataloader', 'convert_norm', 'convert_resize',
        'convert_tactile', 'convert_total', 'transfer', 'forward', 'total'.
        gpu_stats: optional dict from get_gpu_stats() for memory logging."""
        self._batch_count += 1
        self.records.append(timings)

        # Per-batch line to file
        gpu_str = ""
        if gpu_stats is not None:
            gpu_str = (
                f" | GPU: util={gpu_stats['gpu_util_pct']:.0f}% "
                f"alloc={gpu_stats['mem_allocated_gb']:.1f}GB "
                f"used={gpu_stats['mem_used_gb']:.1f}/{gpu_stats['mem_total_gb']:.1f}GB "
                f"peak={gpu_stats['max_mem_allocated_gb']:.1f}GB"
            )
        with open(self.log_path, 'a') as f:
            f.write(
                f"[BATCH {self._batch_count - 1}] "
                f"dl={timings.get('dataloader', 0):.3f}s | "
                f"convert: norm={timings.get('convert_norm', 0):.3f}s "
                f"resize={timings.get('convert_resize', 0):.3f}s "
                f"tac={timings.get('convert_tactile', 0):.3f}s "
                f"total={timings.get('convert_total', 0):.3f}s | "
                f"xfer={timings.get('transfer', 0):.3f}s | "
                f"fwd={timings.get('forward', 0):.3f}s | "
                f"total={timings.get('total', 0):.3f}s"
                f"{gpu_str}\n"
            )


        # Periodic summary
        if self._batch_count % self.summary_interval == 0:
            self._write_summary()

    def _write_summary(self):
        if not self.records:
            return
        n = len(self.records)
        start = max(0, n - self.summary_interval)
        recent = self.records[start:]

        def stats(key):
            vals = sorted([r.get(key, 0) for r in recent])
            if not vals:
                return None
            avg = sum(vals) / len(vals)
            p50 = vals[len(vals) // 2]
            p95 = vals[int(len(vals) * 0.95)] if len(vals) >= 20 else vals[-1]
            return {
                'avg': avg, 'p50': p50, 'p95': p95,
                'min': vals[0], 'max': vals[-1]
            }

        cpu_util = _get_cpu_util()

        # GPU stats for summary
        gpu_stats = get_gpu_stats(self._device) if self._device is not None else None

        with open(self.log_path, 'a') as f:
            f.write(f"\n{'=' * 70}\n")
            f.write(f"[TIMING SUMMARY batches {start}-{n - 1}]\n")
            for key in ['dataloader', 'convert_total', 'convert_norm',
                        'convert_resize', 'convert_tactile',
                        'transfer', 'forward', 'total']:
                s = stats(key)
                if s:
                    f.write(
                        f"  {key:20s}: avg={s['avg']:.3f} "
                        f"p50={s['p50']:.3f} p95={s['p95']:.3f} "
                        f"min={s['min']:.3f} max={s['max']:.3f}\n"
                    )
            if cpu_util is not None:
                f.write(f"  CPU util: ~{cpu_util}%\n")
            if gpu_stats is not None:
                f.write(
                    f"  GPU: util={gpu_stats['gpu_util_pct']:.0f}% "
                    f"alloc={gpu_stats['mem_allocated_gb']:.1f}GB "
                    f"used={gpu_stats['mem_used_gb']:.1f}/{gpu_stats['mem_total_gb']:.1f}GB "
                    f"peak={gpu_stats['max_mem_allocated_gb']:.1f}GB\n"
                )
            f.write(f"{'=' * 70}\n\n")


    def finalize(self):
        """Write a final summary at end of training."""
        self._write_summary()


def visualize_batch(data, batch_idx, epoch, save_dir, use_tactile, max_samples=2):


    """
    Visualize first `max_samples` elements from a training batch after `convert_batch`.

    Produces separate sections per sample containing:
      - 4 camera images (side by side, larger)
      - Low-dim observation state: 5 line subplots (one per joint group)
      - Action chunk: 5 line subplots (one per joint group, with padding marker)
      - Tactile heatmap + mean-force line plot (if use_tactile)
      - Tactile_next heatmap (if use_tactile)

    Saves the figure to ``save_dir``.
    """
    os.makedirs(save_dir, exist_ok=True)

    image   = data["image"]         # [B, N_cam, 3, H, W]
    lowdim  = data["lowdim"]        # [B, T1, D1]
    action  = data["action"]        # [B, T, D_action]
    is_pad  = data["action_mask"]   # [B, T]

    n_samples = min(max_samples, image.shape[0])
    n_cam  = image.shape[1]
    cam_names = ["head_left", "head_right", "wrist_right", "wrist_left"]

    # --- Determine layout using GridSpec ---
    n_joint_groups = len(JOINT_GROUPS)  # 5
    n_rows = n_samples * (1 + n_joint_groups + n_joint_groups)  # per sample: cameras + lowdim groups + action groups
    if use_tactile:
        n_rows += n_samples * 3  # tactile heatmap, tactile mean, tactile_next heatmap

    fig = plt.figure(figsize=(18, 4 * n_rows))
    gs = fig.add_gridspec(n_rows, 1, hspace=0.5)

    row = 0

    # --- Visualize each sample ---
    for s in range(n_samples):
        # ---- Camera row: 4 cameras side by side (larger) ----
        ax = fig.add_subplot(gs[row])
        row += 1
        cam_imgs = image[s].cpu().float()  # [N_cam, 3, H, W]
        cam_imgs = (cam_imgs - cam_imgs.min()) / (cam_imgs.max() - cam_imgs.min() + 1e-6)
        grid = vutils.make_grid(cam_imgs, nrow=n_cam, normalize=False, padding=8, pad_value=1)
        ax.imshow(grid.permute(1, 2, 0).numpy())
        ax.set_title(f"Sample {s} | Cameras: {', '.join(cam_names[:n_cam])}", fontsize=11, fontweight='bold')
        ax.axis("off")

        # ---- Lowdim: 5 subplots (one per joint group) ----
        ld = lowdim[s].cpu().numpy()  # [T1, D1]
        for group_name, indices in JOINT_GROUPS.items():
            ax = fig.add_subplot(gs[row])
            row += 1
            ld_group = ld[:, indices]  # [T1, len(indices)]
            for j in range(ld_group.shape[1]):
                ax.plot(range(ld_group.shape[0]), ld_group[:, j],
                        marker='o', markersize=2, alpha=0.6, label=f"j{indices[j]}")
            ax.set_title(f"Sample {s} | Lowdim — {group_name} ({len(indices)} DOF)", fontsize=9)
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            if len(indices) <= 10:
                ax.legend(fontsize=5, loc='best', ncol=2)

        # ---- Action: 5 subplots (one per joint group) ----
        act = action[s].cpu().numpy()  # [T, D_action]
        pad = is_pad[s].cpu().numpy()  # [T]
        pad_start = int(np.argmax(pad)) if pad.any() else act.shape[0]

        for group_name, indices in JOINT_GROUPS.items():
            ax = fig.add_subplot(gs[row])
            row += 1
            act_group = act[:, indices]  # [T, len(indices)]
            for j in range(act_group.shape[1]):
                ax.plot(range(act_group.shape[0]), act_group[:, j],
                        alpha=0.6, linewidth=0.8, label=f"j{indices[j]}")
            if pad_start < act.shape[0]:
                ax.axvline(x=pad_start - 0.5, color='red', linestyle='--', alpha=0.7,
                           label=f'pad@{pad_start}')
            ax.set_title(f"Sample {s} | Action — {group_name} ({len(indices)} DOF)", fontsize=9)
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            if len(indices) <= 10:
                ax.legend(fontsize=5, loc='best', ncol=2)

        if use_tactile:
            # ---- Tactile heatmap [18, 120] (deltas ⊕ next) ----
            ax = fig.add_subplot(gs[row])
            row += 1
            tac = data["tactile"][s]
            if tac.dim() == 1:
                tac = tac.reshape(18, -1)
            tac_np = tac.cpu().numpy()
            im3 = ax.imshow(tac_np, aspect='auto', cmap='coolwarm')
            ax.set_title(f"Sample {s} | Tactile (past⊕deltas) [{tac_np.shape[0]}, {tac_np.shape[1]}]", fontsize=9)
            ax.set_xlabel("Feature dim (0-59: past | 60-119: deltas)")
            ax.set_ylabel("Timestep (0-17)")
            ax.axvline(x=59.5, color='yellow', linestyle='--', alpha=0.5, label='past | deltas')
            ax.legend(fontsize=5)
            plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)

            # ---- Tactile mean force over time (line plot) ----
            ax = fig.add_subplot(gs[row])
            row += 1
            past_mean = tac_np[:, :60].mean(axis=1)
            delta_mean = tac_np[:, 60:120].mean(axis=1)
            ax.plot(range(tac_np.shape[0]), past_mean, 'b-o', markersize=3, label='past mean')
            ax.plot(range(tac_np.shape[0]), delta_mean, 'r-s', markersize=3, label='delta mean')
            ax.set_title(f"Sample {s} | Tactile mean force over time", fontsize=9)
            ax.set_xlabel("Timestep (0-17)")
            ax.set_ylabel("Mean value")
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)

            # ---- Tactile_next heatmap ----
            ax = fig.add_subplot(gs[row])
            row += 1
            tac_next = data["tactile_next"][s]
            if tac_next.dim() == 1:
                tac_next = tac_next.reshape(18, -1)
            tac_next_np = tac_next.cpu().numpy()
            im4 = ax.imshow(tac_next_np, aspect='auto', cmap='coolwarm')
            ax.set_title(f"Sample {s} | Tactile_next (target) [{tac_next_np.shape[0]}, {tac_next_np.shape[1]}]", fontsize=9)
            ax.set_xlabel("Feature dim")
            ax.set_ylabel("Timestep (0-17)")
            plt.colorbar(im4, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Epoch {epoch} | Batch {batch_idx} | Samples 0-{n_samples-1}", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out_path = os.path.join(save_dir, f"epoch{epoch}_batch{batch_idx}_samples0-{n_samples-1}.png")
    fig.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"[Viz] Saved → {out_path}")



def main(args):
    set_seed(1)

    # Validate mutually exclusive parameters
    resume_path = args['resume_path']
    load_pretrained_for_newtraining = args['load_pretrained_for_newtraining']
    if resume_path is not None and load_pretrained_for_newtraining is not None:
        raise ValueError(
            "ERROR: Cannot use both --load_pretrained_for_newtraining and --resume_path at the same time!\n"
            "Choose one:\n"
            "  --load_pretrained_for_newtraining <path>  : Load pretrained model, start training from scratch\n"
            "  --resume_path <path>                      : Resume training from checkpoint (continues from last epoch/step)\n"
        )

    if resume_path is None and load_pretrained_for_newtraining is None:
        import warnings
        warnings.warn(
            "No --load_pretrained_for_newtraining or --resume_path provided. "
            "Training from scratch (random init).",
            stacklevel=2
        )


    # command line parameters

    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']
    use_tactile = args['use_tactile']
    visualize_batch_flag = args.get('visualize_batch', False)
    visualize_batch_dir  = args.get('visualize_batch_dir', None)
    visualize_n_batches  = args.get('visualize_n_batches', 3)
    doImageTransforms = args.get('doImageTransforms', False)
    disable_normalization = args.get('disable_normalization', False)
    expt_name = args['expt_name']

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = expt_name + '_' + timestamp
    ckpt_dir = os.path.join(ckpt_dir, timestamp)

    if IS_ORIGAMI_TASK:
        ckpt_dir = ckpt_dir + "_ori"
        timestamp = timestamp + "_ori"

    if use_tactile:
        ckpt_dir = ckpt_dir + "_tactile"
        timestamp = timestamp + "_tactile"
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- logging: configure as early as ckpt_dir is known ---
    # accelerate has already exported RANK by the time main() runs.
    _rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    setup_logging(rank=_rank, log_dir=ckpt_dir)
    log.info("=" * 70)
    log.info("run start: expt=%s  ckpt_dir=%s", expt_name, ckpt_dir)
    log.info("=" * 70)
    log.info("CLI args:")
    for _k in sorted(args):
        log.info("  %-34s %s", _k, args[_k])
    log.info("configs.py:")
    log.info("  IS_ORIGAMI_TASK=%s FPS=%s CHUNK_SIZE=%s STATE_DIM=%s BACKBONE=%s LR_BACKBONE=%s",
             IS_ORIGAMI_TASK, FPS, CHUNK_SIZE, STATE_DIM, BACKBONE, LR_BACKBONE)
    log.info("  MASK_FINGERS=%s  MAXDURATION_IN_EPISODE_SEC=%s  PROPRIO_HORIZON=%s",
             MASK_FINGERS, MAXDURATION_IN_EPISODE_SEC, PROPRIOCEPTIVE_TEMPORAL_HORIZON)
    log.info("  FULL_DATASET=%s", FULL_DATASET)
    for _k, _v in DELTA_TIMESTAMPS.items():
        log.info("  DELTA_TIMESTAMPS[%-22s] %d offsets, frame idx %s%s",
                 _k, len(_v), [round(t * FPS) for t in _v[:6]],
                 " ..." if len(_v) > 6 else "")

    # episode_len = 10000
    # camera_names = ['/observe/vision/head/stereo/lefteye/rgb',
    #                 '/observe/vision/head/stereo/righteye/rgb',
    #                 '/observe/vision/right_wrist/fisheye/rgb',
    #                 '/observe/vision/left_wrist/fisheye/rgb']
    
    

    # fixed parameters
    # state_dim = 58
    # lr_backbone = 1e-5
    # backbone = 'resnet18'
    
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': CHUNK_SIZE, #args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': LR_BACKBONE,
                         'backbone': BACKBONE,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': CAMERA_NAMES, #TODO check this order in list
                         'use_tactile': use_tactile,
                         'state_dim': args['state_dim'],
                         'proprioceptive_temporal_horizon': PROPRIOCEPTIVE_TEMPORAL_HORIZON,
                         'backbone_weights': BACKBONE_WEIGHTS,
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': LR_BACKBONE, 'backbone' : LR_BACKBONE, 'num_queries': 1,
                         'camera_names': CAMERA_NAMES,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'batch_size_train': batch_size_train,
        'ckpt_dir': ckpt_dir,
        'episode_len': EPISODE_LEN, #TODO: not sure wheres it is used 
        'state_dim': STATE_DIM,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': CAMERA_NAMES,
        # 'real_robot': not is_sim,
        'use_tactile': use_tactile,
        'resume_path': resume_path,
        'load_pretrained_for_newtraining': load_pretrained_for_newtraining,
        'visualize_batch': visualize_batch_flag,
        'visualize_batch_dir': visualize_batch_dir or os.path.join(ckpt_dir, 'batch_viz'),
        'visualize_n_batches': visualize_n_batches,
        'lr_config': {
            'policy': 'CosineAnnealing',
            'warmup': 'linear',
            # Warmup as a FRACTION of the total schedule, not an absolute step
            # count: 500 fixed steps meant something completely different at
            # 50 episodes vs 500, and at 1 GPU vs 8. Resolved to an absolute
            # number once total_iters is known (see train_bc).
            'warmup_ratio_of_total': float(os.environ.get('WARMUP_RATIO', '0.03')),
            'warmup_iters_min': 100,
            'min_lr_ratio': 1e-1,
        },
        'ckpt_save_epochs': args['ckpt_save_epochs'],
        'tb_log_freq': args['tb_log_freq'],
        'doImageTransforms':  doImageTransforms,
        'disable_normalization': disable_normalization,
        'max_train_steps': args.get('max_train_steps'),
        'max_val_steps': args.get('max_val_steps'),
    }
    
    
    info_log_path = os.path.join(ckpt_dir, f'info_{os.path.basename(ckpt_dir)}.log')
    with open(info_log_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("TRAINING INFO LOG\n")
        f.write("=" * 70 + "\n\n")

        f.write("--- CLI Args ---\n")
        import json
        args_formatted = json.dumps(args, indent=2, default=str)
        f.write(args_formatted + "\n\n")

        f.write("--- System Info ---\n")
        f.write(f"CPU count: {os.cpu_count()}\n")
        f.write(f"CUDA available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
            f.write(f"GPU count: {torch.cuda.device_count()}\n")
        f.write("\n")

        f.write("--- Config ---\n")
        f.write(f"IS_ORIGAMI_TASK: {IS_ORIGAMI_TASK}\n")
        f.write(f"FPS: {FPS}\n")
        f.write(f"CHUNK_SIZE: {CHUNK_SIZE}\n")
        f.write(f"STATE_DIM: {STATE_DIM}\n")
        f.write(f"BACKBONE: {BACKBONE}\n")
        f.write(f"LR_BACKBONE: {LR_BACKBONE}\n")
        f.write(f"CAMERA_NAMES: {CAMERA_NAMES}\n")
        f.write(f"TOLERANCE: {TOLERANCE}\n")
        f.write(f"MASK_FINGERS: {MASK_FINGERS}\n")
        f.write(f"HAND_MASK: {HAND_MASK}\n")
        f.write(f"MAXDURATION_IN_EPISODE_SEC: {MAXDURATION_IN_EPISODE_SEC}\n")
        f.write(f"PROPRIOCEPTIVE_TEMPORAL_HORIZON: {PROPRIOCEPTIVE_TEMPORAL_HORIZON}\n")
        f.write(f"DELTA_TIMESTAMPS: {DELTA_TIMESTAMPS}\n")
        f.write(f"FULL_DATASET: {FULL_DATASET}\n")
        # f.write(f"SEASONS: {SEASONS}\n")
        f.write("\n")

        f.write("--- Policy Config ---\n")
        for k, v in policy_config.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        f.write("--- Training Config ---\n")
        for k, v in config.items():
            if k == 'policy_config':
                continue
            f.write(f"  {k}: {v}\n")
        f.write("\n")



    # Device will be set by Accelerator, but keep a fallback for dataset loading
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(os.cpu_count())

    norm_stats_cache = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    # data['train']['norm_stats_cache'] = norm_stats_cache
    # data['val']['norm_stats_cache']   = norm_stats_cache
    # data_tactile['train']['norm_stats_cache'] = norm_stats_cache
    # data_tactile['val']['norm_stats_cache']   = norm_stats_cache #TODO

    train_dataset = None
    val_dataset = None
    val_dataloader = None
    normalizer = None

    if IS_ORIGAMI_TASK:
        # Set env var so origami_dataset.py can log active keys to the info log file
        os.environ["ORI_INFO_LOG_PATH"] = info_log_path

        # --- Episode-level train/val split (never frame-level: adjacent frames
        #     are near-duplicates and would leak the val set into training) ---
        train_eps, val_eps = plan_train_val_episodes(
            dataset_root=FULL_DATASET,
            max_episodes=MAX_EPISODES,
            val_episodes=VAL_EPISODES,
        )
        config['train_episodes'] = train_eps
        config['val_episodes'] = val_eps

        train_dataset = get_origami_full_dataset(
            dataset_root=FULL_DATASET,
            split= "full",
            TOLERANCE= TOLERANCE,
            delta_timestamps=DELTA_TIMESTAMPS,
            use_tactile=use_tactile,
            max_duration_sec= MAXDURATION_IN_EPISODE_SEC  , #NOTE:;;; becareful
            doImageTransforms = config.get('doImageTransforms', False),
            episodes=train_eps,
            tag="train",
        )

        if val_eps:
            val_dataset = get_origami_full_dataset(
                dataset_root=FULL_DATASET,
                split="full",
                TOLERANCE=TOLERANCE,
                delta_timestamps=DELTA_TIMESTAMPS,
                use_tactile=use_tactile,
                max_duration_sec=MAXDURATION_IN_EPISODE_SEC,
                doImageTransforms=False,   # never augment the val set
                episodes=val_eps,
                tag="val",
            )

        log.info("dataset type=%s  train=%d frames  val=%s frames",
                 type(train_dataset).__name__, len(train_dataset),
                 len(val_dataset) if val_dataset is not None else "-")
        log.info("dataset stats keys: %s", sorted(train_dataset.meta.stats.keys()))

        # --- Per-dimension action loss weights (needs dataset stats) ---
        # policy_config is the same dict object train_bc passes to make_policy,
        # so mutating it here reaches the policy.
        _group_weights = None
        if args.get('loss_group_weights'):
            _group_weights = json.loads(args['loss_group_weights'])
        _dim_weights, _dropped = build_action_dim_weights(
            state_dim=args['state_dim'],
            mode=args.get('loss_dim_weight_mode', 'uniform'),
            group_weights=_group_weights,
            action_stats=(train_dataset.meta.stats.get('action')
                          if args.get('drop_degenerate_action_dims') else None),
        )
        _is_uniform = (args.get('loss_dim_weight_mode', 'uniform') == 'uniform' and not _dropped)
        policy_config['action_dim_weights'] = None if _is_uniform else _dim_weights
        policy_config['tac_weight'] = args.get('tac_weight', 1.0)
        config['loss_dim_weight_mode'] = args.get('loss_dim_weight_mode', 'uniform')
        config['loss_group_weights'] = _group_weights
        config['action_dim_weights'] = policy_config['action_dim_weights']
        config['tac_weight'] = policy_config['tac_weight']
        if _dropped:
            log.info("action dims zeroed in the L1 loss (constant in this dataset): %s", _dropped)
        log.info("action loss weighting: mode=%s group_weights=%s tac_weight=%s",
                 config['loss_dim_weight_mode'], _group_weights, config['tac_weight'])


        # NOTE: Do NOT create DistributedSampler manually here.
        # accelerator.prepare() will automatically add one when shuffle=True.
        # Auto-scale num_workers relative to CPU cores and GPU count
        _num_cpus = os.cpu_count() or 64
        _num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        _auto_workers = max(4, (_num_cpus - 6) // _num_gpus)
        log.info("DataLoader: CPUs=%d GPUs=%d -> num_workers=%d (batch_size=%d, prefetch_factor=10, "
                 "persistent_workers=True, pin_memory=True, drop_last=True, shuffle=True)",
                 _num_cpus, _num_gpus, _auto_workers, batch_size_train)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size_train, 
            shuffle=True,             # accelerator.prepare() will wrap with DistributedSampler
            pin_memory=True,          #impt ~1.2 speedup
            num_workers=_auto_workers,
            persistent_workers=True,    # gives 2.x speed from my observations
            prefetch_factor=10,        # increased from 6 — more buffer to smooth video decode spikes

            drop_last=True,           # important for DDP — avoids uneven batch sizes
        )

        if val_dataset is not None:
            # Deliberately NOT passed to accelerator.prepare(): validation runs
            # on the main process only against the unwrapped model, which keeps
            # the metric exact (no DDP batch padding / duplicate-sample
            # de-duplication to reason about) at the cost of the other ranks
            # idling for a few seconds every VAL_EVERY_N_EPOCHS epochs.
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size_val,
                shuffle=False,
                pin_memory=True,
                num_workers=max(2, _auto_workers // 4),
                persistent_workers=True,
                prefetch_factor=4,
                drop_last=False,
            )
            log.info("val DataLoader: %d frames, %d batches, num_workers=%d (main process only, every %d epochs)",
                     len(val_dataset), len(val_dataloader),
                     val_dataloader.num_workers, VAL_EVERY_N_EPOCHS)





        
         
    else:
        if not use_tactile:
            train_dataset = HaPipelineV2DatasetD020(**data['train'])
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=False, num_workers=4) #, BATCH_SIZE), prefetch_factor=1)

            # val_dataset = HaPipelineV2DatasetD020(**data['val'])
            # val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=False, num_workers=36, prefetch_factor=1)
        else:
            train_dataset = HaPipelineV2DatasetD020(**data_tactile['train'])
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=False, num_workers=4) #, BATCH_SIZE), prefetch_factor=1)

            # val_dataset = HaPipelineV2DatasetD020(**data_tactile['val'])
            # val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=False, num_workers=36, prefetch_factor=1)

        normalizer = train_dataset.get_normalizer()
    #----------------------------------------------------------------

    # === Log dataset & dataloader info to info.log ===
    with open(info_log_path, 'a') as f:
        f.write("--- Dataset Info ---\n")
        f.write(f"IS_ORIGAMI_TASK: {IS_ORIGAMI_TASK}\n")
        if IS_ORIGAMI_TASK:
            f.write(f"Full Dataset path : {FULL_DATASET} ")
            f.write(f"Dataset length (total frames): {len(train_dataset)}\n")
            try:
                f.write(f"Dataset stats keys: {list(train_dataset.meta.stats.keys())}\n")
            except Exception:
                f.write("Dataset stats keys: (could not retrieve)\n")
            try:
                f.write(f"observation.state feature: {train_dataset.meta.features.get('observation.state', 'N/A')}\n")
            except Exception:
                f.write("observation.state feature: (could not retrieve)\n")
        else:
            f.write(f"Dataset type: HaPipelineV2DatasetD020\n")
            f.write(f"Dataset length: {len(train_dataset)}\n")

        f.write(f"\n--- Dataloader Info ---\n")
        f.write(f"batch_size: {batch_size_train}\n")
        f.write(f"num_workers: {train_dataloader.num_workers}\n")
        f.write(f"pin_memory: {train_dataloader.pin_memory}\n")
        f.write(f"shuffle: {not train_dataloader.sampler is None}\n")
        try:
            f.write(f"persistent_workers: {train_dataloader.persistent_workers}\n")
        except Exception:
            f.write(f"persistent_workers: (N/A)\n")
        try:
            f.write(f"prefetch_factor: {train_dataloader.prefetch_factor}\n")
        except Exception:
            f.write(f"prefetch_factor: (N/A)\n")
        f.write(f"batches per epoch: {len(train_dataloader)}\n")
        f.write("\n")

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'normalize.pkl')
    # with open(stats_path, 'wb') as f:
    #     pickle.dump(normalizer, f)  #pickle.dump(preprocessor,f) if IS_ORIGAMI_TASK else 
    
    
    log.info("batches per epoch (pre-shard) = %d", len(train_dataloader))
    # with tqdm(train_dataloader, desc=f"Sanity Check Train Epoch {-1}", leave=False) as tepoch:
    #         for batch_idx, data in enumerate(tepoch): 
    #             continue
    time.sleep(2)
    
    #=======================================================
    # policy_best.ckpt is written inside train_bc whenever the val loss improves,
    # so nothing more to save here.
    best_ckpt_info = train_bc(train_dataloader, normalizer, train_dataset, timestamp,
                              config, old_device=None, info_log_path=info_log_path,
                              val_dataloader=val_dataloader)

    if best_ckpt_info is not None:
        best_epoch, best_val_loss, _ = best_ckpt_info
        log.info("best checkpoint: epoch %s, val loss %.6f", best_epoch, best_val_loss)
    #=======================================================



def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer

def old_forward_pass(data, policy, normalizer, device, use_tactile, epoch=0):
    image_data = data["image"]               # [B, N_cam, 3, H, W]
    qpos_data = data["lowdim"]               # [B, T1, D1]
    action_data = data["action"]            # [B, T, D_action]
    is_pad = data["action_mask"]            # [B, T]

    # normalize
    qpos_data_norm = normalize_obs_lowdim(qpos_data, normalizer)  # [B, T1, D1]
    action_data_norm = normalize_action(action_data, normalizer)  # [B, T, D_action]

    # === apply masking to hand joint
    hand_mask = [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]

    # right_hand mask
    qpos_data_norm = apply_joint_mask(qpos_data_norm, hand_mask, start_index=7)

    # left_hand mask
    qpos_data_norm = apply_joint_mask(qpos_data_norm, hand_mask, start_index=35)

    # flatten
    B, T1, D1 = qpos_data_norm.shape
    qpos_data_norm = qpos_data_norm.view(B, T1 * D1)  # → [B, T1 * D1]

    # move to device
    qpos_data_norm = qpos_data_norm.to(device)
    image_data = image_data.to(device)
    action_data_norm = action_data_norm.to(device)
    is_pad = is_pad.to(device)

    if use_tactile:
        tactile = data["tactile"]                          # [B, T2, D2]
        tactile_norm = normalize_tactile(tactile, normalizer)  # normalize
        B, T2, D2 = tactile_norm.shape
        tactile_norm = tactile_norm.view(B, T2 * D2)                # → [B, T2 * D2]
        tactile_norm = tactile_norm.to(device)                     

        tactile_next = data["tactile_next"]                          # [B, T2, D2]
        tactile_next_norm = normalize_tactile_next(tactile_next, normalizer)  # normalize
        tactile_next_norm = tactile_next_norm.to(device)        
        


        return policy(qpos_data_norm, image_data, action_data_norm, is_pad, device, tactile_norm, tactile_next_norm, epoch)

    return policy(qpos_data_norm, image_data, action_data_norm, is_pad, device)


def origami_forward_pass(data, policy, normalizer, device, use_tactile, epoch=0, log_final_inputs=False,
                         writer=None, global_step=0, return_a_hat=False):
    image_data = data["image"]               # [B, N_cam, 3, H, W]
    qpos_data = data["lowdim"]               # [B, T1, D1]
    action_data = data["action"]            # [B, T, D_action]
    is_pad = data["action_mask"]            # [B, T]


    #------------------------------------ NORMALIZE ??? #NOTE: ------------------------------------------------\
    qpos_data_norm = qpos_data
    action_data_norm = action_data

    # qpos_data_norm = normalize_obs_lowdim(qpos_data, normalizer)
    # action_data_norm = normalize_action(action_data, normalizer)  # [B, T, D_action]

    # qpos_data_norm   = normalizer.normalize("observation.state", qpos_data)  # [B, T1, D1]
    # action_data_norm = normalizer.normalize("action", action_data)           # [B, T, D_action]
    #---------------------------------------------------------------------------------------------------------

    # === apply masking to hand joint
    # hand_mask = [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    if MASK_FINGERS:
        print("before masking : qpos data ", qpos_data_norm)
        qpos_data_norm = apply_joint_mask(qpos_data_norm, HAND_MASK, start_index=7)
        qpos_data_norm = apply_joint_mask(qpos_data_norm, HAND_MASK, start_index=7+22+7) #NOTE: confirm this
        print("after masking : qpos data ", qpos_data_norm)



    # right_hand mask
    # qpos_data_norm = apply_joint_mask(qpos_data_norm, hand_mask, start_index=7)

    # left_hand mask
    # qpos_data_norm = apply_joint_mask(qpos_data_norm, hand_mask, start_index=35)
    #------------------------------------------------------------------------------------

    # flatten
    B, T1, D1 = qpos_data_norm.shape
    qpos_data_norm = qpos_data_norm.view(B, T1 * D1)  # → [B, T1 * D1]

    # move to device...redundant since normalizer 
    # non_blocking=True enables async H2D copy overlap (pin_memory=True on DataLoader)
    qpos_data_norm   = qpos_data_norm.to(device)#, non_blocking=True)
    image_data       = image_data.to(device)# , non_blocking=True)
    action_data_norm = action_data_norm.to(device)#, non_blocking=True)
    is_pad = is_pad.to(device)# , non_blocking=True)

    if use_tactile:
        tactile = data["tactile"]                          # [B, T2, D2]
        tactile_norm = tactile
        # tactile_norm = normalizer.normalize("observation.tactile", tactile)
        # tactile_norm = normalize_tactile(tactile, normalizer)  # normalize
        B, T2, D2 = tactile_norm.shape
        tactile_norm = tactile_norm.view(B, T2 * D2)                # → [B, T2 * D2]
        tactile_norm = tactile_norm.to(device)# , non_blocking=True)                     #TODO sure if the view is correct here, maybe we should keep the tactile as [B, T2, D2] for the model to process it as a sequence

        tactile_next = data["tactile_next"]                          # [B, T2, D2]
        tactile_next_norm = tactile_next
        # tactile_next_norm = normalizer.normalize("observation.tactile_next", tactile_next)
        # tactile_next_norm = normalize_tactile_next(tactile_next, normalizer)  # normalize
        tactile_next_norm = tactile_next_norm.to(device)# , non_blocking=True)

        # Padding mask for the tactile target (built in convert_batch from
        # observation.tactile_is_pad). Used only for the l1_tac denominator.
        tactile_next_pad = data.get("tactile_next_mask", None)
        if tactile_next_pad is not None:
            tactile_next_pad = tactile_next_pad.to(device)

        if log_final_inputs:
            log_final_model_inputs(qpos_data_norm, image_data, action_data_norm, is_pad,
                                   tactile_norm, tactile_next_norm, writer, global_step)

        return policy(qpos_data_norm, image_data, action_data_norm, is_pad, device,
                      tactile_norm, tactile_next_norm,
                      tactile_next_pad=tactile_next_pad, epoch=epoch,
                      return_a_hat=return_a_hat)

    if log_final_inputs:
        log_final_model_inputs(qpos_data_norm, image_data, action_data_norm, is_pad,
                               None, None, writer, global_step)

    return policy(qpos_data_norm, image_data, action_data_norm, is_pad, device,
                  return_a_hat=return_a_hat)



@torch.no_grad()
def origami_validate(val_dataloader, policy, normalizer, device, use_tactile, epoch,
                     max_steps=None):
    """Run the held-out episodes and return a dict of scalar metrics.

    `policy` must be the UNWRAPPED module (accelerator.unwrap_model), and this
    should only be called on the main process -- see the val DataLoader comment
    in main() for why.

    Two different things are reported, in two different spaces, deliberately:

      val/*     the training losses, in NORMALIZED space -- identical formula to
                training, so val and train are directly comparable within a run.
      val_l1/*  per-joint-group mean |error| in PHYSICAL units (radians), after
                denormalizing both the prediction and the target.

    val_l1 has to be denormalized or it is meaningless across runs: the
    per-dimension scale factor is 2/(q99-q01), which on this dataset ranges from
    0.94x to 82x across the 65 action dims. Without this, a val_l1 from an
    unnormalized run and one from a normalized run would be in different units
    and could not be compared -- which is the entire point of running the
    normalization variants. In physical units they are also directly comparable
    against a physical baseline (e.g. "copy the current pose", ~0.044 rad).

    With normalization off, denormalize() is identity, so nothing changes.
    """
    policy.eval()

    # Accumulate sums + counts so the mean is exact regardless of batch sizes.
    sums = {}
    n_valid_steps = 0.0
    n_batches = 0

    for batch_idx, data in enumerate(tqdm(val_dataloader, desc=f"Val epoch {epoch}", leave=False)):
        if max_steps is not None and batch_idx >= max_steps:
            break
        data = convert_batch(data, use_tactile=use_tactile, delta_timestamps=DELTA_TIMESTAMPS,
                             epoch=epoch, batch_idx=batch_idx, normalizer=normalizer)
        data.pop("_timing", None)

        forward_dict, a_hat = origami_forward_pass(
            data, policy, normalizer, device, use_tactile, epoch=epoch, return_a_hat=True)

        actions = data["action"][:, :policy.model.num_queries].to(device)
        is_pad = data["action_mask"][:, :policy.model.num_queries].to(device)

        # Back to physical units before measuring per-joint error. The loss
        # terms above stay in normalized space (that is what the model is
        # trained on); only the reported metric is converted.
        if normalizer is not None:
            a_hat_phys = normalizer.denormalize("action", a_hat.float())
            actions_phys = normalizer.denormalize("action", actions.float())
        else:
            a_hat_phys, actions_phys = a_hat.float(), actions.float()

        valid = (~is_pad).unsqueeze(-1).to(a_hat_phys.dtype)       # [B, T, 1]
        abs_err = (a_hat_phys - actions_phys).abs() * valid        # [B, T, D], radians

        for k, v in forward_dict.items():
            sums[f"val/{k}"] = sums.get(f"val/{k}", 0.0) + float(v.item())
        for group_name, indices in JOINT_GROUPS.items():
            sums[f"val_l1/{group_name}"] = (sums.get(f"val_l1/{group_name}", 0.0)
                                            + abs_err[..., indices].sum().item())
        sums["_abs_err_total"] = sums.get("_abs_err_total", 0.0) + abs_err.sum().item()

        n_valid_steps += float(valid.sum().item())
        n_batches += 1

    policy.train()

    if n_batches == 0:
        return {}

    metrics = {}
    for k, v in sums.items():
        if k.startswith("val/"):
            metrics[k] = v / n_batches                     # loss terms: mean over batches
    denom_steps = max(n_valid_steps, 1.0)
    for group_name, indices in JOINT_GROUPS.items():
        # mean |error| per (timestep, dim) inside this joint group
        metrics[f"val_l1/{group_name}"] = sums[f"val_l1/{group_name}"] / (denom_steps * len(indices))
    metrics["val_l1/all_dims"] = sums["_abs_err_total"] / (denom_steps * STATE_DIM)
    # ^ radians. Reference points on season_POC22061 held-out episodes:
    #   copy current pose frozen 3.3s = 0.044, predict dataset mean = 0.103.
    metrics["val/n_valid_steps"] = n_valid_steps
    return metrics


def train_bc(train_dataloader, normalizer, train_dataset, timestamp, config, old_device,
             info_log_path=None, val_dataloader=None):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    use_tactile = config['use_tactile']
    resume_path = config.get('resume_path', None)
    load_pretrained_for_newtraining = config.get('load_pretrained_for_newtraining', None)
    tb_log_freq = config.get('tb_log_freq', 1)


    # === batch visualization config ===
    viz_enabled    = config.get('visualize_batch', False)
    viz_dir        = config.get('visualize_batch_dir', os.path.join(ckpt_dir, 'batch_viz'))
    viz_n_batches  = config.get('visualize_n_batches', 3)



    set_seed(seed)
    start_epoch = 0
    global_step = 0
    min_val_loss = np.inf
    best_ckpt_info = None

    from transformers import get_cosine_schedule_with_warmup

    # # Initialize Accelerator
    # accelerator =  Accelerator(find_unused_parameters=True,
    #                            gradient_accumulation_steps=1,
    #                            )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    # Pass it to Accelerator
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    # Re-configure logging now that the true rank is known (idempotent).
    setup_logging(rank=accelerator.process_index, log_dir=ckpt_dir)
    log.info("accelerator: num_processes=%d process_index=%d device=%s mixed_precision=%s",
             accelerator.num_processes, accelerator.process_index, device,
             accelerator.mixed_precision)

    policy = make_policy(policy_class, policy_config)
    optimizer = make_optimizer(policy_class, policy)

    # --- Normalization: use recommended modes, or set all to identity if disabled ---
    norm_log = get_logger("norm")
    disable_normalization = config.get('disable_normalization', False)
    if disable_normalization:
        feature_modes = {k: None for k in recommended_modes(use_tactile=use_tactile)}
    else:
        feature_modes = recommended_modes(use_tactile=use_tactile)
        # Per-feature ablation: force selected keys to identity while the rest
        # stay normalized (ORI_NORM_DISABLE_KEYS).
        for _k in NORM_DISABLE_KEYS:
            if _k in feature_modes:
                feature_modes[_k] = None
                if _k == "observation.tactile" and "observation.tactile_next" in feature_modes:
                    feature_modes["observation.tactile_next"] = None
            else:
                raise ValueError(
                    f"ORI_NORM_DISABLE_KEYS contains unknown feature {_k!r}. "
                    f"Valid keys: {sorted(feature_modes)}"
                )
    config['feature_modes'] = feature_modes
    config['norm_disable_keys'] = list(NORM_DISABLE_KEYS)
    normalizer = OriNormalizer(
        stats=train_dataset.meta.stats,
        feature_modes=feature_modes,
        device=device,
    )

    norm_log.info("disable_normalization=%s  norm_disable_keys=%s  image_norm=%s",
                  disable_normalization, NORM_DISABLE_KEYS or "-",
                  os.environ.get("ORI_IMAGE_NORM", "1"))
    if disable_normalization:
        norm_log.info("ALL features in identity/pass-through mode -- no normalization applied anywhere.")
    for _k, _mode in sorted(normalizer.transforms.items()):
        _requested = feature_modes.get(_k)
        _note = "" if _mode == _requested else f"  (DOWNGRADED from {_requested!r})"
        norm_log.info("  %-38s -> %s%s", _k, _mode, _note)

    # Normalization health, reported unconditionally at INFO. The single most
    # useful number is the worst |normalized value| the dataset can produce:
    # anything in the hundreds means a denominator problem or an untamed tail.
    _raw_stats = train_dataset.meta.stats
    for _k, _fs in normalizer._stats.items():
        if _fs.q01 is None or _fs.q99 is None:
            continue
        _denom = (_fs.q99 - _fs.q01).float()
        norm_log.info("  stats[%s]: dim=%d  (q99-q01) min=%.3e median=%.3e max=%.3e",
                      _k, _denom.numel(), _denom.min().item(),
                      _denom.median().item(), _denom.max().item())
        if _fs.degenerate_dims:
            norm_log.info("  stats[%s]: %d degenerate dim(s) %s pinned to unit scale "
                          "(spread < %g) instead of being amplified",
                          _k, len(_fs.degenerate_dims), _fs.degenerate_dims,
                          normalizer.degenerate_spread)
        _stats_key = normalizer.key_aliases.get(_k, _k)
        _src = _raw_stats.get(_stats_key)
        if _src is not None and "min" in _src and "max" in _src:
            _mn = torch.as_tensor(np.asarray(_src["min"], dtype=np.float32)).flatten().to(_denom.device)
            _mx = torch.as_tensor(np.asarray(_src["max"], dtype=np.float32)).flatten().to(_denom.device)
            _lo = (_mn - _fs.q01) / (_denom + 1e-6) * 2 - 1
            _hi = (_mx - _fs.q01) / (_denom + 1e-6) * 2 - 1
            _worst = torch.maximum(_lo.abs(), _hi.abs())
            _clip = normalizer.clip.get(_k)
            norm_log.info("  stats[%s]: worst |normalized| over dataset = %.1f (dim %d)%s",
                          _k, _worst.max().item(), int(_worst.argmax().item()),
                          f", clipped at +/-{_clip:g}" if _clip else "")
            if _clip is None and _worst.max().item() > 20:
                norm_log.warning("  stats[%s]: %d dim(s) exceed |20| after normalization and are "
                                 "NOT clipped -- consider adding this key to the clip dict",
                                 _k, int((_worst > 20).sum().item()))

    # NOTE: normalizer stats are deliberately kept in fp32 under mixed precision.
    # They used to be cast to bf16/fp16 here, which is destructive:
    # bf16 has 8 mantissa bits, so its spacing near 0.56 is ~2e-3, while
    # action[58]'s q99-q01 is 1.3e-5 -- q01 and q99 round to the SAME bf16 value,
    # the denominator collapses to 0, and (x-q01)/(0+1e-6) explodes. fp16's
    # spacing there is ~5e-4, still 40x larger than the real spread.
    # Normalization runs on fp32 inputs before autocast anyway, so there is
    # nothing to gain.

    if accelerator.is_main_process:
        dims_to_exclude = log_problematic_features(
            stats=train_dataset.meta.stats,
            normalizer_transforms=normalizer.transforms,
            log_file_path=Path(ckpt_dir) / "feature_report.log",
        )
        norm_log.info("problematic dims per group: %s",
                      {k: len(v) for k, v in dims_to_exclude.items()})

    if accelerator.is_main_process:
        with open(info_log_path, 'a') as f:
            f.write("\n--- Normalizer Configuration ---\n")
            f.write(f"disable_normalization: {disable_normalization}\n")
            if disable_normalization:
                f.write("  => All features set to identity (pass-through) mode. No normalization applied.\n")
            f.write(normalizer.describe())
            f.write("\n")

        config_save_path = os.path.join(ckpt_dir, 'training_configs.json')
        stats_save_path = os.path.join(ckpt_dir, 'training_stats.pkl')
        with open(config_save_path, 'w') as f:
            # Filter out non-serializable objects
            config_to_save = {k: v for k, v in config.items() 
                            if k not in ['policy_config', 'policy']}
            json.dump(config_to_save, f, indent=2, default=str)

        with open(stats_save_path, 'wb') as f:
            pickle.dump(normalizer._stats, f)

    

    # === 构建 scheduler ===
    total_iters = num_epochs * len(train_dataloader)

    # total_iters is computed BEFORE prepare(), i.e. in "whole dataset" units.
    # AcceleratedScheduler advances the wrapped scheduler num_processes times per
    # optimizer step, so the schedule lives in exactly these units too -- which
    # means warmup_iters here really is `ratio` of the run, on any GPU count.
    _wu_ratio = config['lr_config']['warmup_ratio_of_total']
    warmup_iters = max(config['lr_config']['warmup_iters_min'], int(_wu_ratio * total_iters))
    config['lr_config']['warmup_iters'] = warmup_iters

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_iters,
        num_training_steps=total_iters,
    )
    log.info("scheduler: cosine, total=%d schedule-steps (=%d epochs x %d pre-shard batches)",
             total_iters, num_epochs, len(train_dataloader))
    log.info("  warmup=%d schedule-steps (%.1f%% of total, ratio=%.3f) -> ~%d real optimizer steps on %d rank(s)",
             warmup_iters, 100.0 * warmup_iters / max(total_iters, 1), _wu_ratio,
             warmup_iters // max(accelerator.num_processes, 1), accelerator.num_processes)

    # BACKBONE_WEIGHTS only affects the freshly-built model. Both branches below
    # load a FULL ACT state_dict, which already contains backbones.0.0.body.* and
    # therefore overwrites whatever the backbone was initialised with.
    if BACKBONE_WEIGHTS and (resume_path or load_pretrained_for_newtraining):
        log.warning("BACKBONE_WEIGHTS=%s is redundant here: the checkpoint about to be loaded "
                    "contains the backbone weights and will overwrite it. BACKBONE_WEIGHTS only "
                    "matters when training from scratch.", BACKBONE_WEIGHTS)

    if load_pretrained_for_newtraining is not None and os.path.exists(load_pretrained_for_newtraining):
        log.info("loading pretrained WEIGHTS ONLY (fresh optimizer/scheduler/epoch) from %s",
                 load_pretrained_for_newtraining)
        tmp_checkpoint = torch.load(load_pretrained_for_newtraining, map_location=device)
        policy.load_state_dict(tmp_checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # if 'scheduler' in checkpoint:
        #     scheduler.load_state_dict(checkpoint['scheduler'])
        # start_epoch = checkpoint.get('epoch', 0)
        # global_step = checkpoint.get('global_step', 0)
        # min_val_loss = checkpoint.get('min_val_loss', np.inf)
        # best_ckpt_info = checkpoint.get('best_ckpt_info', None)

    elif load_pretrained_for_newtraining is not None:
        log.error("--load_pretrained_for_newtraining=%s does not exist -- SILENTLY training from "
                  "random init", load_pretrained_for_newtraining)

    if resume_path is not None and os.path.exists(resume_path):
        log.info("[Resume] loading full checkpoint from %s", resume_path)
        checkpoint = torch.load(resume_path, map_location=device)
        log.info("[Resume] checkpoint keys: %s", list(checkpoint.keys()))

        if isinstance(policy, torch.nn.parallel.DistributedDataParallel):
            policy.module.load_state_dict(checkpoint["model"], strict=True)
        else:
            policy.load_state_dict(checkpoint["model"], strict=True)

        # policy.load_state_dict(checkpoint['model'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        # +1: the stored epoch is the last one that COMPLETED, so re-running it
        # would train on it twice. (For a mid-epoch sigterm/globalstep ckpt this
        # forfeits the remainder of that epoch, which is the safer trade.)
        start_epoch = checkpoint.get('epoch', -1) + 1
        global_step = checkpoint.get('global_step', 0)
        min_val_loss = checkpoint.get('min_val_loss', np.inf)
        best_ckpt_info = checkpoint.get('best_ckpt_info', None)
        log.info("[Resume] restored epoch=%d (resuming at %d) global_step=%d min_val_loss=%s lr=%s",
                 checkpoint.get('epoch', -1), start_epoch, global_step, min_val_loss,
                 optimizer.param_groups[0]['lr'])
        if start_epoch >= num_epochs:
            log.warning("[Resume] start_epoch=%d >= num_epochs=%d -- the training loop will not "
                        "run. Raise --num_epochs.", start_epoch, num_epochs)

    elif resume_path is not None:
        log.error("--resume_path=%s does not exist -- SILENTLY training from random init", resume_path)

    # policy = torch.compile(policy, mode="reduce-overhead")

    # === Prepare for distributed training ===
    policy, optimizer, train_dataloader, scheduler = accelerator.prepare(
        policy, optimizer, train_dataloader, scheduler
    )
    log.info("after accelerator.prepare(): %d batches/epoch on this rank (%d ranks)",
             len(train_dataloader), accelerator.num_processes)

    # === Data loader timing instrumentation ===
    timing_log_path = os.path.join(ckpt_dir, 'dataloader_timing.log')
    batch_timing_logger = BatchTimingLogger(log_path=timing_log_path, summary_interval=50, device=device)

    # Write header
    with open(timing_log_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("DATALOADER TIMING LOG\n")
        f.write(f"CPU cores: {os.cpu_count()}\n")
        f.write(f"num_workers: {train_dataloader.num_workers}\n")
        f.write(f"batch_size_train: {config.get('batch_size_train')}")
        f.write("=" * 70 + "\n\n")

    # Wrap dataloader to time each __next__() call
    timed_dataloader = TimingDataLoader(train_dataloader)

    train_history = []
    validation_history = []
    # NOTE: global_step / min_val_loss / best_ckpt_info are initialised above and
    # then overwritten by the resume block. They used to be reset to 0/inf/None
    # right here, which silently discarded everything the resume had just
    # restored -- TensorBoard curves restarted at 0 and overlaid the previous
    # run, and policy_globalstep_*.ckpt names collided. Do not re-initialise.
    epoch_start_idx = 0  # track start index of current epoch in train_history
    log.info("training loop: epochs %d..%d, starting at global_step=%d",
             start_epoch, num_epochs - 1, global_step)

    # === TensorBoard writer ===
    tb_log_dir = os.path.join(ckpt_dir, 'tensorboard', timestamp)
    # Only rank 0 owns the event files; other ranks previously created their own
    # writer in the same directory, producing duplicate/interleaved event files.
    writer = SummaryWriter(log_dir=tb_log_dir) if accelerator.is_main_process else None
    if writer is not None:
        log.info("[TensorBoard] logging to %s", tb_log_dir)
        # log hyperparameters as text
        hparam_str = "\n".join([f"{k}: {v}" for k, v in config.items()
                                if k != 'policy_config'])
        writer.add_text("config/hparams", hparam_str, 0)

    # === saving untrained for debug ===
    # ckpt_path = os.path.join(ckpt_dir, f'untrained_policy.ckpt')
    # torch.save({
    #     'model': policy.state_dict(),
    #     'optimizer': optimizer.state_dict(),
    #     'scheduler': scheduler.state_dict(),
    #     'epoch': -1,
    #     'global_step': global_step,
    #     'min_val_loss': min_val_loss,
    # }, ckpt_path)
    

    # === SIGTERM signal handler: save checkpoint before job is killed ===
    # This allows graceful resume when S2 scheduler sends SIGTERM (phd stop)
    _sigterm_received = False

    def _on_sigterm(signum, frame):
        nonlocal _sigterm_received
        _sigterm_received = True
        if accelerator.is_main_process:
            print(f"\n[SIGTERM] Received signal {signum}, saving emergency checkpoint...")
            # Kill dataloader workers FIRST to avoid "worker is killed" error
            try:
                _dl = train_dataloader._dataloader if hasattr(train_dataloader, '_dataloader') else train_dataloader
                if hasattr(_dl, '_iterator') and _dl._iterator is not None:
                    _dl._iterator._shutdown_workers()
            except Exception:
                pass
            try:
                ckpt_path = os.path.join(ckpt_dir, f'policy_sigterm_step{global_step}.ckpt')
                torch.save({
                    'model': accelerator.unwrap_model(policy).state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step,
                    'min_val_loss': min_val_loss,
                }, ckpt_path)
                print(f"[SIGTERM] Emergency checkpoint saved to {ckpt_path}")
            except Exception as e:
                print(f"[SIGTERM] Failed to save checkpoint: {e}")
        import sys
        sys.exit(0)


    import signal as _signal
    _signal.signal(_signal.SIGTERM, _on_sigterm)

    for epoch in tqdm(range(start_epoch, num_epochs)):
        # Set epoch for DistributedSampler to ensure different shuffling each epoch
        if hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)
        
        step_log = {}

        log.info("---- epoch %d/%d  (global_step=%d, lr=%.3e) ----",
                 epoch, num_epochs - 1, global_step, optimizer.param_groups[0]['lr'])
        if epoch == 75:
            log.info("tactile teacher forcing OFF from this epoch: the second transformer pass "
                     "now consumes the model's own tactile_hat instead of ground-truth tactile_next")
        epoch_start_idx = len(train_history)  # mark start of this epoch's entries

        # ===================== VALIDATION =====================
        # Runs on the main process only, against the unwrapped model, every
        # VAL_EVERY_N_EPOCHS epochs (and on the final epoch). Other ranks wait
        # at the barrier below.
        _do_val = (val_dataloader is not None
                   and (epoch % VAL_EVERY_N_EPOCHS == 0 or epoch == num_epochs - 1))
        if _do_val:
            if accelerator.is_main_process:
                _t_val = time.time()
                val_metrics = origami_validate(
                    val_dataloader, accelerator.unwrap_model(policy), normalizer,
                    device, use_tactile, epoch, max_steps=config.get('max_val_steps'))
                validation_history.append((epoch, val_metrics))

                if val_metrics:
                    epoch_val_loss = val_metrics['val/loss']
                    log.info("VAL epoch %d (%.1fs): loss=%.5f | %s",
                             epoch, time.time() - _t_val, epoch_val_loss,
                             " ".join(f"{k.split('/')[-1]}={v:.5f}"
                                      for k, v in sorted(val_metrics.items())
                                      if k.startswith("val_l1/")))
                    if writer is not None:
                        for k, v in val_metrics.items():
                            writer.add_scalar(k, v, epoch)
                        writer.flush()

                    if epoch_val_loss < min_val_loss:
                        min_val_loss = epoch_val_loss
                        best_ckpt_info = (epoch, min_val_loss,
                                          deepcopy(accelerator.unwrap_model(policy).state_dict()))
                        ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
                        torch.save({
                            'model': best_ckpt_info[2],
                            'epoch': epoch,
                            'global_step': global_step,
                            'min_val_loss': min_val_loss,
                            'val_metrics': val_metrics,
                        }, ckpt_path)
                        log.info("VAL new best (%.5f) -> %s", min_val_loss, ckpt_path)
            accelerator.wait_for_everyone()

        # training
        policy.train()
        optimizer.zero_grad()
        train_losses = []
        with tqdm(timed_dataloader, desc=f"Train Epoch {epoch}", leave=False) as tepoch:
            # for batch_idx, data in enumerate(train_dataloader):
            for batch_idx, data in enumerate(tepoch): 
                # if batch_idx > 3: 
                #     break
                # assert False 

                _t_batch_start = time.time()
                forward_dict = None 
                _batch_timings = {}

                # ── Dataloader time (from TimingDataLoader wrapper) ──
                _batch_timings['dataloader'] = timed_dataloader.last_batch_time

                if IS_ORIGAMI_TASK:
                    # ── convert_batch (timing attached as data["_timing"]) ──
                    data = convert_batch(data, use_tactile=use_tactile, delta_timestamps=DELTA_TIMESTAMPS, 
                                         epoch=epoch, batch_idx=batch_idx, normalizer=normalizer)
                    convert_timing = data.pop("_timing", {})
                    _batch_timings['convert_norm'] = convert_timing.get('norm', 0)
                    _batch_timings['convert_resize'] = convert_timing.get('resize', 0)
                    _batch_timings['convert_tactile'] = convert_timing.get('tactile', 0)
                    _batch_timings['convert_total'] = convert_timing.get('total', 0)


                    # ── forward pass (includes .to(device) transfer + policy forward) ──
                    _t_fwd_start = time.time()
                    log_final = (accelerator.is_main_process
                                 and epoch % 10 == 0 and batch_idx < viz_n_batches)
                    forward_dict = origami_forward_pass(data, policy, normalizer, device, use_tactile, epoch=epoch,
                                                        log_final_inputs=log_final, writer=writer, global_step=global_step)
                    _t_fwd_total = time.time() - _t_fwd_start
                    _batch_timings['forward'] = _t_fwd_total

                    _batch_timings['total'] = time.time() - _t_batch_start

                    # Log per-batch + periodic summary to file (with GPU stats every 50 batches)
                    _gpu_stats = get_gpu_stats(device) if (batch_timing_logger._batch_count % 50 == 0) else None
                    batch_timing_logger.log_batch(_batch_timings, gpu_stats=_gpu_stats)

                else:

                    data = train_dataset.postprocess(data, device, use_tactile) #TODO CHECK THIS functionality
                    forward_dict = old_forward_pass(data, policy, normalizer, device, use_tactile)

                

                #continuing just for debugging 
                #need to time each batch step 


                # === diagnostics: first N batches every 10th epoch, MAIN PROCESS ONLY ===
                # (all ranks used to run these and race on the same PNG/log paths)
                _diag = accelerator.is_main_process and epoch % 10 == 0 and batch_idx < viz_n_batches
                if _diag:
                    try:
                        log_input_stats(data, use_tactile, writer, global_step, info_log_path)
                    except Exception as e:
                        log.warning("could not log input stats for batch %d: %s", batch_idx, e)

                # NOTE: must be AFTER convert_batch so data has "image", "lowdim", "action" keys
                if _diag and viz_enabled:
                    try:
                        visualize_batch(
                            data, batch_idx=batch_idx, epoch=epoch,
                            save_dir=viz_dir, use_tactile=use_tactile,
                        )
                    except Exception as e:
                        log.warning("could not visualize batch %d: %s", batch_idx, e)

                
                # backward
                loss = forward_dict['loss']
                accelerator.backward(loss)

                if _step_gate() and log.isEnabledFor(logging.DEBUG):
                    _gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), float('inf'))
                    log.debug("step %d | %s | grad_norm=%.4f lr=%.3e",
                              global_step,
                              " ".join(f"{k}={v.item():.5f}" for k, v in forward_dict.items()),
                              _gn.item(), optimizer.param_groups[0]['lr'])
                    if not torch.isfinite(_gn):
                        log.error("step %d: non-finite gradient norm (%s) -- optimizer step will "
                                  "corrupt the weights", global_step, _gn.item())

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()


                train_losses.append(loss.item())
                train_history.append(detach_dict(forward_dict))

                tepoch.set_postfix(
                    loss=loss.item(),
                    refresh=False
                )

                # === TensorBoard: per-step logging (only on main process) ===
                if accelerator.is_main_process and global_step % tb_log_freq == 0:
                    writer.add_scalar("train/loss_step", loss.item(), global_step)
                    for k, v in forward_dict.items():
                        if k == 'loss':
                            continue
                        if isinstance(v, torch.Tensor):
                            writer.add_scalar(f"train/{k}_step", v.item(), global_step)
                    # current learning rate
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar("train/lr", current_lr, global_step)

                    # === TensorBoard: GPU stats (every 10 steps to reduce overhead) ===
                    if global_step % 100 == 0:
                        gpu_stats = get_gpu_stats(device)
                        if gpu_stats is not None:
                            writer.add_scalar("gpu/utilization_pct", gpu_stats['gpu_util_pct'], global_step)
                            writer.add_scalar("gpu/mem_allocated_gb", gpu_stats['mem_allocated_gb'], global_step)
                            writer.add_scalar("gpu/mem_reserved_gb", gpu_stats['mem_reserved_gb'], global_step)
                            writer.add_scalar("gpu/mem_used_gb", gpu_stats['mem_used_gb'], global_step)
                            writer.add_scalar("gpu/mem_total_gb", gpu_stats['mem_total_gb'], global_step)
                            writer.add_scalar("gpu/max_mem_allocated_gb", gpu_stats['max_mem_allocated_gb'], global_step)

                global_step += 1

                _max_steps = config.get('max_train_steps')
                if _max_steps is not None and (batch_idx + 1) >= _max_steps:
                    log.warning("--max_train_steps=%d reached, ending epoch %d early (DEBUG)",
                                _max_steps, epoch)
                    if ( global_step % 2000 == 0 ) and accelerator.is_main_process:
                        pass
                    break

                if ( global_step% 2000==0 ) and accelerator.is_main_process:

                    ckpt_path = os.path.join(ckpt_dir, f'policy_globalstep_{global_step}_loss_{loss.item()}.ckpt')
                    torch.save({
                        'model': accelerator.unwrap_model(policy).state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                        'global_step': global_step,
                        'min_val_loss': min_val_loss,
                    }, ckpt_path)
                    log.info("saved step checkpoint -> %s", ckpt_path)

        epoch_summary = compute_dict_mean(train_history[epoch_start_idx:])
        epoch_train_loss = epoch_summary['loss']
        log.info("epoch %d done: %d steps | %s",
                 epoch, len(train_history) - epoch_start_idx,
                 " ".join(f"{k}={v.item():.5f}" for k, v in epoch_summary.items()))

        # === TensorBoard: per-epoch logging (only on main process) ===
        if accelerator.is_main_process:
            for k, v in epoch_summary.items():
                if isinstance(v, torch.Tensor):
                    writer.add_scalar(f"epoch/{k}", v.item(), epoch)
            writer.add_scalar("epoch/avg_loss", epoch_train_loss, epoch)
            writer.add_scalar("epoch/lr", optimizer.param_groups[0]['lr'], epoch)

            # === TensorBoard: per-epoch GPU stats ===
            gpu_stats = get_gpu_stats(device)
            if gpu_stats is not None:
                writer.add_scalar("epoch/gpu_util_pct", gpu_stats['gpu_util_pct'], epoch)
                writer.add_scalar("epoch/gpu_mem_allocated_gb", gpu_stats['mem_allocated_gb'], epoch)
                writer.add_scalar("epoch/gpu_max_mem_gb", gpu_stats['max_mem_allocated_gb'], epoch)
            # Reset peak memory tracker at end of epoch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)

            # log a sample of input images on the first epoch
            if epoch == start_epoch and 'image' in data:
                try:
                    # data['image'] is [B, N_cam, 3, H, W] — take first camera to avoid 5D make_grid error
                    imgs = data['image'][:4]
                    if imgs.dim() == 5:
                        imgs = imgs[:, 0]  # [4, 3, H, W] — first camera only
                    img_grid = vutils.make_grid(imgs.cpu(), nrow=2, normalize=True)
                    writer.add_image("inputs/sample_images", img_grid, epoch)
                except Exception as e:
                    print(f"[TensorBoard] Could not log images: {e}")

            writer.flush()

        if (epoch % config['ckpt_save_epochs'] == 0 or global_step%400==0 ) and accelerator.is_main_process:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_loss_{epoch_train_loss:.3f}.ckpt')
            torch.save({
                'model': accelerator.unwrap_model(policy).state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
                'min_val_loss': min_val_loss,
            }, ckpt_path)
            log.info("saved epoch checkpoint -> %s", ckpt_path)

    if accelerator.is_main_process:
        ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
        torch.save(accelerator.unwrap_model(policy).state_dict(), ckpt_path)
        log.info("saved final weights -> %s", ckpt_path)

        if best_ckpt_info is None:
            # Only possible when VAL_EPISODES is empty. Fall back to the last epoch,
            # but say so -- this is NOT a validated best checkpoint.
            log.warning("no validation was run (VAL_EPISODES empty); "
                        "policy_last.ckpt is the only meaningful checkpoint")
        else:
            best_epoch, min_val_loss, _ = best_ckpt_info
            log.info("training finished: seed=%s, best val loss %.6f at epoch %d "
                     "(saved as policy_best.ckpt)", seed, min_val_loss, best_epoch)

        # === TensorBoard: close writer ===
        if writer is not None:
            writer.add_hparams(
                {
                    "lr": config['policy_config']['lr'],
                    "num_epochs": num_epochs,
                    "seed": seed,
                    "use_tactile": int(use_tactile),
                    "warmup_iters": config['lr_config']['warmup_iters'],
                    "n_val_episodes": len(config.get('val_episodes', [])),
                },
                {
                    "hparam/best_val_loss": float(min_val_loss),
                },
            )
            writer.close()
            log.info("[TensorBoard] writer closed. `tensorboard --logdir %s`",
                     os.path.join(ckpt_dir, 'tensorboard'))

    # === Final timing summary ===
    batch_timing_logger.finalize()
    with open(timing_log_path, 'a') as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write("TRAINING COMPLETE - FINAL TIMING SUMMARY\n")
        f.write(f"Total batches logged: {batch_timing_logger._batch_count}\n")
        f.write("=" * 70 + "\n")

    return best_ckpt_info



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--ckpt_save_epochs', action='store', type=int, help='seed', required=True)

    parser.add_argument('--expt_name', action='store', type=str, help='expt_name', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    # NOTE: no --chunk_size. The action chunk length is CHUNK_SIZE in configs.py,
    # because it also defines DELTA_TIMESTAMPS["action"]; a CLI value could not
    # change the dataset and was silently ignored.
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--use_tactile', action='store_true')
    parser.add_argument('--resume_path', type=str, default=None, help='path to resume checkpoint')
    parser.add_argument('--load_pretrained_for_newtraining', type=str, default=None, help='path to load pretrained ckpt')

    ################## extras
    
    parser.add_argument('--state_dim', action='store', type=int, help='req for network output dim of actions', required=True)
    
    parser.add_argument('--tb_log_freq', action='store', type=int, help='tb_log_freq', required=True)
    
    parser.add_argument('--doImageTransforms', action='store_true',
                        help='Save PNG visualizations of the first N training batches')

    # parser.add_argument('--max_duration_sec', action='store', type=int, help='crop episode if so', default =120)  


    # batch visualization (debug)
    parser.add_argument('--visualize_batch', action='store_true',
                        help='Save PNG visualizations of the first N training batches')
    parser.add_argument('--visualize_batch_dir', type=str, default=None,
                        help='Directory to save batch visualizations (default: <ckpt_dir>/batch_viz)')
    parser.add_argument('--visualize_n_batches', type=int, default=3,
                        help='Number of batches to visualize (default: 3)')

    parser.add_argument('--disable_normalization', action='store_true',
                        help='Disable all feature normalization (identity/pass-through mode)')

    # --- loss weighting ---
    parser.add_argument('--loss_dim_weight_mode', type=str, default='uniform',
                        choices=['uniform', 'group'],
                        help="Per-dimension action L1 weighting. 'uniform' (default) reproduces "
                             "the previous behaviour; 'group' applies --loss_group_weights.")
    parser.add_argument('--loss_group_weights', type=str, default=None,
                        help='JSON dict of joint-group multipliers for --loss_dim_weight_mode group, '
                             'e.g. \'{"left_hand": 2.0, "right_hand": 2.0}\'. Groups not listed keep 1.0. '
                             'Weights are rescaled to mean 1 so the L1 term stays comparable to an '
                             'unweighted run (and so kl_weight / tac_weight keep their meaning).')
    parser.add_argument('--drop_degenerate_action_dims', action='store_true',
                        help='Give zero loss weight to action dims whose q99-q01 is below the '
                             'degenerate threshold (dims 58,59 on this dataset -- they are constant, '
                             'so they add a constant to the loss and no gradient).')
    parser.add_argument('--max_train_steps', type=int, default=None,
                        help='DEBUG: stop each epoch after N optimizer steps. Use for smoke tests; '
                             'note the LR schedule is still sized for the full epoch.')
    parser.add_argument('--max_val_steps', type=int, default=None,
                        help='DEBUG: cap the number of validation batches.')
    parser.add_argument('--tac_weight', type=float, default=1.0,
                        help='Multiplier on the tactile prediction loss (default 1.0, unchanged). '
                             'Sweep this together with --kl_weight: the three terms l1, kl*kl_weight '
                             'and l1_tac*tac_weight were never balanced against each other.')

    main(vars(parser.parse_args()))



# python benchmark_dataloader.py \
#   --dataset_root /home/sr5/sairaj.loke/other/new_data/larger_data_shortgop \
#   --label 'shortgop' \
#   --video_backend torchcodec \
#   --batch_size 16 --num_workers 4
