import torch
import numpy as np
import time 
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
import cv2
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torchvision.utils as vutils
import os
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter

from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from utils import unnormalize_image, normalize_action, denormalize_action, normalize_obs_lowdim, denormalize_obs_lowdim, normalize_tactile, denormalize_tactile, normalize_tactile_next, denormalize_tactile_next, apply_joint_mask
#TOOD add normin dataset transforms?

from policy import ACTPolicy
from dataset.ha_pipelinev2_dataset import HaPipelineV2DatasetD020
from dataset.data import data
from dataset.data_tactile import data_tactile

# from visualize_episodes import save_videos

from dataset.origami_dataset import (
    get_origami_full_dataset, 
    convert_batch , LeRobotNormalizer)

from lerobot.processor import  PolicyProcessorPipeline
from lerobot.policies.factory import make_pre_post_processors
# from lerobot.common.datasets import transforms
# RobotProcessorPipeline for actual h/w inference (unbatched), policy processingis for batched : 
# ref: https://huggingface.co/docs/lerobot/introduction_processors

from train_utils import _stats
from train_eval_utils import JOINT_GROUPS, JOINT_GROUP_COLORS, _detailed_stats, log_input_stats
import IPython
e = IPython.embed


from configs import ( EPISODE_LEN, TOLERANCE, CAMERA_NAMES, STATE_DIM, LR_BACKBONE, BACKBONE, IS_ORIGAMI_TASK,
    FULL_DATASET, DELTA_TIMESTAMPS, CHUNK_SIZE, PROPRIOCEPTIVE_TEMPORAL_HORIZON, MASK_FINGERS, HAND_MASK, FPS, MAXDURATION_IN_EPISODE_SEC )




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


def visualize_batch(data, batch_idx, epoch, save_dir, use_tactile, max_samples=4):

    """
    Visualize sample index 0 from a training batch after `convert_batch`.

    Produces a single figure per batch containing:
      - 4 camera images (side by side)
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

    s = 0  # always visualize sample 0
    n_cam  = image.shape[1]
    cam_names = ["head_left", "head_right", "wrist_right", "wrist_left"]

    # --- Determine layout using GridSpec ---
    n_joint_groups = len(JOINT_GROUPS)  # 5
    n_rows = 1 + n_joint_groups + n_joint_groups  # cameras + lowdim groups + action groups
    if use_tactile:
        n_rows += 3  # tactile heatmap, tactile mean, tactile_next heatmap

    fig = plt.figure(figsize=(18, 3.5 * n_rows))
    gs = fig.add_gridspec(n_rows, 1, hspace=0.6)

    row = 0

    # ---- Row 0: 4 cameras side by side ----
    ax = fig.add_subplot(gs[row])
    row += 1
    cam_imgs = image[s].cpu().float()  # [N_cam, 3, H, W]
    cam_imgs = (cam_imgs - cam_imgs.min()) / (cam_imgs.max() - cam_imgs.min() + 1e-6)
    grid = vutils.make_grid(cam_imgs, nrow=n_cam, normalize=False, padding=4, pad_value=1)
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.set_title(f"Cameras: {', '.join(cam_names[:n_cam])}", fontsize=10)
    ax.axis("off")

    # ---- Lowdim: 5 subplots (one per joint group) ----
    ld = lowdim[s].cpu().numpy()  # [T1, D1]
    for group_name, indices in JOINT_GROUPS.items():
        ax = fig.add_subplot(gs[row])
        row += 1
        ld_group = ld[:, indices]  # [T1, len(indices)]
        for j in range(ld_group.shape[1]):
            ax.plot(range(ld_group.shape[0]), ld_group[:, j],
                    marker='o', markersize=3, alpha=0.7, label=f"j{indices[j]}")
        ax.set_title(f"Lowdim — {group_name} ({len(indices)} DOF)", fontsize=9)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        if len(indices) <= 10:
            ax.legend(fontsize=6, loc='best', ncol=2)

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
                    alpha=0.7, linewidth=1, label=f"j{indices[j]}")
        if pad_start < act.shape[0]:
            ax.axvline(x=pad_start - 0.5, color='red', linestyle='--', alpha=0.7,
                       label=f'pad@{pad_start}')
        ax.set_title(f"Action — {group_name} ({len(indices)} DOF)", fontsize=9)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        if len(indices) <= 10:
            ax.legend(fontsize=6, loc='best', ncol=2)

    if use_tactile:
        # ---- Tactile heatmap [18, 120] (deltas ⊕ next) ----
        ax = fig.add_subplot(gs[row])
        row += 1
        tac = data["tactile"][s]
        if tac.dim() == 1:
            tac = tac.reshape(18, -1)
        tac_np = tac.cpu().numpy()
        im3 = ax.imshow(tac_np, aspect='auto', cmap='coolwarm')
        ax.set_title(f"Tactile (past⊕deltas) [{tac_np.shape[0]}, {tac_np.shape[1]}]", fontsize=9)
        ax.set_xlabel("Feature dim (0-59: past | 60-119: deltas)")
        ax.set_ylabel("Timestep (0-17)")
        ax.axvline(x=59.5, color='yellow', linestyle='--', alpha=0.5, label='past | deltas')
        ax.legend(fontsize=6)
        plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)

        # ---- Tactile mean force over time (line plot) ----
        ax = fig.add_subplot(gs[row])
        row += 1
        # mean across feature dim for past (0:60) and deltas (60:120) separately
        past_mean = tac_np[:, :60].mean(axis=1)
        delta_mean = tac_np[:, 60:120].mean(axis=1)
        ax.plot(range(tac_np.shape[0]), past_mean, 'b-o', markersize=4, label='past mean')
        ax.plot(range(tac_np.shape[0]), delta_mean, 'r-s', markersize=4, label='delta mean')
        ax.set_title("Tactile mean force over time", fontsize=9)
        ax.set_xlabel("Timestep (0-17)")
        ax.set_ylabel("Mean value")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # ---- Tactile_next heatmap ----
        ax = fig.add_subplot(gs[row])
        row += 1
        tac_next = data["tactile_next"][s]
        if tac_next.dim() == 1:
            tac_next = tac_next.reshape(18, -1)
        tac_next_np = tac_next.cpu().numpy()
        im4 = ax.imshow(tac_next_np, aspect='auto', cmap='coolwarm')
        ax.set_title(f"Tactile_next (target) [{tac_next_np.shape[0]}, {tac_next_np.shape[1]}]", fontsize=9)
        ax.set_xlabel("Feature dim")
        ax.set_ylabel("Timestep (0-17)")
        plt.colorbar(im4, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Epoch {epoch} | Batch {batch_idx} | Sample 0", fontsize=14, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = os.path.join(save_dir, f"epoch{epoch}_batch{batch_idx}_sample0.png")
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[Viz] Saved → {out_path}")



def main(args):
    set_seed(1)
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
    resume_path = args['resume_path']
    visualize_batch_flag = args.get('visualize_batch', False)
    visualize_batch_dir  = args.get('visualize_batch_dir', None)
    visualize_n_batches  = args.get('visualize_n_batches', 3)
    doImageTransforms = args.get('doImageTransforms', False) 
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
                         'proprioceptive_temporal_horizon': PROPRIOCEPTIVE_TEMPORAL_HORIZON
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': LR_BACKBONE, 'backbone' : LR_BACKBONE, 'num_queries': 1,
                         'camera_names': CAMERA_NAMES,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
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
        'visualize_batch': visualize_batch_flag,
        'visualize_batch_dir': visualize_batch_dir or os.path.join(ckpt_dir, 'batch_viz'),
        'visualize_n_batches': visualize_n_batches,
        'lr_config': {
            'policy': 'CosineAnnealing',
            'warmup': 'linear',
            'warmup_iters': 500,
            'warmup_ratio': 1.0 / 10,
            'min_lr_ratio': 1e-1,
        },
        'ckpt_save_epochs': args['ckpt_save_epochs'],
        
        'doImageTransforms':  doImageTransforms
    }
    
    
    info_log_path = os.path.join(ckpt_dir, 'info.log')
    with open(info_log_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("TRAINING INFO LOG\n")
        f.write("=" * 70 + "\n\n")

        f.write("--- CLI Args ---\n")
        f.write(str(args) + "\n\n")

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


    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(os.cpu_count())

    norm_stats_cache = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    # data['train']['norm_stats_cache'] = norm_stats_cache
    # data['val']['norm_stats_cache']   = norm_stats_cache
    # data_tactile['train']['norm_stats_cache'] = norm_stats_cache
    # data_tactile['val']['norm_stats_cache']   = norm_stats_cache #TODO

    train_dataset = None 
    normalizer = None 

    if IS_ORIGAMI_TASK:
        
        train_dataset = get_origami_full_dataset(
            dataset_root=FULL_DATASET,
            split= "full",
            TOLERANCE= TOLERANCE, 
            delta_timestamps=DELTA_TIMESTAMPS,
            use_tactile=use_tactile,
            max_duration_sec= MAXDURATION_IN_EPISODE_SEC  , #NOTE:;;; becareful 
            doImageTransforms = config.get('doImageTransforms', False) 
        )

        # valid_dataset = 
        # test_dataset = 
        print(type(train_dataset)) #, vars(train_dataset))
        print("stats keys" , train_dataset.meta.stats.keys())
        
        train_dataloader =  DataLoader(
            train_dataset, 
            batch_size=batch_size_train, 
            pin_memory=False,           # avoid shared memory pressure
            shuffle=True, 
            num_workers=2,#32               # reduced from min(cpu_count, BATCH_SIZE) to avoid SIGBUS
            persistent_workers=True,    # workers cleaned up between epochs, freeing shm
            prefetch_factor=1, #2
        )
         #TODO: check if error if num_workers = 0 and prefetch_factor>=1        
        #----------
        # preprocessor, postprocessor =  make_pre_post_processors(policy_cfg=policy_config, dataset_stats=norm_stats_cache)        #----------
        #TODO: check above dataset stats and others    
        
        normalizer = LeRobotNormalizer(
            train_dataset.meta.stats,
            {
                "observation.state": None, #"gaussian",
                "action": None, # "gaussian",
                "observation.tactile": None, #"gaussian",

                # not changing these yet
                "observation.images.head_left": None,
                "observation.images.head_right": None,
                "observation.images.wrist_left": None,
                "observation.images.wrist_right": None,
            },
            
            device=device,
        ) 
         
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
        f.write(f"\n--- Normalizer Info ---\n")
        if IS_ORIGAMI_TASK:
            f.write(f"Normalizer type: LeRobotNormalizer\n")
            f.write(f"Normalizer transforms: {normalizer.transforms if normalizer else 'None'}\n")
        else:
            f.write(f"Normalizer type: {type(normalizer).__name__}\n")
        f.write("\n")

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'normalize.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(normalizer, f)  #pickle.dump(preprocessor,f) if IS_ORIGAMI_TASK else 

    
    print("batches/epi", len(train_dataloader))
    # with tqdm(train_dataloader, desc=f"Sanity Check Train Epoch {-1}", leave=False) as tepoch:
    #         for batch_idx, data in enumerate(tepoch): 
    #             continue
    time.sleep(2)
    
    #=======================================================
    best_ckpt_info = train_bc(train_dataloader, normalizer, train_dataset, timestamp, 
                              config, device, info_log_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')
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


def origami_forward_pass(data, policy, normalizer, device, use_tactile, epoch=0):
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

    # move to device
    qpos_data_norm   = qpos_data_norm.to(device)
    image_data       = image_data.to(device)
    action_data_norm = action_data_norm.to(device)
    is_pad = is_pad.to(device)

    if use_tactile:
        tactile = data["tactile"]                          # [B, T2, D2]
        tactile_norm = tactile
        # tactile_norm = normalizer.normalize("observation.tactile", tactile)
        # tactile_norm = normalize_tactile(tactile, normalizer)  # normalize    
        B, T2, D2 = tactile_norm.shape
        tactile_norm = tactile_norm.view(B, T2 * D2)                # → [B, T2 * D2]
        tactile_norm = tactile_norm.to(device)                     #TODO sure if the view is correct here, maybe we should keep the tactile as [B, T2, D2] for the model to process it as a sequence

        tactile_next = data["tactile_next"]                          # [B, T2, D2]
        tactile_next_norm = tactile_next
        # tactile_next_norm = normalizer.normalize("observation.tactile_next", tactile_next)
        # tactile_next_norm = normalize_tactile_next(tactile_next, normalizer)  # normalize
        tactile_next_norm = tactile_next_norm.to(device)        
        #TODO: TO/or not to mask tactile ? seems like they didn't???
                           


        return policy(qpos_data_norm, image_data, action_data_norm, is_pad, device, tactile_norm, tactile_next_norm, epoch)

    return policy(qpos_data_norm, image_data, action_data_norm, is_pad, device)



def train_bc(train_dataloader, normalizer, dataset, timestamp, config, device, info_log_path=None):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    use_tactile = config['use_tactile']
    resume_path = config.get('resume_path', None)


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

    policy = make_policy(policy_class, policy_config)
    policy.to(device)
    optimizer = make_optimizer(policy_class, policy)
    

    # === 构建 scheduler ===
    total_iters = num_epochs * len(train_dataloader)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['lr_config']['warmup_iters'],
        num_training_steps=total_iters,
    )

    train_history = []
    validation_history = []
    global_step = 0
    min_val_loss = np.inf
    best_ckpt_info = None
    epoch_start_idx = 0  # track start index of current epoch in train_history

    # === TensorBoard writer ===
    tb_log_dir = os.path.join(ckpt_dir, 'tensorboard', timestamp)
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"[TensorBoard] Logging to {tb_log_dir}")
    # log hyperparameters as text
    hparam_str = "\n".join([f"{k}: {v}" for k, v in config.items()
                            if k != 'policy_config'])
    writer.add_text("config/hparams", hparam_str, 0)

    # === resume ===
    if resume_path is not None and os.path.exists(resume_path):
        print(f"[Resume] Loading checkpoint from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        policy.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', 0)
        min_val_loss = checkpoint.get('min_val_loss', np.inf)
        best_ckpt_info = checkpoint.get('best_ckpt_info', None)


    # ckpt_path = os.path.join(ckpt_dir, f'untrained_policy.ckpt')
    # torch.save({
    #     'model': policy.state_dict(),
    #     'optimizer': optimizer.state_dict(),
    #     'scheduler': scheduler.state_dict(),
    #     'epoch': -1,
    #     'global_step': global_step,
    #     'min_val_loss': min_val_loss,
    # }, ckpt_path)

    

    for epoch in tqdm(range(start_epoch, num_epochs)):
        step_log = {}
        print(f'\nEpoch {epoch}')
        epoch_start_idx = len(train_history)  # mark start of this epoch's entries
        # if epoch % 5 == 0:
        #     # validation
        #     # with torch.inference_mode():
        #     with torch.no_grad():
        #         policy.eval()
        #         epoch_dicts = []
        #         for data in tqdm(val_dataloader, desc="Validation", leave=False):
        #             data = dataset.postprocess(data, device, use_tactile)
        #             forward_dict = forward_pass(data, policy, normalizer, device, use_tactile)
        #             epoch_dicts.append(forward_dict)

        #         epoch_summary = compute_dict_mean(epoch_dicts)
        #         validation_history.append(epoch_summary)

        #         epoch_val_loss = epoch_summary['loss']
        #         if epoch_val_loss < min_val_loss:
        #             min_val_loss = epoch_val_loss
        #             best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))

        #     print(f'Val loss:   {epoch_val_loss:.5f}')
        #     summary_string = ''
        #     for k, v in epoch_summary.items():
        #         summary_string += f'{k}: {v.item():.3f} '
        #     print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        train_losses = []
        with tqdm(train_dataloader, desc=f"Train Epoch {epoch}", leave=False) as tepoch:
            # for batch_idx, data in enumerate(train_dataloader):
            for batch_idx, data in enumerate(tepoch): 
                # if batch_idx > 3: 
                #     break
                # assert False 

                forward_dict = None 
                
                if IS_ORIGAMI_TASK:
                    data = convert_batch(data, use_tactile=use_tactile, delta_timestamps=DELTA_TIMESTAMPS) 
                    forward_dict = origami_forward_pass(data, policy, normalizer, device, use_tactile, epoch=epoch)
                else:
                    data = dataset.postprocess(data, device, use_tactile) #TODO CHECK THIS functionality
                    forward_dict = old_forward_pass(data, policy, normalizer, device, use_tactile)

                # === input stats logging: first 3 batches of the first epoch ===
                if epoch == start_epoch and batch_idx < 3:
                    try:
                        log_input_stats(data, use_tactile, writer, global_step, info_log_path)
                    except Exception as e:
                        print(f"[InputStats] Could not log input stats for batch {batch_idx}: {e}")
                    print("batch idx", batch_idx)
                    # break #TODO remove this 

                # === batch visualization: sample 0, first 3 batches, every 10th epoch ===

                # NOTE: must be AFTER convert_batch so data has "image", "lowdim", "action" keys
                if viz_enabled and epoch % 10 == 0 and batch_idx < viz_n_batches:
                    try:
                        visualize_batch(
                            data, batch_idx=batch_idx, epoch=epoch,
                            save_dir=viz_dir, use_tactile=use_tactile,
                        )
                    except Exception as e:
                        print(f"[Viz] Could not visualize batch {batch_idx}: {e}")

                
                # backward
                loss = forward_dict['loss']
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                train_losses.append(loss.item())
                train_history.append(detach_dict(forward_dict))

                tepoch.set_postfix(
                    loss=loss.item(),
                    refresh=False
                )

                # === TensorBoard: per-step logging ===
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
                if global_step % 10 == 0:
                    gpu_stats = get_gpu_stats(device)
                    if gpu_stats is not None:
                        writer.add_scalar("gpu/utilization_pct", gpu_stats['gpu_util_pct'], global_step)
                        writer.add_scalar("gpu/mem_allocated_gb", gpu_stats['mem_allocated_gb'], global_step)
                        writer.add_scalar("gpu/mem_reserved_gb", gpu_stats['mem_reserved_gb'], global_step)
                        writer.add_scalar("gpu/mem_used_gb", gpu_stats['mem_used_gb'], global_step)
                        writer.add_scalar("gpu/mem_total_gb", gpu_stats['mem_total_gb'], global_step)
                        writer.add_scalar("gpu/max_mem_allocated_gb", gpu_stats['max_mem_allocated_gb'], global_step)

                global_step += 1

        print('esi', epoch_start_idx)
        epoch_summary = compute_dict_mean(train_history[epoch_start_idx:])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')

        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # === TensorBoard: per-epoch logging ===
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
                img_grid = vutils.make_grid(data['image'][:4].cpu(), nrow=2, normalize=True)
                writer.add_image("inputs/sample_images", img_grid, epoch)
            except Exception as e:
                print(f"[TensorBoard] Could not log images: {e}")
        writer.flush()

        if epoch % config['ckpt_save_epochs'] == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_loss_{epoch_train_loss:.3f}.ckpt')
            # torch.save(policy.state_dict(), ckpt_path)
            torch.save({
                'model': policy.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
                'min_val_loss': min_val_loss,
            }, ckpt_path)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    # If no validation was run, best_ckpt_info is None — use last epoch as best
    if best_ckpt_info is None:
        print("[Warning] No validation was run, using last epoch as best checkpoint")
        best_ckpt_info = (num_epochs - 1, epoch_train_loss, deepcopy(policy.state_dict()))

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_val_loss_{min_val_loss}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # === TensorBoard: close writer ===
    writer.add_hparams(
        {
            "lr": config['policy_config']['lr'],
            "num_epochs": num_epochs,
            "seed": seed,
            "use_tactile": int(use_tactile),
        },
        {
            "hparam/best_val_loss": min_val_loss,
        },
    )
    writer.close()
    print(f"[TensorBoard] Writer closed. Run `tensorboard --logdir {os.path.join(ckpt_dir, 'tensorboard')}` to view.")

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
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--use_tactile', action='store_true')
    parser.add_argument('--resume_path', type=str, default=None, help='path to resume checkpoint')

    ################## extras
    
    parser.add_argument('--state_dim', action='store', type=int, help='req for network output dim of actions', required=True)
    
    
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
    main(vars(parser.parse_args()))
