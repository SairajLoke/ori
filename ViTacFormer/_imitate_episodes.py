import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from utils import unnormalize_image, normalize_action, denormalize_action, normalize_obs_lowdim, denormalize_obs_lowdim, normalize_tactile, denormalize_tactile, normalize_tactile_next, denormalize_tactile_next, apply_joint_mask
from policy import ACTPolicy
# from visualize_episodes import save_videos
from dataset.ha_pipelinev2_dataset import HaPipelineV2DatasetD020
from dataset.data import data
from dataset.data_tactile import data_tactile
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm, trange

import IPython
e = IPython.embed

import torchvision.utils as vutils
import os
from PIL import Image
import torchvision.transforms.functional as TF

import cv2



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

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ckpt_dir = os.path.join(ckpt_dir, timestamp)

    if use_tactile:
        ckpt_dir = ckpt_dir + "_tactile"
        timestamp = timestamp + "_tactile"


    os.makedirs(ckpt_dir, exist_ok=True)

    episode_len = 10000
    camera_names = ['/observe/vision/head/stereo/lefteye/rgb','/observe/vision/head/stereo/righteye/rgb','/observe/vision/right_wrist/fisheye/rgb','/observe/vision/left_wrist/fisheye/rgb']

    # fixed parameters
    state_dim = 58
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'use_tactile': use_tactile
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        # 'real_robot': not is_sim,
        'use_tactile': use_tactile,
        'resume_path': resume_path,
        'lr_config': {
            'policy': 'CosineAnnealing',
            'warmup': 'linear',
            'warmup_iters': 1000,
            'warmup_ratio': 1.0 / 10,
            'min_lr_ratio': 1e-1,
        }
    }

    norm_stats_cache = os.path.join(ckpt_dir, 'dataset_stats.pkl')

    data['train']['norm_stats_cache'] = norm_stats_cache
    data['val']['norm_stats_cache'] = norm_stats_cache
    data_tactile['train']['norm_stats_cache'] = norm_stats_cache
    data_tactile['val']['norm_stats_cache'] = norm_stats_cache


    if not use_tactile:
        train_dataset = HaPipelineV2DatasetD020(**data['train'])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=False, num_workers=36, prefetch_factor=1)

        # val_dataset = HaPipelineV2DatasetD020(**data['val'])
        # val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=False, num_workers=36, prefetch_factor=1)
    else:
        train_dataset = HaPipelineV2DatasetD020(**data_tactile['train'])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=False, num_workers=36, prefetch_factor=1)

        # val_dataset = HaPipelineV2DatasetD020(**data_tactile['val'])
        # val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=False, num_workers=36, prefetch_factor=1)

    normalizer = train_dataset.get_normalizer()



    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'normalize.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(normalizer, f)

    best_ckpt_info = train_bc(train_dataloader, normalizer, train_dataset, timestamp, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


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

def forward_pass(data, policy, normalizer, device, use_tactile, epoch=0):
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


def train_bc(train_dataloader, normalizer, dataset, timestamp, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    use_tactile = config['use_tactile']
    resume_path = config.get('resume_path', None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    for epoch in tqdm(range(start_epoch, num_epochs)):
        step_log = {}
        print(f'\nEpoch {epoch}')
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
                data = dataset.postprocess(data, device, use_tactile)
                forward_dict = forward_pass(data, policy, normalizer, device, use_tactile)
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

                global_step += 1

        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')

        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 5 == 0:
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

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_val_loss_{min_val_loss}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')


    return best_ckpt_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
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

    main(vars(parser.parse_args()))
