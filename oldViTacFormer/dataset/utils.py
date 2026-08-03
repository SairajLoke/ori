import fnmatch
import os
from pathlib import Path

import h5py
import IPython
import numpy as np
import matplotlib.pyplot as plt 
import torch
import transforms3d as t3d
try:
    import pytorch3d.transforms as p3d
except:
    print("cannot import pytorch3d")

e = IPython.embed

HAND_ACTION_NAMES = [
    "R_pinky_actuator",
    "R_ring_actuator",
    "R_middle_actuator",
    "R_index_actuator",
    "R_thumb_actuator",
    "R_thumb_angle",
]

HAND_STATE_NAMES = [
    "R_pinky_actuator",
    "R_ring_actuator",
    "R_middle_actuator",
    "R_index_actuator",
    "R_thumb_actuator",
    "R_thumb_angle",
]

HAND_FORCE_STATE_NAMES = [
    "R_pinky_actuator",
    "R_ring_actuator",
    "R_middle_actuator",
    "R_index_actuator",
    "R_thumb_actuator",
]


def get_ext_h5(base_h5, ext):
    return Path(base_h5).parent / (Path(base_h5).stem + f".{ext}.hdf5")


def flatten_list(L):
    return [item for sublist in L for item in sublist]

def normalize_vector(x):
    return x / np.linalg.norm(x, axis=-1)

def rotation_6d_to_matrix_np(d6: np.array) -> np.array:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if np.sum(np.abs(d6)) < 1e-10:
        return np.eye(3)
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = normalize_vector(a1)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = normalize_vector(b2)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)


def matrix_to_rotation_6d_np(matrix: np.array) -> np.array:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.shape[:-2]
    return matrix[..., :2, :].copy().reshape(batch_dim + (6,))


def rotation_to_mat(roation):
    if isinstance(roation, torch.Tensor):
        assert roation.shape[-1] == 6, "torch only support 6d rotation"
        rotation_mat = p3d.rotation_6d_to_matrix(roation)
    else:
        if len(roation) == 3:
            rotation_abs = np.linalg.norm(roation)
            rotation_axis = roation
            rotation_mat = np.eye(3) if rotation_abs < 1e-6 else t3d.axangles.axangle2mat(axis=rotation_axis, angle=rotation_abs)
        elif len(roation) == 6:
            rotation_mat = rotation_6d_to_matrix_np(roation)
    return rotation_mat

def mat_to_rotation(mat, rdof6=True):
    if isinstance(mat, torch.Tensor):
        assert rdof6, "torch only support 6d rotation"
        rdof6 = p3d.matrix_to_rotation_6d(mat)
        return rdof6
    else:
        if rdof6:
            return matrix_to_rotation_6d_np(mat)
        else:
            axis, angle = t3d.axangles.mat2axangle(mat, unit_thresh=1e-4)
            return axis*angle

def action_minus_qpos(action_axisangle, qpos_axisangle, base_axis="G", return_tensor=True):
    """
    action_axisangle: [N, 3 or 6]
    qpos_axisangle: [3]
    base_axis: G means gripper-axis, E means ego-axis 
    """
    assert base_axis in ["E", "G"]
    action_axisangle = action_axisangle.numpy() if \
        isinstance(action_axisangle, torch.Tensor) else action_axisangle
    qpos_axisangle = qpos_axisangle.numpy() if \
        isinstance(qpos_axisangle, torch.Tensor) else qpos_axisangle

    qpos_mat = rotation_to_mat(qpos_axisangle)
    chunk_size = action_axisangle.shape[0]
    rel_action_orit = []
    for i in range(chunk_size):
        action_mat = rotation_to_mat(action_axisangle[i])
        if base_axis == "G":
            relative_action_mat = np.linalg.inv(qpos_mat) @ action_mat
        elif base_axis == "E":
            relative_action_mat = action_mat @ np.linalg.inv(qpos_mat)
        else:
            raise NotImplementedError
        # rel_axis, rel_abs = t3d.axangles.mat2axangle(relative_action_mat)
        # rel_action_orit.append(rel_axis*rel_abs)
        rel_action_orit.append(mat_to_rotation(relative_action_mat, rdof6=(
            action_axisangle.shape[1] == 6
        )))
    
    if return_tensor:
        return torch.from_numpy(np.stack(rel_action_orit, axis=0, dtype=np.float32))
    else:
        return np.stack(rel_action_orit, axis=0, dtype=np.float32)


def xyz_rotation_to_mat(qpos_xyz_axisangle):
    if isinstance(qpos_xyz_axisangle, torch.Tensor):
        # qpos_abs = torch.norm(qpos_xyz_axisangle[..., 3:])
        batch_shape = qpos_xyz_axisangle.shape[:-1]
        qpos_axis = qpos_xyz_axisangle[..., 3:]
        assert qpos_axis.shape[-1] == 6, "xyz_axisangle should be 6d for torch [for grad valid]"
        qpos_mat = p3d.rotation_6d_to_matrix(qpos_axis)
        se3_mat = torch.cat([qpos_mat, qpos_xyz_axisangle[..., :3].unsqueeze(-1)], dim=-1)
        bottom_row = torch.tensor([0, 0, 0, 1], dtype=se3_mat.dtype, device=se3_mat.device)
        bottom_row = bottom_row.view(1, 1, 4).expand(*batch_shape, 1, 4)
        se3_mat = torch.cat([se3_mat, bottom_row], dim=-2)
        return se3_mat
    else:
        qpos_abs = np.linalg.norm(qpos_xyz_axisangle[3:])
        qpos_axis = qpos_xyz_axisangle[3:]
        if len(qpos_axis) == 3:
            qpos_mat = np.eye(3) if qpos_abs < 1e-6 else t3d.axangles.axangle2mat(axis=qpos_axis, angle=qpos_abs)
        elif len(qpos_axis) == 6:
            qpos_mat = rotation_6d_to_matrix_np(qpos_axis)
        return t3d.affines.compose(qpos_xyz_axisangle[:3], qpos_mat, np.ones(3))

def mat_to_xyz_rotation(mat, rdof6=True):
    if isinstance(mat, torch.Tensor):
        xyz = mat[..., :3, 3]
        assert rdof6, "torch only support 6d rotation"
        rdof6 = p3d.matrix_to_rotation_6d(mat[..., :3, :3])
        return torch.cat([xyz, rdof6], dim=-1)
    else:
        xyz = mat[:3, 3]
        if rdof6:
            rdof6 = matrix_to_rotation_6d_np(mat[:3, :3])
            return np.concatenate([xyz, rdof6])
        else:
            axis, angle = t3d.axangles.mat2axangle(mat[:3, :3], unit_thresh=1e-4)
            return np.concatenate([xyz, axis*angle])

def action_minus_qpos_se3(action_xyz_axisangle, qpos_xyz_axisangle, base_axis="G", return_tensor=True):
    """
    action_xyz_axisangle: [N, 6 or 9]
    qpos_xyz_axisangle: [3]
    base_axis: G means gripper-axis, E means ego-axis 
    """
    assert base_axis in ["E", "G"]
    action_xyz_axisangle = action_xyz_axisangle.numpy() if \
        isinstance(action_xyz_axisangle, torch.Tensor) else action_xyz_axisangle
    qpos_xyz_axisangle = qpos_xyz_axisangle.numpy() if \
        isinstance(qpos_xyz_axisangle, torch.Tensor) else qpos_xyz_axisangle

    qpos_mat = xyz_rotation_to_mat(qpos_xyz_axisangle)
    chunk_size = action_xyz_axisangle.shape[0]
    rel_action = []
    for i in range(chunk_size):
        action_mat = xyz_rotation_to_mat(action_xyz_axisangle[i])
        if base_axis == "G":
            relative_action_mat = np.linalg.inv(qpos_mat) @ action_mat
        elif base_axis == "E":
            relative_action_mat = action_mat @ np.linalg.inv(qpos_mat)
        else:
            raise NotImplementedError
        rel_action.append(mat_to_xyz_rotation(relative_action_mat,
                                          rdof6=(action_xyz_axisangle.shape[1] == 9)))
    
    if return_tensor:
        return torch.from_numpy(np.stack(rel_action, axis=0))
    else:
        return np.stack(rel_action, axis=0)
    

def get_norm_stats(
        dataset_path_list, 
        kinematic_metas, 
        return_stats=True, 
        info_dict=None,
        qpos_key="/observations/qpos",
        action_key="/action",
    ):
    relative_conf = None
    if info_dict is not None:
        action_dim = info_dict["action_dim"]
        hand_dim = info_dict["hand_dim"]
        pos_dim = info_dict["pos_dim"]
        orit_dim = info_dict["orit_dim"]
        relative_conf = info_dict.get("relative_conf", None)

    all_qpos_data = []
    all_action_data = []
    all_ref_action_data = []
    all_episode_len = []

    for dataset_path in dataset_path_list:
        if return_stats:
            try:
                with h5py.File(dataset_path, "r") as root:
                    qpos = root[qpos_key][()]
                    # qvel = root['/observations/qvel'][()]
                    if "/base_action" in root:
                        base_action = root["/base_action"][()]
                        base_action = preprocess_base_action(base_action)
                        action = np.concatenate([root["/action"][()], base_action], axis=-1)
                    else:
                        action = root[action_key][()]

                if kinematic_metas is not None:
                    kinematic_h5_fpath = get_ext_h5(dataset_path, "kinematic")
                    with h5py.File(kinematic_h5_fpath, "r") as kinematic_h5f:
                        for mname, mdim in kinematic_metas:
                            meta = kinematic_h5f[f"/observations/kinematic/{mname}"][()]
                            qpos = np.concatenate([qpos, meta], axis=-1)

            except Exception as e:
                print(f"Error loading {dataset_path} in get_norm_stats")
                print(e)
                quit()

            # ref_action = action[:,:action_dim] - qpos[:,:action_dim]
            ref_action = action - qpos
            all_ref_action_data.append(torch.from_numpy(ref_action))

            all_qpos_data.append(torch.from_numpy(qpos))
            all_action_data.append(torch.from_numpy(action))
            all_episode_len.append(len(qpos))
        
        else:
            with h5py.File(dataset_path, "r") as root:
                episode_length = root[qpos_key].shape[0]
                all_episode_len.append(episode_length)

    stats = dict()
    if return_stats:
        all_qpos_data = torch.cat(all_qpos_data, dim=0)
        all_action_data = torch.cat(all_action_data, dim=0)
        all_ref_action_data = torch.cat(all_ref_action_data, dim=0)

        # normalize qpos data
        qpos_mean = all_qpos_data.mean(dim=[0]).float()
        qpos_std = all_qpos_data.std(dim=[0]).float()
        qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping
        qpos_min = all_qpos_data.min(dim=0).values.float()
        qpos_max = all_qpos_data.max(dim=0).values.float()

        # normalize action data
        action_mean = all_action_data.mean(dim=[0]).float()
        action_std = all_action_data.std(dim=[0]).float()
        action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping
        action_min = all_action_data.min(dim=0).values.float()
        action_max = all_action_data.max(dim=0).values.float()
        
        # normalize action-qpos data
        ref_action_mean = all_ref_action_data.mean(dim=[0]).float()
        ref_action_std = all_ref_action_data.std(dim=[0]).float()
        ref_action_std = torch.clip(ref_action_std, 1e-2, np.inf)

        rel_stats = {
            "rel_action_mean":ref_action_mean.numpy(),
            "rel_action_std": ref_action_std.numpy(),
        }
        if relative_conf is not None:
            # collect relative action-mean action-std
            ref_action_mean = all_ref_action_data.mean(dim=[0]).float()
            ref_action_std = all_ref_action_data.std(dim=[0]).float()
            ref_action_std = torch.clip(ref_action_std, 1e-4, np.inf)
            ref_action_min = all_ref_action_data.min(dim=0).values.float()
            ref_action_max = all_ref_action_data.max(dim=0).values.float()
            if relative_conf.get("rel_hand", False):
                action_mean[:hand_dim] = ref_action_mean[:hand_dim]
                action_std[:hand_dim] = ref_action_std[:hand_dim]
            if relative_conf.get("rel_pos", False):
                action_mean[hand_dim:hand_dim+pos_dim] = ref_action_mean[hand_dim:hand_dim+pos_dim]
                action_std[hand_dim:hand_dim+pos_dim] = ref_action_std[hand_dim:hand_dim+pos_dim]
            if relative_conf.get("rel_orit", False):
                action_mean[hand_dim+pos_dim:hand_dim+pos_dim+orit_dim] = ref_action_mean[hand_dim+pos_dim:hand_dim+pos_dim+orit_dim]
                action_std[hand_dim+pos_dim:hand_dim+pos_dim+orit_dim] = ref_action_std[hand_dim+pos_dim:hand_dim+pos_dim+orit_dim]
            if relative_conf.get("rel_ja", False):
                action_mean[hand_dim:] = ref_action_mean[hand_dim:]
                action_std[hand_dim:] = ref_action_std[hand_dim:]

            rel_stats = {
                "rel_action_mean":ref_action_mean.numpy(),
                "rel_action_std": ref_action_std.numpy(),
                "rel_action_min": ref_action_min.numpy(),
                "rel_action_max": ref_action_max.numpy(),
            }
            
        eps = 0.0001
        stats = {
            "action_mean": action_mean.numpy(),
            "action_std": action_std.numpy(),
            "action_min": action_min.numpy() - eps,
            "action_max": action_max.numpy() + eps,
            "qpos_mean": qpos_mean.numpy(),
            "qpos_std": qpos_std.numpy(),
            "qpos_min": qpos_min.numpy() - eps,
            "qpos_max": qpos_max.numpy() + eps,
            "example_qpos": qpos,
        }
        stats.update(rel_stats)

    return stats, all_episode_len

def debug_show_histogram(action_all, prefix="abs"):
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))  


    axs[0].hist(action_all[:,1], bins=1024, color='blue', alpha=0.7, edgecolor='black')  
    axs[0].set_title('x')  
    axs[0].set_xlabel('Value')  
    axs[0].set_ylabel('Frequency')  
    axs[0].set_yscale('log')  


    axs[1].hist(action_all[:,2], bins=1024, color='green', alpha=0.7, edgecolor='black')  
    axs[1].set_title('y')  
    axs[1].set_xlabel('Value')  
    axs[1].set_ylabel('Frequency')
    axs[1].set_yscale('log') 


    axs[2].hist(action_all[:,3], bins=1024, color='red', alpha=0.7, edgecolor='black')  
    axs[2].set_title('z')  
    axs[2].set_xlabel('Value')  
    axs[2].set_ylabel('Frequency')
    axs[2].set_yscale('log') 

    plt.savefig(f'{prefix}_action.png', format='png', dpi=300)
    print(f"save {prefix}_action.png")

def calibrate_linear_vel(base_action, c=None):
    if c is None:
        c = 0.0  # 0.19
    v = base_action[..., 0]
    w = base_action[..., 1]
    base_action = base_action.copy()
    base_action[..., 0] = v - c * w
    return base_action


def smooth_base_action(base_action):
    return np.stack(
        [
            np.convolve(base_action[:, i], np.ones(5) / 5, mode="same")
            for i in range(base_action.shape[1])
        ],
        axis=-1,
    ).astype(np.float32)


def preprocess_base_action(base_action):
    # base_action = calibrate_linear_vel(base_action)
    base_action = smooth_base_action(base_action)

    return base_action


def postprocess_base_action(base_action):
    linear_vel, angular_vel = base_action
    linear_vel *= 1.0
    angular_vel *= 1.0
    # angular_vel = 0
    # if np.abs(linear_vel) < 0.05:
    #     linear_vel = 0
    return np.array([linear_vel, angular_vel])


# env utils
def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


# helper functions


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_name_idx(h5_fname):
    return int(h5_fname.split("_")[-1].split(".")[0])


def find_all_hdf5(dataset_dir, skip_mirrored_data=False):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, "*.h*5"):
            if "features" in filename:
                continue
            if "kinematic" in filename:
                continue
            if "point" in filename:
                continue
            if "stepmeta" in filename:
                continue
            if "tactile" in filename:
                continue
            if "anno" in filename:
                continue
            if skip_mirrored_data and "mirror" in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))        
    return sorted(hdf5_files)

def get_depth_name(camera_name):
    return camera_name.replace("/begin", "")[:-3] + "depth"

def get_true_camera_name(camera_name):
    return camera_name.replace("/begin", "")

def norm_fit_input(norm_arr, src):
    if isinstance(src, np.ndarray):
        if isinstance(norm_arr, torch.Tensor):
            norm_arr = norm_arr.cpu().numpy()
        else:
            norm_arr = norm_arr
    else:
        if not isinstance(norm_arr, torch.Tensor):
            norm_arr = torch.tensor(norm_arr)
        norm_arr = norm_arr.to(src.device)

    if len(norm_arr.shape) > 1:
        norm_arr = norm_arr[0]
    return norm_arr

def normalize_by_policy(x, norm_stats, key, policy):
    if policy == "ACT":
        return normalize_gaussian(x, norm_stats, key)
    elif policy == "Diffusion":
        return normalize_minmax(x, norm_stats, key)
    else:
        raise NotImplementedError
    
def denormalize_by_policy(x, norm_stats, key, policy):
    if policy == "ACT":
        return denormalize_gaussian(x, norm_stats, key)
    elif policy == "Diffusion":
        return denormalize_minmax(x, norm_stats, key)
    else:
        raise NotImplementedError

def normalize_gaussian(x, norm_stats, key):
    mean = norm_fit_input(norm_stats[key + "_mean"], x)
    std = norm_fit_input(norm_stats[key + "_std"], x)
    return (x - mean) / std

def denormalize_gaussian(x, norm_stats, key):
    mean = norm_fit_input(norm_stats[key + "_mean"], x)
    std = norm_fit_input(norm_stats[key + "_std"], x)
    return x * std + mean

def normalize_minmax(x, norm_stats, key):
    min_val = norm_fit_input(norm_stats[key + "_min"], x)
    max_val = norm_fit_input(norm_stats[key + "_max"], x)
    return (x - min_val) / (max_val - min_val) * 2 - 1

def denormalize_minmax(x, norm_stats, key):
    min_val = norm_fit_input(norm_stats[key + "_min"], x)
    max_val = norm_fit_input(norm_stats[key + "_max"], x)
    return (x + 1) / 2 * (max_val - min_val) + min_val
