import os
import time
import h5py
import json
import numpy as np
import cv2
import random
import torch
from pathlib import Path
from collections import defaultdict
from .utils import bgr2hsv, hsv2bgr
from ..utils import action_minus_qpos, action_minus_qpos_se3, \
    normalize_by_policy, xyz_rotation_to_mat, mat_to_xyz_rotation
from ..utils import xyz_rotation_to_mat, mat_to_xyz_rotation
import transforms3d as t3d


class ImageProcess(object):
    def __init__(
        self,
        key,
        camera_preprcess_info,
        norm=True,
        color_type="rgb",
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        aug=False,
        aug_geo=True,
        aug_color=True,
        aug_conf=dict(),
        debug=False,
    ):

        self.key = key
        self.cam_preprocess_info = camera_preprcess_info

        # augmentation param
        self.norm = norm
        self.color_type = color_type
        self.mean = np.array(mean)
        self.std = np.array(std)

        self.aug = aug
        self.aug_conf = aug_conf
        self.aug_geo = aug_geo
        self.aug_color = aug_color
        self.crop_ratio = aug_conf.get("crop_ratio", 0.95)
        self.rotate_range = aug_conf.get("rotate_range", [-3, 3])
        self.brightness_delta = aug_conf.get("brightness_delta", 32)
        self.contrast_lower, self.contrast_upper = aug_conf.get(
            "contrast_range", (0.5, 1.5)
        )
        self.saturation_lower, self.saturation_upper = aug_conf.get(
            "saturation_range", (0.5, 1.5)
        )
        self.hue_delta = aug_conf.get("hue_delta", 18)

        self.debug = debug

    def _convert(self, img, alpha=1, beta=0, asint=True):
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8) if asint else img

    def _random_flags(self):
        contrast_mode = np.random.randint(2)
        # whether to apply brightness distortion
        brightness_flag = np.random.randint(2)
        # whether to apply contrast distortion
        contrast_flag = np.random.randint(2)
        # the mode to convert color from BGR to HSV
        hsv_mode = np.random.randint(4)

        brightness_beta = np.random.uniform(
            -self.brightness_delta, self.brightness_delta
        )
        # the alpha in `self._convert` to be multiplied to image array
        # in contrast distortion
        contrast_alpha = np.random.uniform(self.contrast_lower, self.contrast_upper)
        # the alpha in `self._convert` to be multiplied to image array
        # in saturation distortion to hsv-formatted img
        saturation_alpha = np.random.uniform(
            self.saturation_lower, self.saturation_upper
        )
        # delta of hue to add to image array in hue distortion
        hue_delta = np.random.randint(-self.hue_delta, self.hue_delta)

        return (
            contrast_mode,
            brightness_flag,
            contrast_flag,
            hsv_mode,
            brightness_beta,
            contrast_alpha,
            saturation_alpha,
            hue_delta,
        )

    def __single_img_croprotate(self, img):
        """
        img: array [H, W, 3]
        """
        # random crop
        h, w = img.shape[:2]
        new_hs = random.randint(0, int(h * (1 - self.crop_ratio)))
        new_ws = random.randint(0, int(w * (1 - self.crop_ratio)))
        new_he = int(h * self.crop_ratio)
        new_we = int(w * self.crop_ratio)
        img = img[new_hs : new_hs + new_he, new_ws : new_ws + new_we]
        img = cv2.resize(img, (w, h))

        # random rotate
        degrees = self.rotate_range
        center = (w // 2, h // 2)
        angle = np.random.uniform(degrees[0], degrees[1])
        scale = 1.0  # 缩放因子，1.0表示不缩放
        M = cv2.getRotationMatrix2D(center, angle, scale)  # 计算旋转矩阵
        img = cv2.warpAffine(img, M, (w, h))  # 应用旋转

        # if self.debug:  # NOTE: save images
        #     cv2.imwrite(f"croprotate_img.jpg", img)
        return img

    def __single_img_colorjitter(self, img, jiter_param, img_key=None):
        (
            contrast_mode,
            brightness_flag,
            contrast_flag,
            hsv_mode,
            brightness_beta,
            contrast_alpha,
            saturation_alpha,
            hue_delta,
        ) = jiter_param

        if brightness_flag:
            img = self._convert(img, beta=brightness_beta)
            # if self.debug:
            #     cv2.imwrite("debug_brightness.jpg", img)

        if contrast_mode == 1:
            if contrast_flag:
                img = self._convert(img, alpha=contrast_alpha)
                # if self.debug:
                #     cv2.imwrite("debug_contrast1.jpg", img)

        if hsv_mode:
            # random saturation/hue distortion
            img = bgr2hsv(img)
            if hsv_mode == 1 or hsv_mode == 3:
                # apply saturation distortion to hsv-formatted img
                img[:, :, 1] = self._convert(img[:, :, 1], alpha=saturation_alpha, asint=False)
            if hsv_mode == 2 or hsv_mode == 3:
                # apply hue distortion to hsv-formatted img
                img[:, :, 0] = img[:, :, 0].astype(int) + hue_delta
            img = hsv2bgr(img)
            # if self.debug:
            #     cv2.imwrite("debug_hsv.jpg", img)

        if contrast_mode == 1:
            if contrast_flag:
                img = self._convert(img, alpha=contrast_alpha)
                # if self.debug:
                #     cv2.imwrite("debug_contrast2.jpg", img)

        return img

    def __single_img_preprocess(self, cur_img, cam_preprocess_info):
        # crop
        if cam_preprocess_info.get("crop", None) is not None:
            s_h, s_w, e_h, e_w = cam_preprocess_info["crop"]
            cur_img = cur_img[s_h:e_h, s_w:e_w]
        # transpose (H, W)调换
        if cam_preprocess_info.get("transpose", False):
            cur_img = cur_img.transpose(1, 0, 2)
        # rotate
        if cam_preprocess_info.get("flip", None) is not None:
            flip_axis = cam_preprocess_info["flip"]
            if flip_axis == "x":  # 上下翻转
                cur_img = cv2.flip(cur_img, 0)
            elif flip_axis == "y":  # 左右翻转
                cur_img = cv2.flip(cur_img, 1)
            elif flip_axis == "xy":
                cur_img = cv2.flip(cur_img, -1)
            else:
                raise NotImplementedError
        # resize
        if cam_preprocess_info.get("img_size", None) is not None:
            h, w = cam_preprocess_info["img_size"]
            cur_img = cv2.resize(cur_img, (w, h))
            if len(cur_img.shape) == 2:
                cur_img = np.expand_dims(cur_img, axis=2)

        return cur_img

    def __call__(self, results):
        if self.aug:
            jiter_param = self._random_flags()

        processed_imgs = []
        for cur_img in results[self.key]:
            cur_img = self.__single_img_preprocess(cur_img, self.cam_preprocess_info)

            if self.aug:
                if self.color_type == "rgb":
                    cur_img = cv2.cvtColor(cur_img, cv2.COLOR_RGB2BGR)
                elif self.color_type == "bgr":
                    cur_img = cur_img
                else:
                    raise NotImplementedError

                if self.aug_geo:
                    cur_img = self.__single_img_croprotate(cur_img)
                if self.aug_color:
                    cur_img = self.__single_img_colorjitter(cur_img, jiter_param, img_key=self.key)

                if self.color_type == "rgb":
                    cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
                elif self.color_type == "bgr":
                    cur_img = cur_img
                else:
                    raise NotImplementedError
                cur_img = cur_img.astype(np.float32)

            if self.debug:  # NOTE: save images
                img_name = self.key.replace('/', '_') # remove name unvalid
                if self.color_type == "rgb":
                    image_bgr = cv2.cvtColor(cur_img, cv2.COLOR_RGB2BGR)
                elif self.color_type == "bgr":
                    image_bgr = cur_img
                else:
                    raise NotImplementedError
                cv2.imwrite(f"process_{img_name}.jpg", image_bgr)

            if self.norm:
                cur_img = cur_img / 255.0
                cur_img = (cur_img - self.mean) / self.std
            processed_imgs.append(cur_img)

        processed_imgs = np.array(processed_imgs).transpose(0, 3, 1, 2).astype(np.float32)  # [3, H, W]
        results[self.key] = torch.from_numpy(processed_imgs)
        return results


class ImageBackgroundAugment(object):
    def __init__(
        self,
        key,
        background_src,
        segmentation_src,
        debug=False,
    ):

        self.key = key
        self.background_src = background_src
        self.segmentation_src = segmentation_src
        self.background_img_pathes = list(Path(background_src).glob("*.JPEG"))
        self.segmentation_img_pathes = list(Path(segmentation_src).rglob("*.png"))

        self.segment_img_dict = defaultdict(list)
        for img_path in self.segmentation_img_pathes:
            step_idx = int(img_path.stem.split("-")[-1])
            camera_name = img_path.parent.name.replace("-", "/")
            train_episode_name = img_path.parent.parent.name
            if camera_name in self.key:
                self.segment_img_dict[f"{train_episode_name}@{step_idx}"] = img_path
        print(f"[ImageBackgroundAugment] background count: {len(self.background_img_pathes)}")
        print(f"[ImageBackgroundAugment] camera {self.key} has {len(self.segment_img_dict)} segment images")
        self.debug = debug

    def __call__(self, results):
        fpath = Path(results["data_path"])
        train_episode_name = fpath.parent.name
        for mid, (img, img_idx) in enumerate(zip(results[self.key], results[f"{self.key}/indices"])):
            img_id = f"{train_episode_name}@{img_idx}"
            if img_id not in self.segment_img_dict:
                continue
            seg_img_path = self.segment_img_dict[img_id]
            seg_img = cv2.imread(str(seg_img_path), cv2.IMREAD_UNCHANGED)
            bg_img = cv2.cvtColor(cv2.imread(str(np.random.choice(self.background_img_pathes))),
                                  cv2.COLOR_BGR2RGB)
            bg_img = cv2.resize(bg_img, (img.shape[1], img.shape[0]))
            show_img = img * np.expand_dims(np.array(seg_img > 0), axis=-1) + \
              bg_img * np.expand_dims(np.array(seg_img == 0), axis=-1)
            results[self.key][mid] = show_img.astype(np.uint8)
            if self.debug:
                cv2.imwrite(f"debug_bg_{self.key.replace('/', '-')}_{img_idx}.jpg",
                            cv2.cvtColor(show_img, cv2.COLOR_RGB2BGR))
        return results

class ConcatData(object):
    def __init__(self, keys, output_key, dim=-1):
        self.keys = keys
        self.output_key = output_key
        self.dim = dim

    def __call__(self, results):
        concat_data = []
        for key in self.keys:
            concat_data.append(results[key])
        if isinstance(concat_data[0], torch.Tensor):
            results[self.output_key] = torch.cat(concat_data, dim=self.dim)
        elif isinstance(concat_data[0], np.ndarray):
            results[self.output_key] = np.concatenate(concat_data, axis=self.dim)
        else:
            raise NotImplementedError
        return results


class CopyData(object):
    def __init__(self, key, output_key):
        self.key = key
        self.output_key = output_key

    def __call__(self, results):
        if self.key not in results:
            return results
        results[self.output_key] = results[self.key].copy()
        return results

class RelativeGenerate(object):
    """
    Suitable for action_meta: Remote v2.0
    """
    def __init__(self, key, ref_key, ref_index=-1, meta=None):
        self.key = key
        self.output_key = key + "/rel"
        self.ref_key = ref_key
        self.ref_index = ref_index
        self.meta = meta


    def cal_relative(self, meta, actions, qpos):
        """
        Calculate relative action
        meta: dict action or qpos
        actions: [action_chunk, action_dim] or [history_qpos, qpos_dim], value list to cal relative
        qpos: [qpos_dim] reference qpos
        """

        ref_type = meta["type"]
        if ref_type == "abs":
            return actions
        elif ref_type == "rel_abs":
            actions[:] -= qpos[:]
        elif ref_type == "rel_yaw":
            yaw_axis = meta.get("yaw_axis", "G")
            ref_orit = action_minus_qpos(
                action_axisangle=actions[:, :],
                qpos_axisangle=qpos[:],
                base_axis=yaw_axis,
                return_tensor=True,
            )
            actions[:, :] = ref_orit
        elif ref_type == "rel_se3":
            yaw_axis = meta.get("yaw_axis", "G")
            ref_transform = action_minus_qpos_se3(
                action_xyz_axisangle=actions[:, :],
                qpos_xyz_axisangle=qpos[:],
                base_axis=yaw_axis,
                return_tensor=True,
            )
            actions[:, :] = ref_transform
        else:
            raise NotImplementedError
        return actions

    def __call__(self, results):
        # qpos = results["qpos"]
        if self.key not in results:
            return results
        value = results[self.key].copy()
        ref_value = results[self.ref_key][self.ref_index].copy()
        rel_value = self.cal_relative(self.meta, value, ref_value)
        results[self.output_key] = rel_value
        return results

class NormalizeByPolicy(object):
    def __init__(self, key, policy):
        self.key = key
        self.policy = policy

    def __call__(self, results):
        norm_stats = results["norm_stats"]
        results[self.key] = normalize_by_policy(results[self.key],
                                            norm_stats, self.key,
                                            self.policy)
        return results

class DataMask(object):
    def __init__(self, key, mask):
        self.key = key
        self.mask = mask

    def __call__(self, results):
        if isinstance(results[self.key], torch.Tensor):
            mask = torch.tensor(self.mask).to(results[self.key].device)
        else:
            mask = np.array(self.mask)
        results[self.key] = results[self.key] * mask
        return results

class EnvAddBatchDim(object):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            if key in results:
                value = results[key]
                value = value if isinstance(value, torch.Tensor) else torch.from_numpy(value)
                results[key] = value.unsqueeze(0).cuda()
        return results

class RandomIKProcessV1(object):
    def __init__(self, action_key, obs_key, urdf_path, joints_range, marker_on_eef, **kwagrs) -> None:
        self.action_key = action_key
        self.obs_key = obs_key
        self.output_aja_key = action_key.replace("/tcp_pose_9d", "/joint_angle")
        self.output_valid_key = action_key.replace("/tcp_pose_9d", "/ik_valid")
        self.output_oja_key = obs_key.replace("/tcp_pose_9d", "/joint_angle")
        self.output_key = [self.output_aja_key, self.output_valid_key, self.output_oja_key]
        self.joints_range = joints_range
        self.debug = kwagrs.get("debug", False)

        from ikpy.chain import Chain
        # only use the for realman arm
        self.robot_chain = Chain.from_urdf_file(urdf_path, active_links_mask=[False] + [True] * 7)
        self.marker_on_eef = (
            None
            if marker_on_eef is None
            else t3d.affines.compose(
                marker_on_eef[:3],
                t3d.quaternions.quat2mat(marker_on_eef[3:]),
                [1, 1, 1],
            )
        )

    def ik(self, init_joint, target_mats):
        init_joint = np.array([0, *init_joint])
        joints = []
        for target in target_mats:
            joint = self.robot_chain.inverse_kinematics_frame(target, init_joint)
            if joint is None:
                return np.array([init_joint for _ in target_mats], dtype=np.float32), False
            init_joint = joint
            joints.append(joint)
        return np.array(joints), True

    def marker_mat_to_eef_mat(self, marker_mat):
        return marker_mat @ np.linalg.inv(self.marker_on_eef)

    def __call__(self, results):
        tcp_action = results.get(self.action_key, None)
        tcp_obs = results.get(self.obs_key, None)
        if tcp_action is None or tcp_obs is None:
            return dict()

        start_obs_tcp = tcp_obs[0]
        obs_len = len(tcp_obs)
        start_joints = np.random.uniform(self.joints_range[0], self.joints_range[1])
        start_eef_in_base = self.robot_chain.forward_kinematics([0, *start_joints])
        start_eef_mat = self.marker_mat_to_eef_mat(xyz_rotation_to_mat(start_obs_tcp))

        states_in_base = []
        for i in range(obs_len):
            target_mat = self.marker_mat_to_eef_mat(xyz_rotation_to_mat(tcp_obs[i]))
            eef_in_base = start_eef_in_base @ np.linalg.inv(start_eef_mat) @ target_mat
            states_in_base.append(eef_in_base)
        for i in range(len(tcp_action)):
            target_mat = self.marker_mat_to_eef_mat(xyz_rotation_to_mat(tcp_action[i]))
            eef_in_base = start_eef_in_base @ np.linalg.inv(start_eef_mat) @ target_mat
            states_in_base.append(eef_in_base)
        target_joints, valid = self.ik(start_joints, states_in_base)

        if self.debug:
            import ikpy.utils.plot as plot_utils
            import matplotlib.pyplot as plt
            fig, ax = plot_utils.init_3d_figure()
            # self.robot_chain.plot(my_chain.inverse_kinematics(target_position), ax, target=target_position)
            for target_joint, state_in_base in zip(target_joints, states_in_base):
                self.robot_chain.plot(target_joint, ax, target=state_in_base[:3, 3])
            plt.show()

        is_valid = []
        for target_joint, state_in_base in zip(target_joints, states_in_base):
            forward = self.robot_chain.forward_kinematics(target_joint)
            delta = np.linalg.norm(forward[:3, 3] - state_in_base[:3, 3])
            is_valid.append(delta < 0.01)
        # if not valid:
        #     print(f"ik error: {np.max(ik_deltas)}")
        results.update({
            self.output_aja_key: target_joints[obs_len:, 1:].astype(np.float32),
            self.output_oja_key: target_joints[:obs_len, 1:].astype(np.float32),
            self.output_valid_key: np.array(is_valid[obs_len:]),
        })
        return results

class PoseTrans9D(object):
    def __init__(self, key):
        self.key = key
        self.output_key = key + "_9d"

    def __call__(self, results):
        if self.key not in results:
            return results

        value = results[self.key].copy()
        if value.ndim == 1:
            value_mat = xyz_rotation_to_mat(value)
            value_9d = mat_to_xyz_rotation(value_mat, rdof6=True)
        else:
            value_9d_list = []
            for i in range(value.shape[0]):
                value_mat = xyz_rotation_to_mat(value[i,:])
                value_9d = mat_to_xyz_rotation(value_mat, rdof6=True)
                value_9d_list.append(value_9d)
            value_9d = np.stack(value_9d_list, axis=0)
        results[self.output_key] = value_9d
        return results

class GenerateRGBD(object):
    def __init__(self, rgbd_conf, rgb_key=None, depth_key=None, output_key=None, debug=False):
        if rgb_key is not None:
            rgbd_conf["rgb"] = rgb_key
        if depth_key is not None:
            rgbd_conf["depth"] = depth_key
        if output_key is not None:
            rgbd_conf["name"] = output_key

        self.rgb_key = rgbd_conf["rgb"]
        self.depth_key = rgbd_conf["depth"]
        self.output_key = rgbd_conf["name"]
        self.rgbd_conf = rgbd_conf
        self.debug = debug

    def __call__(self, results):
        s_ts = time.time()

        rgb_value = results[self.rgb_key]   # [1, H, W, 3]
        rgb_value = np.squeeze(rgb_value, axis=0)
        if rgb_value.shape[0] == 3:
            rgb_value = rgb_value.permute(1,2,0)
        depth_value = results[self.depth_key]   # [1, H, W]
        height = self.rgbd_conf["height"]
        width = self.rgbd_conf["width"]

        depth_value[depth_value == -np.inf] = 0
        # clamp depth image to 10 meters to make output image human friendly
        depth_value[depth_value < -10] = -10
        depth_value = depth_value.reshape(height, width, 1)
        depth_value = depth_value.astype(np.float32)

        rgbd_value = np.concatenate([rgb_value, depth_value], axis=-1)  # [H, W, 3]
        results[self.output_key] = rgbd_value

        e_ts = time.time()
        if self.debug:
            print("[GenerateRGBD] cost time: %.4f"%(e_ts - s_ts))

        return results

class GenerateTablePlane(object):
    def __init__(
        self,
        data_type="0.1.0",
        plane_key="plane_func",
        output_key="/observation/points/table_func",
        env=False,
        debug=False,
        **kwargs
    ):
        self.data_type = data_type
        self.plane_key = plane_key
        self.output_key = output_key
        self.env = env
        self.debug = debug

    def __call__(self, results):
        if self.env:
            return results
        s_ts = time.time()
        data_path = results["data_path"]
        if self.data_type == "0.1.0":
            json_path = data_path.replace(".hdf5", ".json")
            with open(json_path, "r") as file:
                json_data = json.load(file)
            plane_func = np.array(json_data[self.plane_key])
            results[self.output_key] = plane_func
        else:
            raise NotImplementedError

        return results

class GenerateImageFlow(object):
    def __init__(
        self,
        key,
        output_name,
        img_type,
        clip_range=None,
        transpose=False,
        debug=False,
    ):
        self.key = key
        self.output_name = output_name
        self.img_type = img_type
        self.clip_range = clip_range
        self.transpose = transpose
        self.debug = debug

    def __call__(self, results):
        if len(results[self.key]) > 1:  #
            flow_imgs = []
            # is_padding = []
            for index in range(1, len(results[self.key])):
                pre_index = index-1  # 相邻帧求
                img_pre = results[self.key][pre_index]
                img_next = results[self.key][index]
                # is_padding.append(results[self.key+"/is_padding"][pre_index] and results[self.key+"/is_padding"][index])
                if self.img_type == "gray":
                    if self.clip_range is not None:
                        img_pre = np.clip(img_pre , self.clip_range[0], self.clip_range[1])
                        img_next = np.clip(img_next, self.clip_range[0], self.clip_range[1])
                        # to 0-255
                        img_pre = (255*(img_pre-self.clip_range[0])/(self.clip_range[1]-self.clip_range[0])).astype(np.uint8)
                        img_next = (255*(img_next-self.clip_range[0])/(self.clip_range[1]-self.clip_range[0])).astype(np.uint8)
                    cv_flow = cv2.calcOpticalFlowFarneback(img_pre, img_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    if self.transpose:
                        cv_flow = np.array(cv_flow).transpose(2, 0, 1)
                    flow_imgs.append(cv_flow)
                    if self.debug:
                        h, w = cv_flow.shape[:2]
                        flow_mag, flow_dir = cv2.cartToPolar(cv_flow[...,0], cv_flow[...,1])
                        flow_mag = np.sqrt(flow_mag)
                        flow_dir = flow_dir * 180 / np.pi

                        # 画出光流箭头
                        img_flow = np.zeros((h, w, 3), np.uint8)
                        for y in range(0, h, 10):
                            for x in range(0, w, 10):
                                fx, fy = cv_flow[y, x]
                                cv2.arrowedLine(img_flow, (x, y), (x + int(fx * 10), y + int(fy * 10)), (0, 255, 0), 1)
                        img_name = self.key.replace('/', '_') # remove name unvalid

                        cv2.imwrite(f"process_{img_name}_{index}.jpg", img_flow)
                else:
                    raise NotImplementedError
            flow_imgs = np.array(flow_imgs)
            results[self.output_name] = torch.from_numpy(flow_imgs)
            # results[self.output_name+"/is_padding"] = is_padding
        return results


class GeneratePlainTaskLang(object):
    def __init__(self, key=None, output_key="/task_lang", task_lang=None, debug=False):
        self.key = key
        self.output_key = output_key
        self.task_lang = task_lang
        assert self.task_lang is not None, "task_lang is required"
        self.debug = debug

    def __call__(self, results):
        results[self.output_key] = self.task_lang
        return results
