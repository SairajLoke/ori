import json
import os
import pickle
import copy
import math
from pathlib import Path

import h5py
import numpy as np
import torch
import time

# from .builder import DATASETS
from .pipelines_v2.compose import Compose
from .utils import find_all_hdf5
from .ha_base_dataset import HaBaseV2Dataset

try:
    from ha_data import TrainMeta
except:
    print("please install ha_data repository")


# @DATASETS.register_module()
class HaPipelineV2DatasetD010(HaBaseV2Dataset):
    def __init__(
        self,
        dataset_config,
        skip_mirrored_data=False,
        test_mode=False,
        seed=None,
        limit_num=None,
        seq_list=None,
        skip=None,
        debug=False,
        norm_stats_cache=None,
        pipeline=None,
        data_type="0.1.0",
        normalize_keys=[],
        length_keys=None,
        stat_pipeline=None,
        stat_sample_step=1,
        stat_worker=6,
        prepare_padding=True,
        action_range=None,
        work_dir=None, # 保存train val list
        **kwargs,
    ):

        super().__init__(
            dataset_config=dataset_config,
            data_type=data_type,
            normalize_keys=normalize_keys,
            norm_stats_cache=norm_stats_cache,
            stat_pipeline=stat_pipeline,
            stat_sample_step=stat_sample_step,
            stat_worker=stat_worker,
        )
        torch.multiprocessing.set_sharing_strategy('file_system')

        self.skip_mirrored_data = skip_mirrored_data
        self.test_mode = test_mode
        self.seed = seed
        self.limit_num = limit_num
        self.norm_stats_cache = norm_stats_cache
        self.seq_list = seq_list
        if seq_list is not None:
            with open(seq_list, "r", encoding="utf-8") as file:
                self.seq_list = [line.strip() for line in file]
        self.skip = skip
        self.work_dir = work_dir
        assert len(self.normalize_keys) > 0, "normalize_keys should not be empty"
        if length_keys is not None:
            self.length_keys = length_keys
        else:
            self.length_keys = self.normalize_keys

        self.data_root = self.dataset_config["data_root"]
        self.root_weights = self.dataset_config.get("root_weights", None)
        self.train_ratio = self.dataset_config.get("train_ratio", None)
        self.kinematic_metas = self.dataset_config.get("kinematic_metas", None)
        self.name_filter = self.dataset_config.get("name_filter", None)
        self.process_range_desc = self.dataset_config.get("process_range_desc", None)
        self.rel_stats_step = self.dataset_config.get("rel_stats_step", None)

        self.prepare_padding = prepare_padding
        self.action_range = action_range

        if isinstance(self.data_root, str):
            self.data_root = [self.data_root]

        origin_data_root = None
        if self.stat_path is not None:
            origin_data_root = copy.deepcopy(self.data_root)
            self.data_root = self.stat_root
            self.root_weights = None # no weighted for stat
        self.init_dataset()

        self.get_stats(stat_pipeline)

        if origin_data_root is not None:
            self.data_root = origin_data_root
            # recover root weights
            self.root_weights = self.dataset_config.get("root_weights", None)
            self.init_dataset()

        self.debug = debug
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

    def init_dataset(self):
        self.init_data_paths()
        self.save_data_paths()

        self.all_episode_len = self.get_all_hdf5_length()
        self.max_episode_len = max(self.all_episode_len)
        print(
            f"Min & Max & Sum & Count episode length: {min(self.all_episode_len)}"
            f" & {max(self.all_episode_len)} & {sum(self.all_episode_len)} & {len(self.all_episode_len)}"
        )
        min_index = self.all_episode_len.index(min(self.all_episode_len))
        max_index = self.all_episode_len.index(max(self.all_episode_len))
        print(
            f"min data path: {self.data_paths[min_index]}, "
            f"max data path: {self.data_paths[max_index]}"
        )

        self.prepare_samples()
        self.save_samples()

    def save_data_paths(self):
        if self.work_dir is None: return
        save_fname = ('val' if self.test_mode else 'train') + f"_list_{self.seed}.txt"
        save_path = os.path.join(self.work_dir, save_fname)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            for p in self.data_paths:
                f.write(f"{p}\n")
        print(f"[HaPipelineV2Dataset] save seq_list {save_path}")

    def save_samples(self):
        if self.work_dir is None or self.root_weights is None: return
        save_fname = ('val' if self.test_mode else 'train') + f"_samples_{self.seed}.pkl"
        save_path = os.path.join(self.work_dir, save_fname)
        with open(save_path, "wb") as f:
            pickle.dump({
                "samples": self.data_paths,
                "root_weights": self.root_weights,
            }, f)

    def init_data_paths(self):
        """
        init data_paths & data_paths_root_index
        """
        self.data_paths = []
        self.data_paths_root_index = []
        total_data_paths = []
        for ti, path in enumerate(self.data_root):
            data_paths = find_all_hdf5(path, self.skip_mirrored_data)
            print(f"{path} Found {len(data_paths)} hdf5 files")
            assert len(data_paths) > 0, f"Found 0 hdf5 files in {path}"
            total_data_paths.extend(data_paths)
            if self.root_weights is not None:
                weight = self.root_weights[ti]
                print(f"Data root weight: {weight}")

            if self.name_filter is not None:
                data_paths = [path for path in data_paths if self.name_filter(path)]

            if self.seq_list is not None:
                used_data_path = []
                for d in data_paths:
                    if d in self.seq_list:
                        used_data_path.append(d)
                data_paths = used_data_path
                print(f"using seq-list get {len(data_paths)} episodes")
            else:
                num_episodes = len(data_paths)
                # random split train-test dataset
                if self.seed is not None:
                    np.random.seed(self.seed)
                    indices = np.random.permutation(num_episodes)
                else:
                    np.random.seed(0)
                    indices = np.arange(num_episodes)
                data_paths = [data_paths[idx] for idx in indices]
                name_indices = [Path(path).stem.split("_")[-1] for path in data_paths]
                print(f"seed: {self.seed}")

                if self.train_ratio is not None:
                    val_ratio = (1 - self.train_ratio) / len(self.data_root)
                    count = int(num_episodes * (1 - val_ratio))
                    if self.test_mode:
                        data_paths = data_paths[count:]
                        print(f"filter names: {name_indices[count:]}")
                        print(f"filter indices: {indices[count:]}")
                    else:
                        data_paths = data_paths[:count]
                        print(f"filter names: {name_indices[:count]}")
                        print(f"filter indices: {indices[:count]}")

            self.data_paths.extend(data_paths)
            self.data_paths_root_index.extend([ti for _ in range(len(data_paths))])

        # print(f"used paths: {self.data_paths}")
        return total_data_paths

    def get_all_hdf5_length(self):
        sample_key = None
        length_list = []
        for h5file in self.data_paths:
            with h5py.File(h5file, "r") as f:
                if sample_key is None:
                    for key in self.length_keys:
                        if key in f:
                            sample_key = key
                            break
                length_list.append(len(f[sample_key]))
        return length_list

    def prepare_samples(self):
        """Filter samples."""
        final_data_paths = []
        final_weights = []
        final_process_range_desc = [None for _ in range(len(self.data_root))]
        episode_samples_len_list = [[] for _ in range(len(self.data_root))]
        weights_by_root = [0 for _ in range(len(self.data_root))]
        for ti, path, episode_len in zip(
            self.data_paths_root_index, self.data_paths, self.all_episode_len
        ):
            cur_data_paths = []

            # can choose different process_range_desc for different root
            # process_range_desc = ['a', 'b']
            # or process_range_desc = [['a', 'b'], None, ['c', 'd'],...] by root
            process_range_desc = (
                self.process_range_desc[ti]
                if (self.process_range_desc is not None)
                and any([isinstance(desc, list) for desc in self.process_range_desc])
                else self.process_range_desc
            )
            final_process_range_desc[ti] = process_range_desc
            if process_range_desc is not None:
                assert self.prepare_padding, "no prepare_padding is not implemented"
                assert self.data_type == "0.1.0"
                # NOTE: read meta_json to filter range task
                assert all(
                    [isinstance(desc, str) for desc in process_range_desc]
                ), "process_range_desc should be a list of strings"

                meta_json = Path(path).parent / (Path(path).stem + f".json")
                with open(str(meta_json), "r") as file:
                    meta_json_data = json.load(file)
                    # step_table = meta_json_data["step_table"]

                process_index = [(path, idx) for idx in range(episode_len)]
                for need_process_info in process_range_desc:
                    if need_process_info in meta_json_data:
                        st, ed = meta_json_data[need_process_info]
                        st = max(0, st)
                        ed = min(episode_len, ed)
                        process_index = [(path, idx) for idx in range(st, ed)]  # NOTE:后者覆盖
                cur_data_paths.extend(process_index)
            else:
                if self.prepare_padding:
                    cur_data_paths.extend([(path, idx) for idx in range(episode_len)])
                else:
                    assert self.action_range is not None, "action_range should be set"
                    cur_data_paths.extend([
                        (path, idx) for idx in range(0, episode_len - self.action_range)
                    ])
            if self.root_weights is not None:
                weight = self.root_weights[ti]
                episode_weights = [weight for _ in range(len(cur_data_paths))]
                final_weights.extend(episode_weights)
                weights_by_root[ti] += sum(episode_weights)
            final_data_paths.extend(cur_data_paths)
            episode_samples_len_list[ti].append(len(cur_data_paths))

        assert len(final_data_paths) <= sum(self.all_episode_len)
        self.data_paths = final_data_paths
        self.episode_samples_len_list = episode_samples_len_list
        self.weights = final_weights if len(final_weights) > 0 else None
        print("process range desc:")
        for ti in range(len(self.data_root)):
            print(f"\t({self.data_root[ti]}): {final_process_range_desc[ti]}")
        print(f"final_data_paths: {len(final_data_paths)}")

        if self.weights is not None:
            print(f"final_weights:\n")
            sum_weights = sum(weights_by_root)
            for ti in range(len(self.data_root)):
                print(
                    f"\t({self.data_root[ti]}) with samples {sum(episode_samples_len_list[ti])}: "
                    f"sum weight {weights_by_root[ti] / sum_weights:.2f}"
                )

        if self.limit_num is not None:
            if isinstance(self.limit_num, int):
                self.data_paths = self.data_paths[: self.limit_num]
                if self.weights is not None:
                    self.weights = self.weights[: self.limit_num]
            else:
                self.data_paths = self.data_paths[self.limit_num[0] : self.limit_num[1]]
                if self.weights is not None:
                    self.weights = self.weights[self.limit_num[0] : self.limit_num[1]]
            print(
                f"limit_num: {self.limit_num}, final_data_paths: {len(self.data_paths)}"
            )

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        data_path, start_ts = self.data_paths[index]

        start_process = time.time()
        inputs = dict(
            data_path=data_path,
            start_ts=start_ts,
            norm_stats=self.norm_stats,
            max_episode_len=self.max_episode_len,
        )
        inputs = self.pipeline(inputs)

        for key, val in inputs.items():
            if isinstance(val, torch.Tensor) and val.ndim == 4 and val.shape[0] == 1:
                inputs[key] = val.squeeze(0)  # 去掉 batch 维


        inputs['pipeline_time'] = time.time() - start_process

        if "file_handle" in inputs:
            inputs["file_handle"].close()
            del inputs["file_handle"]

        return inputs


# @DATASETS.register_module()
class HaPipelineV2DatasetD020(HaPipelineV2DatasetD010):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # sequence-training
        self.multi_frame_conf = kwargs.get("multi_frame_conf", None)
        if self.multi_frame_conf is not None:
            self.with_seq_flag = self.multi_frame_conf.get("with_seq_flag", False)
            self.sequences_split_num = self.multi_frame_conf.get("sequences_split_num", 1)    # 每段sequence拆分数量
            self.start_offset = self.multi_frame_conf.get("start_offset", [0,0])  # 开始取值时
            self.skip_range = self.multi_frame_conf.get("skip_range", None)  # 跳帧范围
            self.skip_prob = self.multi_frame_conf.get("skip_prob", 0.0)    # 跳帧概率
            if self.with_seq_flag:
                self._set_sequence_group_flag()

    def __getitem__(self, index):
        # support sequence training
        if isinstance(index, dict):
            # print(f"[HaPipelineV2DatasetD020] index: {index['idx']} | group_id: {index['group_id']}")
            idx = index["idx"]
            seq_start_flag = index["seq_start_flag"]
        else:
            idx = index
            seq_start_flag = False

        results = super().__getitem__(idx)
        results["seq_start_flag"] = seq_start_flag
        return results

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        if self.sequences_split_num != 1:
            if self.sequences_split_num == "all":
                self.flag = np.array(
                    range(len(self.data_paths)), dtype=np.int64 # 每个为单独的flag
                )
            else:
                bin_counts = np.bincount(self.flag) # 每段数量
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(
                            range(
                                0,
                                bin_counts[curr_flag],
                                math.ceil(
                                    bin_counts[curr_flag]
                                    / self.sequences_split_num
                                ),
                            )
                        )
                        + [bin_counts[curr_flag]]
                    )

                    for sub_seq_idx in (
                        curr_sequence_length[1:] - curr_sequence_length[:-1]
                    ):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                # NOTE： 按指定的seq分段数量进行分段
                assert len(new_flags) == len(self.flag)
                assert (
                    len(np.bincount(new_flags))
                    == len(np.bincount(self.flag)) * self.sequences_split_num
                )
                self.flag = np.array(new_flags, dtype=np.int64)

    def prepare_samples(self):
        """
        reading anno files to filter offline episodes
        """
        final_data_paths = []
        group_inds_by_sample = []
        for idx in range(len(self.data_paths)):
            data_p = self.data_paths[idx]
            anno_p = self.anno_paths[idx]
            episode_len = self.all_episode_len[idx]
            cur_data_paths = []

            process_index = [(data_p, idx) for idx in range(episode_len)]
            if self.process_range_desc is not None:
                with h5py.File(anno_p, "r") as root:
                    for need_process_info in self.process_range_desc:
                        if need_process_info in root:
                            # print(f"need_process_info={need_process_info}, root[need_process_info]={root[need_process_info][:]}")
                            st, ed = root[need_process_info][:].flatten().tolist()
                            ed = min(episode_len, ed)
                            # ed -= 50
                            process_index = [(data_p, idx) for idx in range(st, ed)]
            cur_data_paths.extend(process_index)
            final_data_paths.extend(cur_data_paths)
            group_inds_by_sample.extend([idx for _ in range(len(process_index))])

        assert len(final_data_paths) <= sum(self.all_episode_len)
        self.data_paths = final_data_paths
        self.group_inds_by_sample = np.array(group_inds_by_sample, dtype=np.int64)
        print(f"[HaPipelineV2DatasetD02] process_range_desc: {self.process_range_desc}")
        print(f"[HaPipelineV2DatasetD02] final samples: {len(final_data_paths)}, filter samples: {sum(self.all_episode_len)-len(final_data_paths)}")
        if self.limit_num is not None:
            if isinstance(self.limit_num, int):
                self.data_paths = self.data_paths[: self.limit_num]
                self.group_inds_by_sample = self.group_inds_by_sample[: self.limit_num]
            else:
                self.data_paths = self.data_paths[self.limit_num[0] : self.limit_num[1]]
                self.group_inds_by_sample = self.group_inds_by_sample[self.limit_num[0] : self.limit_num[1]]
            print(
                f"[HaPipelineV2DatasetD02] limit_num: {self.limit_num}, final samples: {len(self.data_paths)}"
            )
        assert self.group_inds_by_sample.shape[0] == len(self.data_paths)

    def get_all_hdf5_length(self):
        sample_key = None
        length_list = []
        for h5file in self.data_paths:
            with h5py.File(h5file, "r") as f:
                if sample_key is None:
                    for key in self.length_keys:
                        index_key = key.split("/")
                        index_key[-1] = "aligned_index"
                        index_key = "/".join(index_key)
                        if index_key in f:
                            sample_key = index_key
                            break
                        elif key in f:
                            sample_key = key
                            break
                length_list.append(len(f[sample_key]))
        return length_list

    def init_data_paths(self):
        self.data_paths = []
        self.anno_paths = []

        for ti, path in enumerate(self.data_root):
            try:
                meta = TrainMeta(path)
            except:
                print(f"[HaPipelineV2DatasetD02] read {path} fail!")
                continue
            # read anno & data
            anno_j, _ = meta.anno.get_from_simple_version(self.data_type.split("."))
            anno_paths = anno_j["url"]
            anno_paths = [os.path.join(path, d) for d in anno_paths]
            # print(len(anno_files), anno_files[0], anno_files[-1])
            data_key = anno_j["depend"]
            data_j, data_version = meta.data.get(data_key.split("/")[-1])
            data_paths = data_j["url"]
            data_paths = [os.path.join(path, d) for d in data_paths]
            print(f"[HaPipelineV2DatasetD02] {path}: total find {len(data_paths)} episodes!")
            if self.seq_list is not None:
                seq_filter_datapath = []
                seq_filter_annopath = []
                for idx, d in enumerate(data_paths):
                    if d in self.seq_list:
                        seq_filter_datapath.append(d)
                        seq_filter_annopath.append(anno_paths[idx])
                data_paths = seq_filter_datapath
                anno_paths = seq_filter_annopath
            else:
                num_episodes = len(data_paths)
                if self.seed is not None:
                    np.random.seed(self.seed)
                    indices = np.random.permutation(num_episodes)
                else:
                    np.random.seed(0)
                    indices = np.arange(num_episodes)

                data_paths = [data_paths[idx] for idx in indices]
                anno_paths = [anno_paths[idx] for idx in indices]
                if self.train_ratio is not None:
                    val_ratio = 1 - self.train_ratio
                    count = int(num_episodes * (1 - val_ratio))
                    if self.test_mode:
                        data_paths = data_paths[count:]
                        anno_paths = anno_paths[count:]
                    else:
                        data_paths = data_paths[:count]
                        anno_paths = anno_paths[:count]

            if self.skip is not None:
                data_paths = data_paths[::2]

            print(f"[HaPipelineV2DatasetD02] {path}: load {len(data_paths)} episodes")
            self.data_paths.extend(data_paths)
            self.anno_paths.extend(anno_paths)

        print(f"[HaPipelineV2DatasetD02] Total load {len(self.data_paths)} episodes")
        print(f"[HaPipelineV2DatasetD02] Using hdf5 paths: {self.data_paths}")
        return self.data_paths


    def postprocess(self, batch, device, use_tactile):

        from torchvision.transforms.functional import resize

        # obs_dict = {}
        result_dict = {}

        camera_names = [
            "/observe/vision/head/stereo/lefteye/rgb",
            "/observe/vision/head/stereo/righteye/rgb",
            "/observe/vision/right_wrist/fisheye/rgb",
            "/observe/vision/left_wrist/fisheye/rgb",
        ]

        resize_shape = (224, 320)  # H, W
        B = batch[camera_names[0]].shape[0]
        cam_images = []

        for cam_name in camera_names:
            image = batch[cam_name].to(device)  # shape: [B, 3, H, W]
            resized_list = []

            for b in range(B):
                img_b = image[b]  # [3, H, W]
                resized = resize(img_b, resize_shape)  # → [3, 224, 320]
                resized_list.append(resized)

            resized_images = torch.stack(resized_list, dim=0)  # [B, 3, 224, 320]
            result_dict[cam_name] = resized_images
            cam_images.append(resized_images)

        result_dict["image"] = torch.stack(cam_images, dim=1)

        qpos_keys = [
            "/state/right_arm/joint_angle",
            "/state/right_hand/joint_angle",
            "/state/left_arm/joint_angle",
            "/state/left_hand/joint_angle",
            "/state/neck/joint_angle",
        ]
        qpos = torch.cat([batch[k] for k in qpos_keys], dim=-1)  # [T，D]
        result_dict["lowdim"] = qpos.to(device)

        if use_tactile:
            tac_key = "/observe/tactile/total_force"
            tactile = batch[tac_key].to(device)
            result_dict["tactile"] = tactile

            tac_next_key = "/observe/tactile/total_force/next"
            tactile_next = batch[tac_next_key].to(device)
            result_dict["tactile_next"] = tactile_next

        action_keys = [
            "/action/right_arm/joint_angle/rel",
            "/action/right_hand/joint_angle/rel",
            "/action/left_arm/joint_angle/rel",
            "/action/left_hand/joint_angle/rel",
            "/action/neck/joint_angle/rel",
        ]
        action = torch.cat([batch[k] for k in action_keys], dim=-1)

        action_abs_keys = [
            "/action/right_arm/joint_angle",
            "/action/right_hand/joint_angle",
            "/action/left_arm/joint_angle",
            "/action/left_hand/joint_angle",
            "/action/neck/joint_angle",
        ]
        action_abs = torch.cat([batch[k] for k in action_abs_keys], dim=-1)


        B, T = action.shape[:2]
        action_pad_mask = torch.zeros((B,T), dtype=torch.bool, device=device)  # [T]

        result_dict["action"] = action.to(device)  # [B, T, D_action]
        result_dict["action_mask"] = action_pad_mask  # [B, T]
        result_dict["action_abs"] = action_abs.to(device)

        return result_dict