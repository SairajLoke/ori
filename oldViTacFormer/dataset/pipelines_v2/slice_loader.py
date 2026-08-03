from typing import Any
import os
import time
import h5py
import numpy as np
import cv2
import json
from pathlib import Path
import torch
import transforms3d


class SliceLoader:
    """
    load slice infor from hdf5 file
    """

    def __init__(self, key, range, step, using_padding=True, padding_value=0, part=None, out_key=None, dtype=None, hf_conf=None, valid_range_desc=None):
        """
        Args:
            key (str): key to load from hdf5 file
            range (tuple): range of slice index
            step (int): step of slice index
            using_padding (bool): whether using_paded value for trainning without is_padding mask
            padding_value (float or str): using_padding value, or ["closest"]
            dtype (numpy class): output np.array data type
            data_type (str): Hesai data type
            valid_range_desc (str): 对比json中的key, 如果有范围限制则范围外的is_padding为True, 范围内的is_padding为False.
            hf_conf (dict): high frequency config
                - index_key: default is aligned_index
                - use_index_value: whether use index_key to get value
        """
        self.key = key
        self.range = range
        self.step = step
        self.using_padding = using_padding
        self.part = part
        self.padding_value = padding_value
        self.dtype = dtype

        self.valid_range_desc = valid_range_desc
        if out_key is None:
            self.out_key = key
        else:
            self.out_key = out_key

        if isinstance(padding_value, str):
            assert padding_value in ["closest"]
        self.VALID = 0
        self.PADDING_START = 1
        self.PADDING_END = 2

        # high frequency conf
        self.hf_conf = None
        if hf_conf is not None:
            self.hf_conf = hf_conf
            index_key = hf_conf.get("index_key", "aligned_index")
            using_index_key = key.split("/")
            using_index_key[-1] = index_key
            self.index_key = "/".join(using_index_key)
            self.use_index_value = hf_conf.get("use_index_value", False)    # 是否使用index_key进行取值
            
    def load_process(self, data):
        if self.part is not None:
            data = data[slice(*self.part)]
        return data

    def __call__(self, results, file_handle=None) -> Any:
        s_t = time.time()
        data_path = results["data_path"]
        start_ts = results["start_ts"]
        file_handle = results.get("file_handle", None)

        if file_handle is None:
            file = h5py.File(data_path, "r")
            results["file_handle"] = file
        else:
            file = file_handle
        
        if self.key not in file:
            raise ValueError(f"[SliceLoader] key {self.key} not in file {data_path}")
        data = file[self.key]

        # NOTE: format data_type
        if self.hf_conf is not None:
            if not self.use_index_value:
                start_ts = file[self.index_key][start_ts]
            else:
                value_data = file[self.key]
                data = file[self.index_key]

        slices = []
        is_padding = []
        indices = []
        for value in range(self.range[0], self.range[1], self.step):
            index = start_ts + value
            if index < 0:
                is_padding.append(self.PADDING_START)
                indices.append(0)
                continue
            if index >= len(data):
                is_padding.append(self.PADDING_END)
                indices.append(len(data) - 1)
                continue
            slice = self.load_process(data[index])
            slices.append(slice)
            is_padding.append(self.VALID)
            indices.append(index)

        assert (
            len(slices) > 0
        ), f"no slice loaded for {data_path}, {self.key}, {start_ts}, {self.range}, {self.step}"

        example = slices[0]
        is_padding = np.array(is_padding)
        for i, p in enumerate(is_padding):
            if p == self.PADDING_START:
                slices.insert(
                    0,
                    (
                        slices[0]
                        if self.padding_value == "closest"
                        else self.padding_value * np.ones_like(example)
                    ),
                )
            elif p == self.PADDING_END:
                slices.append(
                    slices[-1]
                    if self.padding_value == "closest"
                    else self.padding_value * np.ones_like(example)
                )
            elif p == self.VALID:
                continue
        
        if self.using_padding:
            is_padding[:] = False

        if self.valid_range_desc is not None:
            # assert self.data_type == "0.1.0", "valid_range_desc only support data_type 0.1.0"
            meta_json = Path(data_path).parent / (Path(data_path).stem + f".json")
            if os.path.exists(str(meta_json)):
                with open(str(meta_json), "r") as file:
                    meta_json_data = json.load(file)
                if self.valid_range_desc in meta_json_data:
                    st, ed = meta_json_data[self.valid_range_desc]
                    for i in range(len(is_padding)):
                        if (not is_padding[i]) and ((indices[i] < st) or (indices[i] >= ed)):
                            is_padding[i] = True
                    
        if self.hf_conf is not None and self.use_index_value:
            using_value = []
            for idx in slices:
                using_value.append(value_data[idx])
            using_value = np.stack(using_value)
            slices = using_value

        if self.dtype is None:  # return N * data
            results[self.out_key] = np.array(slices) 
        else:
            results[self.out_key] = np.array(slices).astype(self.dtype)

        results[f"{self.out_key}/is_padding"] = (np.array(is_padding) > 0).astype(np.bool_)
        results[f"{self.out_key}/indices"] = np.array(indices)
        return results


class ImageLoader(SliceLoader):
    def __init__(self, key, range, step, using_padding=True, padding_value=0, is_compressed=True, channel_num=3):
        """
        Args:
            key (str): key to load from hdf5 file
            range (tuple): range of slice index
            step (int): step of slice index
            using_padding (bool): whether using_paded value for trainning without is_padding mask
            padding_value (int or str): using_padding value, or ["closest"]
            is_compressed (bool): whether the image is compressed
        """
        super().__init__(key, range, step, using_padding, padding_value)
        self.is_compressed = is_compressed
        self.channel_num = channel_num

    def load_process(self, data):
        if self.channel_num == 3:
            if self.is_compressed:
                data = cv2.imdecode(data, cv2.IMREAD_COLOR)
        elif self.channel_num == 1:
            if self.is_compressed:
                data = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
                if len(data.shape) == 2:
                    data = np.expand_dims(data, axis=2)
        return data 

    def __call__(self, results) -> Any:
        results = super().__call__(results)
        return results


class LoadTaskCode(object):
    def __init__(
        self,
        key,
        task_code_map,
        env=False,
        env_default_code=0,
        debug=False,
    ):  
        self.key = key
        self.task_code_map = task_code_map
        self.env = env
        self.env_default_code = env_default_code
        self.debug = debug

    def __call__(self, results):
        if not self.env:
            hdf5_path = results["data_path"]
            meta_path = hdf5_path.replace("hdf5", "json")
            cur_ts = results["start_ts"]
            cur_code = None
            with open(meta_path, "r") as file:
                meta_data = json.load(file)
                for name, code in self.task_code_map.items():
                    if name in meta_data:
                        st, ed = meta_data[name]
                        if cur_ts >=st and cur_ts<=ed:
                            cur_code = code
                            break
            if cur_code is not None:
                results[self.key] = torch.tensor(cur_code)
        else:
            cur_code = results[self.key]
            if cur_code is None:
                cur_code = [self.env_default_code]
            results[self.key] = torch.tensor(cur_code)

        return results

class RawRGBDLoader(object):
    def __init__(
        self, 
        rgbd_param, 
        output_key="/observation/images/rgbd_dict",
        data_type="0.1.0",
        env=False, 
        debug=False
    ):
        self.rgbd_param = rgbd_param
        self.output_key = output_key
        self.data_type = data_type
        self.env = env
        rgb_key = [rgbd_conf["rgb"] for rgbd_conf in rgbd_param]
        depth_key = [rgbd_conf["depth"] for rgbd_conf in rgbd_param]
        self.key = rgb_key + depth_key
        self.debug = debug
    
    def __call__(self, results):
        """
        return: dict(
            - rgbd name: [H, W, 4]
        )
        """
        s_ts = time.time()
        rgbd_img = dict()

        if not self.env:
            data_path = results["data_path"]
            start_ts = results["start_ts"]
            with h5py.File(data_path, "r") as root:
                for rgbd_conf in self.rgbd_param:
                    rgb_topic = rgbd_conf["rgb"]
                    rgb = np.array(root[rgb_topic][start_ts])
                    height = rgbd_conf["height"]
                    width = rgbd_conf["width"]
                    if self.data_type == "0.1.0":
                        rgb = cv2.imdecode(rgb, 1)
                    elif self.data_type == "sim1.0.0":
                        rgb = rgb.reshape(height, width, -1)[:, :, :3]

                    depth_topic = rgbd_conf["depth"]
                    depth = np.array(root[depth_topic][start_ts], dtype=np.float32)
                    depth[depth == -np.inf] = 0
                    # clamp depth image to 10 meters to make output image human friendly
                    depth[depth < -10] = -10
                    depth = depth.reshape(height, width, 1)

                    cam_name = rgbd_conf["name"]
                    rgbd_img[cam_name] = np.concatenate([rgb, depth], axis=-1)  # [H, W, 4]
        else:
            for rgbd_conf in self.rgbd_param:
                rgb_topic = rgbd_conf["rgb"]
                depth_topic = rgbd_conf["depth"]
                height = rgbd_conf["height"]
                width = rgbd_conf["width"]
                if self.data_type == "0.1.0":
                    rgb = results["raw"][rgb_topic][..., :3]  # [H, W, C]
                    if rgb_topic == "/observation/images/head_rgbd_rgb":    # head rgb
                        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                    depth = results["raw"][depth_topic].astype(np.float32) # [H, W]-uint16
                    depth = np.expand_dims(depth, axis=-1)  # [H, W, 1]
                elif self.data_type == "sim1.0.0":
                    imgs_dict = results.get("raw_images", None)
                    rgb = imgs_dict[rgb_topic][0][..., :3]  # list
                    depth = results["depths"][depth_topic][0]
                    depth = depth.reshape(height, width, 1)

                depth[depth == -np.inf] = 0
                depth[depth < -10] = -10
                cam_name = rgbd_conf["name"]

                rgbd = torch.from_numpy(np.concatenate([rgb, depth], axis=-1)).unsqueeze(dim=0) # [1, H, W, 4]
                rgbd_img[cam_name] = rgbd

        e_ts = time.time()
        if self.debug:
            print("[LoadRawRGBD] cost time: %.4f"%(e_ts - s_ts))
        
        results[self.output_key] = rgbd_img

        return results

class LoadExtrin(object):
    def __init__(
        self,
        key,    # cam_name
        output_key=None,
        data_type="0.1.0",
        env=False,
        debug=False,
    ):
        self.key = key
        if output_key is None:
            self.output_key = f"{key}/extrinsic"
        else:
            self.output_key = output_key
        self.data_type = data_type
        self.env = env
        self.debug=debug

    def __call__(self, results):
        data_path = results["data_path"]
        if self.env:
            extrin_path = "./result.json"   # NOTE: 暂定固定位置的json path
            with open(extrin_path, "r") as file:
                json_data = json.load(file)
                rt = np.array(json_data).astype(np.float32)
            results[self.output_key] = rt
        else:
            if self.data_type == "0.1.0":
                # read extrinsic from json
                json_path = data_path.replace("hdf5", "json")
                with open(json_path, "r") as file:
                    json_data = json.load(file)
                
                rt = np.array(json_data[f"{self.key}/extrinsic"]).astype(np.float32)
                rt = rt.reshape(4,4)
                results[self.output_key] = rt
            else:
                raise NotImplementedError
        
        return results

class LoadDiscriminatorCode(object):
    def __init__(self, output_key, load_type="stage_cls", data_type="0.1.0", debug=False):
        self.output_key = output_key
        assert load_type in ["stage_cls"]
        self.load_type = load_type
        self.data_type = data_type
        self.debug = debug

    def __call__(self, results):
        if self.data_type == "0.1.0":
            data_path = results["data_path"]
            start_ts = results["start_ts"]
            json_path = data_path.replace("hdf5", "json")
            with open(json_path, "r") as file:
                json_data = json.load(file)
                pick, place, seq_end = json_data["pick_n_place"]
                if self.load_type == "stage_cls":
                    if start_ts < pick:
                        results[self.output_key] = 0
                    elif start_ts < place:
                        results[self.output_key] = 1
                    else:
                        results[self.output_key] = 2
                results[self.output_key] = torch.tensor(results[self.output_key])
        else:
            raise NotImplementedError
        
        return results

class LoadTaskCodeV2(object):
    """
    根据给定的TASK_MAP给定task_code
    """
    def __init__(self, task_map, output_key="/task_code", debug=False):
        self.task_map = task_map
        self.output_key = output_key
        self.debug = debug

    def __call__(self, results):
        data_path = results["data_path"]
        task_version = data_path.split("/")[-5]
        task_name = data_path.split("/")[-6]

        task_code_name = f"{task_name}/{task_version}"
        if task_code_name not in self.task_map:
            raise ValueError(f"[LoadTaskCodeV2] task_code_name {task_code_name} not in task_map")
        results[self.output_key] = torch.tensor(self.task_map[task_code_name])

        return results
    
class EndpointLoader(object):
    def __init__(self, key, range, dtype, end_length=100):
        """
        Args:
            key (str): key to load from hdf5 file
            range: whether output single signal or chunk_size length signals
            end_length (int): the number of frames to label as 1 at the end
        """
        self.key =  key
        self.range = range
        self.end_length = end_length
        self.dtype = dtype

    def __call__(self, results, file_handle=None) -> dict:
        data_path = results["data_path"]
        start_ts = results["start_ts"]
        file_handle = results.get("file_handle", None)

        if file_handle is None:
            file = h5py.File(data_path, "r")
            results["file_handle"] = file  # 将文件句柄保存到results中以便后续重用
        else:
            file = file_handle

        if self.key not in file:
            raise ValueError(f"[EndpointLoader] key {self.key} not in file {data_path}")

        data = file[self.key]
        data_length = len(data)
        labels = np.zeros(data_length + self.range[1])
        labels[-(self.range[1] + self.end_length):] = 1

        label = labels[start_ts: start_ts + self.range[1]]

        if self.dtype is None:
            results['endpoint/gt'] = np.array(label)
        else:
            results['endpoint/gt'] = np.array(label).astype(self.dtype)

        return results
