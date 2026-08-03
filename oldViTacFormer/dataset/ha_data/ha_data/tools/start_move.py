import h5py
import bisect
import logging
import os
import numpy as np
import json
from pathlib import Path
from ha_data.data.naming_rule import *
from ha_data.data.train_meta import TrainMeta

def get_start_index(path_list, mode_drop):
    total_frame_num = 0

    thresh_hold = 0.0025
    offset = 5 
    start_key = "anno/start_move/step_table"

    for season_folder in path_list:
        if  not os.path.exists(season_folder):
            print('path not exists', season_folder)
            exit()
        print('Process ', season_folder)
        try:
            with open(os.path.join(season_folder, 'train.json'), 'r') as f:
                meta = json.load(f)
                data_version = meta['content']['data']['default']
                naming = deduce_naming(data_version)
                print(f'Using data version {data_version} from train.json')
        except Exception as e:
            print(f'Failed to read train.json: {e}')
            exit()
        train_meta = TrainMeta(season_folder, naming)
        if train_meta.context.meta == {}:
            print(f'{season_folder}/train.json not exists!!!!!!')
            exit()
        if naming == '':
            print('failed to get default data from', season_folder)
            exit()
        image = naming.IMAGE_PREFIX + "head/stereo/lefteye/rgb"
        j_data, key = train_meta.data.get(naming.DATA_VERSION)
        train_meta.anno.init('data/'+ key)
        for hdf5_p in j_data["url"]:
            h5_path = os.path.join(season_folder, hdf5_p)
            print(h5_path)
            with h5py.File(h5_path, "r") as root:
                seq_length = root[image].shape[0]
                total_frame_num += seq_length
                h5_name = hdf5_p.split('/')[-1][:-5]
                h5_file = os.path.join(season_folder, train_meta.get_anno_h5_path(hdf5_p))
                Path(os.path.dirname(h5_file)).mkdir(exist_ok=True)
                if mode_drop:
                    mode = root['/lbto/mode']
                    mode = mode['mode']
                    index_mode_0 = next(i for i, value in enumerate(mode) if value == 0)
                    dagger_start_key = "anno/dagger/step_table"
                    with h5py.File(h5_file, 'w') as h5:
                        h5.create_dataset(dagger_start_key, data=[index_mode_0, seq_length])
                    print(f"save {h5_name}_anno.hdf5, start idx: {index_mode_0}, end_idx: {seq_length}")
                    continue
                try:
                    for idx in range(seq_length-1):
                        dif = np.sum(np.abs(root[image][idx+1] - root[image][idx]))
                        # print(dif)
                        if dif > thresh_hold:
                            cur_st_idx = max(0, idx)
                            break
                    cur_st_idx += offset
                    if naming.name[:9] == "Train_0_3" or naming.name[:9] == "Train_0_2":
                        # get real index from anno/aligned
                        aligned_idx = root[naming.ALIGN_PREFIX]
                        aligned_st_idx = bisect.bisect_left(aligned_idx, cur_st_idx)
                        if aligned_st_idx < len(aligned_idx) and aligned_idx[aligned_st_idx] < cur_st_idx:
                            aligned_st_idx += 1
                        cur_st_idx = aligned_st_idx
                        seq_length = len(root[naming.ALIGN_PREFIX])
                    with h5py.File(h5_file, 'w') as h5:
                        h5.create_dataset(start_key, data=[cur_st_idx, seq_length])
                    print(f"save {h5_name}_anno.hdf5, start idx: {cur_st_idx}, end_idx: {seq_length}")
                except:
                    print(f"{h5_file} failed!")
                    print(season_folder)
                    print(h5_name)
        train_meta.anno.add()
        train_meta.context.dump()
        print(f'finished {season_folder} start move annotation')

    print(f"find {total_frame_num} frames!")

def get_max_img_length():
    path_list = [
        "",
    ]
    hdf5_list = []
    camera_name_list = [
        '/observation/images/head_stereo_left_rgb',
        '/observation/images/head_stereo_right_rgb',
        '/observation/images/head_rgbd_rgb',
        '/observation/images/right_wrist_rgbd_rgb',
        '/observation/images/chest_rgbd_rgb',
    ]
    image_max_length_list = [0 for _ in range(len(camera_name_list))]
    image_min_length_list = [1e10 for _ in range(len(camera_name_list))]

    for folder_p in path_list:
        cur_hdf5 = [str(p) for p in Path(folder_p).glob("*.hdf5")]
        hdf5_list.extend(cur_hdf5)
    print(f"find {len(hdf5_list)} hdf5 files!")

    for hdf5_p in hdf5_list:
        with h5py.File(hdf5_p, "r") as root:
            for i, c_name in enumerate(camera_name_list):
                max_l = root[c_name].shape[1]
                max_l = max(max_l, image_max_length_list[i])
                image_max_length_list[i] = max_l

                min_l = min(max_l, image_min_length_list[i])
                image_min_length_list[i] = min_l

    print(image_max_length_list)
    print(image_min_length_list)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_list", type=str, default="path_list.txt")
    parser.add_argument("--dagger_mode_drop", type=bool, default=False)
    args = parser.parse_args()
    print(f"Args: {args}")

    # more options
    path_list = args.path_list
    if path_list.endswith(".txt"):
        with open(args.path_list, 'r') as f:
            path_list = f.read().strip().split('\n')
    elif os.path.isdir(path_list):
        path_list = [path_list]

    print('Make anno start_move for the following:\n', len(path_list), path_list)
    get_start_index(path_list, args.dagger_mode_drop)
    # get_max_img_length()
