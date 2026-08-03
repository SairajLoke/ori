import random
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from robotsdl.utils import get_dist_info

from ..runner import iter_runner_type
from ..utils import Registry, build_from_cfg, instantiate_from_config
from .samplers import InfiniteBatchSampler, GroupInBatchSampler
DATASETS = Registry("datasets")
PIPELINES = Registry("pipeline")


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_dataset(cfg, default_args=None):
    from .dataset_wrappers import ConcatDataset
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif "type" in cfg and cfg["type"] == "ConcatDataset":
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg["datasets"]],
            cfg.get("separate_eval", True),
        )
    elif "type" not in cfg and "target" in cfg:
        dataset = instantiate_from_config(cfg)
    elif cfg["type"] in DATASETS._module_dict.keys():
        dataset = build_from_cfg(cfg, DATASETS, default_args)
    else:
        raise ValueError(f'{cfg["type"]}')
    return dataset


def build_dataloader(
    dataset,
    samples_per_gpu,
    workers_per_gpu,
    num_gpus,
    dist,
    seed,
    runner_type,
    shuffle=True,
    pin_memory=True,
):
    rank, world_size = get_dist_info()
    with_seq_flag = getattr(dataset, "with_seq_flag", False)
    if dist:
        # When model is :obj:`DistributedDataParallel`,
        # `batch_size` of :obj:`dataloader` is the
        # number of training samples on each GPU.
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        # When model is obj:`DataParallel`
        # the batch size is samples on all the GPUS
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    if runner_type in iter_runner_type:
        if with_seq_flag:
            batch_sampler = GroupInBatchSampler(
                dataset, batch_size, world_size, rank, seed=seed
            )
        else:
            # this is a batch sampler, which can yield
            # a mini-batch indices each time.
            # it can be used in both `DataParallel` and
            # `DistributedDataParallel`
            batch_sampler = InfiniteBatchSampler(
                dataset, batch_size, world_size, rank, seed=seed, shuffle=shuffle
            )
        batch_size = 1
        sampler = None

    else:
        # TODO: epoch runner
        batch_sampler = None
        sampler = None

    init_fn = (
        partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)
        if seed is not None
        else None
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        # collate_fn=dataset.collate_fn,
        pin_memory=pin_memory,
        worker_init_fn=init_fn,
    )

    return dataloader
