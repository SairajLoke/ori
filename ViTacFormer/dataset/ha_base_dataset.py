import os
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from pathlib import Path
import tqdm
from .pipelines_v2.compose import Compose
from types import SimpleNamespace
from .normalizer import LinearNormalizer, SingleFieldLinearNormalizer


class HaBaseV2Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_config,
        data_type="0.1.0",
        normalize_keys=[],
        norm_stats_cache=None,
        stat_pipeline=None,
        stat_sample_step=1,
        stat_worker=6,
    ):
        self.dataset_config = dataset_config
        self.data_type = data_type

        self.normalize_keys = normalize_keys
        self.norm_stats_cache = norm_stats_cache
        self.stat_worker = stat_worker
        self.stat_sample_step = stat_sample_step
        self.stat_pipeline = stat_pipeline

        self.stat_path = self.dataset_config.get("stat_path", None)
        if self.stat_path is not None and isinstance(self.stat_path, str):
            self.stat_root = [self.stat_path]

        self.norm_stats = dict()

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    # @master_first
    def get_stats(self, stat_pipeline):
        if self.norm_stats_cache is not None and os.path.exists(self.norm_stats_cache):
            self.stat_path = ["None"]
            with open(self.norm_stats_cache, "rb") as f:
                self.norm_stats = pickle.load(f)
        else:
            self.norm_stats = self.get_norm_stats(stat_pipeline)
            self.stat_path = self.data_root
            if self.norm_stats_cache is not None:
                Path(self.norm_stats_cache).parent.mkdir(parents=True, exist_ok=True)
                with open(self.norm_stats_cache, "wb") as f:
                    pickle.dump(self.norm_stats, f)
                os.system(f"chmod 777 {self.norm_stats_cache}")

        # self.normalizer = SimpleNamespace()
        for key in self.normalize_keys:
            stats = self.norm_stats[key]
            # self.normalizer.__dict__[key] = {
            #     "mean": torch.tensor(stats["mean"], dtype=torch.float32),
            #     "std": torch.tensor(stats["std"], dtype=torch.float32),
            #     "min": torch.tensor(stats["min"], dtype=torch.float32),
            #     "max": torch.tensor(stats["max"], dtype=torch.float32),
            # }

            print(
                f"normalized key: {key}"
                f"\n\tmean: " + ", ".join(['{:0.4f}'.format(v) for v in stats['mean']])
                + f"\n\tstd: " + ", ".join(['{:0.4f}'.format(v) for v in stats['std']])
                + f"\n\tmin: " + ", ".join(['{:0.4f}'.format(v) for v in stats['min']])
                + f"\n\tmax: " + ", ".join(['{:0.4f}'.format(v) for v in stats['max']])
            )

    def get_norm_stats(self, stat_pipeline):
        assert stat_pipeline is not None
        self.pipeline = Compose(stat_pipeline)

        indices = list(range(0, len(self), self.stat_sample_step))
        sampler = SubsetRandomSampler(indices)
        statistic_loader = DataLoader(
            self,
            sampler=sampler,
            batch_size=self.stat_worker,
            shuffle=False,
            num_workers=self.stat_worker,
            pin_memory=False,
        )

        data_norm_stats = {}
        for data in tqdm.tqdm(statistic_loader):
            for key in self.normalize_keys:
                if key not in data_norm_stats:
                    data_norm_stats[key] = []
                data_norm_stats[key].append(data[key].numpy())
            del data

        norm_stats = {}
        for key in self.normalize_keys:
            data = np.concatenate(data_norm_stats[key], axis=0)
            data = data.reshape(-1, data.shape[-1])

            data_std = np.std(data, axis=0)
            data_std = np.where(abs(data_std) < 1e-3, 1, data_std)
            norm_stats[key] = {
                "mean": np.mean(data, axis=0),
                "std": data_std,
                "min": np.min(data, axis=0),
                "max": np.max(data, axis=0),
            }
        return norm_stats

    def get_normalizer(self, mode="gaussian", output_max=1.0, output_min=-1.0, range_eps=1e-4, fit_offset=True):

        normalizer = LinearNormalizer()

        for key in self.normalize_keys:
            stats = self.norm_stats[key]
            mean = torch.tensor(stats["mean"], dtype=torch.float32)
            std = torch.tensor(stats["std"], dtype=torch.float32)
            min_ = torch.tensor(stats["min"], dtype=torch.float32)
            max_ = torch.tensor(stats["max"], dtype=torch.float32)

            if mode == "limits":
                if fit_offset:
                    input_range = max_ - min_
                    ignore_dim = input_range < range_eps
                    input_range = input_range.clone()
                    input_range[ignore_dim] = output_max - output_min
                    scale = (output_max - output_min) / input_range
                    offset = output_min - scale * min_
                    offset[ignore_dim] = (output_max + output_min) / 2 - min_[ignore_dim]
                else:
                    output_abs = min(abs(output_min), abs(output_max))
                    input_abs = torch.maximum(torch.abs(min_), torch.abs(max_))
                    ignore_dim = input_abs < range_eps
                    input_abs = input_abs.clone()
                    input_abs[ignore_dim] = output_abs
                    scale = output_abs / input_abs
                    offset = torch.zeros_like(mean)
            elif mode == "gaussian":
                ignore_dim = std < range_eps
                scale = std.clone()
                scale[ignore_dim] = 1.0
                scale = 1.0 / scale

                if fit_offset:
                    offset = -mean * scale
                else:
                    offset = torch.zeros_like(mean)
            else:
                raise ValueError(f"Unsupported normalization mode: {mode}")

            input_stats = {
                "mean": mean,
                "std": std,
                "min": min_,
                "max": max_,
            }

            normalizer[key] = SingleFieldLinearNormalizer.create_manual(
                scale=scale,
                offset=offset,
                input_stats_dict=input_stats
            )

        return normalizer