# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process, is_dist_avail_and_initialized

from .position_encoding import build_position_encoding

import IPython
e = IPython.embed

try:
    from my_utils.ori_logging import get_logger
except ImportError:  # pragma: no cover -- detr used standalone
    import logging as _logging

    def get_logger(name):
        return _logging.getLogger("ori." + name)

log = get_logger("backbone")


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out


def _load_local_backbone_weights(backbone: nn.Module, path: str):
    """Load ResNet weights from a local file with strict=True.

    Expects a plain torchvision state_dict, e.g. resnet18-f37072fd.pth or the
    copy in assets/backbones/. strict=True means a file that does not match the
    architecture exactly is an error -- silently accepting a partial match would
    leave part of the backbone randomly initialised.

    Verified against the stock resnet18-f37072fd.pth (102 keys, byte-identical
    to the copy in assets/) and against the failure cases: a truncated file, a
    resnet34 state_dict, and a full ACT checkpoint all raise RuntimeError rather
    than partially loading.

    (FrozenBatchNorm2d also deletes any num_batches_tracked keys in
    _load_from_state_dict before the strict check, so files saved from a live
    nn.Module load too, even though the published torchvision weights carry
    none.)
    """
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    backbone.load_state_dict(state_dict, strict=True)
    log.info("backbone weights loaded from %s (strict)", path)


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 weights_path: str = None):
        # Upstream ACT/DETR passes `pretrained=is_main_process()`, which loads
        # ImageNet weights on rank 0 ONLY -- every other rank builds a randomly
        # initialised ResNet and relies on DDP's construction-time broadcast to
        # overwrite it. That is fine upstream (single process => always True)
        # but here it is silent, load-bearing luck: any code path that reads the
        # weights before accelerator.prepare() sees garbage on ranks 1..N-1.
        #
        # Instead: every rank loads real pretrained weights, and a barrier makes
        # rank 0 populate the torch-hub cache first so the ranks never race on
        # the download.
        #
        # weights_path (configs.BACKBONE_WEIGHTS / BACKBONE_WEIGHTS env var)
        # bypasses torch hub entirely: the architecture is built with random
        # weights and the state_dict is loaded from your own file. Useful on
        # nodes with no network or no writable TORCH_HOME, and required if you
        # want a backbone pretrained on something other than ImageNet.
        _distributed = is_dist_avail_and_initialized()
        if weights_path:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                weights=None, norm_layer=FrozenBatchNorm2d)
            _load_local_backbone_weights(backbone, weights_path)
        else:
            if _distributed and not is_main_process():
                torch.distributed.barrier()
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                weights="DEFAULT", norm_layer=FrozenBatchNorm2d)
            if _distributed and is_main_process():
                torch.distributed.barrier()
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    weights_path = getattr(args, "backbone_weights", None)
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation,
                        weights_path=weights_path)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels

    log.info("backbone=%s num_channels=%d train_backbone=%s (lr_backbone=%s) "
             "return_interm_layers=%s dilation=%s",
             args.backbone, backbone.num_channels, train_backbone, args.lr_backbone,
             return_interm_layers, args.dilation)
    log.info("  norm_layer=FrozenBatchNorm2d  pos_embed=%s  weights=%s",
             args.position_embedding, weights_path or "torchvision DEFAULT (ImageNet, via torch hub)")
    log.info("  expects ImageNet-standardised input (mean .485/.456/.406, std .229/.224/.225); "
             "applied in convert_batch, disable with ORI_IMAGE_NORM=0")
    return model
