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
    """Load ResNet weights from a local checkpoint instead of the torch hub cache.

    Accepts a bare torchvision resnet state_dict, or a wrapper dict under any of
    'model' / 'state_dict' / 'weights'. Common prefixes are stripped
    automatically, so a full ACT checkpoint (keys like
    'model.backbones.0.0.body.conv1.weight') works as well as a plain
    'resnet18-f37072fd.pth'.

    Note FrozenBatchNorm2d has no num_batches_tracked buffer; its
    _load_from_state_dict drops that key, so a stock torchvision state_dict
    loads cleanly.
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for k in ("model", "state_dict", "weights"):
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected a state_dict, got {type(obj).__name__}")

    target = set(backbone.state_dict().keys())

    # Try the raw keys plus a set of prefix strips; keep whichever matches most.
    def _strip(sd, prefix):
        return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    candidates = [("", obj)]
    prefixes = set()
    for k in obj:
        for anchor in ("conv1.", "layer1."):
            i = k.find(anchor)
            if i > 0:
                prefixes.add(k[:i])
    for pre in prefixes:
        candidates.append((pre, _strip(obj, pre)))

    best_prefix, best_sd, best_hits = "", obj, len(target & set(obj.keys()))
    for pre, sd in candidates[1:]:
        hits = len(target & set(sd.keys()))
        if hits > best_hits:
            best_prefix, best_sd, best_hits = pre, sd, hits

    missing, unexpected = backbone.load_state_dict(best_sd, strict=False)
    missing = [m for m in missing if "num_batches_tracked" not in m]
    log.info("backbone weights loaded from %s", path)
    log.info("  matched %d/%d target tensors%s", best_hits, len(target),
             f" after stripping prefix {best_prefix!r}" if best_prefix else "")
    if unexpected:
        # Harmless: extra tensors in the source we simply do not need.
        log.warning("  %d unexpected key(s) ignored: %s%s",
                    len(unexpected), unexpected[:8], " ..." if len(unexpected) > 8 else "")

    # Equivalent to strict=True, but with a diagnostic instead of an opaque
    # exception. load_state_dict(strict=False) is used only so we can report
    # WHAT is missing; any genuinely missing tensor is still fatal, because
    # otherwise part of the backbone silently stays randomly initialised and the
    # run looks fine until the numbers are quietly worse.
    if best_hits == 0:
        raise ValueError(
            f"{path}: no keys matched the backbone.\n"
            f"  expected e.g. {sorted(target)[:3]}\n"
            f"  got          {sorted(obj)[:3]}"
        )
    if missing:
        raise ValueError(
            f"{path}: {len(missing)} tensor(s) missing -- refusing to train with a "
            f"partially-initialised backbone.\n"
            f"  matched {best_hits}/{len(target)}"
            f"{f' after stripping prefix {best_prefix!r}' if best_prefix else ''}\n"
            f"  missing: {missing[:10]}{' ...' if len(missing) > 10 else ''}"
        )
    return missing, unexpected


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
