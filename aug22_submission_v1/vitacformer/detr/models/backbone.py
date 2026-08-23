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


# torchvision ViT constructors, dispatched on args.backbone the same way
# resnetNN names dispatch to torchvision.models.resnetNN. No small ViT variant
# is available in torchvision (no timm/DINOv2 in this environment); vit_b_16
# is the recommended default -- its 224x224 pretrained resolution matches
# configs.IMAGE_HW exactly, and at patch_size=16 gives 14x14=196 tokens/camera
# vs ResNet18/layer4's 49 -- see concerns.md #4 (the paper occupies under 1%
# of the frame; more, smaller patches is the lever that helps).
_VIT_NAMES = {"vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14"}


class ViTBackbone(nn.Module):
    """Wraps a torchvision VisionTransformer to present the same interface as
    BackboneBase: forward() returns {"0": [B, C, h, w]}, and .num_channels is
    the feature dimension -- the only contract Joiner/input_proj/
    PositionEmbeddingSine actually depend on. Nothing downstream needs to know
    whether the backbone is a CNN or a ViT.

    Built by dropping the classification head and reshaping the ViT's patch
    tokens back into a spatial grid: torchvision's VisionTransformer.forward()
    ends in `x = x[:, 0]; x = self.heads(x)` (CLS token -> class logits); this
    instead keeps `x[:, 1:]` (the patch tokens, dropping CLS) and reshapes
    [B, n_h*n_w, C] -> [B, C, n_h, n_w].
    """

    def __init__(self, name: str, train_backbone: bool, weights_path: str = None,
                 unfrozen_layers: int = None):
        super().__init__()
        ctor = getattr(torchvision.models, name)

        if weights_path:
            vit = ctor(weights=None)
            _load_local_backbone_weights(vit, weights_path)
        else:
            _distributed = is_dist_avail_and_initialized()
            if _distributed and not is_main_process():
                torch.distributed.barrier()
            vit = ctor(weights="DEFAULT")
            if _distributed and is_main_process():
                torch.distributed.barrier()

        self.patch_size = vit.patch_size
        self.image_size = vit.image_size
        self.num_channels = vit.hidden_dim
        self.conv_proj = vit.conv_proj
        self.class_token = vit.class_token
        self.encoder = vit.encoder
        # vit.heads (the classification head) is intentionally dropped: never
        # used, and keeping it would carry ~1000-class weights that mean
        # nothing for this task.

        # Unlike ResNet's BackboneBase (whose freeze loop has been commented
        # out since before this project -- lr_backbone alone controls its
        # effective learning rate, it is never truly frozen), a ViT backbone
        # genuinely respects train_backbone: with a small dataset, a frozen
        # pretrained ViT is usually the right default rather than fine-tuning
        # 86M+ parameters from a few hundred episodes.
        #
        # unfrozen_layers (ORI_VIT_UNFROZEN_LAYERS) is a middle ground: freeze
        # everything except the LAST N transformer blocks (+ the final
        # LayerNorm, which is cheap and renormalises the output distribution
        # that input_proj actually reads -- leaving it frozen while the blocks
        # feeding it change would create a mismatch right at that interface).
        # Only meaningful when train_backbone is True; ORI_LR_BACKBONE must
        # still be > 0 for the unfrozen tensors to receive a nonzero update.
        n_layers = len(self.encoder.layers)
        if not train_backbone:
            for p in self.parameters():
                p.requires_grad_(False)
            log.info("ViT backbone frozen (train_backbone=False, i.e. lr_backbone<=0)")
        elif unfrozen_layers is not None:
            k = max(0, min(unfrozen_layers, n_layers))
            if unfrozen_layers != k:
                log.warning("ORI_VIT_UNFROZEN_LAYERS=%d out of range [0,%d], clamped to %d",
                           unfrozen_layers, n_layers, k)
            for p in self.parameters():
                p.requires_grad_(False)
            trainable_blocks = list(self.encoder.layers.children())[n_layers - k:] if k > 0 else []
            for block in trainable_blocks:
                for p in block.parameters():
                    p.requires_grad_(True)
            for p in self.encoder.ln.parameters():
                p.requires_grad_(True)
            n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in self.parameters())
            log.info("ViT backbone partially unfrozen: last %d/%d encoder blocks + final LN "
                     "(%.1fM/%.1fM params trainable, %.0f%%)",
                     k, n_layers, n_train / 1e6, n_total / 1e6, 100 * n_train / n_total)

    def forward(self, x):
        n, c, h, w = x.shape
        if h != self.image_size or w != self.image_size:
            raise ValueError(
                f"ViTBackbone was built for {self.image_size}x{self.image_size} input "
                f"(the pretrained resolution) but got {h}x{w}. Unlike ResNet, a ViT's "
                f"patch grid and position embeddings are sized for one fixed resolution -- "
                f"set configs.IMAGE_HW to ({self.image_size}, {self.image_size})."
            )
        p = self.patch_size
        n_h, n_w = h // p, w // p

        tokens = self.conv_proj(x)                                          # [n, C, n_h, n_w]
        tokens = tokens.reshape(n, self.num_channels, n_h * n_w).permute(0, 2, 1)  # [n, n_h*n_w, C]

        cls = self.class_token.expand(n, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)                            # [n, 1+n_h*n_w, C]
        tokens = self.encoder(tokens)
        tokens = tokens[:, 1:]                                              # drop CLS -> patches only

        feat = tokens.permute(0, 2, 1).reshape(n, self.num_channels, n_h, n_w)
        return {"0": feat}


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
            log.info("Bacbone randomly initialised")
            if _distributed and not is_main_process():
                torch.distributed.barrier()
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                weights=None, norm_layer=FrozenBatchNorm2d)
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

    if args.backbone in _VIT_NAMES:
        if return_interm_layers:
            raise NotImplementedError(
                "--masks (return_interm_layers) is not implemented for ViT backbones -- "
                "a ViT has one output resolution, not the 4 stages a ResNet exposes."
            )
        backbone = ViTBackbone(args.backbone, train_backbone, weights_path=weights_path,
                               unfrozen_layers=getattr(args, "vit_unfrozen_layers", None))
        log.info("backbone=%s (ViT) num_channels=%d train_backbone=%s (lr_backbone=%s) "
                 "patch_size=%d image_size=%d",
                 args.backbone, backbone.num_channels, train_backbone, args.lr_backbone,
                 backbone.patch_size, backbone.image_size)
    else:
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation,
                            weights_path=weights_path)
        log.info("backbone=%s (CNN) num_channels=%d train_backbone=%s (lr_backbone=%s) "
                 "return_interm_layers=%s dilation=%s",
                 args.backbone, backbone.num_channels, train_backbone, args.lr_backbone,
                 return_interm_layers, args.dilation)
        log.info("  norm_layer=FrozenBatchNorm2d")

    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels

    log.info("  pos_embed=%s  weights=%s",
             args.position_embedding, weights_path or "torchvision DEFAULT (ImageNet, via torch hub)")
    log.info("  expects ImageNet-standardised input (mean .485/.456/.406, std .229/.224/.225); "
             "applied in convert_batch, disable with ORI_IMAGE_NORM=0")
    return model
