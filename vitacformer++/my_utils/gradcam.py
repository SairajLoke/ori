"""
Grad-CAM over the vision backbone, for eyeballing whether the model is
actually looking at the paper/hands or has latched onto something else
(background, table edge, cable clutter).

Hooks the backbone's OUTPUT, not its internals (no reach into
resnet.layer4 or any architecture-specific module), so this keeps working
unchanged when a ViT backbone option is added later -- both are required to
return a dict of {"0": [B, C, h, w]} from BackboneBase-style forward(), which
is the only contract this module depends on.

Usage (see origami_imitate_episodes.py's origami_validate for the call site):

    from my_utils.gradcam import save_gradcam_grid
    save_gradcam_grid(
        policy=accelerator.unwrap_model(policy),   # unwrapped DDP module
        data=data,                                  # one converted batch
        device=device,
        camera_names=CAMERA_NAMES,
        out_dir=os.path.join(ckpt_dir, "gradcam", f"epoch_{epoch}"),
        n_samples=2,
    )

Cost: one extra forward + backward pass on `n_samples` examples, isolated from
the training graph (does not touch optimizer state or accumulate into .grad
of the real training step). Only call this outside the hot loop -- it is not
free, which is why it is opt-in via --gradcam and only runs at the validation
cadence.
"""

from __future__ import annotations

import os
from typing import List

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from my_utils.ori_logging import get_logger

log = get_logger("gradcam")

# Same constants convert_batch uses to standardise images before the backbone.
# Needed here only to undo that transform for display -- Grad-CAM is overlaid
# on the human-viewable [0,1] image, not the standardised one the model sees.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _denorm_for_display(img: torch.Tensor, was_imagenet_normed: bool) -> torch.Tensor:
    """[3,H,W] standardised or [0,1] tensor -> [H,W,3] uint8 numpy-ready tensor."""
    x = img.detach().float().cpu().unsqueeze(0)
    if was_imagenet_normed:
        x = x * _IMAGENET_STD + _IMAGENET_MEAN
    x = x.clamp(0, 1)
    return (x[0].permute(1, 2, 0) * 255).byte()


def _colorize(cam_2d: np.ndarray) -> np.ndarray:
    """[H,W] float in [0,1] -> [H,W,3] uint8 heatmap (blue->red), no matplotlib
    dependency needed for a 3-stop colormap this simple."""
    r = np.clip(1.5 - np.abs(4 * cam_2d - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * cam_2d - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * cam_2d - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


@torch.enable_grad()
def save_gradcam_grid(
    policy,
    data: dict,
    device,
    camera_names: List[str],
    out_dir: str,
    n_samples: int = 2,
    target: str = "l1",
) -> None:
    """Run one isolated forward+backward pass and save a Grad-CAM overlay per
    camera per sample.

    `policy` must be the UNWRAPPED model (accelerator.unwrap_model), and
    `data` the dict convert_batch produced (must include ground-truth
    "action" -- Grad-CAM here explains what the model looked at while
    producing its prediction ERROR against that ground truth, which is more
    diagnostic during training than an unconditional saliency map).

    `target`:
      "l1"    -- backprop the masked action L1 loss (default). Answers
                 "what pixels influenced how WRONG the prediction was".
      "a_hat" -- backprop sum(|a_hat|). Answers "what pixels influenced the
                 prediction at all", regardless of correctness.

    Wrapped in torch.enable_grad() because the only caller (validation) runs
    under torch.no_grad(); this carves out grad tracking for just this call
    without needing the caller to change its decorator.
    """
    os.makedirs(out_dir, exist_ok=True)

    was_training = policy.training
    policy.eval()

    n = min(n_samples, data["image"].shape[0])
    image = data["image"][:n].to(device).clone().requires_grad_(False)
    qpos = data["lowdim"][:n].to(device)
    action = data["action"][:n, : policy.model.num_queries].to(device)
    is_pad = data["action_mask"][:n, : policy.model.num_queries].to(device)
    tactile = data.get("tactile")
    tactile_next = data.get("tactile_next")

    B, T1, D1 = qpos.shape
    qpos_flat = qpos.reshape(B, T1 * D1)

    tactile_flat = tactile_next_in = None
    if tactile is not None:
        tactile = tactile[:n].to(device)
        tactile_next_in = tactile_next[:n].to(device) if tactile_next is not None else None
        Bt, T2, D2 = tactile.shape
        tactile_flat = tactile.reshape(Bt, T2 * D2)

    # --- hook the backbone's OUTPUT, once per camera call ---
    # Joiner is nn.Sequential(BackboneBase_or_ViT, position_embedding);
    # backbones[0][0] is the shared vision module every camera is run through
    # (called once per camera, each call processing the whole batch).
    #
    # Deliberately NOT using register_full_backward_hook: a module invoked
    # multiple times within one forward pass (this one is, once per camera) is
    # a known correctness gotcha for backward hooks -- PyTorch does not
    # guarantee one callback per invocation in that case, and in practice it
    # under-fired here (fewer grads than forward calls). retain_grad() on each
    # captured tensor sidesteps that: it is per-TENSOR, and each camera's
    # activation is a distinct tensor object even though produced by the same
    # module, so reading .grad after backward() is unambiguous.
    target_module = policy.model.backbones[0][0]
    activations = []

    def _fwd_hook(_module, _inp, out):
        feat = out["0"] if isinstance(out, dict) else out
        feat.retain_grad()
        activations.append(feat)

    h_fwd = target_module.register_forward_hook(_fwd_hook)

    # Gradient must flow through the backbone to reach the hooked activations,
    # even for parameters normally frozen (e.g. a frozen ViT backbone once that
    # option exists). Force it on for just this isolated pass, then restore
    # each parameter's original requires_grad -- do NOT leave a frozen backbone
    # trainable after Grad-CAM finishes.
    _orig_requires_grad = [p.requires_grad for p in policy.parameters()]
    try:
        for p in policy.parameters():
            p.requires_grad_(True)

        a_hat, _, (_, _), _ = policy.model(
            qpos_flat, image, None, tactile_flat, action, is_pad,
            tactile_next_in, epoch=999,  # 999: use tactile_hat, not ground truth, for pass 2
        )

        if target == "a_hat":
            scalar = a_hat.abs().sum()
        else:
            err = F.l1_loss(a_hat, action, reduction="none")
            valid = (~is_pad).unsqueeze(-1).to(err.dtype)
            scalar = (err * valid).sum()

        policy.zero_grad(set_to_none=True)
        scalar.backward()

    finally:
        h_fwd.remove()
        for p, orig in zip(policy.parameters(), _orig_requires_grad):
            p.requires_grad_(orig)

    if not activations:
        log.warning("gradcam: no activations captured -- does this policy have backbones?")
        policy.train(was_training)
        return

    # One forward call per CAMERA, each processing the whole batch (n samples)
    # at once -- self.backbones[0](image[:, cam_id]) is called with all n
    # samples together, not once per sample. So len(activations) == n_cams,
    # each activation/grad tensor has batch dim n, not len == n * n_cams.
    grads = [act.grad for act in activations]
    if any(g is None for g in grads):
        log.warning("gradcam: %d/%d camera activations got no gradient -- skipping",
                    sum(g is None for g in grads), len(grads))
        policy.train(was_training)
        return
    n_cams = len(camera_names)
    if len(activations) != n_cams:
        log.warning("gradcam: expected %d activations (one per camera), got %d -- "
                    "skipping (backbone call pattern may have changed)",
                    n_cams, len(activations))
        policy.train(was_training)
        return

    Himg, Wimg = image.shape[-2:]  # image is [B, N_cam, 3, H, W]
    was_imagenet_normed = (
        image.min().item() < -0.01 or image.max().item() > 1.01
    )  # crude but sufficient: identity-normalized images stay in [0,1]

    saved = 0
    for cam_id, cam_name in enumerate(camera_names):
        act = activations[cam_id]   # [n, C, h, w]
        grad = grads[cam_id]        # [n, C, h, w]

        for s in range(n):
            # classic Grad-CAM: global-average-pool the gradient per channel,
            # use it as that channel's importance weight, ReLU the weighted sum.
            weights = grad[s : s + 1].mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * act[s : s + 1]).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=(Himg, Wimg), mode="bilinear", align_corners=False)
            cam = cam[0, 0].detach().cpu().numpy()
            cam = cam / (cam.max() + 1e-8)

            base = _denorm_for_display(image[s, cam_id], was_imagenet_normed).numpy()
            heat = _colorize(cam)
            overlay = (0.55 * base + 0.45 * heat).astype(np.uint8)

            short_name = cam_name.rsplit(".", 1)[-1]
            out_path = os.path.join(out_dir, f"sample{s}_{short_name}.png")
            Image.fromarray(overlay).save(out_path)
            saved += 1

    log.info("gradcam: saved %d overlays -> %s", saved, out_dir)
    policy.train(was_training)
