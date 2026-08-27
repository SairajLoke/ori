"""
Cross-attention map capture (temporal / self / between-modalities), for
inspecting what the model actually attends to.

Hooks nn.MultiheadAttention submodules directly via forward hooks -- no
changes needed to detr_vae.py/transformer.py/policy.py's forward() signatures.
Every attention call in this codebase already computes weights
(need_weights=True by default; only the main transformer's self_attn/
cross_attn_1/cross_attn_2 are gated by --explicit_flash_attn) and discards
them via [0] indexing -- this just keeps a copy instead of throwing it away.

Categories captured (module path on the unwrapped policy.model):
  "self"            transformer.encoder.layers[i].self_attn      (always)
  "cross" (tactile) transformer.encoder.layers[i].cross_attn_1/2 (use_tactile)
  "cross" (fusion)  fusion.cross_attn                            (use_decision_fusion)
  "temporal"        history_attn.layers[0].self_attn             (image_history)

Usage (see origami_imitate_episodes.py's origami_validate for the call site):

    from my_utils.attention_capture import save_attention_maps
    save_attention_maps(
        policy=accelerator.unwrap_model(policy),
        data=data,
        normalizer=normalizer,
        device=device,
        out_dir=os.path.join(ckpt_dir, "attention_maps", f"epoch_{epoch}"),
        n_samples=2,
        use_tactile=use_tactile,
        use_decision_fusion=use_decision_fusion,
        image_history=image_history,
        epoch=epoch,
    )

Cost: one extra forward pass on n_samples examples, no gradients -- cheaper
than gradcam's forward+backward. Only call this outside the hot loop.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from my_utils.ori_logging import get_logger

log = get_logger("attention_capture")


def _iter_attn_modules(model, use_tactile, use_decision_fusion, image_history):
    """Yield (name, module) for every nn.MultiheadAttention worth capturing
    that actually exists on this model instance. Silently skips modules that
    don't exist for this config (e.g. no fusion.cross_attn when
    use_decision_fusion=False) -- that is the normal case, not an error."""
    for i, layer in enumerate(model.transformer.encoder.layers):
        yield f"self_attn_layer{i}", layer.self_attn
        if use_tactile and hasattr(layer, "cross_attn_1"):
            yield f"cross_attn_1_layer{i}", layer.cross_attn_1
            yield f"cross_attn_2_layer{i}", layer.cross_attn_2
    if use_decision_fusion and hasattr(model, "fusion"):
        yield "fusion_cross_attn", model.fusion.cross_attn
    if image_history and hasattr(model, "history_attn"):
        # One shared module reused per camera -- a single hook fires once per
        # camera within one forward call; saved call indices are camera order.
        yield "history_attn", model.history_attn.layers[0].self_attn


def save_attention_maps(policy, data, normalizer, device, out_dir, n_samples=2,
                        use_tactile=False, use_decision_fusion=False,
                        image_history=False, epoch=0):
    """Run one isolated forward pass and save every captured attention map.

    `policy` must be the UNWRAPPED model (accelerator.unwrap_model), and
    `data` the dict convert_batch produced for this validation batch.
    """
    os.makedirs(out_dir, exist_ok=True)

    if not policy.model.transformer.encoder.layers[0].need_weights:
        log.warning("attention_capture: need_weights=False (--explicit_flash_attn was used) "
                    "-- self_attn/cross_attn_1/cross_attn_2 will yield nothing this run; "
                    "history_attn/fusion_cross_attn are unaffected (not gated by that flag).")

    n = min(n_samples, data["image"].shape[0])
    sliced = {k: (v[:n] if torch.is_tensor(v) else v) for k, v in data.items()}

    was_training = policy.training
    policy.eval()

    captured = {}

    def _make_hook(name):
        def _hook(_module, _inp, output):
            captured.setdefault(name, []).append(output[1].detach().cpu().numpy())
        return _hook

    handles = [
        module.register_forward_hook(_make_hook(name))
        for name, module in _iter_attn_modules(policy.model, use_tactile, use_decision_fusion, image_history)
    ]

    try:
        # Deferred import: origami_imitate_episodes.py imports save_attention_maps
        # from this module, so importing origami_forward_pass at module level here
        # would be circular. By the time this function actually runs (only ever
        # called from within origami_validate), that module has finished loading.
        from origami_imitate_episodes import origami_forward_pass
        with torch.no_grad():
            origami_forward_pass(sliced, policy, normalizer, device, use_tactile, epoch=epoch)
    finally:
        for h in handles:
            h.remove()
        policy.train(was_training)

    if not captured:
        log.warning("attention_capture: nothing captured -- no matching attention modules "
                    "found for this model config.")
        return

    saved = 0
    for name, arrays in captured.items():
        for call_idx, arr in enumerate(arrays):
            np.save(os.path.join(out_dir, f"{name}_call{call_idx}.npy"), arr)
            saved += 1

            # arr: [B, seq_q, seq_k] (average_attn_weights=True default, heads
            # already averaged) -- one quick heatmap per sample.
            for s in range(arr.shape[0]):
                fig, ax = plt.subplots(figsize=(4, 4))
                im = ax.imshow(arr[s], aspect="auto", cmap="viridis")
                ax.set_title(f"{name} call{call_idx} sample{s}", fontsize=8)
                fig.colorbar(im, ax=ax, fraction=0.046)
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, f"{name}_call{call_idx}_sample{s}.png"), dpi=100)
                plt.close(fig)

    log.info("attention_capture: saved %d attention map(s) (+ heatmap PNGs) -> %s", saved, out_dir)
