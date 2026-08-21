"""
Shared training & evaluation utilities for the Origami ViTacFormer project.

Contains:
  - Joint group definitions (indices into 65-dim action/state vectors)
  - _detailed_stats:   print shape/min/max/mean/std for a tensor
  - log_input_stats:   log per-batch input stats to console, TensorBoard, and info.log
"""

import torch


# ──────────────────────────────────────────────────────────────────────
# Joint group definitions (indices into the 65-dim action/state vectors)
# ──────────────────────────────────────────────────────────────────────
JOINT_GROUPS = {
    "left_arm":   list(range(0, 7)),
    "left_hand":  list(range(7, 29)),
    "right_arm":  list(range(29, 36)),
    "right_hand": list(range(36, 58)),
    "motor":      list(range(58, 65)),
}

JOINT_GROUP_NAMES = {
    "left_arm":   [f"left_arm_j{i}" for i in range(7)],
    "left_hand":  [f"left_hand_j{i}" for i in range(22)],
    "right_arm":  [f"right_arm_j{i}" for i in range(7)],
    "right_hand": [f"right_hand_j{i}" for i in range(22)],
    "motor":      [f"motor_j{i}" for i in range(7)],
}

JOINT_GROUP_COLORS = {
    "left_arm":   "tab:blue",
    "left_hand":  "tab:orange",
    "right_arm":  "tab:green",
    "right_hand": "tab:red",
    "motor":      "tab:purple",
}


# ──────────────────────────────────────────────────────────────────────
# Per-dimension action loss weights
# ──────────────────────────────────────────────────────────────────────
FINGER_SPANS = [("thumb", 0, 5), ("index", 5, 9), ("middle", 9, 13),
                ("ring", 13, 17), ("pinky", 17, 22)]
HAND_OFFSETS = {"left_hand": 7, "right_hand": 36}


def load_action_weight_spec(path):
    """Read the JSON written by tools/compute_action_weights.py."""
    import json
    with open(path) as f:
        spec = json.load(f)
    missing = {"fingers", "groups"} - set(spec)
    if missing:
        raise ValueError(f"{path}: missing key(s) {sorted(missing)}")
    unknown = set(spec["fingers"]) - {n for n, _, _ in FINGER_SPANS}
    if unknown:
        raise ValueError(f"{path}: unknown finger(s) {sorted(unknown)}")
    return spec


def constant_action_dims(spec):
    """{dim: value} for action dims the dataset holds constant. Predicting them
    wastes gradient, but they still have to be EMITTED (the reply is fixed at 65
    columns and the evaluator checks jumps/velocity), so the value matters."""
    return {int(k): float(v) for k, v in (spec.get("constant_dims") or {}).items()}


def build_predicted_action_dims(state_dim, mode, weight_spec=None):
    """Which contract columns the model predicts.

      "all"        every dim (65). Dims the loss zeroes are still emitted.
      "active"     drop the dims the dataset holds constant, plus ring+pinky.
                   Lower-dimensional output, but the dropped joints can then
                   only be EMITTED, not actuated -- see unfilled_action_plan().

    Returns (indices, dropped) with indices in ascending contract order; the
    order is load-bearing, since it maps model output position -> reply column.
    """
    if mode == "all":
        return list(range(state_dim)), []
    if mode != "active":
        raise ValueError(f"Unknown action_dims_mode {mode!r} (expected 'all' or 'active')")
    drop = set(constant_action_dims(weight_spec or {}))
    for off in HAND_OFFSETS.values():
        for name, a, b in FINGER_SPANS:
            if name in ("ring", "pinky"):
                drop.update(range(off + a, off + b))
    drop = {i for i in drop if i < state_dim}
    return [i for i in range(state_dim) if i not in drop], sorted(drop)


def build_action_dim_weights(state_dim, mode="uniform", group_weights=None,
                             action_stats=None, degenerate_spread=1e-3,
                             weight_spec=None):
    """Build a [state_dim] weight vector for the per-dimension action L1 loss.

    Modes:
      "uniform"  every dimension weighted 1.0 -- identical to the previous
                 behaviour, and the default.
      "group"    per-joint-group multipliers, e.g.
                 {"left_hand": 2.0, "right_hand": 2.0}. Groups not listed keep
                 1.0. This is the knob for "the fingers matter more than the
                 arms".

    Independently, if `action_stats` is given, any dimension whose
    (q99 - q01) is below `degenerate_spread` gets weight 0. Measured on this
    dataset that is action dims [58, 59] -- spread 1.3e-5 and 2.9e-5, i.e.
    constant. They contribute a constant to the loss and no gradient signal, so
    including them only dilutes the mean.

    The result is rescaled so that the mean weight over NON-ZERO dims is 1.0.
    That keeps the magnitude of the L1 term comparable to an unweighted run,
    which matters because it is summed against kl_weight * KL and tac_weight *
    L1_tac -- rescaling the L1 term silently reweights those too.

    Returns a plain list of floats (JSON-serialisable for the config log).
    """
    weights = [1.0] * state_dim

    if mode == "group":
        gw = {g: 1.0 for g in JOINT_GROUPS}
        gw.update(group_weights or {})
        unknown = set(gw) - set(JOINT_GROUPS)
        if unknown:
            raise ValueError(f"Unknown joint group(s) {sorted(unknown)}. "
                             f"Valid: {sorted(JOINT_GROUPS)}")
        for group_name, indices in JOINT_GROUPS.items():
            for i in indices:
                if i < state_dim:
                    weights[i] = float(gw[group_name])
    elif mode == "file":
        # Per-finger weights derived from dataset torque, plus per-group weights.
        # Fingers are set first, then any group weight listed for a hand scales
        # that whole hand on top -- so the two knobs compose rather than clash.
        if weight_spec is None:
            raise ValueError("mode='file' needs weight_spec (see load_action_weight_spec)")
        fw = weight_spec["fingers"]
        gw = weight_spec.get("groups") or {}
        for hand, off in HAND_OFFSETS.items():
            for name, a, b in FINGER_SPANS:
                if name not in fw:
                    continue
                for i in range(off + a, min(off + b, state_dim)):
                    weights[i] = float(fw[name]) * float(gw.get(hand, 1.0))
        for group_name, indices in JOINT_GROUPS.items():
            if group_name in HAND_OFFSETS or group_name not in gw:
                continue          # hands already handled above
            for i in indices:
                if i < state_dim:
                    weights[i] = float(gw[group_name])
    elif mode != "uniform":
        raise ValueError(
            f"Unknown loss_dim_weight_mode {mode!r} (expected 'uniform', 'group' or 'file')")

    dropped = []
    # dims the spec marks constant carry no gradient signal; they are emitted
    # from constant_action_dims() instead of being predicted
    for i in constant_action_dims(weight_spec or {}):
        if i < state_dim and weights[i] != 0.0:
            weights[i] = 0.0
            dropped.append(i)
    if action_stats is not None:
        q01 = action_stats.get("q01")
        q99 = action_stats.get("q99")
        if q01 is not None and q99 is not None:
            for i in range(min(state_dim, len(q01))):
                if float(q99[i]) - float(q01[i]) < degenerate_spread:
                    weights[i] = 0.0
                    dropped.append(i)

    nonzero = [w for w in weights if w > 0]
    if not nonzero:
        raise ValueError("All action dimensions ended up with zero weight.")
    scale = len(nonzero) / sum(nonzero)
    weights = [w * scale for w in weights]

    return weights, dropped


# ──────────────────────────────────────────────────────────────────────
# Detailed stats printer
# ──────────────────────────────────────────────────────────────────────
def _detailed_stats(name, tensor):
    """Print detailed stats (shape, min, max, mean, std) for a tensor."""
    if not isinstance(tensor, torch.Tensor):
        print(f"  [{name}] Not a tensor: {type(tensor)}")
        return
    if tensor.numel() == 0:
        print(f"  [{name}] Empty tensor, shape={list(tensor.shape)}")
        return
    t_min = tensor.min().item()
    t_max = tensor.max().item()
    t_mean = tensor.float().mean().item()
    t_std = tensor.float().std().item() if tensor.numel() > 1 else 0.0
    status = ""
    if tensor.dtype.is_floating_point:
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            status = " ⚠️ (NaN/Inf Detected!)"
    print(f"  [{name}] Shape: {list(tensor.shape)} | dtype: {tensor.dtype} | "
          f"Min: {t_min:.4f} | Max: {t_max:.4f} | Mean: {t_mean:.4f} | Std: {t_std:.4f}{status}")


# ──────────────────────────────────────────────────────────────────────
# Input stats logger (first 3 batches of training)
# ──────────────────────────────────────────────────────────────────────
def log_input_stats(data, use_tactile, writer, global_step, log_file=None):
    """
    Log detailed input stats for a converted batch to console, TensorBoard, and optionally info.log.

    Called for the first 3 batches of training to verify data integrity.
    """
    separator = "=" * 70
    header = f"\n[Input Stats] global_step={global_step}"
    print(separator)
    print(header)
    print(separator)

    lines = [separator, header, separator]

    def _log(name, tensor):
        _detailed_stats(name, tensor)
        if isinstance(tensor, torch.Tensor) and tensor.numel() > 0 and writer is not None:
            try:
                writer.add_histogram(f"input_stats/{name}", tensor.cpu(), global_step)
            except Exception:
                pass
        if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
            t_min = tensor.min().item()
            t_max = tensor.max().item()
            t_mean = tensor.float().mean().item()
            t_std = tensor.float().std().item() if tensor.numel() > 1 else 0.0
            lines.append(f"  [{name}] Shape: {list(tensor.shape)} | "
                         f"Min: {t_min:.4f} | Max: {t_max:.4f} | "
                         f"Mean: {t_mean:.4f} | Std: {t_std:.4f}")
        else:
            lines.append(f"  [{name}] Not a tensor or empty: {type(tensor)}")

    _log("image", data["image"][0])
    _log("lowdim", data["lowdim"][0])
    _log("action", data["action"][0])
    _log("action_mask", data["action_mask"][0])
    if use_tactile:
        _log("tactile", data["tactile"][0])
        _log("tactile_next", data["tactile_next"][0])

    # Per-joint-group stats for lowdim and action
    for group_name, indices in JOINT_GROUPS.items():
        ld_group = data["lowdim"][..., indices]
        act_group = data["action"][..., indices]
        _log(f"lowdim/{group_name}", ld_group)
        _log(f"action/{group_name}", act_group)

    print(separator)
    lines.append(separator)

    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write("\n".join(lines) + "\n")


# ──────────────────────────────────────────────────────────────────────
# Final model input logger (after all transformations)
# ──────────────────────────────────────────────────────────────────────
def log_final_model_inputs(qpos_norm, image_data, action_norm, is_pad,
                           tactile_norm=None, tactile_next_norm=None,
                           writer=None, global_step=0):
    """
    Log the EXACT inputs that go into the model forward pass.
    Called AFTER flattening, masking, and device transfer.

    This ensures logged values match what the model actually consumes.
    """
    separator = "=" * 70
    header = f"\n[FINAL Model Inputs] global_step={global_step} (after flattening & device transfer)"
    print(separator)
    print(header)
    print(separator)

    lines = [separator, header, separator]

    def _log_final(name, tensor):
        _detailed_stats(name, tensor)
        if isinstance(tensor, torch.Tensor) and tensor.numel() > 0 and writer is not None:
            try:
                writer.add_histogram(f"final_model_inputs/{name}", tensor.cpu(), global_step)
            except Exception:
                pass
        if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
            t_min = tensor.min().item()
            t_max = tensor.max().item()
            t_mean = tensor.float().mean().item()
            t_std = tensor.float().std().item() if tensor.numel() > 1 else 0.0
            lines.append(f"  [{name}] Shape: {list(tensor.shape)} | "
                         f"Min: {t_min:.4f} | Max: {t_max:.4f} | "
                         f"Mean: {t_mean:.4f} | Std: {t_std:.4f}")
        else:
            lines.append(f"  [{name}] Not a tensor or empty: {type(tensor)}")

    _log_final("qpos_flat (model input)", qpos_norm)
    _log_final("image (model input)", image_data)
    _log_final("action_flat (model input)", action_norm)
    _log_final("is_pad (model input)", is_pad)

    if tactile_norm is not None:
        _log_final("tactile_flat (model input)", tactile_norm)
    if tactile_next_norm is not None:
        _log_final("tactile_next_flat (model input)", tactile_next_norm)

    print(separator)
    lines.append(separator)
    print("\n".join(lines))