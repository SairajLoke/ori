"""
Log problematic features that may cause issues during normalization.
Call this during training setup to document which dims are broken/stuck/degenerate.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional


def log_problematic_features(
    stats: Dict,
    normalizer_transforms: Dict[str, Optional[str]],
    log_file_path: Path,
    eps: float = 1e-6,
):
    """
    Inspect stats and log which features are problematic:
    - Zero variance (constant values)
    - Tiny denominators (q99 ≈ q01)
    - Extreme outliers
    - Should be excluded from model input
    
    Args:
        stats: dataset.meta.stats dict with mean/std/min/max/q01/q99 per feature
        normalizer_transforms: dict of {feature_key: mode} from normalizer
        log_file_path: Path to write log file
        eps: epsilon threshold for "near-zero"
    """
    
    log_file = Path(log_file_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, 'w') as f:
        f.write("=" * 90 + "\n")
        f.write("PROBLEMATIC FEATURES REPORT (Post-Normalization)\n")
        f.write("=" * 90 + "\n\n")
        
        # Track which dims to exclude
        dims_to_exclude = {
            "qpos": [],
            "action": [],
            "joint_torque": [],
            "tactile": [],
        }
        
        # ========== ZERO VARIANCE FEATURES ==========
        f.write("[CRITICAL] ZERO-VARIANCE FEATURES (Divide by ~0 after normalization)\n")
        f.write("-" * 90 + "\n")
        
        zero_var_found = False
        for key, key_stats in stats.items():
            if key_stats is None:
                continue
            
            std = np.asarray(key_stats.get("std", []))
            if std is None or len(std) == 0:
                continue
            
            # Find dims with std ~= 0
            zero_dims = np.where(std < eps)[0]
            if len(zero_dims) > 0:
                zero_var_found = True
                f.write(f"  {key}: dims {list(zero_dims)} (std ≈ 0)\n")
                
                if "qpos" in key or "state" in key:
                    dims_to_exclude["qpos"].extend(zero_dims.tolist())
                elif "action" in key:
                    dims_to_exclude["action"].extend(zero_dims.tolist())
                elif "joint_torque" in key:
                    dims_to_exclude["joint_torque"].extend(zero_dims.tolist())
                elif "tactile" in key:
                    dims_to_exclude["tactile"].extend(zero_dims.tolist())
        
        if not zero_var_found:
            f.write("  ✓ None found\n")
        f.write("\n")
        
        # ========== TINY QUANTILE DENOMINATORS ==========
        f.write("[HIGH RISK] TINY QUANTILE DENOMINATORS (q99 - q01 << threshold)\n")
        f.write("-" * 90 + "\n")
        
        tiny_denom_found = False
        for key, key_stats in stats.items():
            if key_stats is None:
                continue
            
            mode = normalizer_transforms.get(key, None)
            if mode != "quantile":
                continue
            
            q01 = np.asarray(key_stats.get("q01", []))
            q99 = np.asarray(key_stats.get("q99", []))
            
            if q01 is None or q99 is None or len(q01) == 0:
                continue
            
            denom = q99 - q01
            tiny_dims = np.where(denom < 0.01)[0]  # denom < 0.01 is problematic
            
            if len(tiny_dims) > 0:
                tiny_denom_found = True
                f.write(f"  {key}: dims {list(tiny_dims)}\n")
                for dim in tiny_dims:
                    f.write(f"      dim[{dim}]: q99-q01 = {denom[dim]:.2e}\n")
                
                if "action" in key:
                    dims_to_exclude["action"].extend(tiny_dims.tolist())
                elif "qpos" in key or "state" in key:
                    dims_to_exclude["qpos"].extend(tiny_dims.tolist())
        
        if not tiny_denom_found:
            f.write("  ✓ None found\n")
        f.write("\n")
        
        # ========== EXTREME OUTLIERS ==========
        f.write("[MODERATE] EXTREME OUTLIERS (>10 std from mean)\n")
        f.write("-" * 90 + "\n")
        
        outlier_found = False
        for key, key_stats in stats.items():
            if key_stats is None:
                continue
            
            mean = np.asarray(key_stats.get("mean", []))
            std = np.asarray(key_stats.get("std", []))
            min_val = np.asarray(key_stats.get("min", []))
            max_val = np.asarray(key_stats.get("max", []))
            
            if len(mean) == 0 or len(std) == 0:
                continue
            
            # Distance to nearest outlier boundary
            dist_to_min = np.abs(mean - min_val) / (std + eps)
            dist_to_max = np.abs(mean - max_val) / (std + eps)
            max_dist = np.maximum(dist_to_min, dist_to_max)
            
            outlier_dims = np.where(max_dist > 10)[0]
            if len(outlier_dims) > 0:
                outlier_found = True
                f.write(f"  {key}: dims {list(outlier_dims)} have outliers >10 std away\n")
        
        if not outlier_found:
            f.write("  ✓ None found (quantile normalization handles this)\n")
        f.write("\n")
        
        # ========== SUMMARY & RECOMMENDATIONS ==========
        f.write("=" * 90 + "\n")
        f.write("RECOMMENDED ACTIONS\n")
        f.write("=" * 90 + "\n\n")
        
        # Clean up duplicates
        for key in dims_to_exclude:
            dims_to_exclude[key] = sorted(list(set(dims_to_exclude[key])))
        
        f.write("Code to zero out problematic dims after normalization:\n\n")
        f.write("```python\n")
        f.write("# After normalizer.normalize():\n")
        
        if dims_to_exclude["qpos"]:
            f.write(f"qpos[:, :, {dims_to_exclude['qpos']}] = 0.0\n")
        if dims_to_exclude["action"]:
            f.write(f"action[:, :, {dims_to_exclude['action']}] = 0.0\n")
        if dims_to_exclude["joint_torque"]:
            f.write(f"joint_torque[:, :, {dims_to_exclude['joint_torque']}] = 0.0\n")
        if dims_to_exclude["tactile"]:
            f.write(f"tactile[:, :, {dims_to_exclude['tactile']}] = 0.0\n")
        
        f.write("```\n\n")
        
        # Summary counts
        total_to_exclude = sum(len(v) for v in dims_to_exclude.values())
        f.write(f"Total dims to exclude: {total_to_exclude}\n")
        f.write(f"  - qpos: {len(dims_to_exclude['qpos'])}\n")
        f.write(f"  - action: {len(dims_to_exclude['action'])}\n")
        f.write(f"  - joint_torque: {len(dims_to_exclude['joint_torque'])}\n")
        f.write(f"  - tactile: {len(dims_to_exclude['tactile'])}\n")
        
        f.write("\n" + "=" * 90 + "\n")
    
    print(f"✓ Problematic features logged to: {log_file}")
    return dims_to_exclude


def log_normalization_report(
    batch,
    normalized_batch,
    epoch: int,
    batch_idx: int,
    log_file_path: Path,
):
    """
    Log before/after normalization stats for a batch.
    Call this once per epoch to monitor normalization behavior.
    
    Args:
        batch: dict with raw tensors
        normalized_batch: dict with normalized tensors
        epoch: current epoch
        batch_idx: current batch index
        log_file_path: Path to append to
    """
    
    if epoch != 0 or batch_idx != 0:
        return  # Only log first batch of first epoch
    
    log_file = Path(log_file_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, 'a') as f:
        f.write("\n" + "=" * 90 + "\n")
        f.write("FIRST BATCH NORMALIZATION CHECK (Epoch 0, Batch 0)\n")
        f.write("=" * 90 + "\n\n")
        
        keys_to_check = [
            ("observation.state", "qpos", 0.5, 1.5),
            ("action", "action", 0.5, 1.5),
            ("observation.state.joint_torque", "joint_torque", 0.3, 1.0),
            ("observation.tactile", "tactile", 0.3, 1.5),
        ]
        
        all_passed = True
        for key, name, expected_std_min, expected_std_max in keys_to_check:
            if key not in batch or key not in normalized_batch:
                continue
            
            raw = batch[key]
            norm = normalized_batch[key]
            
            f.write(f"[{key}]\n")
            f.write(f"  Before norm: shape={raw.shape}, min={raw.min():.4f}, max={raw.max():.4f}, "
                   f"mean={raw.mean():.4f}, std={raw.std():.4f}\n")
            f.write(f"  After norm:  shape={norm.shape}, min={norm.min():.4f}, max={norm.max():.4f}, "
                   f"mean={norm.mean():.4f}, std={norm.std():.4f}\n")
            
            norm_std = norm.std().item()
            
            # Validation checks
            status = "✓ PASS"
            
            # Check for explosion (sudden huge std)
            if norm_std > 100:
                f.write(f"  ❌ FAIL: Huge std after normalization! {norm_std:.2f} >> 100\n")
                f.write(f"     Likely cause: zero denominator in quantile normalization\n")
                status = "❌ FAIL"
                all_passed = False
            
            # Check if std is too small (may indicate identity/broken normalization)
            elif norm_std < expected_std_min:
                f.write(f"  ⚠️  WARNING: std too small ({norm_std:.4f} < {expected_std_min})\n")
                f.write(f"     May indicate: feature is constant, or normalization mode is 'identity'\n")
            
            # Check if std is reasonable
            elif expected_std_min <= norm_std <= expected_std_max:
                f.write(f"  ✓ PASS: std is in expected range [{expected_std_min}, {expected_std_max}]\n")
            else:
                f.write(f"  ⚠️  INFO: std is {norm_std:.4f} (expected ~{(expected_std_min + expected_std_max)/2:.2f})\n")
            
            f.write("\n")
        
        f.write("=" * 90 + "\n")
        if all_passed:
            f.write("✓ NORMALIZATION STATUS: ALL CHECKS PASSED\n")
        else:
            f.write("❌ NORMALIZATION STATUS: FAILURES DETECTED - Review above logs\n")
        f.write("=" * 90 + "\n")