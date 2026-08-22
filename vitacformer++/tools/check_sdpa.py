"""Is fused attention actually running? (needs CUDA -- run on the training box)

`--explicit_flash_attn` sets need_weights=False, which makes the fused path
REACHABLE. Whether PyTorch then takes it depends on dtype, head_dim, masks and
the installed kernels, so this measures rather than assumes:

  1. profiles a real forward and reports which attention kernels fired
  2. peak CUDA memory both ways -- the unfused path materialises [B*heads, T, T]
     per attention call, so a drop is direct evidence the matrices are gone
  3. wall time both ways

    python tools/check_sdpa.py --batch 64 --seq 200
    python tools/check_sdpa.py --batch 64 --seq 1024   # e.g. unpooled DINOv2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "detr"))

FUSED_HINTS = ("flash", "efficient_attention", "fmha", "mem_eff", "cudnn_attention")
UNFUSED_HINTS = ("bmm", "softmax", "baddbmm")


def build(seq, d_model, nhead, ffn, layers, explicit_flash_attn, device):
    from detr.models.transformer import TransformerEncoderLayer, TransformerEncoder
    lyr = TransformerEncoderLayer(d_model, nhead, ffn, 0.0, "relu", False, False,
                                  need_weights=not explicit_flash_attn)
    return TransformerEncoder(lyr, layers, None).to(device).eval()


def run(enc, x, iters):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(3):
            enc(x)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(iters):
            enc(x)
        torch.cuda.synchronize()
    return (time.time() - t0) / iters, torch.cuda.max_memory_allocated() / 2**20


def kernels(enc, x):
    from torch.profiler import profile, ProfilerActivity
    with torch.no_grad(), profile(activities=[ProfilerActivity.CUDA]) as prof:
        enc(x)
    names = [e.key.lower() for e in prof.key_averages()]
    return ([n for n in names if any(h in n for h in FUSED_HINTS)],
            [n for n in names if any(h in n for h in UNFUSED_HINTS)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seq", type=int, default=200, help="200 = current; 1024 = unpooled DINOv2")
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--ffn", type=int, default=3200)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available -- run this on the training box."); return 1
    dev = torch.device("cuda")
    dt = getattr(torch, args.dtype)
    print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__} | dtype={args.dtype}")
    print(f"backends: flash={torch.backends.cuda.flash_sdp_enabled()} "
          f"mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()} "
          f"math={torch.backends.cuda.math_sdp_enabled()}")
    print(f"shape: seq={args.seq} batch={args.batch} d_model={args.d_model} "
          f"heads={args.nhead} head_dim={args.d_model // args.nhead}\n")

    x = torch.randn(args.seq, args.batch, args.d_model, device=dev, dtype=dt)
    print(f"{'setting':28s}{'ms/iter':>10s}{'peak MB':>10s}   attention kernels")
    print("-" * 82)
    res = {}
    for label, efa in (("default (need_weights=True)", False),
                       ("--explicit_flash_attn", True)):
        enc = build(args.seq, args.d_model, args.nhead, args.ffn, args.layers, efa, dev).to(dt)
        ms, mem = run(enc, x, args.iters)
        fused, unfused = kernels(enc, x)
        res[efa] = (ms, mem)
        tag = f"FUSED: {sorted(set(fused))[:2]}" if fused else f"unfused: {sorted(set(unfused))[:3]}"
        print(f"{label:28s}{ms * 1e3:10.2f}{mem:10.0f}   {tag}")
    print("-" * 82)
    (ms0, m0), (ms1, m1) = res[False], res[True]
    print(f"\nspeedup {ms0 / max(ms1, 1e-9):.2f}x   memory {m0 / max(m1, 1e-9):.2f}x "
          f"({m0 - m1:+.0f} MB)")
    # the unfused path holds B*heads*T*T floats per attention call
    theo = args.batch * args.nhead * args.seq * args.seq * (2 if dt != torch.float32 else 4) / 2**20
    print(f"expected saving if the [T,T] matrices are gone: ~{theo:.0f} MB per attention call")
    print("\nfused kernels present under --explicit_flash_attn => it is really being used.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
