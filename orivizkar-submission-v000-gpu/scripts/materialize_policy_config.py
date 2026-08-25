#!/usr/bin/env python3
"""Backfill a real "policy_config" key into training_configs.json for
checkpoints that predate it, parsed from that run's info_*.log. Unlike the
server's runtime fallback (vitac_policy_server.py:_policy_config_from_info_log),
this writes the recovered dict to disk -- the Dockerfile only COPYs
training_configs.json, not info_*.log, so the runtime fallback can't see it
inside the image. Idempotent: skips any checkpoint that already has the key.
"""
import argparse, ast, glob, json, os, re, sys

def recover(ckpt_dir: str) -> dict:
    logs = sorted(glob.glob(os.path.join(ckpt_dir, "info_*.log")))
    if not logs:
        raise KeyError(f"no info_*.log in {ckpt_dir}")
    pc, inside = {}, False
    for line in open(logs[0]):
        line = line.rstrip("\n")
        if line.startswith("---"):
            inside = "Policy Config" in line
            continue
        m = re.match(r"\s\s(\w+):\s(.*)$", line) if inside else None
        if m:
            try:
                pc[m.group(1)] = ast.literal_eval(m.group(2))
            except (ValueError, SyntaxError):
                pc[m.group(1)] = m.group(2)
    missing = {"camera_names", "backbone", "hidden_dim", "state_dim"} - set(pc)
    if missing:
        raise KeyError(f"{logs[0]} Policy Config block missing {sorted(missing)}")
    return pc

def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints_dir")
    args = p.parse_args()
    for d in sorted(glob.glob(os.path.join(args.checkpoints_dir, "*"))):
        cfg_path = os.path.join(d, "training_configs.json")
        if not os.path.isdir(d) or not os.path.isfile(cfg_path):
            continue
        cfg = json.load(open(cfg_path))
        if "policy_config" in cfg:
            print(f"skip  {os.path.basename(d)}: already has policy_config")
            continue
        try:
            cfg["policy_config"] = recover(d)
        except KeyError as e:
            print(f"WARN  {os.path.basename(d)}: {e}")
            continue
        json.dump(cfg, open(cfg_path, "w"), indent=2)
        print(f"fixed {os.path.basename(d)}: wrote {len(cfg['policy_config'])} policy_config keys")

if __name__ == "__main__":
    sys.exit(main())
