#!/usr/bin/env python3
"""Merge selected dataset folders into a new repository.

Requirements (run once in your environment):
    pip install lerobot-edit-dataset  # provides the `lerobot-edit-dataset` CLI

Usage:
    python getdatainmobile.py

The script expects the following files/variables:
* `checks_success.txt` – a plain‑text file with one folder name per line.
* Two source roots where the original datasets live (you can edit the variables
  `SRC_ROOTS` below).
* `NEW_ROOT` – where the merged dataset will be created.
* `NEW_REPO_ID` – name of the new HuggingFace‑style repo (any string).
"""

import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Configuration – edit these paths to match your environment
# ---------------------------------------------------------------------
# File containing the list of folder names you want to keep
CHECKS_FILE = Path("merge_list.txt")

# Two source directories where the original data folders are located
SRC_ROOTS = [
    Path("/home/sr5/sairaj.loke/other/new_data"),
]

# Destination for the merged dataset
NEW_ROOT = Path("/home/sr5/sairaj.loke/other/new_data/larger_data")
NEW_REPO_ID = "larger_data"

# ---------------------------------------------------------------------

def read_success_names() -> list[str]:
    """Read folder names from `checks_success.txt` (one per line)."""
    if not CHECKS_FILE.is_file():
        sys.exit(f"❌ Checks file not found: {CHECKS_FILE}")
    with CHECKS_FILE.open() as f:
        names = [line.strip() + "/lerobot3.0" for line in f if line.strip()]

    print("names: ", names)
    return names

def locate_folders(names: list[str]) -> list[Path]:
    """Search the two source roots for folders whose name matches any entry in `names`.
    Returns a list of absolute paths to the matching folders.
    """
    found = []
    for root in SRC_ROOTS:
        for name in names:
            candidate = root / name
            if candidate.is_dir():
                found.append(candidate)
    if not found:
        sys.exit("❌ No matching folders were found in the provided source roots.")
    return found

def build_lerobot_command(folders: list[Path]):
    """Construct the `lerobot-edit-dataset` command.
    The CLI expects a JSON‑like string for the `--operation.roots` and
    `--operation.repo_ids` arguments.
    """
    # Convert Path objects to strings suitable for the CLI
    roots_str = ", ".join([f"'{str(p)}'" for p in folders])
    # The repo IDs are just the folder names (relative to the source root)
    repo_ids = [f"{f.parent.name}/{f.name}" for f in folders]

    repo_ids_str = ", ".join([f"'{rid}'" for rid in repo_ids])

    cmd = [
        "lerobot-edit-dataset",
        "--new_root", str(NEW_ROOT),
        "--new_repo_id", NEW_REPO_ID,
        "--operation.type", "merge",
        "--operation.roots", f"[{roots_str}]",
        "--operation.repo_ids", f"[{repo_ids_str}]",
        "--push_to_hub", "false",
        # "--operation.ignore_missing_meta", "true",
    ]

    return cmd

def main():
    names = read_success_names()
    folders = locate_folders(names)
    print("folders" , len(folders), folders)
    cmd = build_lerobot_command(folders)
    print("🔧 Running command:")
    print(" ".join(cmd))
    # Execute the command – we use `subprocess.run` so the user can see any errors.
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"❌ lerobot-edit-dataset failed with exit code {result.returncode}")
    print("✅ Merge completed successfully.")

if __name__ == "__main__":
    main()
