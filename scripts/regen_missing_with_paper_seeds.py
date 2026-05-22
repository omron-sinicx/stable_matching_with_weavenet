#!/usr/bin/env python3
"""Regenerate val/test combos absent from the legacy paper datasets, using
paper-default seeds (135789 for validation, 2456 for test).

Run AFTER scripts/replace_with_legacy_data.py — that script hardlinks legacy
NPZs and falls back to sha256-derived copies for combos the paper upload did
not cover. This script overwrites those fallback NPZs (and their .txt list
file) with paper-seed-generated data so the published zip is internally
consistent: every (dist, size, split) was either taken bit-exact from the
paper authors or generated with the paper's default seeds.
"""
import os
import shutil
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEGACY_ROOT = Path("/workspace/docs/papers/legacy_code/datasets")
DATA_ROOT = Path("/workspace/data/shared")

LEGACY_DISTS = ["UU", "UD", "DD", "GG", "LL"]
CURRENT_SIZES = [3, 5, 7, 9, 10, 20, 30, 100]


def legacy_has(split: str, dist: str, size: int) -> bool:
    d = LEGACY_ROOT / split / dist / f"size-{size:02d}"
    if not d.is_dir():
        return False
    files = list(d.glob("instance_size-*_*.npz"))
    return len(files) >= 1000  # paper aimed for 1000 per combo


def main():
    missing = []  # list of (split, dist, size)
    for split in ("validation", "test"):
        for dist in LEGACY_DISTS:
            for size in CURRENT_SIZES:
                if not legacy_has(split, dist, size):
                    missing.append((split, dist, size))

    print(f"Found {len(missing)} combos absent from legacy (will regen with paper seeds):")
    for split, dist, size in missing:
        print(f"  {split:<10s} {dist}{size}x{size}")

    for split, dist, size in missing:
        tag = f"{dist}{size}x{size}"
        target_dir = DATA_ROOT / split / tag
        target_txt = DATA_ROOT / split / f"{tag}.txt"
        # Clean stale sha256 NPZs so the regenerated dir is exactly N samples.
        if target_dir.exists():
            shutil.rmtree(target_dir)
        if target_txt.exists():
            target_txt.unlink()
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_valtest_data.py"),
            "--out", str(DATA_ROOT),
            "--types", dist,
            "--sizes", str(size),
            "--splits", split,
            "--val", "1000",   # match paper's n_samples=1000
            "--test", "1000",
            "--paper-seeds",
        ]
        print(f"\n$ {' '.join(cmd)}")
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
