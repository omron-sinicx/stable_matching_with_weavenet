#!/usr/bin/env python3
"""Replace val/test data with legacy paper data via hardlinks.

For each (split, dist, size) combination present in
docs/papers/legacy_code/datasets/, hardlink the 1000 legacy NPZ files
into data/shared/{validation,test}/{XY}{N}x{N}/{NNNNN}.npz and write
the corresponding .txt list. For combinations absent from legacy data,
fall back to the sha256-derived NPZ files in the *_sha256_backup
directories.
"""
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEGACY_ROOT = Path("/workspace/docs/papers/legacy_code/datasets")
DATA_ROOT = Path("/workspace/data/shared")

# Distribution codes used by both legacy and current naming
LEGACY_DISTS = ["UU", "UD", "DD", "GG", "LL"]
# Sizes used by current configs (legacy may cover a subset)
CURRENT_SIZES = [3, 5, 7, 9, 10, 20, 30, 100]


def find_legacy_npz(split: str, dist: str, size: int):
    """Return sorted list of legacy NPZ paths or [] if combo missing."""
    d = LEGACY_ROOT / split / dist / f"size-{size:02d}"
    if not d.is_dir():
        return []
    return sorted(d.glob("instance_size-*_*.npz"))


def restore_from_backup(split: str, dist: str, size: int) -> bool:
    """Hardlink from sha256 backup if available. Returns True on success."""
    backup_dir = DATA_ROOT / f"{split}_sha256_backup" / f"{dist}{size}x{size}"
    if not backup_dir.is_dir():
        return False
    target_dir = DATA_ROOT / split / f"{dist}{size}x{size}"
    target_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(backup_dir.glob("*.npz"))
    for src in files:
        dst = target_dir / src.name
        if not dst.exists():
            os.link(src, dst)
    txt = DATA_ROOT / split / f"{dist}{size}x{size}.txt"
    txt.write_text("".join(f"{dist}{size}x{size}/{p.name}\n" for p in files))
    return True


def main():
    legacy_count = 0
    sha256_count = 0
    missing = []
    for split in ("validation", "test"):
        for dist in LEGACY_DISTS:
            for size in CURRENT_SIZES:
                combo = f"{split}/{dist}{size}x{size}"
                legacy_files = find_legacy_npz(split, dist, size)
                if legacy_files:
                    target_dir = DATA_ROOT / split / f"{dist}{size}x{size}"
                    target_dir.mkdir(parents=True, exist_ok=True)
                    indices = []
                    for src in legacy_files:
                        # extract trailing 4-digit index
                        m = re.search(r"_(\d{4})\.npz$", src.name)
                        idx = int(m.group(1)) if m else 0
                        indices.append(idx)
                        dst = target_dir / f"{idx:05d}.npz"
                        if not dst.exists():
                            os.link(src, dst)
                    txt = DATA_ROOT / split / f"{dist}{size}x{size}.txt"
                    lines = [
                        f"{dist}{size}x{size}/{idx:05d}.npz\n" for idx in indices
                    ]
                    txt.write_text("".join(lines))
                    print(f"  legacy  {combo}  ({len(legacy_files)} files)")
                    legacy_count += 1
                elif restore_from_backup(split, dist, size):
                    print(f"  sha256  {combo}  (fallback)")
                    sha256_count += 1
                else:
                    missing.append(combo)
                    print(f"  MISSING {combo}")
    print()
    print(f"Summary: legacy={legacy_count}, sha256_fallback={sha256_count}, missing={len(missing)}")
    if missing:
        print("Missing combos:")
        for c in missing:
            print(f"  - {c}")


if __name__ == "__main__":
    main()
