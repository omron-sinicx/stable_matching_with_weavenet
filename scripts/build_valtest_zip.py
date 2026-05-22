#!/usr/bin/env python3
"""Build stable_matching_val-test.zip for redistribution.

Walks data/shared/{validation,test}/ and packs the .txt list files plus the
{XY}{N}x{N}/*.npz files into a single zip with a top-level
stable_matching_val-test/ folder. Includes README.md describing provenance.

Usage:
    python scripts/build_valtest_zip.py --out data/local/stable_matching_val-test.zip
"""
import argparse
import sys
import time
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path("/workspace/data/shared")
TOP = "stable_matching_val-test"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "local" / "stable_matching_val-test.zip",
        help="Output zip path (default: data/local/stable_matching_val-test.zip)",
    )
    ap.add_argument(
        "--readme",
        type=Path,
        default=PROJECT_ROOT / "docs" / "papers" / "zip_README_draft.md",
        help="Path to README.md to embed in the zip root.",
    )
    ap.add_argument(
        "--compress-level",
        type=int,
        default=6,
        help="DEFLATE level 0-9 (default 6 — sane trade-off).",
    )
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    started = time.time()
    n_files = 0
    n_bytes_in = 0

    with zipfile.ZipFile(
        args.out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=args.compress_level
    ) as zf:
        # 1. README
        zf.write(args.readme, arcname=f"{TOP}/README.md")
        n_files += 1
        n_bytes_in += args.readme.stat().st_size

        # 2. validation/ and test/
        for split in ("validation", "test"):
            split_root = DATA_ROOT / split
            if not split_root.is_dir():
                print(f"  ! {split_root} missing — skipping", file=sys.stderr)
                continue
            print(f"  packing {split}/ ...", flush=True)
            # .txt list files
            for txt in sorted(split_root.glob("*.txt")):
                zf.write(txt, arcname=f"{TOP}/{split}/{txt.name}")
                n_files += 1
                n_bytes_in += txt.stat().st_size
            # combo dirs and their NPZs
            for combo_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
                npzs = sorted(combo_dir.glob("*.npz"))
                for npz in npzs:
                    arc = f"{TOP}/{split}/{combo_dir.name}/{npz.name}"
                    zf.write(npz, arcname=arc)
                    n_files += 1
                    n_bytes_in += npz.stat().st_size
                print(f"    + {split}/{combo_dir.name}  ({len(npzs)} files)", flush=True)

    elapsed = time.time() - started
    out_size = args.out.stat().st_size
    print()
    print(f"done in {elapsed:.0f}s")
    print(f"  files:      {n_files}")
    print(f"  raw bytes:  {n_bytes_in/1e9:.2f} GB")
    print(f"  zip bytes:  {out_size/1e9:.2f} GB  ({100*out_size/n_bytes_in:.0f}%)")
    print(f"  output:     {args.out}")


if __name__ == "__main__":
    main()
