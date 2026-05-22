#!/usr/bin/env python
"""Regenerate validation/test NPZ datasets for stable_matching_with_weavenet.

Reproduces the schema of the (lost) stable_matching_val-test.zip referenced by
README.md so the existing data loader (src/datamodules/stable_matching.py::NpzDataset)
keeps working.

Schema per .npz file:
    sab              (N, M)    float32  side-a satisfaction (values in {0.1..1.0})
    sba              (M, N)    float32  side-b satisfaction
    matches          (K, N, M) float32  K stable matchings (binary)
    fairness         (K,)      float32  per-match balance score (cost form)
    satisfaction     (K,)      float32  per-match total satisfaction (a + b)
    gs_matches       (2, N, M) float32  GS a-optimal then b-optimal
    SexEqualityCost  scalar    float32  min over `matches`
    EgalitarianCost  scalar    float32  min over `matches`

Per user direction (2026-05-22): K=2 for all sizes (GS man-/woman-optimal only),
including size=100 where exhaustive enumeration is infeasible.

Seeds are derived from sha256("{type}/{n}x{n}/{split}/{idx}") so re-runs are
reproducible without checking in giant binaries.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from datamodules.stable_matching import UniversalSMIGenerator  # noqa: E402
from weavenet.metric import (  # noqa: E402
    balance_score,
    egalitarian_score,
    is_stable,
    sexequality_cost,
)
from weavenet.preference import PreferenceFormat, sat2cost  # noqa: E402

# Note: weavenet 1.0.x sexequality_cost has a typo (`cba` instead of `cba_t`) on
# the satisfaction->cost branch, so we pre-convert and always pass cost format.

DISTRIBS = {
    "UU": ("U", "U"),
    "DD": ("D", "D"),
    "GG": ("G", "G"),
    "UD": ("U", "D"),
    "LL": ("L", "L"),
}
SIZES = (3, 5, 7, 9, 10, 20, 30, 100)
SPLITS = (("validation", 200), ("test", 1000))


def derive_seed(distrib_key: str, n: int, split: str, idx: int) -> int:
    h = hashlib.sha256(f"{distrib_key}/{n}x{n}/{split}/{idx}".encode()).digest()
    return int.from_bytes(h[:4], "little")


def sat_to_rank(sat: np.ndarray) -> np.ndarray:
    """satisfaction (higher = preferred) -> rank (0 = most preferred)."""
    order = np.argsort(-sat, axis=-1)
    rank = np.empty_like(order)
    rows = np.arange(order.shape[0])[:, None]
    cols = np.arange(order.shape[1])[None, :]
    rank[rows, order] = cols
    return rank


def gale_shapley(rank_a: np.ndarray, rank_b: np.ndarray) -> np.ndarray:
    """Proposer-optimal stable matching when side a proposes.

    rank_a[i, j] = rank of j in a-agent i's preferences.
    rank_b[j, i] = rank of i in b-agent j's preferences.
    Returns (N, M) binary matching.
    """
    N, M = rank_a.shape
    prefs_a = np.argsort(rank_a, axis=1)  # most-preferred-first
    next_idx = np.zeros(N, dtype=np.int32)
    partner_a = np.full(N, -1, dtype=np.int32)
    partner_b = np.full(M, -1, dtype=np.int32)
    free = list(range(N))
    while free:
        i = free.pop()
        if next_idx[i] >= M:
            continue
        j = int(prefs_a[i, next_idx[i]])
        next_idx[i] += 1
        curr = partner_b[j]
        if curr == -1:
            partner_b[j] = i
            partner_a[i] = j
        elif rank_b[j, i] < rank_b[j, curr]:
            partner_b[j] = i
            partner_a[i] = j
            partner_a[curr] = -1
            free.append(int(curr))
        else:
            free.append(i)
    out = np.zeros((N, M), dtype=np.float32)
    for i in range(N):
        if partner_a[i] != -1:
            out[i, partner_a[i]] = 1.0
    return out


def generate_instance(distrib_key: str, n: int, split: str, idx: int) -> dict:
    seed = derive_seed(distrib_key, n, split, idx)
    np.random.seed(seed)
    random.seed(seed)

    distrib_m, distrib_w = DISTRIBS[distrib_key]
    gen = UniversalSMIGenerator(
        distrib_m=distrib_m,
        distrib_w=distrib_w,
        sigma_m=0.4,
        sigma_w=0.4,
        N_range_m=(n, n),
        N_range_w=(n, n),
        transform=True,
        dtype=np.float32,
        samples_per_epoch=1,
    )
    sab, sba, na, nb = gen[0]
    assert na == n and nb == n, (na, nb, n)

    rank_a = sat_to_rank(sab)            # (N, M)
    rank_b = sat_to_rank(sba)            # (M, N)
    m_a_opt = gale_shapley(rank_a, rank_b)            # a-optimal, (N, M)
    m_b_opt = gale_shapley(rank_b, rank_a).T          # b-optimal -> (N, M)

    gs = np.stack([m_a_opt, m_b_opt], axis=0).astype(np.float32)  # (2, N, M)
    matches = gs.copy()

    m_batch = torch.from_numpy(matches).float()
    # repeat (not expand) → contiguous: weavenet.preference.batch_sum uses .view()
    sab_b = torch.from_numpy(sab).unsqueeze(0).repeat(matches.shape[0], 1, 1).float()
    sba_t_b = torch.from_numpy(sba.T).unsqueeze(0).repeat(matches.shape[0], 1, 1).float()
    cab_b = sat2cost(sab_b, dim=-1)
    cba_t_b = sat2cost(sba_t_b, dim=-2)

    stable = is_stable(m_batch, sab_b, sba_t_b).numpy()
    if not stable.all():
        raise RuntimeError(
            f"GS solution flagged unstable: {distrib_key} n={n} split={split} idx={idx} stable={stable}"
        )

    se = sexequality_cost(m_batch, cab_b, cba_t_b, PreferenceFormat.cost).numpy()
    egal = egalitarian_score(m_batch, cab_b, cba_t_b, PreferenceFormat.cost).numpy()
    bal = balance_score(m_batch, cab_b, cba_t_b, PreferenceFormat.cost).numpy()

    sat_total = (m_batch * sab_b).sum(dim=(-1, -2)) + (m_batch * sba_t_b).sum(dim=(-1, -2))

    return {
        "sab": sab.astype(np.float32),
        "sba": sba.astype(np.float32),
        "matches": matches.astype(np.float32),
        "fairness": bal.astype(np.float32),
        "satisfaction": sat_total.numpy().astype(np.float32),
        "gs_matches": gs.astype(np.float32),
        "SexEqualityCost": np.float32(se.min()),
        "EgalitarianCost": np.float32(egal.min()),
    }


def write_split(out_root: Path, distrib_key: str, n: int, split: str, count: int) -> None:
    tag = f"{distrib_key}{n}x{n}"
    sub_dir = out_root / split / tag
    sub_dir.mkdir(parents=True, exist_ok=True)
    list_path = out_root / split / f"{tag}.txt"
    lines = []
    for idx in range(count):
        sample = generate_instance(distrib_key, n, split, idx)
        npz_rel = f"{tag}/{idx:05d}.npz"
        npz_abs = out_root / split / npz_rel
        np.savez(npz_abs, **sample)
        lines.append(npz_rel)
    list_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "shared",
        help="Output root. Will write {out}/validation/ and {out}/test/.",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=list(DISTRIBS.keys()),
        choices=list(DISTRIBS.keys()),
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    parser.add_argument("--val", type=int, default=200, help="samples per (type, size) for validation")
    parser.add_argument("--test", type=int, default=1000, help="samples per (type, size) for test")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Generate only 2 val + 3 test samples per pair (sanity check).",
    )
    args = parser.parse_args()

    val_count = 2 if args.smoke else args.val
    test_count = 3 if args.smoke else args.test

    out_root: Path = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[generate] root={out_root}")
    for dk in args.types:
        for n in args.sizes:
            for split, cnt in (("validation", val_count), ("test", test_count)):
                tag = f"{dk}{n}x{n}"
                print(f"  {split:<10s} {tag:<10s} n={cnt}", flush=True)
                write_split(out_root, dk, n, split, cnt)
    print("[generate] done.")


if __name__ == "__main__":
    main()
