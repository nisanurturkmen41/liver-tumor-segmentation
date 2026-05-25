"""
cv_split.py
-----------
Deterministic patient-level 5-fold cross-validation split for MSD Task03_Liver.

CRITICAL: The split is at PATIENT level. Mixing slices/patches from the same
patient between train and validation causes data leakage and invalidates
metrics. This is a classic mistake in medical imaging.

Usage:
    python src/cv_split.py --processed_dir /path/to/processed --out folds.json
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import List, Dict

import numpy as np


def list_processed_patients(processed_dir: Path) -> List[str]:
    """Find patient IDs with both processed image AND label."""
    # Filenames produced by Week 1 pipeline:
    #   liver_<id>_processed.nii.gz       (image)
    #   liver_<id>_label_processed.nii.gz (mask)
    pat = re.compile(r"^liver_(\d+)_processed\.nii\.gz$")

    images = {}
    for p in processed_dir.iterdir():
        m = pat.match(p.name)
        if m:
            images[m.group(1)] = p

    patients = []
    for pid in images:
        lbl = processed_dir / f"liver_{pid}_label_processed.nii.gz"
        if lbl.exists():
            patients.append(pid)
        else:
            print(f"[WARN] No matching label for liver_{pid}, skipping.")

    patients.sort(key=lambda x: int(x))
    return patients


def make_folds(patients: List[str], n_folds: int = 5, seed: int = 42) -> Dict:
    rng = np.random.default_rng(seed)
    shuffled = patients.copy()
    rng.shuffle(shuffled)

    folds = {f"fold_{i}": {"train": [], "val": []} for i in range(n_folds)}
    val_chunks = np.array_split(shuffled, n_folds)

    for i, val in enumerate(val_chunks):
        val_set = set(val.tolist())
        folds[f"fold_{i}"]["val"] = sorted(val_set, key=lambda x: int(x))
        folds[f"fold_{i}"]["train"] = sorted(
            [p for p in patients if p not in val_set], key=lambda x: int(x)
        )
    return folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, required=True)
    ap.add_argument("--out", type=str, default="folds.json")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    processed_dir = Path(args.processed_dir)
    patients = list_processed_patients(processed_dir)
    print(f"Found {len(patients)} processed patients.")

    folds = make_folds(patients, n_folds=args.n_folds, seed=args.seed)

    seen = []
    for k, v in folds.items():
        seen.extend(v["val"])
        print(f"{k}: train={len(v['train'])}, val={len(v['val'])}")
    assert sorted(seen, key=lambda x: int(x)) == patients, \
        "Fold split is invalid: val sets do not cover all patients exactly once."

    with open(args.out, "w") as f:
        json.dump(folds, f, indent=2)
    print(f"Saved fold spec to {args.out}")


if __name__ == "__main__":
    main()
