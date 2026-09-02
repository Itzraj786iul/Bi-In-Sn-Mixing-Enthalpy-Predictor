"""Composition overlap between experimental and synthetic datasets.

    python scripts/analyze_ml_overlap.py

Loads the three primary CSV files, computes nearest synthetic composition
distance for each experimental point, prints a summary, and writes
reports/ml_overlap_analysis.csv.

No ML training. No dataset modification.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORT = ROOT / "reports" / "ml_overlap_analysis.csv"

THRESHOLDS = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.012, 0.02, 0.03)


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


def min_comp_distance(exp_xyz: np.ndarray, syn_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diff = exp_xyz[:, None, :] - syn_xyz[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=2))
    idx = dist.argmin(axis=1)
    return dist[np.arange(len(exp_xyz)), idx], idx


def nearest_on_same_section_and_T(exp: pd.DataFrame, syn: pd.DataFrame) -> np.ndarray:
    out = np.empty(len(exp))
    for i in range(len(exp)):
        row = exp.iloc[i]
        sub = syn[(syn["cross_section"] == row["cross_section"]) & (syn["temperature_K"] == row["temperature_K"])]
        r = row[["xBi", "xIn", "xSn"]].to_numpy(dtype=float)
        s = sub[["xBi", "xIn", "xSn"]].to_numpy(dtype=float)
        out[i] = np.sqrt(((r[None, :] - s) ** 2).sum(axis=1)).min()
    return out


def count_below(dist: np.ndarray, thresholds=THRESHOLDS) -> None:
    n = len(dist)
    for t in thresholds:
        print(f"  <= {t:g}: {int((dist <= t).sum())} / {n}")


def main() -> int:
    exp = load("original_experimental_data.csv")
    syn_a = load("synthetic_cross_sections.csv")
    syn_b = load("synthetic_full_ternary.csv")

    exp_xyz = exp[["xBi", "xIn", "xSn"]].to_numpy(dtype=float)
    a_xyz = syn_a[["xBi", "xIn", "xSn"]].to_numpy(dtype=float)
    b_xyz = syn_b[["xBi", "xIn", "xSn"]].to_numpy(dtype=float)

    d_a, idx_a = min_comp_distance(exp_xyz, a_xyz)
    d_b, idx_b = min_comp_distance(exp_xyz, b_xyz)
    d_a_sec_t = nearest_on_same_section_and_T(exp, syn_a)

    result = exp.copy()
    result["nearest_A_dist"] = d_a
    result["nearest_A_dist_same_secT"] = d_a_sec_t
    result["nearest_A_id"] = syn_a.iloc[idx_a]["id"].values
    result["nearest_B_dist"] = d_b
    result["nearest_B_id"] = syn_b.iloc[idx_b]["id"].values
    result["nearest_A_region"] = syn_a.iloc[idx_a]["region_type"].values
    result["nearest_B_region"] = syn_b.iloc[idx_b]["region_type"].values

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(REPORT, index=False)

    print("ML overlap analysis")
    print("=" * 60)
    print(f"Experimental points: {len(exp)}")
    print(f"Dataset A: {len(syn_a)}  |  Dataset B: {len(syn_b)}")
    print()
    print("Experimental structure (cross-section x temperature):")
    print(exp.groupby(["cross_section", "temperature_K"]).size().to_string())
    print()
    print("Nearest Dataset A (same cross-section + same temperature):")
    print(f"  min    = {d_a_sec_t.min():.6f}")
    print(f"  median = {np.median(d_a_sec_t):.6f}")
    print(f"  max    = {d_a_sec_t.max():.6f}")
    print("  counts:")
    count_below(d_a_sec_t)
    print()
    print("Nearest Dataset B (any composition, any temperature):")
    print(f"  min    = {d_b.min():.6f}")
    print(f"  median = {np.median(d_b):.6f}")
    print(f"  max    = {d_b.max():.6f}")
    print("  counts:")
    count_below(d_b)
    print()
    print(f"Wrote {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
