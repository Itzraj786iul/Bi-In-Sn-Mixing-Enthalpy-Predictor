"""
Rebuild Dataset A and Dataset B from the experimental CSV and the RKM model.

    python scripts/generate_synthetic_data.py

Does NOT overwrite data/original_experimental_data.csv or data/extraction_audit.csv.
DOES overwrite the two synthetic CSV files. Random seed = 42.

Do not run this unless you intend to replace the current synthetic datasets.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rkm_model import rkm_delta_mix_h

RANDOM_SEED = 42
XIN_GRID_MIN = 0.09
XIN_GRID_MAX = 0.91
XIN_GRID_STEP = 0.001
SIMPLEX_STEP = 0.01
NEAR_CS_DISTANCE = 0.03
ANCHOR_DISTANCE = 0.012
ALLOWED_TEMPERATURES = {767, 813, 855}
DATA_DIR = ROOT / "data"

# Table III footnote a, u(ΔmixH) in J/mol.
SERIES_U_J = {
    1: 23.3,
    2: 19.7,
    3: 16.08,
    4: 18.17,
    5: 14.4,
    6: 12.7,
    7: 13.2,
    8: 10.5,
    9: 7.9,
}

CS_RATIO_LABELED = {
    "(Sn0.33Bi0.67)1-xInx": (0.67, 0.33),
    "(Sn0.50Bi0.50)1-xInx": (0.50, 0.50),
    "(Sn0.67Bi0.33)1-xInx": (0.33, 0.67),
}

SERIES_FOR_SECTION_TEMP = {
    ("(Sn0.33Bi0.67)1-xInx", 767): 1,
    ("(Sn0.33Bi0.67)1-xInx", 813): 2,
    ("(Sn0.33Bi0.67)1-xInx", 855): 3,
    ("(Sn0.50Bi0.50)1-xInx", 767): 4,
    ("(Sn0.50Bi0.50)1-xInx", 813): 5,
    ("(Sn0.50Bi0.50)1-xInx", 855): 6,
    ("(Sn0.67Bi0.33)1-xInx", 767): 7,
    ("(Sn0.67Bi0.33)1-xInx", 813): 8,
    ("(Sn0.67Bi0.33)1-xInx", 855): 9,
}


def bath_ratio(exp_df: pd.DataFrame, cross_section: str) -> float:
    sub = exp_df[exp_df["cross_section"] == cross_section]
    return float(sub["xBi"].iloc[0] / (sub["xBi"].iloc[0] + sub["xSn"].iloc[0]))


def distance_to_cross_section_line(x_bi, x_in, x_sn, r_bi: float) -> np.ndarray:
    x_bi = np.asarray(x_bi, dtype=float)
    x_in = np.asarray(x_in, dtype=float)
    x_sn = np.asarray(x_sn, dtype=float)
    p = np.stack([x_bi, x_in, x_sn], axis=-1)
    a = np.array([r_bi, 0.0, 1.0 - r_bi])
    b = np.array([0.0, 1.0, 0.0])
    ab = b - a
    ap = p - a
    t = np.clip(np.sum(ap * ab, axis=-1) / np.dot(ab, ab), 0.0, 1.0)
    proj = a + t[..., None] * ab
    return np.linalg.norm(p - proj, axis=-1)


def classify_region(x_bi, x_in, x_sn, exp_df: pd.DataFrame):
    cs_ratios = [bath_ratio(exp_df, cs) for cs in CS_RATIO_LABELED]
    d_cs = np.min(
        np.stack([distance_to_cross_section_line(x_bi, x_in, x_sn, r) for r in cs_ratios], axis=0),
        axis=0,
    )
    exp_pts = exp_df[["xBi", "xIn", "xSn"]].to_numpy()
    p = np.stack([np.asarray(x_bi), np.asarray(x_in), np.asarray(x_sn)], axis=-1)
    d_matrix = np.sqrt(((p[:, None, :] - exp_pts[None, :, :]) ** 2).sum(axis=2))
    d_exp = d_matrix.min(axis=1)
    nearest_idx = d_matrix.argmin(axis=1)

    region = np.full(len(p), "unsampled_ternary_region", dtype=object)
    region[d_cs <= NEAR_CS_DISTANCE] = "near_experimental_cross_section"
    region[d_exp <= ANCHOR_DISTANCE] = "experimental_anchor"
    return region, d_exp, nearest_idx


def attach_experimental_anchors(df: pd.DataFrame, exp_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    region, d_exp, nearest_idx = classify_region(out["xBi"].values, out["xIn"].values, out["xSn"].values, exp_df)
    nearest = exp_df.iloc[nearest_idx].reset_index(drop=True)
    out["region_type"] = region
    out["nearest_experimental_id"] = nearest["id"].values
    out["distance_to_nearest_experimental_point"] = d_exp
    out["experimental_anchor_value"] = nearest["integral_mixing_enthalpy_J_per_mol"].values
    out["synthetic_minus_experimental"] = out["delta_mix_H_J_per_mol"] - out["experimental_anchor_value"]
    return out


def make_cross_section_synthetic(exp_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    x_in_grid = np.round(np.arange(XIN_GRID_MIN, XIN_GRID_MAX + 1e-12, XIN_GRID_STEP), 10)
    rows = []
    for cs in CS_RATIO_LABELED:
        r_bi = bath_ratio(exp_df, cs)
        r_sn = 1.0 - r_bi
        for T in sorted(ALLOWED_TEMPERATURES):
            sid = SERIES_FOR_SECTION_TEMP[(cs, T)]
            sigma = float(SERIES_U_J[sid])
            for x_in in x_in_grid:
                x_bi = r_bi * (1.0 - x_in)
                x_sn = r_sn * (1.0 - x_in)
                base = float(rkm_delta_mix_h(x_bi, x_in, x_sn, T))
                eps = float(rng.normal(0.0, sigma))
                rows.append(
                    {
                        "source": "synthetic",
                        "synthetic": True,
                        "generation_method": "RKM_interpolation_plus_noise",
                        "series_id": sid,
                        "cross_section": cs,
                        "temperature_K": T,
                        "xBi": x_bi,
                        "xIn": float(x_in),
                        "xSn": x_sn,
                        "delta_mix_H_J_per_mol": base + eps,
                        "base_model_prediction_J_per_mol": base,
                        "noise_added_J_per_mol": eps,
                        "uncertainty_J_per_mol": sigma,
                        "uncertainty_basis": (
                            f"Table III footnote a, Series {sid}: "
                            f"u(ΔmixH)={sigma / 1000.0} kJ/mol = {sigma:.2f} J/mol used as σ of N(0, σ)."
                        ),
                        "notes": (
                            "Model-generated along an experimentally investigated cross-section. "
                            "NOT an experimental measurement."
                        ),
                    }
                )
    return attach_experimental_anchors(pd.DataFrame(rows), exp_df)


def simplex_grid(step: float) -> np.ndarray:
    n = int(round(1.0 / step))
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            pts.append((i / n, j / n, k / n))
    return np.array(pts, dtype=float)


def make_full_ternary_synthetic(exp_df: pd.DataFrame) -> pd.DataFrame:
    comps = simplex_grid(SIMPLEX_STEP)
    rows = []
    for T in sorted(ALLOWED_TEMPERATURES):
        x_bi, x_in, x_sn = comps[:, 0], comps[:, 1], comps[:, 2]
        base = np.asarray(rkm_delta_mix_h(x_bi, x_in, x_sn, T), dtype=float)
        for k in range(len(comps)):
            rows.append(
                {
                    "source": "synthetic",
                    "synthetic": True,
                    "generation_method": "RKM_full_ternary_grid",
                    "series_id": np.nan,
                    "cross_section": "",
                    "temperature_K": T,
                    "xBi": float(x_bi[k]),
                    "xIn": float(x_in[k]),
                    "xSn": float(x_sn[k]),
                    "delta_mix_H_J_per_mol": float(base[k]),
                    "base_model_prediction_J_per_mol": float(base[k]),
                    "noise_added_J_per_mol": 0.0,
                    "uncertainty_J_per_mol": 0.0,
                    "uncertainty_basis": (
                        "noise=0; temperature-independent Table IV RKM evaluated at the listed T "
                        "for bookkeeping only (paper: almost temperature-independent)."
                    ),
                    "notes": (
                        "Full-ternary RKM grid point. NOT experimentally validated except near "
                        "the three measured cross-sections. NOT an experimental measurement."
                    ),
                }
            )
    return attach_experimental_anchors(pd.DataFrame(rows), exp_df)


def finalize(syn_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = syn_df.copy()
    out.insert(0, "id", [f"{prefix}_{i:05d}" for i in range(1, len(out) + 1)])
    out["delta_mix_H_kJ_per_mol"] = out["delta_mix_H_J_per_mol"] / 1000.0
    cols = [
        "id",
        "source",
        "synthetic",
        "generation_method",
        "series_id",
        "temperature_K",
        "xBi",
        "xIn",
        "xSn",
        "delta_mix_H_J_per_mol",
        "delta_mix_H_kJ_per_mol",
        "base_model_prediction_J_per_mol",
        "noise_added_J_per_mol",
        "uncertainty_J_per_mol",
        "cross_section",
        "region_type",
        "nearest_experimental_id",
        "distance_to_nearest_experimental_point",
        "experimental_anchor_value",
        "synthetic_minus_experimental",
        "uncertainty_basis",
        "notes",
    ]
    return out[cols]


def main() -> None:
    exp_path = DATA_DIR / "original_experimental_data.csv"
    if not exp_path.exists():
        raise SystemExit("Missing data/original_experimental_data.csv. Do not invent experimental rows.")

    exp_df = pd.read_csv(exp_path)
    rng_a = np.random.default_rng(RANDOM_SEED)

    syn_a = finalize(make_cross_section_synthetic(exp_df, rng_a), "SYN_A")
    syn_b = finalize(make_full_ternary_synthetic(exp_df), "SYN_B")

    syn_a.to_csv(DATA_DIR / "synthetic_cross_sections.csv", index=False)
    syn_b.to_csv(DATA_DIR / "synthetic_full_ternary.csv", index=False)

    print("Wrote Dataset A and Dataset B.")
    print(f"Dataset A: {len(syn_a)} rows")
    print(f"Dataset B: {len(syn_b)} rows")
    print("WARNING: Synthetic data are model-generated and are NOT experimental observations.")


if __name__ == "__main__":
    main()
