"""
Build final ML surrogate predictions over Dataset B (full ternary grid).

    python scripts/build_final_surrogate.py

Trains frozen final Poly D2 on 104 experimental observations only.
Predicts Dataset B for surrogate exploration — NOT experimental validation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rkm_model import rkm_delta_mix_h

FEATURE_COLS = ["xBi", "xIn", "temperature_K"]
EXP_TARGET = "integral_mixing_enthalpy_J_per_mol"
DEGREE = 2
REPORT_PATH = ROOT / "reports" / "final_surrogate_analysis.md"
PRED_PATH = ROOT / "reports" / "final_surrogate_predictions.csv"
FIG_DIR = ROOT / "figures"

LOCSO_REF = {"MAE": 41.10, "RMSE": 56.34, "R2": 0.9712}
FIVEFOLD_REF = {"MAE": 30.34, "RMSE": 40.02, "R2": 0.9855}
RKM_EXP_REF = {"MAE": 57.13, "RMSE": 73.49, "R2": 0.9510}
LOCSO_PRED_PATH = ROOT / "reports" / "ml_composition_representation_predictions.csv"


def load_experimental() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "original_experimental_data.csv")


def load_dataset_b() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "synthetic_full_ternary.csv")


def features(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURE_COLS].to_numpy(dtype=float)


def train_final_model(exp: pd.DataFrame) -> tuple[PolynomialFeatures, LinearRegression]:
    X = features(exp)
    y = exp[EXP_TARGET].to_numpy(dtype=float)
    poly = PolynomialFeatures(degree=DEGREE, include_bias=True)
    X_poly = poly.fit_transform(X)
    coef, _, _, _ = np.linalg.lstsq(X_poly, y, rcond=None)
    model = LinearRegression(fit_intercept=False)
    model.coef_ = coef
    model.intercept_ = 0.0
    return poly, model


def predict(poly: PolynomialFeatures, model: LinearRegression, df: pd.DataFrame) -> np.ndarray:
    return model.predict(poly.transform(features(df)))


def rkm_predict_df(df: pd.DataFrame) -> np.ndarray:
    return np.array([rkm_delta_mix_h(r.xBi, r.xIn, r.xSn) for r in df.itertuples()])


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "N": int(len(y_true))}


def plot_xy_scatter(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    filename: str,
    cbar_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(df["xBi"], df["xIn"], c=df[value_col], s=8, cmap="viridis", alpha=0.85)
    ax.set_xlabel("xBi")
    ax.set_ylabel("xIn")
    ax.set_title(title)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150)
    plt.close(fig)


def write_report(
    out_df: pd.DataFrame,
    exp_locsos: pd.DataFrame,
) -> None:
    n_rows = len(out_df)
    ml_min, ml_max = out_df["ml_prediction"].min(), out_df["ml_prediction"].max()
    rkm_min, rkm_max = out_df["rkm_prediction"].min(), out_df["rkm_prediction"].max()
    diff_min, diff_max = out_df["difference_ml_minus_rkm"].min(), out_df["difference_ml_minus_rkm"].max()
    mean_abs = float(out_df["absolute_difference"].mean())
    max_abs = float(out_df["absolute_difference"].max())

    lines = [
        "# Final ML Surrogate Analysis (Dataset B)",
        "",
        "## Important disclaimer",
        "Dataset B is **RKM-generated** and covers many compositions **without experimental measurements**.",
        "ML predictions on Dataset B are **surrogate / model-exploration outputs only**.",
        "They are **not experimentally validated** in unsampled ternary regions.",
        "",
        "## Final frozen model",
        "PolynomialFeatures(degree=2) + minimum-norm OLS",
        "",
        "Features: xBi, xIn, temperature_K",
        "",
        "Trained on: 104 original experimental observations only.",
        "",
        "## Experimentally validated performance (104 calorimetry points only)",
        "",
        "These metrics come from **prior validation** on experimental data. Dataset B is not used here.",
        "",
        "### Primary validation — LOCSO (Step 3H, Poly D2 2-var + T)",
        f"- ML: MAE = {LOCSO_REF['MAE']:.2f} J/mol, RMSE = {LOCSO_REF['RMSE']:.2f} J/mol, R² = {LOCSO_REF['R2']:.4f}",
        f"- RKM: MAE = {RKM_EXP_REF['MAE']:.2f} J/mol, RMSE = {RKM_EXP_REF['RMSE']:.2f} J/mol, R² = {RKM_EXP_REF['R2']:.4f}",
        "",
        "### Secondary validation — random 5-fold (Step 3J)",
        f"- ML: MAE = {FIVEFOLD_REF['MAE']:.2f} J/mol, RMSE = {FIVEFOLD_REF['RMSE']:.2f} J/mol, R² = {FIVEFOLD_REF['R2']:.4f}",
        "",
        "### LOCSO ML vs RKM on same 104 points (from Step 3H predictions file)",
        "",
    ]

    ml_locso = calculate_metrics(
        exp_locsos["experimental_delta_mixH"],
        exp_locsos["ml_prediction"],
    )
    rkm_locso = calculate_metrics(
        exp_locsos["experimental_delta_mixH"],
        exp_locsos["rkm_prediction"],
    )
    lines.append(
        f"- ML: MAE = {ml_locso['MAE']:.2f}, RMSE = {ml_locso['RMSE']:.2f}, R² = {ml_locso['R2']:.4f}"
    )
    lines.append(
        f"- RKM: MAE = {rkm_locso['MAE']:.2f}, RMSE = {rkm_locso['RMSE']:.2f}, R² = {rkm_locso['R2']:.4f}"
    )
    lines.append("")

    lines.extend(
        [
            "## Dataset B surrogate predictions (NOT experimental validation)",
            "",
            f"- Rows predicted: **{n_rows}**",
            f"- ML prediction range: {ml_min:.2f} to {ml_max:.2f} J/mol",
            f"- RKM prediction range: {rkm_min:.2f} to {rkm_max:.2f} J/mol",
            f"- ML − RKM difference range: {diff_min:.2f} to {diff_max:.2f} J/mol",
            f"- Mean |ML − RKM|: {mean_abs:.2f} J/mol",
            f"- Maximum |ML − RKM|: {max_abs:.2f} J/mol",
            "",
            "### By temperature",
            "",
            "| T (K) | N | ML min | ML max | RKM min | RKM max | Mean |ML−RKM| | Max |ML−RKM| |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for temp, grp in out_df.groupby("temperature_K", sort=True):
        lines.append(
            f"| {int(temp)} | {len(grp)} | {grp['ml_prediction'].min():.2f} | {grp['ml_prediction'].max():.2f} | "
            f"{grp['rkm_prediction'].min():.2f} | {grp['rkm_prediction'].max():.2f} | "
            f"{grp['absolute_difference'].mean():.2f} | {grp['absolute_difference'].max():.2f} |"
        )

    lines.extend(["", "### Largest |ML − RKM| compositions", ""])
    top = out_df.nlargest(10, "absolute_difference")
    lines.append("| id | T (K) | xBi | xIn | xSn | region_type | ML | RKM | ML−RKM |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |")
    for _, row in top.iterrows():
        lines.append(
            f"| {row['id']} | {int(row['temperature_K'])} | {row['xBi']:.3f} | {row['xIn']:.3f} | "
            f"{row['xSn']:.3f} | {row['region_type']} | {row['ml_prediction']:.1f} | "
            f"{row['rkm_prediction']:.1f} | {row['difference_ml_minus_rkm']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Figures (Cartesian xBi–xIn maps; surrogate exploration only)",
            "",
            "- `figures/ml_surrogate_dmixH_767K.png`",
            "- `figures/ml_surrogate_dmixH_813K.png`",
            "- `figures/ml_surrogate_dmixH_855K.png`",
            "- `figures/ml_surrogate_diff_rkm_813K.png`",
            "",
        ]
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


def main() -> int:
    exp = load_experimental()
    syn_b = load_dataset_b()
    print(f"Experimental N = {len(exp)}, Dataset B N = {len(syn_b)}")

    poly, model = train_final_model(exp)

    out = syn_b.copy()
    out["ml_prediction"] = predict(poly, model, syn_b)
    out["rkm_prediction"] = rkm_predict_df(syn_b)
    out["difference_ml_minus_rkm"] = out["ml_prediction"] - out["rkm_prediction"]
    out["absolute_difference"] = out["difference_ml_minus_rkm"].abs()

    out.to_csv(PRED_PATH, index=False)
    print(f"Wrote {PRED_PATH}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for temp in (767, 813, 855):
        sub = out[out["temperature_K"] == temp]
        plot_xy_scatter(
            sub,
            "ml_prediction",
            f"ML surrogate ΔmixH at {temp} K (Dataset B — not experimentally validated)",
            f"ml_surrogate_dmixH_{temp}K.png",
            "ML ΔmixH (J/mol)",
        )
    sub813 = out[out["temperature_K"] == 813]
    plot_xy_scatter(
        sub813,
        "difference_ml_minus_rkm",
        "ML − RKM at 813 K (Dataset B surrogate exploration)",
        "ml_surrogate_diff_rkm_813K.png",
        "ML − RKM (J/mol)",
    )
    print(f"Wrote figures to {FIG_DIR}")

    exp_locsos = pd.read_csv(LOCSO_PRED_PATH)
    exp_locsos = exp_locsos[exp_locsos["feature_set"] == "composition_temperature_2vars"].copy()

    write_report(out, exp_locsos)

    print(f"Dataset B rows: {len(out)}")
    print(f"ML range: [{out['ml_prediction'].min():.2f}, {out['ml_prediction'].max():.2f}]")
    print(f"RKM range: [{out['rkm_prediction'].min():.2f}, {out['rkm_prediction'].max():.2f}]")
    print(f"Mean |ML-RKM|: {out['absolute_difference'].mean():.2f}")
    print(f"Max |ML-RKM|: {out['absolute_difference'].max():.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
