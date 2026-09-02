"""
Diagnostic analysis of existing LOCSO out-of-fold predictions.

    python scripts/analyze_ml_models.py

No model training. Reads prediction CSVs from Steps 3B–3D and summarizes
where each model succeeds or fails on the 104 experimental observations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "ml_model_diagnostics.md"
CSV_PATH = ROOT / "reports" / "ml_model_diagnostics.csv"
FIG_DIR = ROOT / "figures"

TRAINING_TYPE = "experimental_only"
MODELS = {
    "RKM": "rkm_prediction",
    "Linear Regression": "linear_prediction",
    "Polynomial Degree 2": "poly2_prediction",
    "Polynomial Degree 3": "poly3_prediction",
    "Random Forest": "rf_prediction",
}


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_pred - y_true
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual, ddof=0)),
        "max_abs_residual": float(np.max(np.abs(residual))),
        "N": int(len(y_true)),
    }


def grouped_metrics(df: pd.DataFrame, pred_col: str, group_col: str) -> pd.DataFrame:
    rows = []
    for key, grp in df.groupby(group_col, sort=True):
        m = calculate_metrics(grp["experimental_delta_mixH"], grp[pred_col])
        rows.append({"group": key, "MAE": m["MAE"], "RMSE": m["RMSE"], "N": m["N"]})
    return pd.DataFrame(rows)


def load_oof_predictions() -> pd.DataFrame:
    baseline = pd.read_csv(ROOT / "reports" / "ml_baseline_predictions.csv")
    poly = pd.read_csv(ROOT / "reports" / "ml_polynomial_predictions.csv")
    rf = pd.read_csv(ROOT / "reports" / "ml_random_forest_predictions.csv")

    linear = baseline[baseline["training_type"] == TRAINING_TYPE].copy()
    poly2 = poly[(poly["training_type"] == TRAINING_TYPE) & (poly["model_degree"] == 2)].copy()
    poly3 = poly[(poly["training_type"] == TRAINING_TYPE) & (poly["model_degree"] == 3)].copy()
    forest = rf[rf["training_type"] == TRAINING_TYPE].copy()

    for name, frame in [("linear", linear), ("poly2", poly2), ("poly3", poly3), ("rf", forest)]:
        if len(frame) != 104:
            raise RuntimeError(f"{name}: expected 104 rows, got {len(frame)}")
        if frame["experiment_id"].duplicated().any():
            raise RuntimeError(f"{name}: duplicate experiment_id")

    base_cols = [
        "experiment_id",
        "cross_section",
        "temperature_K",
        "xBi",
        "xIn",
        "xSn",
        "experimental_delta_mixH",
        "rkm_prediction",
        "fold",
        "held_out_section",
    ]
    out = linear[base_cols].copy()
    out["linear_prediction"] = linear["ml_prediction"].to_numpy()
    out["poly2_prediction"] = poly2.set_index("experiment_id").loc[out["experiment_id"], "ml_prediction"].to_numpy()
    out["poly3_prediction"] = poly3.set_index("experiment_id").loc[out["experiment_id"], "ml_prediction"].to_numpy()
    out["rf_prediction"] = forest.set_index("experiment_id").loc[out["experiment_id"], "ml_prediction"].to_numpy()

    for pred_col in ["rkm_prediction", "linear_prediction", "poly2_prediction", "poly3_prediction", "rf_prediction"]:
        out[f"{pred_col.replace('_prediction', '')}_residual"] = out[pred_col] - out["experimental_delta_mixH"]

    return out


def worst_observations(df: pd.DataFrame, pred_col: str, n: int = 5) -> pd.DataFrame:
    residual_col = pred_col.replace("_prediction", "_residual")
    ranked = df.assign(abs_residual=df[residual_col].abs()).sort_values("abs_residual", ascending=False).head(n)
    return ranked[
        [
            "experiment_id",
            "cross_section",
            "temperature_K",
            "xIn",
            "experimental_delta_mixH",
            pred_col,
            residual_col,
        ]
    ]


def short_section(section: str) -> str:
    if "0.33Bi0.67" in section:
        return "Bi-rich (0.33Bi0.67)"
    if "0.50Bi0.50" in section:
        return "Equiatomic (0.50Bi0.50)"
    if "0.67Bi0.33" in section:
        return "Sn-rich (0.67Bi0.33)"
    return section


def plot_pred_vs_exp(df: pd.DataFrame, pred_col: str, title: str, filename: str) -> None:
    y = df["experimental_delta_mixH"].to_numpy()
    p = df[pred_col].to_numpy()
    lo = min(y.min(), p.min()) - 50
    hi = max(y.max(), p.max()) + 50

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, p, alpha=0.75, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Experimental ΔmixH (J/mol)")
    ax.set_ylabel("Predicted ΔmixH (J/mol)")
    ax.set_title(title)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150)
    plt.close(fig)


def plot_residual_vs_xin(df: pd.DataFrame, residual_col: str, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for section, grp in df.groupby("cross_section"):
        ax.scatter(grp["xIn"], grp[residual_col], alpha=0.75, label=short_section(section), s=35)
    ax.axhline(0.0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xlabel("xIn")
    ax.set_ylabel("Residual (prediction − experiment) J/mol")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150)
    plt.close(fig)


def poly2_vs_rkm_summary(df: pd.DataFrame) -> list[str]:
    lines = ["## Polynomial Degree 2 vs RKM", ""]
    df = df.copy()
    df["poly2_abs"] = df["poly2_residual"].abs()
    df["rkm_abs"] = df["rkm_residual"].abs()
    df["poly2_better"] = df["poly2_abs"] < df["rkm_abs"]

    n_better = int(df["poly2_better"].sum())
    lines.append(f"- Polynomial D2 has lower absolute residual on **{n_better}/104** points ({100 * n_better / 104:.1f}%).")
    lines.append("")

    lines.append("### By cross-section")
    lines.append("")
    lines.append("| Cross-section | N | Poly2 MAE | RKM MAE | Poly2 better (count) |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for section, grp in df.groupby("cross_section", sort=True):
        p_mae = calculate_metrics(grp["experimental_delta_mixH"], grp["poly2_prediction"])["MAE"]
        r_mae = calculate_metrics(grp["experimental_delta_mixH"], grp["rkm_prediction"])["MAE"]
        better = int(grp["poly2_better"].sum())
        lines.append(f"| `{section}` | {len(grp)} | {p_mae:.2f} | {r_mae:.2f} | {better}/{len(grp)} |")
    lines.append("")

    lines.append("### By temperature")
    lines.append("")
    lines.append("| T (K) | N | Poly2 MAE | RKM MAE | Poly2 better (count) |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for temp, grp in df.groupby("temperature_K", sort=True):
        p_mae = calculate_metrics(grp["experimental_delta_mixH"], grp["poly2_prediction"])["MAE"]
        r_mae = calculate_metrics(grp["experimental_delta_mixH"], grp["rkm_prediction"])["MAE"]
        better = int(grp["poly2_better"].sum())
        lines.append(f"| {int(temp)} | {len(grp)} | {p_mae:.2f} | {r_mae:.2f} | {better}/{len(grp)} |")
    lines.append("")

    lines.append("### By LOCSO fold (held-out section)")
    lines.append("")
    lines.append("| Fold | Held-out section | Poly2 MAE | RKM MAE | Poly2 better (count) |")
    lines.append("| ---: | --- | ---: | ---: | ---: |")
    for fold, grp in df.groupby("fold", sort=True):
        p_mae = calculate_metrics(grp["experimental_delta_mixH"], grp["poly2_prediction"])["MAE"]
        r_mae = calculate_metrics(grp["experimental_delta_mixH"], grp["rkm_prediction"])["MAE"]
        better = int(grp["poly2_better"].sum())
        held = grp["held_out_section"].iloc[0]
        lines.append(f"| {fold} | `{held}` | {p_mae:.2f} | {r_mae:.2f} | {better}/{len(grp)} |")
    lines.append("")

    return lines


def write_report(df: pd.DataFrame, model_stats: dict[str, dict]) -> None:
    lines = [
        "# ML Model Diagnostics",
        "",
        "Analysis of **104 out-of-fold experimental predictions** from LOCSO",
        f"(`{TRAINING_TYPE}` training setup). No new models were trained.",
        "",
        "Residual = prediction − experimental ΔmixH.",
        "",
        "## Overall metrics",
        "",
        "| Model | MAE | RMSE | R² | Residual mean | Residual std | Max |residual| |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for model_name, pred_col in MODELS.items():
        m = model_stats[model_name]["overall"]
        lines.append(
            f"| {model_name} | {m['MAE']:.2f} | {m['RMSE']:.2f} | {m['R2']:.4f} | "
            f"{m['residual_mean']:.2f} | {m['residual_std']:.2f} | {m['max_abs_residual']:.2f} |"
        )
    lines.append("")

    for model_name, pred_col in MODELS.items():
        lines.append(f"## {model_name}")
        lines.append("")

        lines.append("### MAE / RMSE by cross-section")
        lines.append("")
        sec = model_stats[model_name]["by_section"]
        lines.append("| Cross-section | N | MAE | RMSE |")
        lines.append("| --- | ---: | ---: | ---: |")
        for _, row in sec.iterrows():
            lines.append(f"| `{row['group']}` | {int(row['N'])} | {row['MAE']:.2f} | {row['RMSE']:.2f} |")
        worst_sec = sec.loc[sec["MAE"].idxmax(), "group"]
        lines.append("")
        lines.append(f"**Worst cross-section (MAE):** `{worst_sec}`")
        lines.append("")

        lines.append("### MAE / RMSE by temperature")
        lines.append("")
        temp = model_stats[model_name]["by_temperature"]
        lines.append("| T (K) | N | MAE | RMSE |")
        lines.append("| --- | ---: | ---: | ---: |")
        for _, row in temp.iterrows():
            lines.append(f"| {int(row['group'])} | {int(row['N'])} | {row['MAE']:.2f} | {row['RMSE']:.2f} |")
        worst_temp = int(temp.loc[temp["MAE"].idxmax(), "group"])
        lines.append("")
        lines.append(f"**Worst temperature (MAE):** {worst_temp} K")
        lines.append("")

        lines.append("### Largest absolute residuals")
        lines.append("")
        worst = model_stats[model_name]["worst"]
        lines.append("| experiment_id | cross_section | T (K) | xIn | experimental | prediction | residual |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for _, row in worst.iterrows():
            resid_col = pred_col.replace("_prediction", "_residual")
            lines.append(
                f"| {row['experiment_id']} | `{row['cross_section']}` | {int(row['temperature_K'])} | "
                f"{row['xIn']:.4f} | {row['experimental_delta_mixH']:.1f} | "
                f"{row[pred_col]:.1f} | {row[resid_col]:.1f} |"
            )
        lines.append("")

    lines.extend(poly2_vs_rkm_summary(df))

    lines.extend(
        [
            "## Figures",
            "",
            "- `figures/ml_diag_pred_vs_exp_rkm.png`",
            "- `figures/ml_diag_pred_vs_exp_poly2.png`",
            "- `figures/ml_diag_pred_vs_exp_rf.png`",
            "- `figures/ml_diag_residual_vs_xin_poly2.png`",
            "- `figures/ml_diag_residual_vs_xin_rkm.png`",
            "",
        ]
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


def main() -> int:
    df = load_oof_predictions()

    model_stats: dict[str, dict] = {}
    for model_name, pred_col in MODELS.items():
        model_stats[model_name] = {
            "overall": calculate_metrics(df["experimental_delta_mixH"], df[pred_col]),
            "by_section": grouped_metrics(df, pred_col, "cross_section"),
            "by_temperature": grouped_metrics(df, pred_col, "temperature_K"),
            "worst": worst_observations(df, pred_col),
        }

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_pred_vs_exp(df, "rkm_prediction", "RKM: predicted vs experimental", "ml_diag_pred_vs_exp_rkm.png")
    plot_pred_vs_exp(
        df,
        "poly2_prediction",
        "Polynomial degree 2: predicted vs experimental",
        "ml_diag_pred_vs_exp_poly2.png",
    )
    plot_pred_vs_exp(
        df,
        "rf_prediction",
        "Random Forest: predicted vs experimental",
        "ml_diag_pred_vs_exp_rf.png",
    )
    plot_residual_vs_xin(
        df,
        "poly2_residual",
        "Polynomial degree 2: residual vs xIn",
        "ml_diag_residual_vs_xin_poly2.png",
    )
    plot_residual_vs_xin(
        df,
        "rkm_residual",
        "RKM: residual vs xIn",
        "ml_diag_residual_vs_xin_rkm.png",
    )
    print(f"Wrote figures to {FIG_DIR}")

    out_cols = [
        "experiment_id",
        "cross_section",
        "temperature_K",
        "xBi",
        "xIn",
        "xSn",
        "experimental_delta_mixH",
        "fold",
        "held_out_section",
        "rkm_prediction",
        "rkm_residual",
        "linear_prediction",
        "linear_residual",
        "poly2_prediction",
        "poly2_residual",
        "poly3_prediction",
        "poly3_residual",
        "rf_prediction",
        "rf_residual",
    ]
    df[out_cols].to_csv(CSV_PATH, index=False)
    print(f"Wrote {CSV_PATH}")

    write_report(df, model_stats)

    print("\nOverall MAE summary:")
    for model_name in MODELS:
        m = model_stats[model_name]["overall"]
        print(f"  {model_name}: MAE={m['MAE']:.2f}, RMSE={m['RMSE']:.2f}, R²={m['R2']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
