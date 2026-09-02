"""
Response-surface and physical-behavior analysis for the frozen final Poly D2 model.

    python scripts/analyze_final_surface.py

Uses fixed coefficients from Step 4A — does NOT refit the model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rkm_model import rkm_delta_mix_h

REPORT_PATH = ROOT / "reports" / "final_surface_analysis.md"
COEF_PATH = ROOT / "reports" / "final_model_coefficients.csv"
FIG_DIR = ROOT / "figures"
EXP_PATH = ROOT / "data" / "original_experimental_data.csv"
EXP_TARGET = "integral_mixing_enthalpy_J_per_mol"
TEMPERATURES = (767, 813, 855)
GRID_STEP = 0.01
NEAR_EXP_DIST = 0.03

LOCSO_REF = {"MAE": 41.10, "RMSE": 56.34, "R2": 0.9712}
RKM_REF = {"MAE": 57.13, "RMSE": 73.49, "R2": 0.9510}


def load_coefficients() -> dict[str, float]:
    coef_df = pd.read_csv(COEF_PATH)
    key_map = {
        "intercept": "intercept",
        "xBi": "xBi",
        "xIn": "xIn",
        "temperature_K": "T",
        "xBi²": "xBi2",
        "xBi·xIn": "xBi_xIn",
        "xBi·temperature_K": "xBi_T",
        "xIn²": "xIn2",
        "xIn·temperature_K": "xIn_T",
        "temperature_K²": "T2",
    }
    out = {}
    for _, row in coef_df.iterrows():
        out[key_map[row["term_display"]]] = float(row["coefficient"])
    return out


def predict_ml(x_bi: np.ndarray, x_in: np.ndarray, t: np.ndarray, c: dict[str, float]) -> np.ndarray:
    return (
        c["intercept"]
        + c["xBi"] * x_bi
        + c["xIn"] * x_in
        + c["T"] * t
        + c["xBi2"] * x_bi**2
        + c["xBi_xIn"] * x_bi * x_in
        + c["xBi_T"] * x_bi * t
        + c["xIn2"] * x_in**2
        + c["xIn_T"] * x_in * t
        + c["T2"] * t**2
    )


def temperature_terms(x_bi: np.ndarray, x_in: np.ndarray, t: np.ndarray, c: dict[str, float]) -> np.ndarray:
    """Pure temperature-related part of the fitted equation at fixed composition."""
    return c["T"] * t + c["xBi_T"] * x_bi * t + c["xIn_T"] * x_in * t + c["T2"] * t**2


def bi_in_interaction_term(x_bi: np.ndarray, x_in: np.ndarray, c: dict[str, float]) -> np.ndarray:
    return c["xBi_xIn"] * x_bi * x_in


def build_ternary_grid(step: float = GRID_STEP) -> pd.DataFrame:
    vals = np.arange(0.0, 1.0 + step / 2, step)
    rows = []
    for x_bi in vals:
        for x_in in vals:
            if x_bi + x_in <= 1.0 + 1e-9:
                x_sn = 1.0 - x_bi - x_in
                rows.append({"xBi": x_bi, "xIn": x_in, "xSn": x_sn})
    return pd.DataFrame(rows)


def add_predictions(df: pd.DataFrame, c: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    t = out["temperature_K"].to_numpy(dtype=float)
    x_bi = out["xBi"].to_numpy(dtype=float)
    x_in = out["xIn"].to_numpy(dtype=float)
    out["ml_prediction"] = predict_ml(x_bi, x_in, t, c)
    x_sn = out["xSn"].to_numpy(dtype=float)
    out["rkm_prediction"] = np.array(
        [rkm_delta_mix_h(b, i, s) for b, i, s in zip(x_bi, x_in, x_sn, strict=True)]
    )
    out["ml_minus_rkm"] = out["ml_prediction"] - out["rkm_prediction"]
    out["bi_in_interaction_contribution"] = bi_in_interaction_term(x_bi, x_in, c)
    out["temperature_term_contribution"] = temperature_terms(x_bi, x_in, t, c)
    return out


def mark_near_experimental(grid: pd.DataFrame, exp: pd.DataFrame) -> pd.DataFrame:
    """Flag grid points within NEAR_EXP_DIST of any experimental (xBi,xIn) at same T."""
    flags = []
    for _, row in grid.iterrows():
        same_t = exp[exp["temperature_K"] == row["temperature_K"]]
        dist = np.sqrt((same_t["xBi"] - row["xBi"]) ** 2 + (same_t["xIn"] - row["xIn"]) ** 2)
        flags.append(bool((dist <= NEAR_EXP_DIST).any()))
    out = grid.copy()
    out["near_experimental"] = flags
    out["region_label"] = np.where(out["near_experimental"], "near_measured", "unsampled_extrapolation")
    return out


def plot_scatter(df: pd.DataFrame, color_col: str, title: str, filename: str, cbar_label: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(df["xBi"], df["xIn"], c=df[color_col], s=10, cmap="viridis", alpha=0.9)
    exp_overlay = df[df["near_experimental"]]
    if len(exp_overlay):
        ax.scatter(
            exp_overlay["xBi"].unique()[:1],
            exp_overlay["xIn"].unique()[:1],
            s=0,
            label="_nolegend_",
        )
    ax.set_xlabel("xBi")
    ax.set_ylabel("xIn")
    ax.set_title(title)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150)
    plt.close(fig)


def plot_temperature_effect(exp: pd.DataFrame, c: dict[str, float]) -> None:
    """ML vs RKM vs T for one mid-series point per cross-section."""
    reps = []
    for section in sorted(exp["cross_section"].unique()):
        g = exp[exp["cross_section"] == section]
        reps.append(g.iloc[len(g) // 2])
    reps = pd.DataFrame(reps)

    t_line = np.linspace(767, 855, 50)
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in reps.iterrows():
        ml_line = predict_ml(
            np.full_like(t_line, row["xBi"]),
            np.full_like(t_line, row["xIn"]),
            t_line,
            c,
        )
        rkm_val = rkm_delta_mix_h(row["xBi"], row["xIn"], row["xSn"])
        section = str(row["cross_section"])[:18]
        ax.plot(t_line, ml_line, label=f"ML {section}…")
        ax.axhline(rkm_val, linestyle="--", alpha=0.5)

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Predicted ΔmixH (J/mol)")
    ax.set_title("Temperature effect: ML curves vs RKM (horizontal, T-independent)")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "final_surface_temperature_effect.png", dpi=150)
    plt.close(fig)


def summarize_range(series: pd.Series) -> str:
    return f"{series.min():.2f} to {series.max():.2f}"


def write_report(
    exp: pd.DataFrame,
    grids: dict[int, pd.DataFrame],
    c: dict[str, float],
) -> None:
    exp_ml = predict_ml(
        exp["xBi"].to_numpy(),
        exp["xIn"].to_numpy(),
        exp["temperature_K"].to_numpy(),
        c,
    )
    exp_stats = {
        "xBi": (exp["xBi"].min(), exp["xBi"].max()),
        "xIn": (exp["xIn"].min(), exp["xIn"].max()),
        "xSn": (exp["xSn"].min(), exp["xSn"].max()),
        "T": (exp["temperature_K"].min(), exp["temperature_K"].max()),
        "exp_dmixH": (exp[EXP_TARGET].min(), exp[EXP_TARGET].max()),
    }

    lines = [
        "# Final ML Surface Analysis (Step 4B)",
        "",
        "## Scope",
        "Response-surface analysis of the **frozen** Poly D2 model (Step 4A coefficients).",
        "No refitting. Coefficients are **empirical ML weights**, not RKM thermodynamic parameters.",
        "",
        "## Primary validated performance (unchanged)",
        f"- LOCSO OOF: MAE = {LOCSO_REF['MAE']:.2f}, RMSE = {LOCSO_REF['RMSE']:.2f}, R² = {LOCSO_REF['R2']:.4f}",
        f"- RKM benchmark: MAE = {RKM_REF['MAE']:.2f}, RMSE = {RKM_REF['RMSE']:.2f}, R² = {RKM_REF['R2']:.4f}",
        "",
        "## A. Composition response",
        "",
        f"Ternary grid: xBi ≥ 0, xIn ≥ 0, xBi+xIn ≤ 1, step = {GRID_STEP}.",
        "",
        "The **xBi·xIn** term (−9182.903 J/mol per unit product) modulates curvature between Bi and In.",
        "Its **contribution at a composition** is coefficient × xBi × xIn (not the raw coefficient alone).",
        "Raw coefficient magnitude must **not** be read as physical importance (different variable scales).",
        "",
        "| T (K) | ML min | ML max | xBi·xIn term min | xBi·xIn term max |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]

    for temp, g in grids.items():
        lines.append(
            f"| {temp} | {g['ml_prediction'].min():.2f} | {g['ml_prediction'].max():.2f} | "
            f"{g['bi_in_interaction_contribution'].min():.2f} | {g['bi_in_interaction_contribution'].max():.2f} |"
        )

    lines.extend(["", "## B. Temperature response", ""])
    for temp in TEMPERATURES:
        g = grids[temp]
        lines.append(
            f"- **{temp} K:** ML range {summarize_range(g['ml_prediction'])} J/mol; "
            f"mean temperature-term contribution {g['temperature_term_contribution'].mean():.2f} J/mol"
        )
    t767 = grids[767]["ml_prediction"].to_numpy()
    t855 = grids[855]["ml_prediction"].to_numpy()
    delta_t = t855 - t767
    lines.append(
        f"- Mean ML change 767→855 K (same compositions): {delta_t.mean():.2f} J/mol; "
        f"max |change| = {np.max(np.abs(delta_t)):.2f} J/mol"
    )
    lines.append("")
    lines.append(
        "The model captures **temperature-dependent variation present in the 104 calorimetry points**. "
        "This is **not** a claim of fundamental thermodynamic T-dependence discovered by ML."
    )
    lines.extend(
        [
            "",
            "## C. Experimental-region analysis",
            "",
            f"104 experiments span: xBi {exp_stats['xBi'][0]:.3f}–{exp_stats['xBi'][1]:.3f}, "
            f"xIn {exp_stats['xIn'][0]:.3f}–{exp_stats['xIn'][1]:.3f}, "
            f"T {int(exp_stats['T'][0])}–{int(exp_stats['T'][1])} K.",
            f"Experimental ΔmixH: {exp_stats['exp_dmixH'][0]:.1f} to {exp_stats['exp_dmixH'][1]:.1f} J/mol (all exothermic).",
            "",
            f"Near-experimental grid points: within {NEAR_EXP_DIST} in (xBi,xIn) of a measured point at the same T.",
            "",
        ]
    )

    for temp, g in grids.items():
        near = g[g["near_experimental"]]
        pct_pos = 100.0 * (near["ml_prediction"] > 0).mean() if len(near) else 0.0
        lines.append(f"### {temp} K — near-measured region (N = {len(near)})")
        lines.append(f"- ML range: {summarize_range(near['ml_prediction'])} J/mol")
        lines.append(f"- ML mean: {near['ml_prediction'].mean():.2f} J/mol")
        lines.append(f"- Fraction ML > 0: {pct_pos:.1f}%")
        lines.append("")

    lines.append(f"- ML on the **104 experimental rows** (in-sample check): range {exp_ml.min():.2f} to {exp_ml.max():.2f} J/mol")
    lines.append("")

    g813 = grids[813]
    lines.extend(
        [
            "## D. RKM comparison at 813 K",
            "",
            f"- ML range: {summarize_range(g813['ml_prediction'])} J/mol",
            f"- RKM range: {summarize_range(g813['rkm_prediction'])} J/mol",
            f"- ML−RKM range: {summarize_range(g813['ml_minus_rkm'])} J/mol",
            f"- Mean |ML−RKM|: {g813['ml_minus_rkm'].abs().mean():.2f} J/mol",
            f"- Max |ML−RKM|: {g813['ml_minus_rkm'].abs().max():.2f} J/mol",
            "",
            "### Largest |ML−RKM| at 813 K",
            "",
            "| xBi | xIn | xSn | region | ML | RKM | ML−RKM |",
            "| ---: | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    top = g813.assign(abs_diff=g813["ml_minus_rkm"].abs()).nlargest(8, "abs_diff")
    for _, row in top.iterrows():
        lines.append(
            f"| {row['xBi']:.3f} | {row['xIn']:.3f} | {row['xSn']:.3f} | {row['region_label']} | "
            f"{row['ml_prediction']:.1f} | {row['rkm_prediction']:.1f} | {row['ml_minus_rkm']:.1f} |"
        )

    full_pos_813 = g813[g813["ml_prediction"] > 0]
    near_pos = full_pos_813[full_pos_813["near_experimental"]]
    lines.extend(
        [
            "",
            "## E. Physical sanity checks",
            "",
            f"- At 813 K, **{len(full_pos_813)}** simplex grid points ({100*len(full_pos_813)/len(g813):.1f}%) have ML ΔmixH > 0.",
            f"- Of those, **{len(near_pos)}** overlap the near-measured region — "
            f"{'some positive predictions occur near experiments' if len(near_pos) else 'positive regions are outside measured compositions'}.",
            "- Extreme ML values occur primarily at **unsampled** compositions (especially Bi–In rich, low xSn).",
            "- **Extrapolation warning:** predictions far from the three measured cross-sections are **not experimentally validated**.",
            "- ML is **not** claimed physically superior to RKM; RKM remains the physics-based benchmark.",
            "",
            "## Figures",
            "",
            "- `figures/final_surface_767K.png`",
            "- `figures/final_surface_813K.png`",
            "- `figures/final_surface_855K.png`",
            "- `figures/final_surface_temperature_effect.png`",
            "- `figures/final_surface_vs_rkm_813K.png`",
            "",
        ]
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


def main() -> int:
    c = load_coefficients()
    exp = pd.read_csv(EXP_PATH)
    base_grid = build_ternary_grid()

    grids: dict[int, pd.DataFrame] = {}
    for temp in TEMPERATURES:
        g = base_grid.copy()
        g["temperature_K"] = temp
        g = add_predictions(g, c)
        g = mark_near_experimental(g, exp)
        grids[temp] = g

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for temp in TEMPERATURES:
        g = grids[temp]
        plot_scatter(
            g,
            "ml_prediction",
            f"Final ML ΔmixH at {temp} K (simplex grid)",
            f"final_surface_{temp}K.png",
            "ML ΔmixH (J/mol)",
        )
    plot_scatter(
        grids[813],
        "ml_minus_rkm",
        "ML − RKM at 813 K (unsampled regions not experimentally validated)",
        "final_surface_vs_rkm_813K.png",
        "ML − RKM (J/mol)",
    )
    plot_temperature_effect(exp, c)
    print(f"Wrote figures to {FIG_DIR}")

    write_report(exp, grids, c)

    g813 = grids[813]
    print(f"813 K ML range: [{g813['ml_prediction'].min():.2f}, {g813['ml_prediction'].max():.2f}]")
    print(f"813 K mean |ML-RKM|: {g813['ml_minus_rkm'].abs().mean():.2f}")
    print(f"813 K positive ML fraction: {(g813['ml_prediction']>0).mean()*100:.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
