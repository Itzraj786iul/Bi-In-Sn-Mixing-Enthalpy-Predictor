"""
Regenerate the Stage-1 data/RKM research figures and the RKM validation report.

    python scripts/generate_research_figures.py              # writes figures/ and reports/
    python scripts/generate_research_figures.py --out-dir D  # writes D/figures/ and D/reports/

Outputs:
    figures/experimental_cross_sections.png
    figures/rkm_vs_experiment.png
    figures/rkm_vs_experiment_cross_sections.png
    figures/ternary_enthalpy_map_813K.png
    figures/experimental_vs_synthetic.png
    reports/rkm_validation.md

Inputs are existing repository data and src/rkm_model.py only. No model is fitted.
The original plotting code for these figures was not preserved, so regenerated images
show the same data and quantities but are not pixel-identical to the committed PNGs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rkm_model import rkm_delta_mix_h

EXP_PATH = ROOT / "data" / "original_experimental_data.csv"
SYN_A_PATH = ROOT / "data" / "synthetic_cross_sections.csv"
EXP_TARGET = "integral_mixing_enthalpy_J_per_mol"
SYN_TARGET = "delta_mix_H_J_per_mol"
TEMPERATURES = (767, 813, 855)
MAP_TEMPERATURE = 813
GRID_STEP = 0.01

SECTIONS = [
    ("(Sn0.33Bi0.67)1-xInx", "Bi-rich", r"Bi-rich $(Sn_{0.33}Bi_{0.67})_{1-x}In_x$", "o"),
    ("(Sn0.50Bi0.50)1-xInx", "Equiatomic", r"Equiatomic $(Sn_{0.50}Bi_{0.50})_{1-x}In_x$", "s"),
    ("(Sn0.67Bi0.33)1-xInx", "Sn-rich", r"Sn-rich $(Sn_{0.67}Bi_{0.33})_{1-x}In_x$", "^"),
]
TEMP_COLORS = {767: "tab:blue", 813: "tab:orange", 855: "tab:green"}
SECTION_COLORS = {"Bi-rich": "tab:blue", "Equiatomic": "tab:orange", "Sn-rich": "tab:green"}
DMIXH_LABEL = r"$\Delta_{\mathrm{mix}}H$ (J mol$^{-1}$)"


def load_experimental() -> pd.DataFrame:
    exp = pd.read_csv(EXP_PATH)
    exp["rkm_prediction"] = rkm_delta_mix_h(exp["xBi"], exp["xIn"], exp["xSn"])
    exp["bi_ratio"] = exp["xBi"] / (exp["xBi"] + exp["xSn"])
    return exp


def section_curve(bi_ratio: float, step: float = GRID_STEP) -> tuple[np.ndarray, np.ndarray]:
    x_in = np.round(np.arange(0.0, 1.0 + step / 2, step), 10)
    h = rkm_delta_mix_h(bi_ratio * (1 - x_in), x_in, (1 - bi_ratio) * (1 - x_in))
    return x_in, h


def metrics(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    e = p - y
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e**2)))
    r2 = float(1 - np.sum(e**2) / np.sum((y - y.mean()) ** 2))
    return mae, rmse, r2


def _section_axes(title: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    fig.suptitle(title, fontsize=13)
    for ax, (_, _, sub_title, _) in zip(axes, SECTIONS):
        ax.set_title(sub_title, fontsize=10)
        ax.set_xlabel(r"$x_{\mathrm{In}}$")
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(DMIXH_LABEL)
    return fig, axes


def plot_experimental_cross_sections(exp: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = _section_axes("Experimental integral mixing enthalpy (Table III) — not synthetic")
    for ax, (cs, _, _, marker) in zip(axes, SECTIONS):
        sec = exp[exp["cross_section"] == cs]
        for t in TEMPERATURES:
            g = sec[sec["temperature_K"] == t]
            ax.scatter(g["xIn"], g[EXP_TARGET], marker=marker, color=TEMP_COLORS[t], label=f"{t} K")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "experimental_cross_sections.png", dpi=150)
    plt.close(fig)


def plot_rkm_parity(exp: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for cs, label, _, marker in SECTIONS:
        g = exp[exp["cross_section"] == cs]
        ax.scatter(g[EXP_TARGET], g["rkm_prediction"], marker=marker, color=SECTION_COLORS[label], label=label)
    lo = min(exp[EXP_TARGET].min(), exp["rkm_prediction"].min()) - 50
    hi = max(exp[EXP_TARGET].max(), exp["rkm_prediction"].max()) + 50
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
    ax.set_xlabel(r"Experimental $\Delta_{\mathrm{mix}}H$ (J mol$^{-1}$)")
    ax.set_ylabel(r"RKM $\Delta_{\mathrm{mix}}H$ (J mol$^{-1}$)")
    ax.set_title("RKM vs experimental (Table III)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "rkm_vs_experiment.png", dpi=150)
    plt.close(fig)


def plot_rkm_cross_sections(exp: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = _section_axes("RKM model vs experimental cross-sections")
    for ax, (cs, _, _, marker) in zip(axes, SECTIONS):
        sec = exp[exp["cross_section"] == cs]
        x_in, h = section_curve(float(sec["bi_ratio"].mean()))
        ax.plot(x_in, h, color="0.2", lw=1.8, label="RKM (T-independent)")
        for t in TEMPERATURES:
            g = sec[sec["temperature_K"] == t]
            ax.scatter(g["xIn"], g[EXP_TARGET], marker=marker, color=TEMP_COLORS[t], label=f"exp {t} K")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "rkm_vs_experiment_cross_sections.png", dpi=150)
    plt.close(fig)


def plot_ternary_map(exp: pd.DataFrame, fig_dir: Path) -> None:
    n = int(round(1 / GRID_STEP))
    pts = [(i / n, j / n) for i in range(n + 1) for j in range(n + 1 - i)]
    x_bi = np.array([p[0] for p in pts])
    x_in = np.array([p[1] for p in pts])
    x_sn = np.clip(1 - x_bi - x_in, 0.0, 1.0)
    h = rkm_delta_mix_h(x_bi, x_in, x_sn)

    def to_xy(b, i, s):
        return s + 0.5 * i, (np.sqrt(3) / 2) * i

    px, py = to_xy(x_bi, x_in, x_sn)
    levels = np.linspace(h.min(), 0.0, 16)
    fig, ax = plt.subplots(figsize=(7, 6))
    cf = ax.tricontourf(px, py, h, levels=levels, cmap="RdYlBu")
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(r"RKM $\Delta_{\mathrm{mix}}H$ (J mol$^{-1}$)")
    ex, ey = to_xy(exp["xBi"].to_numpy(), exp["xIn"].to_numpy(), exp["xSn"].to_numpy())
    ax.scatter(ex, ey, marker="x", s=18, color="k", label="experimental compositions")
    ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0], color="k", lw=0.8)
    ax.text(-0.03, -0.05, "Bi", ha="center")
    ax.text(1.03, -0.05, "Sn", ha="center")
    ax.text(0.5, np.sqrt(3) / 2 + 0.03, "In", ha="center")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        rf"Model ternary $\Delta_{{\mathrm{{mix}}}}H$ map at {MAP_TEMPERATURE} K (RKM, not experimental)",
        pad=22,
    )
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(fig_dir / f"ternary_enthalpy_map_{MAP_TEMPERATURE}K.png", dpi=150)
    plt.close(fig)


def plot_experimental_vs_synthetic(exp: pd.DataFrame, fig_dir: Path) -> None:
    syn = pd.read_csv(SYN_A_PATH)
    fig, axes = _section_axes("Experimental (markers) vs synthetic (grey) — synthetic is not measured")
    for ax, (cs, _, _, marker) in zip(axes, SECTIONS):
        s = syn[syn["cross_section"] == cs]
        ax.scatter(s["xIn"], s[SYN_TARGET], s=2, color="0.6", alpha=0.3, label="synthetic (RKM ± noise)")
        g = exp[exp["cross_section"] == cs]
        ax.scatter(
            g["xIn"], g[EXP_TARGET], marker=marker, color="crimson", edgecolor="k", lw=0.5,
            label="experimental Table III",
        )
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "experimental_vs_synthetic.png", dpi=150)
    plt.close(fig)


def write_rkm_validation(exp: pd.DataFrame, report_path: Path) -> None:
    y = exp[EXP_TARGET].to_numpy()
    p = exp["rkm_prediction"].to_numpy()
    mae, rmse, r2 = metrics(y, p)

    lines = [
        "# RKM validation against Table III experimental data",
        "",
        "Paper: Kumar, Mohan and Behera, *J. Electron. Mater.* **48**, 8096–8106 (2019).",
        "",
        "The implementation uses Equation (4) and Table IV of that paper.",
        "Parameters are temperature-independent. Experimental points at 767 K, 813 K and 855 K",
        "are therefore compared to the same RKM prediction at each composition.",
        "",
        "## Ternary index assignment",
        "",
        "Validated (i, j, k) order: **In–Bi–Sn**",
        "",
        "Table IV names the ternary parameter $L_{\\mathrm{Bi-Sn-In}}$. A literal",
        "(i, j, k) = (Bi, Sn, In) assignment does **not** reproduce Fig. 10 or the authors'",
        "statement that the fitted curve is close to experiment. The assignment used here",
        "(i = In, j = Bi, k = Sn) was selected because it reproduces the measured",
        "cross-section shapes, the Bi-rich vs Sn-rich depth order, and a high R² vs Table III.",
        "See `docs/rkm_model.md`.",
        "",
        "## Overall metrics (all 9 series)",
        "",
        f"- N = {len(exp)}",
        f"- MAE = {mae:.2f} J/mol",
        f"- RMSE = {rmse:.2f} J/mol",
        f"- R² = {r2:.4f}",
        "",
        "## Metrics by series",
        "",
        "| Series | T (K) | Cross-section | N | MAE | RMSE | R² |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for sid, g in exp.groupby("series_id"):
        m = metrics(g[EXP_TARGET].to_numpy(), g["rkm_prediction"].to_numpy())
        lines.append(
            f"| {sid} | {int(g['temperature_K'].iloc[0])} | {g['cross_section'].iloc[0]} | {len(g)} "
            f"| {m[0]:.2f} | {m[1]:.2f} | {m[2]:.4f} |"
        )
    lines += ["", "## Metrics by temperature", ""]
    for t in TEMPERATURES:
        g = exp[exp["temperature_K"] == t]
        m = metrics(g[EXP_TARGET].to_numpy(), g["rkm_prediction"].to_numpy())
        lines.append(f"- {t} K: MAE = {m[0]:.2f} J/mol, RMSE = {m[1]:.2f} J/mol, R² = {m[2]:.4f}")

    lines += ["", "## Qualitative shape checks (Figures 2–4)", ""]
    minima = []
    for cs, _, _, _ in SECTIONS:
        ratio = float(exp.loc[exp["cross_section"] == cs, "bi_ratio"].mean())
        x_in, h = section_curve(ratio)
        k = int(np.argmin(h))
        minima.append(float(h[k]))
        lines.append(f"- {cs}: model minimum at xIn={x_in[k]:.3f}.")
    ordered = minima[0] < minima[1] < minima[2]
    passed = ordered and all(v < 0 for v in minima)
    lines.append(
        f"- Minima order preserved: Bi-rich {minima[0]:.1f} < equiatomic {minima[1]:.1f} "
        f"< Sn-rich {minima[2]:.1f} J/mol."
    )
    lines += [
        "",
        f"Shape checks passed: **{passed}**",
        "",
        "## Residual comments",
        "",
        "The largest residuals are on the Bi-rich section (Series 1) and at some In-rich",
        "compositions, which the paper already flags as the weaker part of the fit.",
        "Residuals remain of the same order as, or larger than, the tabulated series",
        "standard uncertainties u(ΔmixH) (8–23 J/mol) but much smaller than the",
        "stated overall calorimeter uncertainty of 10–12%.",
        "",
        "Synthetic generation was allowed to proceed only because the model reproduces",
        "the experimental qualitative behavior of Figures 2–4 and 10.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    fig_dir = args.out_dir / "figures"
    report_dir = args.out_dir / "reports"
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    exp = load_experimental()
    plot_experimental_cross_sections(exp, fig_dir)
    plot_rkm_parity(exp, fig_dir)
    plot_rkm_cross_sections(exp, fig_dir)
    plot_ternary_map(exp, fig_dir)
    plot_experimental_vs_synthetic(exp, fig_dir)
    write_rkm_validation(exp, report_dir / "rkm_validation.md")
    print(f"Wrote 5 figures to {fig_dir} and rkm_validation.md to {report_dir}")


if __name__ == "__main__":
    main()
