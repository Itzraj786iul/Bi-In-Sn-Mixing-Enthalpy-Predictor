"""
Path-aware residual-learning experiment (controlled; experimental-only LOCSO).

    python scripts/step_6_path_aware_residual.py

Step 2 (reports/residual_diagnostic_analysis.md) diagnosed the Sn-rich LOCSO
failure of RKM + residual Poly D2 as composition-dependent extrapolation across
composition paths of different Bi/(Bi+Sn) ratio. This script tests one change:
adding the Bi/Sn composition coordinate

    bi_sn_fraction = xBi / (xBi + xSn)

to the residual model's inputs.

    Model A (control): residual ~ PolyD2(xBi, xIn, T)                  (Step 1)
    Model B:           residual ~ PolyD2(xBi, xIn, T, bi_sn_fraction)

Everything else is identical: 104 experimental observations, RKM baseline,
target, LOCSO folds, degree, minimum-norm least-squares fit and metrics.
No synthetic data are loaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_composition_representation import (  # noqa: E402
    EXP_TARGET,
    calculate_metrics,
    get_locsos_folds,
    predict_polynomial,
    train_polynomial,
)
from step_4_residual_learning import DIRECT_POLY_D2_REF, SECTION_LABELS, load_experimental  # noqa: E402
from validate_dataset import COMPOSITION_TOL, CROSS_SECTION_RATIO, RATIO_TOL  # noqa: E402

STEP1_PATH = ROOT / "reports" / "residual_learning_results.csv"
REPORT_PATH = ROOT / "reports" / "path_aware_residual_analysis.md"
PRED_PATH = ROOT / "reports" / "path_aware_residual_results.csv"
FIG_PATH = ROOT / "figures" / "path_aware_residual_by_section.png"

MODELS = {
    "A": ["xBi", "xIn", "temperature_K"],
    "B": ["xBi", "xIn", "temperature_K", "bi_sn_fraction"],
}
MODEL_NAMES = {"RKM": "RKM", "A": "Current residual Poly D2", "B": "Path-aware residual Poly D2"}
STEP1_REF = {"MAE": 44.66, "RMSE": 55.75, "R2": 0.9718}
RKM_REF = {"MAE": 57.13, "RMSE": 73.49, "R2": 0.9510}
SECTION_ORDER = ["Bi-rich", "Equiatomic", "Sn-rich"]
SECTION_COLORS = {"Bi-rich": "tab:blue", "Equiatomic": "tab:green", "Sn-rich": "tab:red"}


def add_path_coordinate(exp: pd.DataFrame) -> pd.DataFrame:
    total = exp["xBi"] + exp["xIn"] + exp["xSn"]
    if not np.allclose(total, 1.0, atol=COMPOSITION_TOL):
        raise RuntimeError("xBi + xIn + xSn != 1 within COMPOSITION_TOL")
    denom = exp["xBi"] + exp["xSn"]
    if not (denom > 0).all():
        raise RuntimeError("xBi + xSn must be > 0 for every observation")
    exp = exp.copy()
    exp["bi_sn_fraction"] = exp["xBi"] / denom
    for section, labelled in CROSS_SECTION_RATIO.items():
        vals = exp.loc[exp["cross_section"] == section, "bi_sn_fraction"]
        if not (np.abs(vals - labelled) < RATIO_TOL).all():
            raise RuntimeError(f"{section}: bi_sn_fraction differs from {labelled} by more than {RATIO_TOL}")
    return exp


def run_locso(exp: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    folds = get_locsos_folds(exp)
    if len(folds) != 3:
        raise RuntimeError(f"Expected 3 LOCSO folds, got {len(folds)}")

    parts, fold_info = [], []
    for fold_idx, (held_out, train, test) in enumerate(folds, start=1):
        if set(train["id"]) & set(test["id"]) or held_out in set(train["cross_section"]):
            raise RuntimeError(f"Fold {fold_idx}: held-out data present in training set")

        out = test[["id", "cross_section", "temperature_K", "xBi", "xIn", "xSn", "bi_sn_fraction"]].copy()
        out["experimental_delta_mixH"] = test[EXP_TARGET].to_numpy(float)
        out["rkm_prediction"] = test["rkm_prediction"].to_numpy(float)
        out["residual"] = test["residual"].to_numpy(float)
        info = {
            "fold": fold_idx,
            "held_out": SECTION_LABELS[held_out],
            "n_train": len(train),
            "n_test": len(test),
            "train_fractions": sorted({round(float(v), 4) for v in train["bi_sn_fraction"]}),
        }
        for key, cols in MODELS.items():
            poly, model = train_polynomial(train[cols].to_numpy(float), train["residual"].to_numpy(float))
            X_poly = poly.transform(train[cols].to_numpy(float))
            sv = np.linalg.svd(X_poly, compute_uv=False)
            info[key] = {
                "n_terms": int(poly.n_output_features_),
                "rank": int(np.linalg.matrix_rank(X_poly)),
                "cond": float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
                "train_mae": calculate_metrics(train["residual"], predict_polynomial(poly, model, train[cols].to_numpy(float)))["MAE"],
            }
            pred = predict_polynomial(poly, model, test[cols].to_numpy(float))
            out[f"predicted_residual_{key}"] = pred
            out[f"hybrid_prediction_{key}"] = out["rkm_prediction"] + pred
        out["fold"] = fold_idx
        out["held_out_section"] = held_out
        parts.append(out)
        fold_info.append(info)

    pred = pd.concat(parts, ignore_index=True).rename(columns={"id": "experiment_id"})
    if len(pred) != 104 or pred["experiment_id"].duplicated().any():
        raise RuntimeError("Each observation must be predicted exactly once")
    pred.insert(2, "section", pred["cross_section"].map(SECTION_LABELS))
    pred["rkm_error"] = pred["experimental_delta_mixH"] - pred["rkm_prediction"]
    for key in MODELS:
        pred[f"hybrid_error_{key}"] = pred["experimental_delta_mixH"] - pred[f"hybrid_prediction_{key}"]
    return pred, fold_info


def verify_against_step1(pred: pd.DataFrame) -> float:
    step1 = pd.read_csv(STEP1_PATH).set_index("experiment_id")
    a = pred.set_index("experiment_id")["predicted_residual_A"]
    diff = float(np.max(np.abs(a - step1.loc[a.index, "predicted_residual"])))
    if diff > 1e-6:
        raise RuntimeError(f"Model A does not reproduce Step 1 predictions (max diff {diff:.3e})")
    return diff


def verify_direct_poly(exp: pd.DataFrame) -> dict:
    preds = []
    for _, train, test in get_locsos_folds(exp):
        poly, model = train_polynomial(train[MODELS["A"]].to_numpy(float), train[EXP_TARGET].to_numpy(float))
        preds.append(pd.Series(predict_polynomial(poly, model, test[MODELS["A"]].to_numpy(float)), index=test.index))
    p = pd.concat(preds).loc[exp.index]
    return calculate_metrics(exp[EXP_TARGET], p)


def metrics_for(df: pd.DataFrame) -> dict[str, dict]:
    y = df["experimental_delta_mixH"]
    out = {"RKM": calculate_metrics(y, df["rkm_prediction"])}
    for key in MODELS:
        out[key] = calculate_metrics(y, df[f"hybrid_prediction_{key}"])
    return out


def residual_diag(df: pd.DataFrame, key: str) -> dict:
    a, p = df["residual"].to_numpy(), df[f"predicted_residual_{key}"].to_numpy()
    e = a - p
    return {
        "corr": float(np.corrcoef(a, p)[0, 1]),
        "MAE": float(np.mean(np.abs(e))),
        "RMSE": float(np.sqrt(np.mean(e**2))),
        "mean_err": float(np.mean(e)),
        "max_abs": float(np.max(np.abs(e))),
    }


def make_figure(pred: pd.DataFrame) -> None:
    temps = sorted(pred["temperature_K"].unique())
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True)
    for i, sec in enumerate(SECTION_ORDER):
        for j, t in enumerate(temps):
            ax = axes[i, j]
            g = pred[(pred["section"] == sec) & (pred["temperature_K"] == t)].sort_values("xIn")
            ax.plot(g["xIn"], g["residual"], "o-", c=SECTION_COLORS[sec], label="actual residual")
            ax.plot(g["xIn"], g["predicted_residual_A"], "s--", c="tab:gray", label="Model A (xBi, xIn, T)")
            ax.plot(g["xIn"], g["predicted_residual_B"], "^:", c="black", label="Model B (+ bi_sn_fraction)")
            ax.axhline(0, color="black", lw=0.6)
            ax.set_title(f"{sec} held out, {int(t)} K", fontsize=10)
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel("Residual (J/mol)")
            if i == 2:
                ax.set_xlabel("xIn")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Actual vs held-out predicted residual (LOCSO), Model A vs Model B")
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)


def interpretation(pooled, by_sec, by_t, diag, hi, pred) -> list[str]:
    sB, sA = diag["B"]["Sn-rich"], diag["A"]["Sn-rich"]
    sn = pred[pred["section"] == "Sn-rich"]
    n_pos = int((sn["hybrid_error_B"] > 0).sum())
    bias_share_B = sB["mean_err"] ** 2 / sB["RMSE"] ** 2
    bias_share_A = sA["mean_err"] ** 2 / sA["RMSE"] ** 2
    lo = sn[sn["xIn"] < 0.25]
    hi_sn = sn[sn["xIn"] > 0.80]
    sec = by_sec
    return [
        "**Observed results**",
        "",
        f"1. Pooled LOCSO error falls from MAE {pooled['A']['MAE']:.2f} / RMSE {pooled['A']['RMSE']:.2f} (Model A) to "
        f"{pooled['B']['MAE']:.2f} / {pooled['B']['RMSE']:.2f} J/mol (Model B); R² {pooled['A']['R2']:.4f} → {pooled['B']['R2']:.4f}.",
        f"2. Almost all of the pooled gain comes from the Bi-rich fold (MAE {sec['Bi-rich']['A']['MAE']:.2f} → "
        f"{sec['Bi-rich']['B']['MAE']:.2f}). Equiatomic gets slightly worse ({sec['Equiatomic']['A']['MAE']:.2f} → "
        f"{sec['Equiatomic']['B']['MAE']:.2f}) and Sn-rich is essentially unchanged ({sec['Sn-rich']['A']['MAE']:.2f} → "
        f"{sec['Sn-rich']['B']['MAE']:.2f}), still above RKM ({sec['Sn-rich']['RKM']['MAE']:.2f}).",
        f"3. In the Sn-rich fold the *type* of error changes. Model A had little bias but the wrong shape "
        f"(corr {sA['corr']:.2f}, bias² = {100*bias_share_A:.0f}% of MSE). Model B follows the shape much better "
        f"(corr {sB['corr']:.2f}) but is offset: all {n_pos} of {len(sn)} Sn-rich predicted residuals are below the actual "
        f"residual (mean error {sB['mean_err']:+.2f} J/mol, bias² = {100*bias_share_B:.0f}% of MSE), i.e. the Model B "
        "hybrid is more exothermic than measured at every Sn-rich point.",
        f"4. The Sn-rich offset grows toward the In corner: mean Model B error {lo['hybrid_error_B'].mean():+.1f} J/mol for "
        f"xIn < 0.25 versus {hi_sn['hybrid_error_B'].mean():+.1f} J/mol for xIn > 0.80. At xIn > 0.80 Model B still "
        "predicts negative Sn-rich residuals at all three temperatures while the measured ones are positive (table above).",
        f"5. By temperature, Model B is better than Model A at 767 K ({by_t[767]['A']['MAE']:.2f} → {by_t[767]['B']['MAE']:.2f}) "
        f"and 813 K ({by_t[813]['A']['MAE']:.2f} → {by_t[813]['B']['MAE']:.2f}) and worse at 855 K "
        f"({by_t[855]['A']['MAE']:.2f} → {by_t[855]['B']['MAE']:.2f}).",
        f"6. For the Bi-rich fold Model B tracks the measured residual closely up to xIn ≈ 0.8 (corr "
        f"{diag['B']['Bi-rich']['corr']:.2f}) but overshoots at the In-rich end (xIn > 0.80 mean predicted "
        f"{hi.loc[('Bi-rich', 767), 'predicted_residual_B']:.0f} vs actual {hi.loc[('Bi-rich', 767), 'residual']:.0f} J/mol at 767 K).",
        "",
        "**Interpretation**",
        "",
        "- The Bi/Sn composition coordinate lets the residual model represent a residual that changes from path to "
        "path, and this clearly helps the Bi-rich extrapolation. It does **not** solve the Sn-rich problem: it "
        "replaces a shape error with a level (bias) error of about the same size.",
        "- The two outer folds respond very differently to the same change, so the pooled improvement should not be "
        "read as a general improvement in cross-path extrapolation. With only three paths, each conclusion about "
        "an outer path rests on a single fold.",
        "- Temperature is not the target of this change and the per-temperature differences are mixed; they are "
        "reported for completeness only.",
        "",
        "**Hypotheses (not tested here)**",
        "",
        "- Because each training fold contains only two Bi/Sn values and both design matrices are rank-deficient, "
        "the across-path behaviour used for the held-out path is partly set by the minimum-norm rule. The Sn-rich "
        "offset and the Bi-rich overshoot near the In corner may reflect this rather than a property of the data.",
        "- The measured In-rich residual changes roughly linearly with Bi/Sn ratio across the three paths (Step 2). "
        "Model B does not reproduce that trend when extrapolating to the Sn-rich path, possibly because its "
        "bi_sn_fraction terms interact with xBi, xIn and T in ways that two paths cannot constrain.",
    ]


def conclusion(pooled, by_sec, diag) -> list[str]:
    snA, snB, snR = (by_sec["Sn-rich"][k]["MAE"] for k in ("A", "B", "RKM"))
    return [
        f"The experiment's primary question — whether an explicit Bi/Sn composition coordinate reduces the Sn-rich "
        f"LOCSO error toward or below RKM — is answered **no**: Sn-rich MAE {snA:.2f} (Model A) → {snB:.2f} J/mol "
        f"(Model B), versus RKM {snR:.2f}. The path-aware model improves pooled LOCSO metrics "
        f"(MAE {pooled['A']['MAE']:.2f} → {pooled['B']['MAE']:.2f}, RMSE {pooled['A']['RMSE']:.2f} → "
        f"{pooled['B']['RMSE']:.2f}), but this gain comes from the Bi-rich fold, with a small loss on the "
        "Equiatomic fold.",
        "",
        "These results do not support adopting the path-aware residual model as a replacement for the current "
        "model, and they do not show that the path-aware feature is generally better. They show that the "
        "representation changes how the model extrapolates across paths, with a large benefit for one outer path, "
        "no benefit for the other, and a structural identifiability limit (two Bi/Sn values per training fold) that "
        "any continuation would need to address. The current final model, the existing direct Poly D2 model, RKM "
        "and the Step 1 residual results are unchanged.",
    ]


def mt(m: dict) -> str:
    return f"{m['MAE']:.2f} | {m['RMSE']:.2f} | {m['R2']:.4f}"


def write_report(exp, pred, fold_info, step1_diff, direct_m) -> None:
    pooled = metrics_for(pred)
    by_sec = {s: metrics_for(g) for s, g in pred.groupby("section")}
    by_t = {int(t): metrics_for(g) for t, g in pred.groupby("temperature_K")}
    diag = {k: {"Overall": residual_diag(pred, k), **{s: residual_diag(g, k) for s, g in pred.groupby("section")}} for k in MODELS}
    frac = exp.groupby("cross_section")["bi_sn_fraction"].agg(["mean", "min", "max", "count"])
    hi = pred[pred["xIn"] > 0.80].groupby(["section", "temperature_K"])[["residual", "predicted_residual_A", "predicted_residual_B"]].mean()

    snA, snB, snR = by_sec["Sn-rich"]["A"]["MAE"], by_sec["Sn-rich"]["B"]["MAE"], by_sec["Sn-rich"]["RKM"]["MAE"]
    sn_improves = snB < snA
    sn_beats_rkm = snB < snR
    pooled_better = pooled["B"]["MAE"] < pooled["A"]["MAE"]
    worse_secs = [s for s in SECTION_ORDER if s != "Sn-rich" and by_sec[s]["B"]["MAE"] > by_sec[s]["A"]["MAE"]]

    L: list[str] = []
    L += [
        "# Path-Aware Residual Learning",
        "",
        "Script: `scripts/step_6_path_aware_residual.py`. Per-observation output: `reports/path_aware_residual_results.csv`. "
        "Figure: `figures/path_aware_residual_by_section.png`.",
        "",
        "## Objective",
        "",
        "Step 2 (`reports/residual_diagnostic_analysis.md`) found that RKM + residual Poly D2 is worse than RKM on "
        "the Sn-rich held-out section (MAE 42.88 → 54.70 J/mol) and attributed this to composition-dependent "
        "extrapolation: the three experimental paths differ in Bi/(Bi+Sn) ratio, the RKM residual changes with that "
        "ratio (most strongly at In-rich compositions), and the residual model has no explicit input for it. This "
        "experiment tests whether adding a Bi/Sn composition coordinate to the residual model reduces the Sn-rich "
        "LOCSO error. It is a single controlled change; no other model, dataset or report was modified, and no "
        "synthetic data were used.",
        "",
        "## Feature definition",
        "",
        "```text",
        "bi_sn_fraction = xBi / (xBi + xSn)",
        "```",
        "",
        "Each experimental cross-section is prepared by adding In to a fixed Bi–Sn starting alloy, so along a path "
        "xBi and xSn shrink together and their ratio stays constant. `bi_sn_fraction` is therefore constant along "
        "each path and identifies which path a composition lies on (a composition-path representation). It is a "
        "coordinate, not a thermodynamic law. Note that it is a deterministic function of the existing inputs "
        "(bi_sn_fraction = xBi / (1 − xIn)), so it adds no new information about the alloy; it changes only which "
        "functions a degree-2 polynomial can express.",
        "",
        f"Checks: xBi + xIn + xSn = 1 within the project tolerance ({COMPOSITION_TOL:g}) for all 104 rows; "
        f"xBi + xSn > 0 for all rows (minimum {float((exp['xBi'] + exp['xSn']).min()):.4f}).",
        "",
        "| Cross-section | Label | n | Mean bi_sn_fraction | Min | Max |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for cs, r in frac.iterrows():
        L.append(f"| `{cs}` | {SECTION_LABELS[cs]} | {int(r['count'])} | {r['mean']:.4f} | {r['min']:.4f} | {r['max']:.4f} |")

    L += [
        "",
        "## Validation protocol",
        "",
        "Leave-one-cross-section-out with the same three folds as Step 1. In each fold the residual model "
        "(residual = ΔmixH_exp − ΔmixH_RKM) is fitted only on the two training sections and predicts the residual "
        "of the held-out section; hybrid = RKM + predicted residual. Both models use `PolynomialFeatures(degree=2)` "
        "and the same minimum-norm least-squares fit (`np.linalg.lstsq`), no regularisation and no tuning. The only "
        "difference is the extra input `bi_sn_fraction` in Model B.",
        "",
        "| Fold | Held out | Train N | Test N | Train bi_sn_fraction values | Model A terms / rank | Model B terms / rank | Model B condition no. |",
        "| ---: | --- | ---: | ---: | --- | --- | --- | ---: |",
    ]
    for f in fold_info:
        L.append(
            f"| {f['fold']} | {f['held_out']} | {f['n_train']} | {f['n_test']} | {', '.join(f'{v:.4f}' for v in f['train_fractions'])} | "
            f"{f['A']['n_terms']} / {f['A']['rank']} | {f['B']['n_terms']} / {f['B']['rank']} | {f['B']['cond']:.2e} |"
        )
    L += [
        "",
        "**Important structural limitation.** Neither design matrix has full rank in any fold, and `lstsq` returns "
        "the minimum-norm solution in both cases (the extremely large condition numbers reflect exact rank "
        "deficiency, not noise).",
        "",
        "- Model A (rank 9 of 10): every training point lies on one of two straight lines in the (xBi, xIn) plane "
        "(xBi = r·(1 − xIn) for the two training ratios r). The product of the two line equations is a degree-2 "
        "polynomial that is zero on every training point, so one quadratic direction is undetermined. This was "
        "already true in Step 1.",
        "- Model B (rank 11 of 15): in addition, each training fold contains only **two** values of `bi_sn_fraction`, "
        "so bi_sn_fraction² is an exact linear combination of the constant and bi_sn_fraction, and further "
        "products with bi_sn_fraction are tied to existing terms on the two lines.",
        "",
        "Consequently, how either model varies *across* paths beyond what two paths can determine is fixed by the "
        "minimum-norm rule, not by the data. For the outer held-out paths (Bi-rich, Sn-rich) this matters because "
        "their predictions are extrapolations in the Bi/Sn coordinate.",
        "",
        "Integrity checks (all passed): 104 observations; 3 folds; held-out section absent from every training set; "
        f"Model A reproduces the Step 1 predicted residuals exactly (max |difference| = {step1_diff:.1e} J/mol); "
        f"RKM metrics reproduce the frozen benchmark; direct Poly D2 (xBi, xIn, T) LOCSO recomputed as "
        f"MAE {direct_m['MAE']:.2f}, RMSE {direct_m['RMSE']:.2f}, R² {direct_m['R2']:.4f} (unchanged).",
        "",
        "## Results",
        "",
        "Pooled LOCSO (104 out-of-fold predictions):",
        "",
        "| Model | MAE (J/mol) | RMSE (J/mol) | R² |",
        "| --- | ---: | ---: | ---: |",
    ]
    for k in ["RKM", "A", "B"]:
        L.append(f"| {MODEL_NAMES[k]} | {mt(pooled[k])} |")

    L += [
        "",
        "### Cross-section results",
        "",
        "MAE (J/mol) on each held-out section:",
        "",
        "| Section | RKM | Current Hybrid | Path-aware Hybrid |",
        "| --- | ---: | ---: | ---: |",
    ]
    for s in SECTION_ORDER:
        m = by_sec[s]
        L.append(f"| {s} | {m['RKM']['MAE']:.2f} | {m['A']['MAE']:.2f} | {m['B']['MAE']:.2f} |")
    L += ["", "RMSE / R² per section:", "",
          "| Section | RKM RMSE | Current RMSE | Path-aware RMSE | RKM R² | Current R² | Path-aware R² |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s in SECTION_ORDER:
        m = by_sec[s]
        L.append(
            f"| {s} | {m['RKM']['RMSE']:.2f} | {m['A']['RMSE']:.2f} | {m['B']['RMSE']:.2f} | "
            f"{m['RKM']['R2']:.4f} | {m['A']['R2']:.4f} | {m['B']['R2']:.4f} |"
        )

    L += [
        "",
        "### Temperature results",
        "",
        "MAE (J/mol), pooled over the three held-out folds:",
        "",
        "| Temperature | RKM | Current Hybrid | Path-aware Hybrid |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for t, m in by_t.items():
        L.append(f"| {t} K | {m['RKM']['MAE']:.2f} | {m['A']['MAE']:.2f} | {m['B']['MAE']:.2f} |")

    L += [
        "",
        "### Residual-prediction diagnostics",
        "",
        "Error = actual residual − predicted residual (equal to the hybrid error on ΔmixH).",
        "",
        "| Scope | Model | corr(actual, predicted) | MAE | RMSE | Mean error | Max abs. error |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scope in ["Overall", *SECTION_ORDER]:
        for k in MODELS:
            d = diag[k][scope]
            L.append(
                f"| {scope} | {k} | {d['corr']:.3f} | {d['MAE']:.2f} | {d['RMSE']:.2f} | {d['mean_err']:+.2f} | {d['max_abs']:.2f} |"
            )

    L += [
        "",
        "## Sn-rich diagnosis",
        "",
        "![Actual vs predicted residual](../figures/path_aware_residual_by_section.png)",
        "",
        f"**Observed.** Sn-rich held-out MAE: RKM {snR:.2f}, Model A {snA:.2f}, Model B {snB:.2f} J/mol. "
        + (
            f"The path-aware representation {'reduces' if sn_improves else 'does not reduce'} the Sn-rich error relative to "
            f"Model A ({snB - snA:+.2f} J/mol) and {'is below' if sn_beats_rkm else 'remains above'} the RKM baseline "
            f"({snB - snR:+.2f} J/mol)."
        ),
        f"Correlation between actual and predicted Sn-rich residual: Model A {diag['A']['Sn-rich']['corr']:.3f}, "
        f"Model B {diag['B']['Sn-rich']['corr']:.3f}. Mean error: A {diag['A']['Sn-rich']['mean_err']:+.2f}, "
        f"B {diag['B']['Sn-rich']['mean_err']:+.2f} J/mol. Max abs. error: A {diag['A']['Sn-rich']['max_abs']:.2f}, "
        f"B {diag['B']['Sn-rich']['max_abs']:.2f} J/mol.",
        "",
        "Mean residual at In-rich compositions (xIn > 0.80) for each held-out section — actual vs held-out "
        "predictions:",
        "",
        "| Held-out section | T (K) | Actual | Model A | Model B |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for (s, t), r in hi.reindex(SECTION_ORDER, level=0).iterrows():
        L.append(f"| {s} | {int(t)} | {r['residual']:.1f} | {r['predicted_residual_A']:.1f} | {r['predicted_residual_B']:.1f} |")

    L += [
        "",
        "## Trade-offs",
        "",
    ]
    for s in SECTION_ORDER:
        m = by_sec[s]
        L.append(
            f"- {s}: Model A {m['A']['MAE']:.2f} → Model B {m['B']['MAE']:.2f} J/mol "
            f"({m['B']['MAE'] - m['A']['MAE']:+.2f}); RKM {m['RKM']['MAE']:.2f}."
        )
    L += [
        f"- Pooled: Model A MAE {pooled['A']['MAE']:.2f} → Model B {pooled['B']['MAE']:.2f} J/mol "
        f"({pooled['B']['MAE'] - pooled['A']['MAE']:+.2f}); RMSE {pooled['A']['RMSE']:.2f} → {pooled['B']['RMSE']:.2f}.",
        "",
        (f"Sections where Model B is worse than Model A: {', '.join(worse_secs)}." if worse_secs
         else "Model B is not worse than Model A on the Bi-rich or Equiatomic section."),
        "",
        "## Interpretation",
        "",
        *interpretation(pooled, by_sec, by_t, diag, hi, pred),
        "",
        "## Scientific conclusion",
        "",
        *conclusion(pooled, by_sec, diag),
        "",
    ]
    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")
    return pooled, by_sec, by_t, diag, hi


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    exp = add_path_coordinate(load_experimental())
    print(f"N = {len(exp)}; synthetic datasets not loaded")
    print(exp.groupby("cross_section")["bi_sn_fraction"].agg(["mean", "min", "max"]))

    pred, fold_info = run_locso(exp)
    step1_diff = verify_against_step1(pred)
    direct_m = verify_direct_poly(exp)
    for f in fold_info:
        print(f)

    pooled = metrics_for(pred)
    for k in ["RKM", "A", "B"]:
        print(f"{MODEL_NAMES[k]:30s} {mt(pooled[k])}")
    for ref, k in ((RKM_REF, "RKM"), (STEP1_REF, "A")):
        if abs(pooled[k]["MAE"] - ref["MAE"]) > 0.01 or abs(pooled[k]["RMSE"] - ref["RMSE"]) > 0.01:
            raise RuntimeError(f"{k} does not reproduce reference metrics {ref}")
    if abs(direct_m["MAE"] - DIRECT_POLY_D2_REF["MAE"]) > 0.01:
        raise RuntimeError("Direct Poly D2 LOCSO result changed")
    print(f"Step 1 reproduced (max diff {step1_diff:.1e}); direct Poly D2 {mt(direct_m)}")

    for s, g in pred.groupby("section"):
        m = metrics_for(g)
        print(f"  {s:11s} RKM {m['RKM']['MAE']:.2f}  A {m['A']['MAE']:.2f}  B {m['B']['MAE']:.2f}")
    for t, g in pred.groupby("temperature_K"):
        m = metrics_for(g)
        print(f"  {int(t)} K      RKM {m['RKM']['MAE']:.2f}  A {m['A']['MAE']:.2f}  B {m['B']['MAE']:.2f}")
    for k in MODELS:
        print(k, {s: {kk: round(v, 3) for kk, v in residual_diag(g, k).items()} for s, g in pred.groupby("section")})

    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(PRED_PATH, index=False)
    make_figure(pred)
    write_report(exp, pred, fold_info, step1_diff, direct_m)
    print(f"Wrote {PRED_PATH}\nWrote {REPORT_PATH}\nWrote {FIG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
