"""
Controlled validation comparison of RKM, direct Poly D2 and the two residual hybrids.

    python scripts/step_8_controlled_model_comparison.py

Two validation regimes on the same 104 experimental observations:
  A) LOCSO (3 folds, one composition path held out) - cross-path extrapolation; primary.
  B) Stratified random 5-fold (stratified by cross-section, fixed seed) - within-path
     interpolation/generalisation. Not a replacement for LOCSO.

Models (existing definitions, unchanged):
  RKM            fixed src/rkm_model.py prediction, never refitted
  Direct Poly D2 dmixH_exp ~ PolyD2(xBi, xIn, T)
  Model A        RKM + [residual ~ PolyD2(xBi, xIn, T)]
  Model B        RKM + [residual ~ PolyD2(xBi, xIn, T, bi_sn_fraction)]

All fitting uses the existing minimum-norm least-squares Poly D2 code. Only
data/original_experimental_data.csv is loaded; no synthetic data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

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
from step_6_path_aware_residual import MODELS as RESIDUAL_FEATURES, add_path_coordinate  # noqa: E402
from validate_dataset import RANDOM_SEED  # noqa: E402

STEP1_PATH = ROOT / "reports" / "residual_learning_results.csv"
STEP3_PATH = ROOT / "reports" / "path_aware_residual_results.csv"
REPORT_PATH = ROOT / "reports" / "controlled_model_comparison.md"
CSV_PATH = ROOT / "reports" / "controlled_model_comparison.csv"
FIG_PATH = ROOT / "figures" / "model_validation_comparison.png"

N_SPLITS = 5
MODEL_KEYS = ["RKM", "Direct", "A", "B"]
MODEL_NAMES = {
    "RKM": "RKM",
    "Direct": "Direct Poly D2",
    "A": "Residual Model A (RKM + PolyD2 residual)",
    "B": "Path-aware Model B (RKM + PolyD2 residual + bi_sn_fraction)",
}
SHORT = {"RKM": "RKM", "Direct": "Direct Poly D2", "A": "Residual Model A", "B": "Path-aware Model B"}
COLORS = {"RKM": "0.45", "Direct": "tab:purple", "A": "tab:orange", "B": "tab:cyan"}
SECTIONS = ["Bi-rich", "Equiatomic", "Sn-rich"]
BENCHMARKS = {
    "RKM": {"MAE": 57.13, "RMSE": 73.49, "R2": 0.9510},
    "Direct": DIRECT_POLY_D2_REF,
    "A": {"MAE": 44.66, "RMSE": 55.75, "R2": 0.9718},
    "B": {"MAE": 36.86, "RMSE": 44.76, "R2": 0.9818},
}


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Fit Direct / A / B on `train` only; RKM is the fixed precomputed prediction."""
    preds = {"RKM": test["rkm_prediction"].to_numpy(float)}
    ranks = {}
    specs = {
        "Direct": (RESIDUAL_FEATURES["A"], EXP_TARGET, False),
        "A": (RESIDUAL_FEATURES["A"], "residual", True),
        "B": (RESIDUAL_FEATURES["B"], "residual", True),
    }
    for key, (cols, target, add_rkm) in specs.items():
        X_train = train[cols].to_numpy(float)
        poly, model = train_polynomial(X_train, train[target].to_numpy(float))
        X_poly = poly.transform(X_train)
        ranks[key] = (int(np.linalg.matrix_rank(X_poly)), int(X_poly.shape[1]))
        p = predict_polynomial(poly, model, test[cols].to_numpy(float))
        preds[key] = p + test["rkm_prediction"].to_numpy(float) if add_rkm else p
    return preds, ranks


def run_scheme(exp: pd.DataFrame, folds: list[tuple[str, pd.DataFrame, pd.DataFrame]], scheme: str):
    oof, fold_rows, rank_rows = [], [], []
    for i, (label, train, test) in enumerate(folds, start=1):
        if set(train["id"]) & set(test["id"]):
            raise RuntimeError(f"{scheme} fold {i}: held-out observation in training set")
        preds, ranks = fit_predict(train, test)
        part = test[["id", "section", "temperature_K", "xBi", "xIn", "xSn", "bi_sn_fraction", EXP_TARGET, "rkm_prediction"]].copy()
        for k in MODEL_KEYS:
            part[f"pred_{k}"] = preds[k]
        part["fold"] = i
        oof.append(part)
        counts = test["section"].value_counts().reindex(SECTIONS, fill_value=0)
        for k in MODEL_KEYS:
            m = calculate_metrics(test[EXP_TARGET], preds[k])
            fold_rows.append({
                "validation_scheme": scheme, "fold": i, "model": SHORT[k],
                "held_out_section": label, "section": "all",
                "test_n_Bi_rich": counts["Bi-rich"], "test_n_Equiatomic": counts["Equiatomic"], "test_n_Sn_rich": counts["Sn-rich"],
                "MAE": m["MAE"], "RMSE": m["RMSE"], "R2": m["R2"],
                "train_size": len(train), "test_size": len(test),
            })
        rank_rows.append({"fold": i, "label": label, **{k: f"{r[0]}/{r[1]}" for k, r in ranks.items()}})
    oof = pd.concat(oof, ignore_index=True)
    if len(oof) != 104 or oof["id"].duplicated().any():
        raise RuntimeError(f"{scheme}: each observation must be predicted exactly once")
    return oof, pd.DataFrame(fold_rows), pd.DataFrame(rank_rows)


def pooled_rows(oof: pd.DataFrame, scheme: str, fold_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in MODEL_KEYS:
        groups = [("all", oof)] + [(s, oof[oof["section"] == s]) for s in SECTIONS]
        for sec, g in groups:
            m = calculate_metrics(g[EXP_TARGET], g[f"pred_{k}"])
            rows.append({
                "validation_scheme": scheme, "fold": "pooled", "model": SHORT[k],
                "held_out_section": "", "section": sec,
                "test_n_Bi_rich": int((g["section"] == "Bi-rich").sum()),
                "test_n_Equiatomic": int((g["section"] == "Equiatomic").sum()),
                "test_n_Sn_rich": int((g["section"] == "Sn-rich").sum()),
                "MAE": m["MAE"], "RMSE": m["RMSE"], "R2": m["R2"],
                "train_size": "", "test_size": len(g),
            })
    return pd.DataFrame(rows)


def stability(fold_df: pd.DataFrame) -> pd.DataFrame:
    g = fold_df.groupby("model", sort=False)["MAE"]
    return pd.DataFrame({"mean": g.mean(), "sd": g.std(ddof=1), "min": g.min(), "max": g.max()})


def verify_locso(exp: pd.DataFrame, oof: pd.DataFrame) -> dict:
    checks = {}
    for k in MODEL_KEYS:
        m = calculate_metrics(oof[EXP_TARGET], oof[f"pred_{k}"])
        ref = BENCHMARKS[k]
        if abs(m["MAE"] - ref["MAE"]) > 0.01 or abs(m["RMSE"] - ref["RMSE"]) > 0.01 or abs(m["R2"] - ref["R2"]) > 0.0001:
            raise RuntimeError(f"LOCSO benchmark mismatch for {k}: {m} vs {ref}")
        checks[k] = m
    s1 = pd.read_csv(STEP1_PATH).set_index("experiment_id")
    s3 = pd.read_csv(STEP3_PATH).set_index("experiment_id")
    o = oof.set_index("id")
    dA = float(np.max(np.abs(o["pred_A"] - s1.loc[o.index, "hybrid_prediction"])))
    dB = float(np.max(np.abs(o["pred_B"] - s3.loc[o.index, "hybrid_prediction_B"])))
    if dA > 1e-6 or dB > 1e-6:
        raise RuntimeError(f"Model A/B LOCSO predictions differ from Step 1/3 outputs ({dA:.2e}, {dB:.2e})")
    checks["max_diff_A"], checks["max_diff_B"] = dA, dB
    return checks


def make_figure(locso_pooled, locso_fold, rand_pooled, rand_fold) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    x = np.arange(len(MODEL_KEYS))
    markers = {"Bi-rich": "o", "Equiatomic": "s", "Sn-rich": "^"}

    ax = axes[0]
    for i, k in enumerate(MODEL_KEYS):
        v = locso_pooled[(locso_pooled["model"] == SHORT[k]) & (locso_pooled["section"] == "all")]["MAE"].iloc[0]
        ax.bar(i, v, 0.6, color=COLORS[k], alpha=0.85)
        ax.text(i, 3, f"{v:.1f}", ha="center", fontsize=10, color="white", fontweight="bold")
        for sec in SECTIONS:
            fv = locso_fold[(locso_fold["model"] == SHORT[k]) & (locso_fold["held_out_section"] == sec)]["MAE"].iloc[0]
            ax.scatter(i + {"Bi-rich": -0.18, "Equiatomic": 0, "Sn-rich": 0.18}[sec], fv, marker=markers[sec],
                       c="black", s=30, zorder=3, label=f"held-out {sec}" if i == 0 else None)
    ax.set_title("A. LOCSO (primary): predict an unseen composition path\nbar = pooled MAE (104), markers = held-out path MAE", fontsize=10)
    ax.set_ylabel("MAE (J/mol)")

    ax = axes[1]
    for i, k in enumerate(MODEL_KEYS):
        v = rand_pooled[(rand_pooled["model"] == SHORT[k]) & (rand_pooled["section"] == "all")]["MAE"].iloc[0]
        ax.bar(i, v, 0.6, color=COLORS[k], alpha=0.85)
        ax.text(i, 3, f"{v:.1f}", ha="center", fontsize=10, color="white", fontweight="bold")
        fv = rand_fold[rand_fold["model"] == SHORT[k]]["MAE"].to_numpy()
        ax.scatter(i + np.linspace(-0.2, 0.2, len(fv)), fv, marker="D", facecolors="none", edgecolors="black", s=24,
                   zorder=3, label="fold MAE" if i == 0 else None)
    ax.set_title(f"B. Stratified random {N_SPLITS}-fold (seed {RANDOM_SEED}): all paths in training\n"
                 "bar = pooled MAE (104), markers = fold MAE", fontsize=10)

    for ax in axes:
        ax.set_xticks(x, [SHORT[k].replace(" Model", "\nModel").replace("Direct ", "Direct\n") for k in MODEL_KEYS], fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Model MAE under two different validation questions (not directly interchangeable)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    exp = add_path_coordinate(load_experimental())
    if len(exp) != 104:
        raise RuntimeError("Expected 104 observations")
    exp["section"] = exp["cross_section"].map(SECTION_LABELS)
    print(f"N = {len(exp)}; synthetic data not loaded; RANDOM_SEED = {RANDOM_SEED}")

    locso_folds = [(SECTION_LABELS[h], tr, te) for h, tr, te in get_locsos_folds(exp)]
    loc_oof, loc_fold, loc_rank = run_scheme(exp, locso_folds, "LOCSO")
    checks = verify_locso(exp, loc_oof)
    print("LOCSO benchmarks reproduced:", {k: v for k, v in checks.items()})

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    rand_folds = [
        ("mixed", exp.iloc[tr].copy(), exp.iloc[te].copy())
        for tr, te in skf.split(exp, exp["cross_section"])
    ]
    rnd_oof, rnd_fold, rnd_rank = run_scheme(exp, rand_folds, f"Stratified random {N_SPLITS}-fold")

    loc_pooled = pooled_rows(loc_oof, "LOCSO", loc_fold)
    rnd_pooled = pooled_rows(rnd_oof, f"Stratified random {N_SPLITS}-fold", rnd_fold)
    out = pd.concat([loc_fold, loc_pooled, rnd_fold, rnd_pooled], ignore_index=True)
    out.to_csv(CSV_PATH, index=False)

    pd.set_option("display.width", 220)
    print(loc_fold[["fold", "model", "held_out_section", "MAE", "RMSE", "R2", "train_size", "test_size"]].round(3).to_string())
    print(rnd_fold[["fold", "model", "test_n_Bi_rich", "test_n_Equiatomic", "test_n_Sn_rich", "MAE", "RMSE", "R2", "train_size", "test_size"]].round(3).to_string())
    print(loc_pooled.round(3).to_string())
    print(rnd_pooled.round(3).to_string())
    print("LOCSO ranks\n", loc_rank, "\nRandom ranks\n", rnd_rank)
    print("Stability LOCSO\n", stability(loc_fold).round(2), "\nStability random\n", stability(rnd_fold).round(2))

    make_figure(loc_pooled, loc_fold, rnd_pooled, rnd_fold)
    write_report(loc_fold, loc_pooled, rnd_fold, rnd_pooled, loc_rank, rnd_rank, checks)
    print(f"Wrote {CSV_PATH}\nWrote {REPORT_PATH}\nWrote {FIG_PATH}")
    return 0


def write_report(loc_fold, loc_pooled, rnd_fold, rnd_pooled, loc_rank, rnd_rank, checks) -> None:
    def pooled(df, k, sec="all", metric="MAE"):
        return float(df[(df["model"] == SHORT[k]) & (df["section"] == sec)][metric].iloc[0])

    def locso_sec(k, sec, metric="MAE"):
        return float(loc_fold[(loc_fold["model"] == SHORT[k]) & (loc_fold["held_out_section"] == sec)][metric].iloc[0])

    st_loc, st_rnd = stability(loc_fold), stability(rnd_fold)
    worst = {k: max(locso_sec(k, s) for s in SECTIONS) for k in MODEL_KEYS}
    best = {k: min(locso_sec(k, s) for s in SECTIONS) for k in MODEL_KEYS}
    sd_sec = {k: float(np.std([locso_sec(k, s) for s in SECTIONS], ddof=1)) for k in MODEL_KEYS}
    best_pooled_loc = min(MODEL_KEYS, key=lambda k: pooled(loc_pooled, k))
    best_worst_loc = min(MODEL_KEYS, key=lambda k: worst[k])
    tr_sizes = sorted(set(rnd_fold["train_size"]))
    te_sizes = sorted(set(rnd_fold["test_size"]))

    L: list[str] = [
        "# Controlled Model Comparison",
        "",
        "Script: `scripts/step_8_controlled_model_comparison.py`. Machine-readable results: "
        "`reports/controlled_model_comparison.csv`. Figure: `figures/model_validation_comparison.png`.",
        "",
        "## Objective",
        "",
        "Steps 1–4 showed that the 104 experimental observations lie on three composition paths with different "
        "Bi/(Bi+Sn) ratios, and that the RKM residual has a strong path-dependent component, especially near the "
        "In-rich corner (`reports/cross_section_residual_structure.md`). A model's measured performance therefore "
        "depends on *which question* the validation asks. This report evaluates the four existing models, "
        "unchanged, under two validation regimes on the same data:",
        "",
        "- **Validation A — LOCSO** (primary): can the model predict a composition path it has never seen?",
        "- **Validation B — stratified random 5-fold**: how well does the model interpolate/generalise when the "
        "training data contain observations from all three paths?",
        "",
        "These are different questions. Validation B is not a replacement for LOCSO and is not presented as more "
        "rigorous.",
        "",
        "## Models",
        "",
        "| Model | Target fitted | Inputs | Prediction |",
        "| --- | --- | --- | --- |",
        "| RKM | none (fixed Table IV parameters, never refitted) | xBi, xIn, xSn | ΔmixH_RKM (temperature-independent) |",
        "| Direct Poly D2 | ΔmixH_exp | PolyD2(xBi, xIn, T) | fitted value |",
        "| Residual Model A | ΔmixH_exp − ΔmixH_RKM | PolyD2(xBi, xIn, T) | RKM + predicted residual |",
        "| Path-aware Model B | ΔmixH_exp − ΔmixH_RKM | PolyD2(xBi, xIn, T, bi_sn_fraction), bi_sn_fraction = xBi/(xBi+xSn) | RKM + predicted residual |",
        "",
        "All fitted models use the existing `PolynomialFeatures(degree=2)` + minimum-norm `np.linalg.lstsq` code "
        "(`scripts/analyze_composition_representation.py`), no regularisation and no tuning. In every fold the "
        "polynomial is fitted on training observations only; RKM predictions come from the unchanged "
        "`src/rkm_model.py`. Only `data/original_experimental_data.csv` (104 rows) is used; no synthetic data, no "
        "RKM-generated training targets.",
        "",
        "## Validation A — LOCSO",
        "",
        "Benchmark reproduction (all within rounding of the frozen values): "
        + "; ".join(f"{SHORT[k]} MAE {checks[k]['MAE']:.2f} / RMSE {checks[k]['RMSE']:.2f} / R² {checks[k]['R2']:.4f}" for k in MODEL_KEYS)
        + f". Model A and Model B held-out predictions match the Step 1 / Step 3 output files to "
        f"{max(checks['max_diff_A'], checks['max_diff_B']):.0e} J/mol.",
        "",
        "Fold-wise results:",
        "",
        "| Fold | Held-out path | Train / test | Model | MAE | RMSE | R² |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for _, r in loc_fold.iterrows():
        L.append(f"| {r['fold']} | {r['held_out_section']} | {r['train_size']} / {r['test_size']} | {r['model']} | {r['MAE']:.2f} | {r['RMSE']:.2f} | {r['R2']:.4f} |")
    L += ["", "Pooled over all 104 held-out predictions:", "", "| Model | MAE | RMSE | R² |", "| --- | ---: | ---: | ---: |"]
    for k in MODEL_KEYS:
        L.append(f"| {SHORT[k]} | {pooled(loc_pooled, k):.2f} | {pooled(loc_pooled, k, metric='RMSE'):.2f} | {pooled(loc_pooled, k, metric='R2'):.4f} |")

    L += [
        "",
        f"## Validation B — Stratified Random {N_SPLITS}-Fold",
        "",
        f"`sklearn.model_selection.StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_SEED})`, "
        f"stratified on the cross-section label (seed = project `RANDOM_SEED` from `scripts/validate_dataset.py`). "
        f"Training sizes {', '.join(map(str, tr_sizes))}; test sizes {', '.join(map(str, te_sizes))}. Every fold's "
        "test set contains all three paths, and every observation is predicted exactly once. For the residual "
        "models the residual is computed from the fixed RKM prediction and the residual model is fitted on the "
        "training observations only.",
        "",
        "| Fold | Train / test | Test n (Bi-rich / Equiatomic / Sn-rich) | Model | MAE | RMSE | R² |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for _, r in rnd_fold.iterrows():
        L.append(
            f"| {r['fold']} | {r['train_size']} / {r['test_size']} | {r['test_n_Bi_rich']} / {r['test_n_Equiatomic']} / {r['test_n_Sn_rich']} | "
            f"{r['model']} | {r['MAE']:.2f} | {r['RMSE']:.2f} | {r['R2']:.4f} |"
        )
    L += ["", "Pooled over all 104 held-out predictions, with per-path MAE of those same predictions:", "",
          "| Model | MAE | RMSE | R² | Bi-rich MAE | Equiatomic MAE | Sn-rich MAE |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for k in MODEL_KEYS:
        L.append(
            f"| {SHORT[k]} | {pooled(rnd_pooled, k):.2f} | {pooled(rnd_pooled, k, metric='RMSE'):.2f} | {pooled(rnd_pooled, k, metric='R2'):.4f} | "
            + " | ".join(f"{pooled(rnd_pooled, k, s):.2f}" for s in SECTIONS) + " |"
        )
    L += [
        "",
        "RKM is identical under both schemes because it is never fitted.",
        "",
        "Design-matrix rank (rank / number of polynomial terms) in the training data of each fold:",
        "",
        "| Scheme | Direct Poly D2 | Model A | Model B |",
        "| --- | --- | --- | --- |",
        f"| LOCSO (two paths in training) | {loc_rank['Direct'].iloc[0]} | {loc_rank['A'].iloc[0]} | {loc_rank['B'].iloc[0]} |",
        f"| Random 5-fold (three paths in training) | {rnd_rank['Direct'].iloc[0]} | {rnd_rank['A'].iloc[0]} | {rnd_rank['B'].iloc[0]} |",
        "",
        "The ranks were identical in every fold of each scheme. With two training paths, all points lie on two "
        "straight lines in the (xBi, xIn) plane, so one quadratic direction (and, for Model B, more) is "
        "undetermined. With three paths the (xBi, xIn, T) polynomial is full rank; Model B keeps one exact "
        "dependency because bi_sn_fraction·xIn = bi_sn_fraction − xBi holds identically.",
        "",
        "## Cross-section stability",
        "",
        "LOCSO, MAE (J/mol) of each held-out path:",
        "",
        "| Model | Pooled MAE | Bi-rich | Equiatomic | Sn-rich | Best-section MAE | Worst-section MAE | SD of section MAEs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for k in MODEL_KEYS:
        L.append(
            f"| {SHORT[k]} | {pooled(loc_pooled, k):.2f} | " + " | ".join(f"{locso_sec(k, s):.2f}" for s in SECTIONS)
            + f" | {best[k]:.2f} | {worst[k]:.2f} | {sd_sec[k]:.2f} |"
        )
    L += [
        "",
        "Fold-MAE stability (descriptive only; not a significance test; LOCSO has only 3 folds):",
        "",
        "| Model | LOCSO mean | LOCSO SD | LOCSO min | LOCSO max | Random mean | Random SD | Random min | Random max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for k in MODEL_KEYS:
        a, b = st_loc.loc[SHORT[k]], st_rnd.loc[SHORT[k]]
        L.append(
            f"| {SHORT[k]} | {a['mean']:.2f} | {a['sd']:.2f} | {a['min']:.2f} | {a['max']:.2f} | "
            f"{b['mean']:.2f} | {b['sd']:.2f} | {b['min']:.2f} | {b['max']:.2f} |"
        )

    L += [
        "",
        "Summary across both questions (pooled MAE, J/mol):",
        "",
        f"| Model | LOCSO | Random {N_SPLITS}-fold | LOCSO worst path |",
        "| --- | ---: | ---: | ---: |",
    ]
    for k in MODEL_KEYS:
        L.append(f"| {SHORT[k]} | {pooled(loc_pooled, k):.2f} | {pooled(rnd_pooled, k):.2f} | {worst[k]:.2f} |")

    L += [
        "",
        "![Model validation comparison](../figures/model_validation_comparison.png)",
        "",
        "## Interpretation",
        "",
        "### Cross-path extrapolation",
        "",
        "**Observed.**",
        "",
        f"- Lowest pooled LOCSO MAE: {SHORT[best_pooled_loc]} ({pooled(loc_pooled, best_pooled_loc):.2f} J/mol). "
        f"Lowest worst-path LOCSO MAE: {SHORT[best_worst_loc]} ({worst[best_worst_loc]:.2f} J/mol). These are "
        "different models.",
        f"- Model B is best on the Bi-rich path ({locso_sec('B', 'Bi-rich'):.2f}) but worst of all four on the "
        f"Sn-rich path ({locso_sec('B', 'Sn-rich'):.2f}); its section MAEs span {best['B']:.2f}–{worst['B']:.2f} "
        f"(SD {sd_sec['B']:.2f}).",
        f"- Model A is best on the Equiatomic path ({locso_sec('A', 'Equiatomic'):.2f}) — the only held-out path that "
        f"lies between the two training paths — and worse than RKM on Sn-rich ({locso_sec('A', 'Sn-rich'):.2f} vs "
        f"{locso_sec('RKM', 'Sn-rich'):.2f}).",
        f"- Direct Poly D2 has the narrowest section spread ({best['Direct']:.2f}–{worst['Direct']:.2f}, SD "
        f"{sd_sec['Direct']:.2f}) and is the only fitted model below RKM on every held-out path, but it is not the "
        "best on any single path except Sn-rich.",
        f"- RKM, which is not fitted, ranges from {best['RKM']:.2f} (Equiatomic) to {worst['RKM']:.2f} (Bi-rich).",
        "",
        "**Interpretation.** Under LOCSO no model is uniformly best. The ranking depends on which path is held out, "
        "and pooled MAE hides this: Model B's pooled advantage comes from one fold. This is consistent with Step 4: "
        "the residual contains a large path-dependent level that must be extrapolated from two paths, and different "
        "model forms extrapolate it differently, with no way to check the extrapolation from the training data "
        "alone (rank-deficient designs).",
        "",
        "### Within-path interpolation/generalization",
        "",
        "**Observed.**",
        "",
        f"- With all three paths represented in training, every fitted model improves relative to LOCSO: Direct "
        f"{pooled(loc_pooled, 'Direct'):.2f} → {pooled(rnd_pooled, 'Direct'):.2f}, Model A {pooled(loc_pooled, 'A'):.2f} → "
        f"{pooled(rnd_pooled, 'A'):.2f}, Model B {pooled(loc_pooled, 'B'):.2f} → {pooled(rnd_pooled, 'B'):.2f} J/mol.",
        f"- The largest change is Model B on Sn-rich: {locso_sec('B', 'Sn-rich'):.2f} (LOCSO) → "
        f"{pooled(rnd_pooled, 'B', 'Sn-rich'):.2f} J/mol (random 5-fold).",
        f"- Fold-to-fold variation is small for all fitted models (fold-MAE SD {st_rnd.loc[SHORT['Direct'], 'sd']:.2f}, "
        f"{st_rnd.loc[SHORT['A'], 'sd']:.2f}, {st_rnd.loc[SHORT['B'], 'sd']:.2f} for Direct, A, B).",
        f"- Under random 5-fold, Model B's largest per-path error is on the Equiatomic path "
        f"({pooled(rnd_pooled, 'B', 'Equiatomic'):.2f}), about the same as its LOCSO Equiatomic MAE "
        f"({locso_sec('B', 'Equiatomic'):.2f}); Direct Poly D2's Equiatomic MAE "
        f"({pooled(rnd_pooled, 'Direct', 'Equiatomic'):.2f}) is about the same as RKM's ({pooled(rnd_pooled, 'RKM', 'Equiatomic'):.2f}).",
        "",
        "**Interpretation.** When every path is present in training, the path-dependent level of the residual "
        "only needs to be *interpolated* along each path, and a model with an explicit Bi/Sn composition "
        "coordinate can represent a separate level per path (its design becomes identifiable in that direction). "
        "That explains why Model B gains most under random 5-fold and why its Sn-rich error collapses there. "
        "It is a statement about representing the three measured paths, not about predicting a fourth.",
        "",
        "Why the two schemes disagree: random 5-fold test points always have close neighbours on the *same* "
        "path in the training set (typically adjacent titration points), so the path-dependent residual level is "
        "seen during training. LOCSO removes the entire path, so that level must be extrapolated across the "
        "Bi/Sn coordinate from only two other paths. The gap between the two columns is therefore a measure of how "
        "much each model depends on having seen the path.",
        "",
        "## Scientific implications",
        "",
        "**Supported by these results**",
        "",
        "- All three fitted models reduce pooled error relative to RKM under both regimes on these 104 observations.",
        "- Model B represents the three measured paths much better than the other models (random 5-fold), "
        "consistent with the strong path-dependent residual structure found in Step 4.",
        "- Cross-path extrapolation remains unresolved: under LOCSO the best model depends on the held-out path, "
        "and no fitted model is below RKM on every path except Direct Poly D2, whose margin on the Equiatomic and "
        "Sn-rich paths is small.",
        "- Performance claims for unmeasured Bi/Sn ratios should be based on LOCSO, with per-path results reported "
        "alongside pooled values.",
        "",
        "**Not supported by these results**",
        "",
        "- That Model B is universally superior: it is worst on the Sn-rich LOCSO fold.",
        "- That random 5-fold is more rigorous than LOCSO: it answers an easier, different question "
        "(interpolation within measured paths) and LOCSO remains the primary cross-path validation.",
        "- That synthetic data are needed: nothing here tests synthetic data, and the limitation identified is the "
        "number of independent experimental composition paths (three), not the number of points per path.",
        "- That RKM is thermodynamically inferior: RKM is a fixed, temperature-independent model from binary "
        "parameters with no fitting to these data; fitted models having lower MAE on the data they are calibrated "
        "against does not imply a better thermodynamic description.",
        "- R² is reported as the coefficient of determination on held-out predictions. It is not an accuracy "
        "measure, and with ΔmixH spanning ≈ −64 to −1413 J/mol it stays high even when errors of 50–100 J/mol occur.",
        "",
        "No model is selected or replaced in this step. The website and the final model are unchanged.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
