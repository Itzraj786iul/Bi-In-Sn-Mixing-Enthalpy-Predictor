"""
Ternary surface grid and experimental overlay for the web application.
"""

from __future__ import annotations

import csv
from pathlib import Path

from model import predict_delta_mix_h, predict_delta_mix_h_batch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP_CSV = PROJECT_ROOT / "data" / "original_experimental_data.csv"

GRID_STEP = 0.01
SURFACE_TEMPERATURES = (767, 813, 855)

CROSS_SECTION_LABELS = {
    "(Sn0.67Bi0.33)1-xInx": "Sn-rich cross-section",
    "(Sn0.50Bi0.50)1-xInx": "Equiatomic Bi/Sn cross-section",
    "(Sn0.33Bi0.67)1-xInx": "Bi-rich cross-section",
}

_experiments: list[dict] | None = None


def load_experiments() -> list[dict]:
    global _experiments
    if _experiments is not None:
        return _experiments

    rows: list[dict] = []
    with EXP_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            x_bi = float(row["xBi"])
            x_in = float(row["xIn"])
            x_sn = float(row["xSn"])
            temperature_k = float(row["temperature_K"])
            exp_dmix_h = float(row["integral_mixing_enthalpy_J_per_mol"])
            cross_section = row["cross_section"]
            rows.append(
                {
                    "id": row["id"],
                    "series_id": int(row["series_id"]),
                    "cross_section": cross_section,
                    "cross_section_label": CROSS_SECTION_LABELS.get(
                        cross_section, cross_section
                    ),
                    "temperature_K": temperature_k,
                    "xBi": x_bi,
                    "xIn": x_in,
                    "xSn": x_sn,
                    "experimental_delta_mix_H_J_mol": exp_dmix_h,
                    "ml_prediction_J_mol": predict_delta_mix_h(x_bi, x_in, temperature_k),
                }
            )

    _experiments = rows
    return rows


def build_ternary_grid(step: float = GRID_STEP) -> tuple[list[float], list[float], list[float]]:
    values = [round(i * step, 10) for i in range(int(round(1 / step)) + 1)]
    x_bi_list: list[float] = []
    x_in_list: list[float] = []
    x_sn_list: list[float] = []

    for x_bi in values:
        for x_in in values:
            if x_bi + x_in <= 1.0 + 1e-9:
                x_sn = 1.0 - x_bi - x_in
                x_bi_list.append(x_bi)
                x_in_list.append(x_in)
                x_sn_list.append(x_sn)

    return x_bi_list, x_in_list, x_sn_list


def build_surface(temperature_k: float) -> dict:
    if temperature_k not in SURFACE_TEMPERATURES:
        allowed = ", ".join(str(t) for t in SURFACE_TEMPERATURES)
        raise ValueError(f"Temperature must be one of: {allowed} K.")

    x_bi, x_in, x_sn = build_ternary_grid()
    delta_mix_h = predict_delta_mix_h_batch(x_bi, x_in, temperature_k)

    experiments = load_experiments()
    at_temperature = [row for row in experiments if row["temperature_K"] == temperature_k]

    return {
        "temperature_K": temperature_k,
        "grid_step": GRID_STEP,
        "n_grid_points": len(x_bi),
        "xBi": x_bi,
        "xIn": x_in,
        "xSn": x_sn,
        "delta_mix_H_J_mol": delta_mix_h,
        "experimental_points": at_temperature,
        "experimental_count_at_temperature": len(at_temperature),
        "experimental_total": len(experiments),
    }
