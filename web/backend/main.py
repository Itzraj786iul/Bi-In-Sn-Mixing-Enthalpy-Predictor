"""
FastAPI backend for Bi-In-Sn mixing enthalpy prediction.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from comparison import build_comparison
from model import predict
from surface import build_surface

DEFAULT_FRONTEND_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)

# Vercel preview deployments use unique *.vercel.app subdomains.
VERCEL_ORIGIN_REGEX = r"https://.*\.vercel\.app"


def allowed_cors_origins() -> list[str]:
    origins = list(DEFAULT_FRONTEND_ORIGINS)
    configured = os.environ.get("FRONTEND_ORIGIN", "").strip()
    if configured:
        for origin in configured.split(","):
            normalized = origin.strip().rstrip("/")
            if normalized and normalized not in origins:
                origins.append(normalized)
    return origins


app = FastAPI(
    title="Bi-In-Sn Mixing Enthalpy API",
    description="Temperature-aware Polynomial Degree-2 surrogate (104 experimental observations).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_cors_origins(),
    allow_origin_regex=VERCEL_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    xBi: float = Field(..., ge=0.0, le=1.0, description="Bismuth mole fraction")
    xIn: float = Field(..., ge=0.0, le=1.0, description="Indium mole fraction")
    temperature_K: float = Field(..., description="Temperature in kelvin")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "message": "Bi-In-Sn mixing enthalpy backend is running"}


@app.post("/predict")
def predict_endpoint(body: PredictRequest) -> dict:
    try:
        return predict(body.xBi, body.xIn, body.temperature_K)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/surface")
def surface_endpoint(
    temperature_K: float = Query(813, description="Surface temperature in kelvin"),
) -> dict:
    try:
        return build_surface(temperature_K)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/comparison")
def comparison_endpoint(
    temperature_K: float = Query(813, description="Surface temperature in kelvin"),
    mode: str = Query("ml", description="Comparison mode: ml, rkm, or difference"),
) -> dict:
    try:
        return build_comparison(temperature_K, mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
