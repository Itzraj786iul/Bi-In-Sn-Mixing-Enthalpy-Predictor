# Bi-In-Sn Mixing Enthalpy Web Application

Scientific web interface for the **frozen** experimental-only Direct Polynomial Degree-2 surrogate, the final model.

- **Frontend:** React + Vite (`web/frontend`)
- **Backend:** FastAPI (`web/backend`)
- **Model:** Hard-coded coefficients in `backend/model.py` (not refit), taken from `reports/final_model_coefficients.csv`. The model was trained only on the 104 experimental observations; the backend loads no synthetic data.

## Backend setup

```bash
cd web/backend
python -m pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### Health check

```bash
curl http://127.0.0.1:8000/health
```

Expected:

```json
{"status":"ok","message":"Bi-In-Sn mixing enthalpy backend is running"}
```

### Example prediction

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"xBi\": 0.2509, \"xIn\": 0.4982, \"temperature_K\": 813}"
```

Example response (ML surrogate; `delta_mix_H_J_mol` shown rounded to 4 decimals):

```json
{
  "xBi": 0.2509,
  "xIn": 0.4982,
  "xSn": 0.2509,
  "temperature_K": 813.0,
  "delta_mix_H_J_mol": -1014.5761,
  "mixing_type": "Exothermic"
}
```

### Benchmark composition

xBi = 0.2509, xIn = 0.4982, xSn = 0.2509, T = 813 K (experimental row EXP_0051, measured ΔmixH = −1017.0 J/mol):

| Quantity | Value |
|----------|------:|
| ML (frozen Poly D2) | −1014.5761 J/mol |
| RKM | −1028.5461 J/mol |
| ML − RKM | +13.9700 J/mol |

## Frontend setup

In a second terminal:

```bash
cd web/frontend
npm install
npm run dev
```

Open http://localhost:5173

The Vite dev server proxies `/predict`, `/health`, `/surface`, and `/comparison` to the FastAPI backend on port 8000.

### Composition map endpoint

```bash
curl "http://127.0.0.1:8000/surface?temperature_K=813"
```

Returns the ternary grid (step 0.01), ML predictions at the requested temperature, and experimental points filtered to that temperature.

### ML vs RKM comparison endpoint

```bash
curl "http://127.0.0.1:8000/comparison?temperature_K=813&mode=ml"
curl "http://127.0.0.1:8000/comparison?temperature_K=813&mode=rkm"
curl "http://127.0.0.1:8000/comparison?temperature_K=813&mode=difference"
```

Modes: `ml`, `rkm`, `difference` (ML − RKM). Uses the frozen ML surrogate and `src/rkm_model.py`.

## Deployment

The frontend and backend are **separate services** in production:

| Service | Build / start | Serves |
|---------|---------------|--------|
| Frontend | `npm run build` → static files in `dist/` | React UI |
| Backend | `uvicorn main:app --host 0.0.0.0 --port 8000` | FastAPI JSON API |

### Environment variables

**Frontend** (set at build time; `VITE_*` values are public in the browser bundle):

| Variable | Development | Production |
|----------|-------------|------------|
| `VITE_API_BASE_URL` | `http://127.0.0.1:8000` or empty (use Vite proxy) | `<deployed backend HTTPS URL>` |

Copy `frontend/.env.example` to `frontend/.env.local` for local overrides.

**Backend**:

| Variable | Development | Production |
|----------|-------------|------------|
| `FRONTEND_ORIGIN` | `http://localhost:5173` (optional) | `<deployed frontend HTTPS URL>` |

`http://localhost:5173` and `http://127.0.0.1:5173` are always allowed for local development. Set `FRONTEND_ORIGIN` to add the production frontend origin. Do not use unrestricted `allow_origins=["*"]`.

Copy `backend/.env.example` for reference. No secrets are required for this application.

### Build and run (production-style)

```bash
# Backend
cd web/backend
python -m pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000

# Frontend (set API URL to your backend before building)
cd web/frontend
npm install
npm run build
# Serve dist/ with any static file host
```

### Local development

With an empty `VITE_API_BASE_URL`, `npm run dev` uses the Vite proxy to `http://127.0.0.1:8000`. With `VITE_API_BASE_URL=http://127.0.0.1:8000`, the frontend calls the backend directly (requires CORS).

## Model summary

**Features:** xBi, xIn, temperature_K (xSn = 1 − xBi − xIn)

**Output:** integral molar mixing enthalpy ΔmixH (J/mol)

**Coefficients:** the backend uses the full-precision fitted coefficient for the T² term (`0.0009741759040480247`). The website displays coefficients to 6 decimal places for readability.

**Experimental evaluation (104 experimental points):**

| Model | Evaluation protocol | MAE | RMSE | R² |
|-------|---------------------|----:|-----:|---:|
| Direct Poly D2 (final) | LOCSO, held-out cross-section | 41.10 J/mol | 56.34 J/mol | 0.9712 |
| RKM | Evaluated on the 104 observations using the published parameters | 57.13 J/mol | 73.49 J/mol | 0.9510 |

These are not strictly like-for-like out-of-sample scores. The ML result uses cross-section holdout. RKM is never refitted in this project, but its ternary parameters were fitted to these experimental measurements by the original authors. The RKM value is therefore a physics-model reference rather than an independent LOCSO validation score.

**Input limits in the web form:**

- 0 ≤ xBi, xIn ≤ 1
- xBi + xIn ≤ 1
- 767 K ≤ T ≤ 855 K

Predictions outside the measured cross-sections are surrogate extrapolation and are not experimentally validated.

**Thermodynamic boundary limitation:** the fitted polynomial is an empirical predictive surrogate and was not constrained to satisfy ΔmixH = 0 at the pure components. It therefore returns non-zero endpoint values (for example, about −392 J/mol for pure Bi at 767 K). These boundary extrapolations are not physically valid pure-component thermodynamic predictions.

## Project layout

```
web/
  backend/
    main.py          # FastAPI routes
    model.py         # Frozen polynomial equation
    requirements.txt
  frontend/
    src/App.jsx      # Prediction UI
    ...
  README.md
```

The main semester project (data, RKM, ML scripts, reports) remains in the repository root and is not modified by the web app.
