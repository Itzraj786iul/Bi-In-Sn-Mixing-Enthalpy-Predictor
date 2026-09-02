# Bi-In-Sn Mixing Enthalpy Web Application

Scientific web interface for the **frozen** Polynomial Degree-2 surrogate (Step 4A).

- **Frontend:** React + Vite (`web/frontend`)
- **Backend:** FastAPI (`web/backend`)
- **Model:** Hard-coded coefficients in `backend/model.py` (not refit)

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

Example response (ML surrogate):

```json
{
  "xBi": 0.2509,
  "xIn": 0.4982,
  "xSn": 0.2509,
  "temperature_K": 813.0,
  "delta_mix_H_J_mol": -1014.69,
  "mixing_type": "Exothermic"
}
```

Note: RKM at the same composition is approximately −1028.55 J/mol; the ML model is trained on experimental data and differs slightly.

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

**Primary validated performance (LOCSO, 104 experimental points):**

| Metric | Value |
|--------|------:|
| MAE | 41.10 J/mol |
| RMSE | 56.34 J/mol |
| R² | 0.9712 |

**Input limits in the web form:**

- 0 ≤ xBi, xIn ≤ 1
- xBi + xIn ≤ 1
- 767 K ≤ T ≤ 855 K

Predictions outside the measured cross-sections are surrogate extrapolation and are not experimentally validated.

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
