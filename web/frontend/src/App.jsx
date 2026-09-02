import { useMemo, useState } from "react";
import TernaryMap from "./TernaryMap.jsx";
import ComparisonMap from "./ComparisonMap.jsx";
import {
  FinalSurrogateSection,
  Hero,
  MethodologySection,
  ResearchProvenanceSection,
  ScientificContextSection,
  SectionHeader,
  SiteFooter,
  ValidationSnapshot,
  mixingTypeLabel,
} from "./ScientificBlocks.jsx";
import { apiUrl } from "./api.js";
import { formatApiError, getFetchErrorMessage } from "./ternaryShared.js";
const T_MIN = 767;
const T_MAX = 855;

function validateClient(xBi, xIn, temperature) {
  const xBiNum = Number(xBi);
  const xInNum = Number(xIn);
  const tNum = Number(temperature);

  if (Number.isNaN(xBiNum) || Number.isNaN(xInNum) || Number.isNaN(tNum)) {
    return "All fields must be valid numbers.";
  }
  if (xBiNum < 0 || xBiNum > 1) {
    return "Bi mole fraction must be between 0 and 1.";
  }
  if (xInNum < 0 || xInNum > 1) {
    return "In mole fraction must be between 0 and 1.";
  }
  if (xBiNum + xInNum > 1) {
    return "Bi and In mole fractions must sum to at most 1 (Sn = 1 − xBi − xIn).";
  }
  if (tNum < T_MIN || tNum > T_MAX) {
    return `Temperature must be between ${T_MIN} K and ${T_MAX} K.`;
  }
  return null;
}

export default function App() {
  const [xBi, setXBi] = useState("0.2509");
  const [xIn, setXIn] = useState("0.4982");
  const [temperature, setTemperature] = useState("813");
  const [mapTemperature, setMapTemperature] = useState(813);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const xSnPreview = useMemo(() => {
    const bi = Number(xBi);
    const ind = Number(xIn);
    if (Number.isNaN(bi) || Number.isNaN(ind)) return "—";
    if (bi + ind > 1) return "invalid (sum > 1)";
    return (1 - bi - ind).toFixed(4);
  }, [xBi, xIn]);

  async function handlePredict(event) {
    event.preventDefault();
    setError("");
    setResult(null);

    const validationError = validateClient(xBi, xIn, temperature);
    if (validationError) {
      setError(validationError);
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(apiUrl("/predict"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          xBi: Number(xBi),
          xIn: Number(xIn),
          temperature_K: Number(temperature),
        }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(formatApiError(data.detail, "Prediction request failed."));
      }
      setResult(data);
    } catch (err) {
      setError(getFetchErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }

  function handleMapCompositionSelect(xBiValue, xInValue, tempK) {
    setXBi(String(Number(xBiValue).toFixed(4)));
    setXIn(String(Number(xInValue).toFixed(4)));
    setTemperature(String(tempK));
    setMapTemperature(tempK);
    setResult(null);
    setError("");
  }

  return (
    <div className="page">
      <Hero />
      <ValidationSnapshot />

      <SectionHeader
        label="01 / Predict"
        title="Estimate mixing enthalpy"
        description="Specify alloy composition and temperature to evaluate the frozen Polynomial Degree-2 surrogate."
      />

      <main className="predict-grid">
        <section className="panel panel-inputs">
          <form onSubmit={handlePredict} className="form">
            <label>
              <span className="field-label">xBi</span>
              <span className="field-hint">Bi mole fraction</span>
              <input
                type="number"
                step="any"
                min="0"
                max="1"
                value={xBi}
                onChange={(e) => setXBi(e.target.value)}
                required
              />
            </label>

            <label>
              <span className="field-label">xIn</span>
              <span className="field-hint">In mole fraction</span>
              <input
                type="number"
                step="any"
                min="0"
                max="1"
                value={xIn}
                onChange={(e) => setXIn(e.target.value)}
                required
              />
            </label>

            <label>
              <span className="field-label">xSn</span>
              <span className="field-hint">Calculated: 1 − xBi − xIn</span>
              <input type="text" value={xSnPreview} readOnly className="readonly mono" />
            </label>

            <label>
              <span className="field-label">Temperature</span>
              <span className="field-hint">767–855 K (experimental range)</span>
              <input
                type="number"
                step="any"
                min={T_MIN}
                max={T_MAX}
                value={temperature}
                onChange={(e) => setTemperature(e.target.value)}
                required
              />
            </label>

            {error && (
              <p className="error" role="alert">
                {error}
              </p>
            )}

            <button type="submit" className="primary-btn" disabled={loading}>
              {loading ? "Computing…" : "Compute ΔmixH"}
            </button>
          </form>
        </section>

        <section className="panel panel-result">
          {result ? (
            <div className="result-display">
              <p className="result-label">ΔmixH</p>
              <p className="result-value mono">{result.delta_mix_H_J_mol.toFixed(2)}</p>
              <p className="result-unit">J/mol</p>
              <p
                className={`result-behavior ${
                  result.mixing_type === "Exothermic"
                    ? "behavior-exo"
                    : result.mixing_type === "Endothermic"
                      ? "behavior-endo"
                      : "behavior-zero"
                }`}
              >
                {mixingTypeLabel(result.mixing_type)}
              </p>
              <dl className="result-meta">
                <div>
                  <dt>xBi</dt>
                  <dd className="mono">{result.xBi.toFixed(4)}</dd>
                </div>
                <div>
                  <dt>xIn</dt>
                  <dd className="mono">{result.xIn.toFixed(4)}</dd>
                </div>
                <div>
                  <dt>xSn</dt>
                  <dd className="mono">{result.xSn.toFixed(4)}</dd>
                </div>
                <div>
                  <dt>Temperature</dt>
                  <dd className="mono">{result.temperature_K.toFixed(1)} K</dd>
                </div>
              </dl>
            </div>
          ) : (
            <div className="result-empty">
              <p className="result-label">ΔmixH</p>
              <p className="placeholder">
                Enter composition and temperature, then compute to obtain integral molar mixing
                enthalpy from the frozen surrogate.
              </p>
            </div>
          )}
        </section>
      </main>

      <SectionHeader
        label="02 / Explore composition"
        title="Composition-dependent mixing enthalpy"
        description="Explore the surrogate surface across the Bi–In–Sn ternary composition space at selected temperatures."
      />

      <TernaryMap
        mapTemperature={mapTemperature}
        onMapTemperatureChange={setMapTemperature}
        onSelectComposition={handleMapCompositionSelect}
      />

      <SectionHeader
        label="03 / Compare models"
        title="Two models. One thermodynamic system."
        description="Compare the experimentally validated Polynomial Degree-2 surrogate with the Redlich-Kister-Muggianu thermodynamic model across the Bi–In–Sn composition space."
      />

      <ComparisonMap />

      <FinalSurrogateSection />
      <MethodologySection />
      <ScientificContextSection />
      <ResearchProvenanceSection />
      <SiteFooter />
    </div>
  );
}
