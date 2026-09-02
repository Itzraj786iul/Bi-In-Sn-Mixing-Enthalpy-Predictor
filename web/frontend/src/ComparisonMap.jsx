import { useCallback, useEffect, useMemo, useState } from "react";
import Plot from "react-plotly.js";
import {
  ExperimentalCoveragePanel,
  ModelValidationTable,
} from "./ScientificBlocks.jsx";
import { apiUrl } from "./api.js";
import {
  SURFACE_TEMPS,
  buildComparisonExperimentalTraces,
  getComparisonErrorMessage,
  getFetchErrorMessage,
  getTernaryLayout,
} from "./ternaryShared.js";

const MODES = [
  { id: "ml", label: "ML surrogate" },
  { id: "rkm", label: "RKM" },
  { id: "difference", label: "ML − RKM" },
];

function formatRange([min, max]) {
  return `[${min.toFixed(1)}, ${max.toFixed(1)}] J/mol`;
}

export default function ComparisonMap() {
  const [temperature, setTemperature] = useState(813);
  const [mode, setMode] = useState("ml");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [fetchKey, setFetchKey] = useState(0);

  const loadComparison = useCallback(() => {
    setFetchKey((key) => key + 1);
  }, []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");

    fetch(apiUrl(`/comparison?temperature_K=${temperature}&mode=difference`))
      .then(async (response) => {
        let payload = {};
        try {
          payload = await response.json();
        } catch {
          payload = {};
        }
        if (!response.ok) {
          throw new Error(getComparisonErrorMessage(response, payload));
        }
        if (!cancelled) setData(payload);
      })
      .catch((err) => {
        if (!cancelled) {
          setData(null);
          setError(getFetchErrorMessage(err));
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [temperature, fetchKey]);

  const surfaceValues = useMemo(() => {
    if (!data) return [];
    if (mode === "ml") return data.delta_mix_H_ml || [];
    if (mode === "rkm") return data.delta_mix_H_rkm || [];
    return data.difference || [];
  }, [data, mode]);

  const colorSettings = useMemo(() => {
    if (!surfaceValues.length) {
      return { colorscale: "Viridis", cmin: null, cmax: null, colorbarTitle: "ΔmixH (J/mol)" };
    }

    if (mode === "difference") {
      const maxAbs = Math.max(...surfaceValues.map((v) => Math.abs(v)));
      return {
        colorscale: "RdBu",
        cmin: -maxAbs,
        cmax: maxAbs,
        colorbarTitle: "ML − RKM (J/mol)",
      };
    }

    return {
      colorscale: "Viridis",
      cmin: Math.min(...surfaceValues),
      cmax: Math.max(...surfaceValues),
      colorbarTitle: "ΔmixH (J/mol)",
    };
  }, [surfaceValues, mode]);

  const surfaceLabel =
    mode === "ml" ? "ML surrogate" : mode === "rkm" ? "RKM thermodynamic model" : "ML − RKM";

  const plotData = useMemo(() => {
    if (!data) return [];

    const hoverTemplate =
      mode === "difference"
        ? `Bi: %{a:.4f}<br>In: %{b:.4f}<br>Sn: %{c:.4f}<br>Temperature: ${temperature} K<br>ML − RKM: %{marker.color:.2f} J/mol<extra></extra>`
        : `Bi: %{a:.4f}<br>In: %{b:.4f}<br>Sn: %{c:.4f}<br>Temperature: ${temperature} K<br>${surfaceLabel}: %{marker.color:.2f} J/mol<extra></extra>`;

    return [
      {
        type: "scatterternary",
        mode: "markers",
        name: surfaceLabel,
        a: data.xBi,
        b: data.xIn,
        c: data.xSn,
        marker: {
          size: 5,
          color: surfaceValues,
          colorscale: colorSettings.colorscale,
          cmin: colorSettings.cmin,
          cmax: colorSettings.cmax,
          colorbar: {
            title: { text: colorSettings.colorbarTitle },
            len: 0.9,
          },
          opacity: 0.95,
        },
        hovertemplate: hoverTemplate,
        showlegend: false,
      },
      ...buildComparisonExperimentalTraces(data.experimental_points || []),
    ];
  }, [data, surfaceValues, colorSettings, mode, surfaceLabel, temperature]);

  const stats = data?.statistics;

  return (
    <section className="comparison-section" aria-labelledby="comparison-map-title">
      <ModelValidationTable />

      <div className="comparison-controls">
        <div className="control-group">
          <span className="control-label">Model view</span>
          <div className="segmented-control" role="group" aria-label="Comparison model view">
            {MODES.map((item) => (
              <button
                key={item.id}
                type="button"
                className={item.id === mode ? "segment active" : "segment"}
                onClick={() => setMode(item.id)}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>

        <div className="control-group">
          <span className="control-label">Temperature</span>
          <div className="segmented-control" role="group" aria-label="Comparison temperature">
            {SURFACE_TEMPS.map((temp) => (
              <button
                key={temp}
                type="button"
                className={temp === temperature ? "segment active" : "segment"}
                onClick={() => setTemperature(temp)}
              >
                {temp} K
              </button>
            ))}
          </div>
        </div>
      </div>

      {mode === "difference" && (
        <div className="difference-interpretation">
          <h3 className="difference-interpretation-title">Model disagreement</h3>
          <p className="difference-interpretation-formula mono">Difference = ML − RKM</p>
          <dl className="difference-interpretation-list">
            <div>
              <dt>Negative</dt>
              <dd>ML predicts a more exothermic value than RKM.</dd>
            </div>
            <div>
              <dt>Positive</dt>
              <dd>ML predicts a less exothermic value than RKM.</dd>
            </div>
          </dl>
          <p className="difference-interpretation-note">
            Large differences in unsampled compositions indicate model disagreement, not
            experimental confirmation of either model.
          </p>
        </div>
      )}

      {loading && (
        <p className="state-message comparison-loading">Computing composition surface…</p>
      )}

      {error && (
        <div className="comparison-error" role="alert">
          <p className="comparison-error-title">Comparison data unavailable</p>
          <p className="comparison-error-body">{error}</p>
          <button type="button" className="retry-btn" onClick={loadComparison}>
            Retry
          </button>
        </div>
      )}

      {!loading && !error && data && (
        <>
          <div className="comparison-map-layout">
            <div className="comparison-map-main">
              <div className="map-header">
                <p className="map-eyebrow">Composition space</p>
                <h3 className="map-title" id="comparison-map-title">
                  Bi–In–Sn
                </h3>
                <p className="map-meta">
                  Selected temperature: <span className="mono">{temperature}</span> K
                </p>
              </div>

              <div className="ternary-plot comparison-plot">
                <Plot
                  data={plotData}
                  layout={getTernaryLayout({ height: 620 })}
                  config={{
                    displayModeBar: true,
                    modeBarButtonsToRemove: ["lasso2d", "select2d"],
                    responsive: true,
                  }}
                  style={{ width: "100%", height: "100%" }}
                  useResizeHandler
                />
              </div>
            </div>

            <ExperimentalCoveragePanel
              temperature={temperature}
              observationCount={data.experimental_count_at_temperature}
            />
          </div>

          {stats && (
            <div className="comparison-readout">
              <p className="comparison-readout-disclaimer">
                Full-simplex model comparison — not experimental validation.
              </p>
              <dl className="comparison-readout-grid">
                <div>
                  <dt>ML surface</dt>
                  <dd className="mono">range: {formatRange(stats.ml_range_J_mol)}</dd>
                </div>
                <div>
                  <dt>RKM surface</dt>
                  <dd className="mono">range: {formatRange(stats.rkm_range_J_mol)}</dd>
                </div>
                <div>
                  <dt>Model disagreement</dt>
                  <dd className="mono">
                    mean |ML − RKM|: {stats.mean_abs_difference_J_mol.toFixed(1)} J/mol
                  </dd>
                </div>
                <div>
                  <dt className="sr-only">Maximum disagreement</dt>
                  <dd className="mono">
                    maximum |ML − RKM|: {stats.max_abs_difference_J_mol.toFixed(1)} J/mol
                  </dd>
                </div>
              </dl>
            </div>
          )}
        </>
      )}
    </section>
  );
}
