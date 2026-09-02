import { useCallback, useEffect, useMemo, useState } from "react";
import Plot from "react-plotly.js";
import {
  CrossSectionLegend,
  ValidatedExploratoryBlocks,
} from "./ScientificBlocks.jsx";
import { apiUrl } from "./api.js";
import {
  SURFACE_TEMPS,
  buildMlExperimentalTraces,
  formatApiError,
  getFetchErrorMessage,
  getTernaryLayout,
} from "./ternaryShared.js";

export default function TernaryMap({ mapTemperature, onMapTemperatureChange, onSelectComposition }) {
  const [surface, setSurface] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");

    fetch(apiUrl(`/surface?temperature_K=${mapTemperature}`))
      .then(async (response) => {
        const data = await response.json();
        if (!response.ok) {
          throw new Error(formatApiError(data.detail, "Could not load composition map."));
        }
        if (!cancelled) setSurface(data);
      })
      .catch((err) => {
        if (!cancelled) {
          setSurface(null);
          setError(getFetchErrorMessage(err, "Could not load composition map."));
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [mapTemperature]);

  const colorRange = useMemo(() => {
    if (!surface?.delta_mix_H_J_mol?.length) return [null, null];
    const values = surface.delta_mix_H_J_mol;
    return [Math.min(...values), Math.max(...values)];
  }, [surface]);

  const plotData = useMemo(() => {
    if (!surface) return [];

    return [
      {
        type: "scatterternary",
        mode: "markers",
        name: "ML surrogate",
        a: surface.xBi,
        b: surface.xIn,
        c: surface.xSn,
        marker: {
          size: 5,
          color: surface.delta_mix_H_J_mol,
          colorscale: "Viridis",
          cmin: colorRange[0],
          cmax: colorRange[1],
          colorbar: {
            title: { text: "ΔmixH (J/mol)" },
            len: 0.9,
          },
          opacity: 0.95,
        },
        hovertemplate:
          `Bi: %{a:.4f}<br>In: %{b:.4f}<br>Sn: %{c:.4f}<br>Temperature: ${mapTemperature} K<br>ΔmixH: %{marker.color:.2f} J/mol<extra></extra>`,
        showlegend: false,
      },
      ...buildMlExperimentalTraces(surface.experimental_points || []),
    ];
  }, [surface, colorRange, mapTemperature]);

  const handlePlotClick = useCallback(
    (event) => {
      const point = event.points?.[0];
      if (!point || point.data.name !== "ML surrogate") return;
      onSelectComposition?.(point.a, point.b, mapTemperature);
    },
    [mapTemperature, onSelectComposition]
  );

  return (
    <section className="explore-section" aria-labelledby="explore-map-title">
      <ValidatedExploratoryBlocks />

      <div className="explore-controls">
        <div className="control-group">
          <span className="control-label">Temperature</span>
          <div className="segmented-control" role="group" aria-label="Surface temperature">
            {SURFACE_TEMPS.map((temp) => (
              <button
                key={temp}
                type="button"
                className={temp === mapTemperature ? "segment active" : "segment"}
                onClick={() => onMapTemperatureChange(temp)}
              >
                {temp} K
              </button>
            ))}
          </div>
        </div>
      </div>

      {loading && (
        <p className="state-message explore-loading">Computing composition surface…</p>
      )}
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}

      {!loading && !error && surface && (
        <div className="explore-map-layout">
          <div className="explore-map-main">
            <div className="map-header">
              <p className="map-eyebrow">Composition space</p>
              <h3 className="map-title" id="explore-map-title">
                ML surrogate surface
              </h3>
              <p className="map-meta">
                Selected temperature: <span className="mono">{mapTemperature}</span> K ·{" "}
                <span className="mono">{surface.experimental_count_at_temperature}</span> of{" "}
                <span className="mono">{surface.experimental_total}</span> experimental observations
              </p>
              <p className="map-hint">Click any surface point to populate the prediction inputs.</p>
            </div>

            <div className="ternary-plot viz-frame">
              <Plot
                data={plotData}
                layout={getTernaryLayout({ height: 600 })}
                config={{
                  displayModeBar: true,
                  modeBarButtonsToRemove: ["lasso2d", "select2d"],
                  responsive: true,
                }}
                style={{ width: "100%", height: "100%" }}
                onClick={handlePlotClick}
                useResizeHandler
              />
            </div>
          </div>

          <aside className="explore-sidebar">
            <CrossSectionLegend />
          </aside>
        </div>
      )}
    </section>
  );
}
