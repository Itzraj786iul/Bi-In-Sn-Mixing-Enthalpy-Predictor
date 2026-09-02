export const SURFACE_TEMPS = [767, 813, 855];

export const BACKEND_UNAVAILABLE_MSG =
  "Unable to load the prediction service. Please check that the backend is running.";

export function formatApiError(detail, fallback = "Request failed.") {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((entry) => entry.msg || JSON.stringify(entry)).join(" ");
  }
  return fallback;
}

export function getFetchErrorMessage(err, fallback = BACKEND_UNAVAILABLE_MSG) {
  if (err instanceof TypeError || err?.message === "Failed to fetch") {
    return BACKEND_UNAVAILABLE_MSG;
  }
  return err?.message || fallback;
}

export const COMPARISON_UNAVAILABLE_MSG =
  "Unable to load the comparison surface. Check that the prediction backend is running and try again.";

export function getComparisonErrorMessage(response, payload) {
  const detail = payload?.detail;
  if (response?.status === 404 || detail === "Not Found") {
    return COMPARISON_UNAVAILABLE_MSG;
  }
  return formatApiError(detail, COMPARISON_UNAVAILABLE_MSG);
}

export const SERIES_STYLES = {
  "Sn-rich cross-section": { symbol: "circle", color: "#1e3a5f" },
  "Equiatomic Bi/Sn cross-section": { symbol: "diamond", color: "#b45309" },
  "Bi-rich cross-section": { symbol: "square", color: "#047857" },
};

export function groupExperimental(points) {
  const groups = {};
  for (const point of points) {
    const label = point.cross_section_label;
    if (!groups[label]) groups[label] = [];
    groups[label].push(point);
  }
  return groups;
}

export function getTernaryLayout(overrides = {}) {
  return {
    margin: { l: 24, r: 24, t: 24, b: 24 },
    paper_bgcolor: "#faf8f5",
    plot_bgcolor: "#faf8f5",
    ternary: {
      sum: 1,
      bgcolor: "#f5f2ed",
      aaxis: {
        title: { text: "Bi", font: { family: "Source Sans 3, sans-serif", size: 13 } },
        min: 0,
        linewidth: 1,
        gridcolor: "#ddd6cc",
        tickfont: { family: "IBM Plex Mono, monospace", size: 11 },
      },
      baxis: {
        title: { text: "In", font: { family: "Source Sans 3, sans-serif", size: 13 } },
        min: 0,
        linewidth: 1,
        gridcolor: "#ddd6cc",
        tickfont: { family: "IBM Plex Mono, monospace", size: 11 },
      },
      caxis: {
        title: { text: "Sn", font: { family: "Source Sans 3, sans-serif", size: 13 } },
        min: 0,
        linewidth: 1,
        gridcolor: "#ddd6cc",
        tickfont: { family: "IBM Plex Mono, monospace", size: 11 },
      },
    },
    legend: {
      orientation: "h",
      y: -0.08,
      x: 0,
      font: { size: 11, family: "Source Sans 3, sans-serif" },
    },
    height: 560,
    ...overrides,
  };
}

export function buildMlExperimentalTraces(points) {
  const traces = [];
  const grouped = groupExperimental(points);
  for (const [label, group] of Object.entries(grouped)) {
    const style = SERIES_STYLES[label] || { symbol: "circle-open", color: "#334155" };
    traces.push({
      type: "scatterternary",
      mode: "markers",
      name: label,
      a: group.map((p) => p.xBi),
      b: group.map((p) => p.xIn),
      c: group.map((p) => p.xSn),
      marker: {
        size: 10,
        symbol: style.symbol,
        color: style.color,
        line: { width: 1.5, color: "#ffffff" },
      },
      customdata: group.map((p) => [
        p.id,
        p.temperature_K,
        p.experimental_delta_mix_H_J_mol,
        p.ml_prediction_J_mol,
      ]),
      hovertemplate:
        "Experiment: %{customdata[0]}<br>" +
        "Series: " +
        label +
        "<br>" +
        "Temperature: %{customdata[1]:.0f} K<br>" +
        "Bi: %{a:.4f}<br>In: %{b:.4f}<br>Sn: %{c:.4f}<br>" +
        "Experimental ΔmixH: %{customdata[2]:.1f} J/mol<br>" +
        "ML prediction: %{customdata[3]:.2f} J/mol<extra></extra>",
    });
  }
  return traces;
}

export function buildComparisonExperimentalTraces(points) {
  const traces = [];
  const grouped = groupExperimental(points);
  for (const [label, group] of Object.entries(grouped)) {
    const style = SERIES_STYLES[label] || { symbol: "circle-open", color: "#334155" };
    traces.push({
      type: "scatterternary",
      mode: "markers",
      name: label,
      a: group.map((p) => p.xBi),
      b: group.map((p) => p.xIn),
      c: group.map((p) => p.xSn),
      marker: {
        size: 10,
        symbol: style.symbol,
        color: style.color,
        line: { width: 1.5, color: "#ffffff" },
      },
      customdata: group.map((p) => [
        p.id,
        p.cross_section_label,
        p.temperature_K,
        p.experimental_delta_mix_H_J_mol,
        p.ml_prediction_J_mol,
        p.rkm_prediction_J_mol,
      ]),
      hovertemplate:
        "Experiment: %{customdata[0]}<br>" +
        "Cross-section: %{customdata[1]}<br>" +
        "Temperature: %{customdata[2]:.0f} K<br>" +
        "Bi: %{a:.4f}<br>In: %{b:.4f}<br>Sn: %{c:.4f}<br>" +
        "Experimental ΔmixH: %{customdata[3]:.1f} J/mol<br>" +
        "ML prediction: %{customdata[4]:.2f} J/mol<br>" +
        "RKM prediction: %{customdata[5]:.2f} J/mol<extra></extra>",
    });
  }
  return traces;
}
