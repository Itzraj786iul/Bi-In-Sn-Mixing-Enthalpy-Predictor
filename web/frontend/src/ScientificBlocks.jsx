import { useState } from "react";

export function Hero() {
  return (
    <header className="hero">
      <div className="hero-top">
        <p className="hero-eyebrow">BI–IN–SN / THERMODYNAMIC MODELING</p>
        <a className="hero-research-link" href="#research-provenance">
          Research
        </a>
      </div>
      <h1 className="hero-title">
        Mixing Enthalpy
        <br />
        of Liquid Bi–In–Sn
      </h1>
      <p className="hero-subtitle">
        An experimentally grounded computational framework for predicting and exploring integral
        molar mixing enthalpy.
      </p>
      <p className="hero-body">
        This interface evaluates the integral molar mixing enthalpy (ΔmixH, J/mol) of the liquid
        Bi–In–Sn alloy system from composition and temperature, using a Polynomial Degree-2 ML
        surrogate trained on 104 experimental calorimetry observations across three composition
        cross-sections at 767, 813, and 855 K.
      </p>
    </header>
  );
}

export function ValidationSnapshot() {
  return (
    <section className="metrics-strip" aria-label="Primary validation metrics">
      <div className="metric">
        <span className="metric-value mono">104</span>
        <span className="metric-label">Calorimetry observations</span>
      </div>
      <div className="metric-divider" aria-hidden="true" />
      <div className="metric">
        <span className="metric-value mono">41.10</span>
        <span className="metric-unit">J/mol</span>
        <span className="metric-label">LOCSO MAE</span>
      </div>
      <div className="metric-divider" aria-hidden="true" />
      <div className="metric">
        <span className="metric-value mono">56.34</span>
        <span className="metric-unit">J/mol</span>
        <span className="metric-label">LOCSO RMSE</span>
      </div>
      <div className="metric-divider" aria-hidden="true" />
      <div className="metric">
        <span className="metric-value mono">0.9712</span>
        <span className="metric-label">LOCSO R²</span>
      </div>
      <p className="metrics-note">
        Primary validation: leave-one-cross-section-out evaluation on 104 experimental
        observations.
      </p>
    </section>
  );
}

export function SectionHeader({ label, title, description }) {
  return (
    <div className="section-header">
      <p className="section-label">{label}</p>
      <h2 className="section-title">{title}</h2>
      {description && <p className="section-desc">{description}</p>}
    </div>
  );
}

export function ValidatedExploratoryBlocks() {
  return (
    <div className="status-blocks">
      <div className="status-block status-observed">
        <span className="status-tag">Experimentally observed</span>
        <p>
          Model performance is evaluated against the 104 experimental calorimetry observations
          using LOCSO.
        </p>
      </div>
      <div className="status-block status-exploration">
        <span className="status-tag">Surrogate exploration</span>
        <p>
          Predictions at unsampled ternary compositions are model estimates and are not
          experimentally validated.
        </p>
      </div>
    </div>
  );
}

export const CROSS_SECTIONS = [
  {
    id: "01",
    formula: (
      <>
        (Sn<sub>0.67</sub>Bi<sub>0.33</sub>)<sub>1−x</sub>In<sub>x</sub>
      </>
    ),
    label: "Sn-rich",
    swatchClass: "legend-circle",
  },
  {
    id: "02",
    formula: (
      <>
        (Sn<sub>0.50</sub>Bi<sub>0.50</sub>)<sub>1−x</sub>In<sub>x</sub>
      </>
    ),
    label: "Equiatomic",
    swatchClass: "legend-diamond",
  },
  {
    id: "03",
    formula: (
      <>
        (Sn<sub>0.33</sub>Bi<sub>0.67</sub>)<sub>1−x</sub>In<sub>x</sub>
      </>
    ),
    label: "Bi-rich",
    swatchClass: "legend-square",
  },
];

export function CrossSectionLegend() {
  return (
    <div className="cross-section-legend">
      <p className="cross-section-legend-title">Experimental cross-sections</p>
      <ul>
        {CROSS_SECTIONS.map((item) => (
          <li key={item.id}>
            <span className="cross-section-id mono">{item.id}</span>
            <span className={`legend-swatch ${item.swatchClass}`} aria-hidden="true" />
            <span className="cross-section-formula">{item.formula}</span>
            <span className="cross-section-name">{item.label}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function ModelValidationTable() {
  return (
    <div className="validation-table-wrap comparison-validation-strip">
      <p className="validation-table-caption">
        Primary experimental benchmark on 104 calorimetry observations.
      </p>
      <table className="validation-table">
        <thead>
          <tr>
            <th scope="col" />
            <th scope="col">ML surrogate</th>
            <th scope="col">RKM</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th scope="row">LOCSO MAE</th>
            <td className="mono">41.10 J/mol</td>
            <td className="mono">57.13 J/mol</td>
          </tr>
          <tr>
            <th scope="row">LOCSO RMSE</th>
            <td className="mono">56.34 J/mol</td>
            <td className="mono">73.49 J/mol</td>
          </tr>
          <tr>
            <th scope="row">LOCSO R²</th>
            <td className="mono">0.9712</td>
            <td className="mono">0.9510</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

export function ExperimentalCoveragePanel({ temperature, observationCount }) {
  return (
    <aside className="experimental-coverage" aria-label="Experimental coverage">
      <p className="experimental-coverage-label">Experimental coverage</p>
      <p className="experimental-coverage-count">
        At the selected temperature ({temperature} K):{" "}
        <span className="mono">{observationCount ?? "—"}</span> calorimetry observations
      </p>
      <ul className="experimental-coverage-list">
        {CROSS_SECTIONS.map((item) => (
          <li key={item.id}>
            <span className={`legend-swatch ${item.swatchClass}`} aria-hidden="true" />
            <span className="experimental-coverage-formula">{item.formula}</span>
            <span className="experimental-coverage-name">{item.label}</span>
          </li>
        ))}
      </ul>
      <p className="experimental-coverage-note">
        Measurements follow these cross-sections only; the full ternary simplex was not
        experimentally mapped.
      </p>
    </aside>
  );
}

const FROZEN_EQUATION = `ΔmixH =
  1477.018848
  − 1633.602767·xBi
  − 2198.688284·xIn
  − 2.629607·temperature_K
  − 1875.297539·xBi²
  − 9182.903143·xBi·xIn
  + 4.021166·xBi·temperature_K
  + 535.330326·xIn²
  + 2.025539·xIn·temperature_K
  + 0.000974·temperature_K²`;

export function FinalSurrogateSection() {
  const [expanded, setExpanded] = useState(false);

  return (
    <section className="surrogate-section" id="final-surrogate">
      <p className="section-label">Model / Final surrogate</p>
      <h2 className="section-title">Temperature-aware Polynomial Degree-2 model</h2>
      <div className="equation-block generic-equation">
        <p className="equation-label">Generic form</p>
        <pre className="equation-text">
          ΔmixH ={"\n"}
          β₀ + β₁xBi + β₂xIn + β₃T{"\n"}
          + β₄xBi² + β₅xBi·xIn{"\n"}
          + β₆xBi·T + β₇xIn²{"\n"}
          + β₈xIn·T + β₉T²
        </pre>
      </div>
      <ul className="surrogate-meta">
        <li>
          <span className="mono">10</span> fitted terms
        </li>
        <li>
          Features: <span className="mono">xBi</span>, <span className="mono">xIn</span>,{" "}
          <span className="mono">temperature_K</span>
        </li>
        <li>
          <span className="mono">xSn = 1 − xBi − xIn</span>
        </li>
      </ul>
      <button
        type="button"
        className="expand-btn"
        aria-expanded={expanded}
        onClick={() => setExpanded((open) => !open)}
      >
        {expanded ? "Hide fitted equation" : "View fitted equation"}
      </button>
      {expanded && (
        <div className="equation-block fitted-equation">
          <pre className="equation-text mono">{FROZEN_EQUATION}</pre>
          <p className="equation-disclaimer">
            These coefficients are empirical regression parameters and are not Redlich-Kister
            thermodynamic interaction parameters.
          </p>
        </div>
      )}
    </section>
  );
}

export function MethodologySection() {
  const steps = [
    { title: "Experiment", detail: "104 calorimetry observations" },
    { title: "Thermodynamic model", detail: "Redlich-Kister-Muggianu" },
    { title: "Synthetic data", detail: "RKM-generated training data" },
    { title: "Machine learning", detail: "Polynomial Degree-2 surrogate" },
    { title: "Validation", detail: "Leave-One-Cross-Section-Out" },
  ];

  return (
    <section className="methodology-section">
      <p className="section-label">Methodology</p>
      <h2 className="section-title">Computational pipeline</h2>
      <div className="pipeline">
        {steps.map((step, index) => (
          <div key={step.title} className="pipeline-item">
            {index > 0 && <div className="pipeline-arrow" aria-hidden="true" />}
            <div className="pipeline-step">
              <h3>{step.title}</h3>
              <p>{step.detail}</p>
            </div>
          </div>
        ))}
      </div>
      <p className="methodology-note">
        LOCSO tests generalization to an unseen experimental composition cross-section, providing
        a stricter evaluation than a random split when neighboring compositions are present.
      </p>
    </section>
  );
}

export function ScientificContextSection() {
  return (
    <section className="context-section">
      <p className="section-label">Scientific context</p>
      <h2 className="section-title">Why mixing enthalpy?</h2>
      <p>
        Integral molar mixing enthalpy characterizes the energetic effect of combining alloy
        constituents in the liquid state. For the Bi–In–Sn system, ΔmixH reflects interactions
        among bismuth, indium, and tin atoms during mixing and provides a thermodynamic descriptor
        relevant to alloy behavior along the measured composition cross-sections.
      </p>
    </section>
  );
}

const PAPER_DOI = "https://doi.org/10.1007/s11664-019-07646-0";

const PROVENANCE_CROSS_SECTIONS = [
  {
    formula: (
      <>
        (Sn<sub>0.67</sub>Bi<sub>0.33</sub>)<sub>1−x</sub>In<sub>x</sub>
      </>
    ),
    label: "Sn-rich",
  },
  {
    formula: (
      <>
        (Sn<sub>0.50</sub>Bi<sub>0.50</sub>)<sub>1−x</sub>In<sub>x</sub>
      </>
    ),
    label: "Equiatomic Bi/Sn",
  },
  {
    formula: (
      <>
        (Sn<sub>0.33</sub>Bi<sub>0.67</sub>)<sub>1−x</sub>In<sub>x</sub>
      </>
    ),
    label: "Bi-rich",
  },
];

const PROVENANCE_STEPS = [
  { num: "01", title: "Experimental calorimetry", detail: "104 observations from the source study." },
  { num: "02", title: "Thermodynamic model", detail: "Redlich-Kister-Muggianu representation." },
  {
    num: "03",
    title: "Synthetic data",
    detail: "Generated from the implemented RKM model for ML development.",
  },
  {
    num: "04",
    title: "ML surrogate",
    detail: (
      <>
        Polynomial Degree-2 regression using <span className="mono">xBi</span>,{" "}
        <span className="mono">xIn</span>, <span className="mono">temperature_K</span> with{" "}
        <span className="mono">xSn = 1 − xBi − xIn</span>.
      </>
    ),
  },
  {
    num: "05",
    title: "Validation",
    detail: "Leave-One-Cross-Section-Out validation on the experimental data.",
  },
];

export function ResearchProvenanceSection() {
  return (
    <section className="research-section" id="research-provenance">
      <p className="section-label">Research / Provenance</p>
      <h2 className="section-title">From experimental calorimetry to computational prediction</h2>
      <p className="research-intro">
        This application builds on experimental measurements of mixing enthalpy in the liquid
        Bi–In–Sn system and develops an experimentally validated machine-learning surrogate for
        interactive composition and temperature exploration.
      </p>

      <div className="provenance-block">
        <h3 className="provenance-label">Source study</h3>
        <p className="provenance-title">
          Measurements of Mixing Enthalpy for a Lead-Free Solder Bi-In-Sn System
        </p>
        <p className="provenance-authors">M.R. Kumar, S. Mohan, and C.K. Behera</p>
        <p className="provenance-meta">
          <em>Journal of Electronic Materials</em>, Vol. 48, No. 12 (2019), pp. 8096–8106
        </p>
        <p className="provenance-summary">
          The source study measured integral molar mixing enthalpy of liquid Bi–In–Sn by
          drop-solution calorimetry at 767 K, 813 K, and 855 K along three composition
          cross-sections. Partial quantities of indium mixing were integrated from heat-flow
          curves; integral molar mixing enthalpies were derived and analysed with a
          Redlich–Kister–Muggianu polynomial fit.
        </p>
      </div>

      <div className="provenance-block">
        <h3 className="provenance-label">Experimental data</h3>
        <p className="provenance-stat">
          <span className="mono provenance-stat-value">104</span>
          <span className="provenance-stat-caption">Calorimetry observations</span>
        </p>
        <ul className="provenance-list">
          <li>Three composition cross-sections:</li>
        </ul>
        <ul className="cross-section-list">
          {PROVENANCE_CROSS_SECTIONS.map((item) => (
            <li key={item.label}>
              <span className="cross-section-formula">{item.formula}</span>
              <span className="cross-section-name">{item.label}</span>
            </li>
          ))}
        </ul>
        <ul className="provenance-list">
          <li>
            Temperatures: <span className="mono">767 K</span>, <span className="mono">813 K</span>,{" "}
            <span className="mono">855 K</span>
          </li>
          <li>
            Target: integral molar mixing enthalpy ΔmixH (<span className="mono">J/mol</span>)
          </li>
        </ul>
      </div>

      <div className="provenance-block">
        <h3 className="provenance-label">Computational workflow</h3>
        <div className="provenance-workflow">
          {PROVENANCE_STEPS.map((step, index) => (
            <div key={step.num} className="provenance-workflow-item">
              {index > 0 && <div className="provenance-workflow-arrow" aria-hidden="true" />}
              <div className="provenance-workflow-step">
                <span className="provenance-step-num mono">{step.num}</span>
                <h4>{step.title}</h4>
                <p>{step.detail}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="provenance-block">
        <h3 className="provenance-label">Primary experimental validation</h3>
        <p className="provenance-subheading">Polynomial Degree-2 ML surrogate</p>
        <ul className="provenance-metrics">
          <li>
            LOCSO MAE: <span className="mono">41.10 J/mol</span>
          </li>
          <li>
            LOCSO RMSE: <span className="mono">56.34 J/mol</span>
          </li>
          <li>
            LOCSO R²: <span className="mono">0.9712</span>
          </li>
        </ul>
        <p className="provenance-subheading">RKM benchmark</p>
        <ul className="provenance-metrics">
          <li>
            MAE: <span className="mono">57.13 J/mol</span>
          </li>
          <li>
            RMSE: <span className="mono">73.49 J/mol</span>
          </li>
          <li>
            R²: <span className="mono">0.9510</span>
          </li>
        </ul>
        <p className="provenance-note">
          These metrics are evaluated against the experimental observations. They do not constitute
          validation of predictions at unsampled ternary compositions.
        </p>
      </div>

      <div className="provenance-block">
        <h3 className="provenance-label">Model transparency</h3>
        <p>
          The deployed ML surrogate is the frozen Polynomial Degree-2 model obtained from the
          project workflow. Its fitted equation is exposed in the{" "}
          <a href="#final-surrogate">Model / Final surrogate</a> section above for transparency.
        </p>
        <p>
          The polynomial coefficients are empirical regression parameters; they are not
          Redlich-Kister thermodynamic interaction parameters.
        </p>
      </div>

      <div className="provenance-block provenance-limitations">
        <h3 className="provenance-label">Scope / Limitations</h3>
        <ul className="provenance-list">
          <li>The experimental dataset provides the validation basis.</li>
          <li>
            LOCSO evaluates generalization to an unseen composition cross-section.
          </li>
          <li>
            The full ternary maps contain compositions that were not directly measured in the
            experimental study.
          </li>
          <li>
            Predictions in those unsampled regions are surrogate estimates.
          </li>
          <li>
            Differences between ML and RKM in unsampled regions indicate model disagreement and
            should not be interpreted as experimentally established thermodynamic behavior.
          </li>
        </ul>
      </div>

      <div className="provenance-block provenance-reference">
        <h3 className="provenance-label">Reference</h3>
        <p className="citation-text">
          M.R. Kumar, S. Mohan, and C.K. Behera, &ldquo;Measurements of Mixing Enthalpy for a
          Lead-Free Solder Bi-In-Sn System,&rdquo; <em>Journal of Electronic Materials</em>, Vol. 48,
          No. 12 (2019), pp. 8096–8106.
        </p>
        <p>
          <a href={PAPER_DOI} target="_blank" rel="noopener noreferrer">
            Source study (DOI)
          </a>
        </p>
      </div>
    </section>
  );
}

export function SiteFooter() {
  return (
    <footer className="site-footer">
      <p className="site-footer-title">Bi–In–Sn Mixing Enthalpy Predictor</p>
      <p className="site-footer-tagline">Experimental thermodynamics × computational modeling</p>
    </footer>
  );
}

export function mixingTypeLabel(mixingType) {
  if (mixingType === "Exothermic") return "EXOTHERMIC";
  if (mixingType === "Endothermic") return "ENDOTHERMIC";
  return "APPROXIMATELY ZERO";
}
