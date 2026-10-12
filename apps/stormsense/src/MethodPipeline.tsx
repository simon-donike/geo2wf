import { useState } from "react";
import { stamp } from "./data";
import { showcaseManifest } from "./showcaseData";
import "./pipeline.css";

const record = showcaseManifest.recorded_prediction;
const stages = [
  [
    "GOES infrared",
    "Retrieve the observation",
    "Retrieve the available ABI infrared scan and its physical temperature channels. These are the scientific inputs, separate from the GeoColor display image.",
    `GOES-${record.imagery.satellite} · acquired ${stamp(record.imagery.end)}`,
  ],
  [
    "Preprocessing",
    "Align and normalize",
    "Reproject to the storm-centered grid, crop to 192 × 192 pixels, check validity, normalize the ten infrared channels, and add distance, solar context, and the validity plane.",
    `${(record.imagery.valid_fraction * 100).toFixed(0)}% valid coverage in this recorded crop`,
  ],
  [
    "Model inference",
    "Estimate surface winds",
    "This Nolo hindcast uses an encoder-only checkpoint to estimate maximum wind and wind radii. Ian’s separate Method example shows how the joint architecture also produces a dense wind field.",
    record.model_version,
  ],
  [
    "Saved estimates",
    "Record the result",
    "Store estimates, source references, model version, and generation time in the runner’s SQLite record. This example was computed retrospectively; acquisition and generation are different times.",
    `Generated ${stamp(record.generated_at)}`,
  ],
  [
    "Data release",
    "Publish immutable assets",
    "Export the numerical record and display imagery. Upload and verify the immutable assets in R2 before advancing the latest-release pointer.",
    `Pinned source release: ${showcaseManifest.source_release}`,
  ],
  [
    "Website",
    "Explore the published record",
    "The browser reads published data through a read-only endpoint. Maps, charts, and the timeline share a selected time. Serving the website does not require the inference runner to remain online.",
    "This showcase reads a bundled copy of the historical release.",
  ],
];

export default function MethodPipeline() {
  const [selected, setSelected] = useState(0);
  const [name, action, description, evidence] = stages[selected];
  return (
    <div className="method-pipeline" data-tour="pipeline">
      <div className="pipeline-intro">
        <span className="eyebrow">
          Recorded Nolo hindcast · 28 September 2026
        </span>
        <h2>From observation to a published estimate.</h2>
        <p>
          Select a stage to follow the data. The flow animation explains the
          process; it is not a live job monitor.
        </p>
      </div>
      <ol
        className="pipeline-stages"
        aria-label="Inference and publication stages"
      >
        {stages.map(([title, subtitle], i) => (
          <li key={title}>
            <button
              aria-pressed={selected === i}
              onClick={() => setSelected(i)}
              aria-controls="pipeline-stage-detail"
            >
              <span className="pipeline-number">
                {String(i + 1).padStart(2, "0")} <i aria-hidden="true">→</i>
              </span>
              <strong>{title}</strong>
              <small>{subtitle}</small>
            </button>
          </li>
        ))}
      </ol>
      <section
        id="pipeline-stage-detail"
        className="pipeline-detail"
        aria-live="polite"
        aria-label="Selected pipeline stage"
      >
        <span className="eyebrow">{name}</span>
        <h3>{action}</h3>
        <p>{description}</p>
        <small>{evidence}</small>
      </section>
      <div className="pipeline-display-branch">
        <span className="eyebrow">Separate display path</span>
        <p>
          <strong>NASA GIBS GeoColor</strong> <span aria-hidden="true">→</span>{" "}
          saved map images <span aria-hidden="true">→</span> website backdrop
        </p>
        <small>
          GeoColor does not feed the model. Its acquisition time is recorded
          separately from the ABI inference scan.
        </small>
      </div>
      <p className="pipeline-footnote">
        Training uses SAR and observed storm properties as supervision.
        Inference uses infrared imagery and context. The recorded example is a
        hindcast, not evidence of prospective forecast skill.
      </p>
    </div>
  );
}
