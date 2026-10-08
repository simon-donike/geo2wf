import type { ReactNode } from "react";
import { stamp, wind } from "./data";
import type { MethodExample, MethodMode } from "./methodExample";

// Sources: pinned joint resolved-config.yaml + BottleneckUNetMLPRegressor;
// historical ScalarAdapter.loss_terms/training_step + pinned encoder config.json.
export default function MethodTraining({
  mode,
  network,
  example,
  unit,
}: {
  mode: MethodMode;
  network: ReactNode;
  example: MethodExample;
  unit: "kt" | "m/s";
}) {
  const joint = mode === "joint";
  const pair = example.training;
  return (
    <div className="method-training">
      <div className="method-training-intro">
        <h2>Learn from the difference.</h2>
        <p>
          Predict, compare with observations, then send the error backward to
          update the model.
        </p>
        <span>TRAINING SCHEMATIC · REPEATED ACROSS TRAINING BATCHES</span>
      </div>
      <div className="method-training-grid">
        <div className="method-training-model">
          <div className="method-part-label">
            <span>01</span> PREDICT
          </div>
          <div className="method-training-input">
            Satellite bands + center / time context{" "}
            <span aria-hidden="true">→</span>
          </div>
          <div
            className="method-training-flow-key"
            aria-label="Animation legend"
          >
            <span>
              <i /> Forward prediction
            </span>
            <span>
              <i /> Backward gradients
            </span>
          </div>
          {network}
          <p>
            Labels enter the loss; the network receives imagery and context.
          </p>
        </div>
        <div className="method-training-comparison">
          <div className="method-part-label">
            <span>02</span> COMPARE WITH OBSERVATIONS
          </div>
          {joint && (
            <article className="method-loss-card field-loss">
              <div className="method-training-fields">
                <figure>
                  <figcaption>
                    SAR wind field <small>Observed</small>
                  </figcaption>
                  <div className="method-training-raster">
                    <img
                      src={pair.sar_image}
                      alt="Real SAR-derived wind-speed field for Hurricane Ian, aligned to the prediction grid"
                    />
                  </div>
                </figure>
                <figure>
                  <figcaption>
                    Predicted wind field <small>Joint model</small>
                  </figcaption>
                  <div className="method-training-raster">
                    <img
                      src={pair.prediction_image}
                      alt="Joint model predicted wind-speed field on the same grid and color scale as SAR"
                    />
                  </div>
                </figure>
              </div>
              <div
                className="method-colorbar"
                style={{
                  background: `linear-gradient(to right, ${pair.palette.join(",")})`,
                }}
              />
              <div className="method-scale">
                <span>0</span>
                <span>Wind speed · {unit}</span>
                <span>{wind(pair.max_ms, unit)}</span>
              </div>
              <p className="method-training-pair-note">
                Ian · {stamp(pair.sar_time)} · {pair.sar_sensor}. Same grid and
                color scale; hatching marks excluded pixels. Held-out example,
                shown for comparison.
              </p>
              <div className="method-loss-result">
                <strong>
                  L<sub>field</sub>
                </strong>
                <span>Masked Huber · δ = 2 m/s</span>
              </div>
              <div className="method-training-gradient-map">
                <div className="method-training-raster">
                  <img
                    src={pair.gradient_image}
                    alt="Actual field-loss gradient with respect to predicted wind speed; cyan means increase and orange means decrease"
                  />
                </div>
                <div>
                  <strong>Where the error pushes back</strong>
                  <p>
                    Actual ∂L<sub>field</sub>/∂prediction, over{" "}
                    {pair.valid_pixels.toLocaleString("en-US")} valid pixels.
                  </p>
                  <div className="method-gradient-signs">
                    <span>← Increase wind</span>
                    <span>Decrease wind →</span>
                  </div>
                  <div
                    className="method-colorbar"
                    style={{
                      background: `linear-gradient(to right, ${pair.gradient_palette.join(",")})`,
                    }}
                  />
                </div>
              </div>
            </article>
          )}
          <article className="method-loss-card">
            <div className="method-loss-pair">
              <span>MLP maximum wind</span>
              <b aria-hidden="true">↔</b>
              <span>Best-track maximum wind</span>
            </div>
            <div className="method-loss-result">
              <strong>
                L<sub>wind</sub>
              </strong>
              <span>Huber · δ = 5 m/s</span>
            </div>
            <p>
              Compare the direct scalar prediction with the matched best-track
              intensity.
            </p>
          </article>
          <article className="method-loss-card">
            <div className="method-loss-pair">
              <span>MLP storm structure</span>
              <b aria-hidden="true">↔</b>
              <span>Best-track structure</span>
            </div>
            <div className="method-loss-result">
              <strong>
                L<sub>structure</sub>
              </strong>
              <span>Smooth L1 · β = 20 km</span>
            </div>
            <p>
              Compare available size and radius labels; missing labels
              contribute no error.
            </p>
          </article>
          <small className="method-training-source">
            {joint
              ? "Joint supervision: SAR + IBTrACS best track."
              : "Encoder fine-tuning: ATCF best track. No field loss or decoder."}
          </small>
        </div>
      </div>
      <div className="method-training-update">
        <div className="method-part-label">
          <span>03</span> UPDATE THE WEIGHTS
        </div>
        <div
          className="method-training-equation"
          aria-label={
            joint
              ? "Total loss equals field loss plus wind loss plus 0.25 times structure loss"
              : "Total loss equals wind loss plus 0.25 times structure loss"
          }
        >
          <span>L = </span>
          {joint && (
            <>
              <span className="field-term">
                L<sub>field</sub>
              </span>
              <span> + </span>
            </>
          )}
          <span>
            L<sub>wind</sub>
          </span>
          <span> + </span>
          <span>
            0.25 L<sub>structure</sub>
          </span>
        </div>
        <div className="method-gradient-return" aria-hidden="true">
          <i />
          <span>← GRADIENTS BACK TO THE MODEL</span>
        </div>
        <p>
          {joint
            ? "The field loss updates the decoder and shared encoder. Scalar losses update the MLP and shared encoder."
            : "Scalar losses update both the MLP and encoder. This fine-tuning starts from the joint model’s encoder and scalar weights."}{" "}
          AdamW applies the update; the next batch repeats the cycle.
        </p>
      </div>
      <details className="method-training-details">
        <summary>Exact loss definitions</summary>
        <div>
          <p>
            Let e = prediction − target. Huber uses ½e² when |e| ≤ δ, otherwise
            δ(|e| − ½δ). Smooth L1 uses ½e²/β when |e| &lt; β, otherwise |e| −
            ½β. These two losses have different scaling.
          </p>
          <p>
            {joint && "Field loss averages Huber over valid pixels. "}Wind loss
            averages Huber over the batch. Structure loss averages Smooth L1
            over all valid structure entries, then receives a weight of 0.25.
          </p>
          <p>
            Structure outputs are eye size, RMW, R34, R50 and R64, in
            kilometres. Wind radii use equivalent-area radii from quadrant
            labels.{" "}
            {joint
              ? "Available IBTrACS structure labels supervise the joint model."
              : "The ATCF fine-tuning dataset has no eye-size labels, so that output is masked. Below-threshold wind radii are explicitly labelled zero; unavailable radii are masked."}
          </p>
          <p>
            Losses use physical units: m/s for wind and km for structure,
            regardless of the display unit. The weights and thresholds shown
            match the two example checkpoints. The SAR comparison and
            output-gradient map are computed from the held-out Ian pair. The
            pixel derivative is clip(prediction − target, −δ, δ) divided by the
            number of valid pixels; excluded pixels have zero gradient. The map
            uses a symmetric ±δ / valid-pixel-count scale. Animated arrows
            illustrate backpropagation routes, not measured parameter-gradient
            magnitudes. No weights are updated here.
          </p>
        </div>
      </details>
    </div>
  );
}
