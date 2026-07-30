# Residual diffusion

`ERA5ResidualDiffusion` keeps diffusion while changing the generated variable
from absolute wind speed to a signed correction around a dense baseline.

\[
r = v_{SAR} - v_{base}
\]

The baseline is either ERA5 wind speed or the prediction from a frozen
`ERA5ResidualRegressor` checkpoint. The latter is the intended stacked model:
the deterministic network commits to one broad reconstruction and diffusion
models the remaining structure.

## Residual transform

Residuals are strongly concentrated near zero but have a rare intense positive
tail. A linear 0--80 m/s target mapping would compress useful corrections into
a very narrow interval. The model therefore uses an odd, invertible asinh map:

\[
z = \frac{\operatorname{asinh}(r/s)}{\operatorname{asinh}(c/s)},
\quad r \in [-c,c]
\]

The checked-in preset uses `s = 5 m/s` and `c = 80 m/s`. Zero remains exactly
zero, small corrections receive useful resolution, and the full intense tail
still maps into diffusion space `[-1,1]`.

## Masks and conditioning

On observed pixels, the target is the transformed physical SAR-minus-baseline
residual. Outside the SAR swath, the target is zero residual with the configured
weak weight. Invalid baseline pixels receive zero loss.

The denoiser receives:

```text
1 noisy residual
+ 24 existing prepared GEO/ERA/solar/mask channels
+ 1 exact baseline in target normalization
+ 1 baseline-valid mask
= 27 U-Net input channels
```

Sampling inverts the residual transform, adds the result to the same baseline
in m/s, applies physical wind bounds, and only then maps to normalized wind for
the existing image and eye-structure metrics.
 Validation additionally logs `baseline_mae_ms` and
`mae_skill_vs_baseline`, so a deterministic stack is measured directly against
the frozen reconstruction it is meant to improve.

## Probabilistic refinement

The deterministic-baseline preset keeps epsilon diffusion as the generative
objective and adds controls aimed at coherent, plausible samples:

- Min-SNR weighting prevents easy, nearly clean timesteps from dominating.
- Observed SAR and off-swath zero-residual losses are normalized separately, so
  swath area does not change the configured anchor strength.
- Inner-core and high-wind pixels receive extra noise-loss weight; high-gradient
  pixels receive only a small additional emphasis.
- At low and medium noise levels, the clean residual estimate receives weak
  gradient-magnitude, log-spectrum, low-frequency consistency, and total-variation
  losses.

The spectrum term compares amplitude rather than phase. The low-frequency term
keeps samples tied to the broad deterministic reconstruction, while the weak
total-variation term suppresses pixel-scale ringing in the correction without
smoothing the deterministic baseline itself.

Ten percent condition dropout trains an unconditional branch without changing
the U-Net shape. Sampling then uses classifier-free guidance; the preset starts
at `guidance_scale: 1.5`. Higher guidance generally favors condition fidelity
and lower guidance preserves more diversity.

Validation uses four stable latent members on its first reconstruction batch.
It reports CRPS, spread, pairwise diversity, ensemble-mean and best-member MAE,
gradient sharpness ratio, log-spectrum error, and a composite
`probabilistic_refinement_score`. Checkpoints use the composite score. Inspect
individual members rather than the ensemble mean when judging sharpness,
because averaging valid alternatives is expected to blur them.
The preset targets a gradient ratio of `0.9`, so checkpoint selection favors a
slightly smoother field than the SAR observation.

## Configuration

The portable ERA5-baseline preset is:

```bash
python train.py --config configs/config_geo_sar_10bands_era5_diffusion_residual.yaml
```

To stack diffusion on a trained deterministic model, use the dedicated preset:

```bash
GEO2WF_BASELINE_CKPT=/path/to/deterministic.ckpt \
  python train.py --config configs/config_geo_sar_10bands_era5_diffusion_residual_deterministic.yaml
```

The same path can be written into
`model.residual.baseline.checkpoint_path` instead. The deterministic module is loaded as a frozen child,
kept in evaluation mode, excluded from the optimizer, and saved with the
residual-diffusion checkpoint for reproducibility.
