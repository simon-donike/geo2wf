# Stage 2: residual diffusion

`ERA5ResidualDiffusion` keeps diffusion while changing the generated variable from absolute wind speed to a signed correction around a dense baseline.

In the main two-stage system, that baseline is the frozen output of the deterministic Stage 1 model:

\[
r = v_{\mathrm{SAR}} - v_{\mathrm{base}}
\]

[Read the complete Stage 1 → Stage 2 handoff first.](two-stage.md)

The implementation can also use ERA5 directly as a portable ablation, but the deterministic checkpoint is the intended stacked workflow.

## Residual transform

Residuals concentrate near zero but have a rare intense positive tail. A linear mapping would compress useful small corrections. The model uses an odd, invertible asinh transform:

\[
z = \frac{\operatorname{asinh}(r/s)}{\operatorname{asinh}(c/s)},
\quad r \in [-c,c]
\]

The checked-in preset uses `s = 5 m/s` and `c = 80 m/s`. Zero remains exactly zero, small corrections receive useful resolution, and the full tail maps into diffusion space `[-1,1]`.

## Inputs, masks, and output

On observed pixels, the target is the transformed physical SAR-minus-baseline residual. Outside the SAR swath, the target is zero residual with a weak configured weight. Invalid baseline pixels receive zero loss.

The denoiser receives:

```text
1 noisy residual
+ 24 prepared GEO / ERA5 / geometry / solar / mask channels
+ 1 exact frozen baseline
+ 1 baseline-valid mask
= 27 U-Net input channels
```

Sampling inverts the residual transform, adds the result to the same baseline in m/s, applies physical wind bounds, and only then maps to normalized wind for image and storm-structure metrics.

Validation reports `baseline_mae_ms` and `mae_skill_vs_baseline`, measuring the sampled refinement directly against the frozen field it is supposed to improve.

## Probabilistic refinement

The deterministic-baseline preset keeps epsilon diffusion as the generative objective and adds:

- Min-SNR weighting across noise levels;
- separately normalized SAR and off-swath losses;
- extra emphasis for inner-core and high-wind pixels;
- weak gradient, spectrum, low-frequency, and total-variation losses at low-to-medium noise; and
- classifier-free guidance through condition dropout.

The spectrum term compares amplitude rather than phase. The low-frequency term keeps members tied to the broad baseline, while total variation suppresses pixel-scale ringing in the correction without smoothing Stage 1 itself.

Ten percent condition dropout trains an unconditional branch without changing U-Net shape. The preset starts sampling at `guidance_scale: 1.5`; higher values generally favor condition fidelity, while lower values preserve more diversity.

## Validation ensemble

Validation uses four stable latent members on its first reconstruction batch and reports:

- CRPS and ensemble spread;
- pairwise diversity;
- ensemble-mean and best-member MAE;
- gradient sharpness ratio;
- log-spectrum error; and
- `probabilistic_refinement_score`.

Checkpoints use the composite score. Inspect individual members when judging sharpness because averaging plausible alternatives is expected to blur them.

## Train Stage 2

```bash
GEO2WF_BASELINE_CKPT=/path/to/deterministic.ckpt \
  python train.py \
  --config configs/config_geo_sar_10bands_era5_diffusion_residual_deterministic.yaml
```

The deterministic module is loaded as a frozen child, kept in evaluation mode, excluded from the optimizer, and saved with the residual-diffusion checkpoint.

For the ERA5-only ablation:

```bash
python train.py \
  --config configs/config_geo_sar_10bands_era5_diffusion_residual.yaml
```

Continue to [Sampling](sampling.md) or [Evaluation](../experiments/evaluation.md).
