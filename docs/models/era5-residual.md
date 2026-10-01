# Field U-Net

`ERA5ResidualRegressor` reconstructs one surface wind-speed field. With ERA5,
it predicts a physical correction to the reanalysis anchor:

\[
\hat v = v_{\mathrm{ERA5}} + f_\theta(x, m).
\]

Without ERA5, the same model family predicts absolute wind. Scalar intensity
and radii diagnosed from this image provide the field-only baseline in the
[paper comparison](../results.md#architecture-ablation).

## Inputs and architecture

The ERA5 configuration receives 23 data channels: ten GEO, nine ERA5
source/derived, storm-center distance, and three solar-time fields. The model
appends condition validity, target-normalized ERA5 wind, and ERA5 validity,
giving **26 U-Net inputs**. The GEO-only model receives 14 data channels plus
the condition mask.

The residual U-Net uses GroupNorm, SiLU, strided downsampling, bilinear
upsampling, and encoder skips. Its zero-initialized residual head initially
reproduces ERA5. Predictions are clamped nonnegative with no default upper cap.

## Objective and diagnostics

The field-only presets use Huber loss in physical m/s, with a 2 m/s transition.
ERA5 runs supervise the common SAR/ERA5-valid footprint. High-wind weights
increase smoothly from 1× to 8× across 25–50 m/s. A robust top-0.5% inner-core
peak term has weight 0.05. A separate 0.05-weighted off-swath term discourages
unsupported corrections where ERA5 is valid; it is disabled without ERA5.
Radial-profile and exceedance-area loss weights are zero in the default.

Validation reports physical errors, high-wind errors, structural diagnostics,
and ERA5 skill where applicable. The default scheduler monitors
`val/peak_structure_score`, while the experiment selects checkpoints by
`val/eye_structure_score`. These are different from the training objective;
inspect the resolved config when comparing runs.

## Train with ERA5

```bash
uv run geo2wf-train experiment=intensity_comparison_unet \
  data.root=/path/to/paired \
  data.ibtracs_file=/path/to/ibtracs.ALL.list.v04r01.csv
```

This experiment uses the joint data module to keep the cohort aligned with
scalar comparisons; it still learns only the field.

## Train without ERA5

Use `experiment=intensity_comparison_unet_no_era5` with the same local paths.
This disables ERA5 inputs, residual addition, off-swath anchoring, and ERA5
comparison metrics, while retaining the cohort-availability filter.

For local checkpoint use, see [commands](../reference/commands.md); for metric
interpretation, see [evaluation](../experiments/evaluation.md).
