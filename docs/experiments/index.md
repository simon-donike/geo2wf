# Choose an experiment

The modular workflow composes a data choice and model choice. Start with the
main two-stage pair:

```bash
# Stage 1
uv run geo2wf-train \
  data=geo_sar_common10_era5 \
  model=deterministic_residual

# Stage 2
GEO2WF_BASELINE_CKPT=/path/to/stage1.ckpt \
uv run geo2wf-train \
  data=geo_sar_common10_era5 \
  model=residual_diffusion_deterministic_baseline
```

## Principal model choices

| Choice | Scientific role | Baseline |
|---|---|---|
| `deterministic_residual` | Stage 1 physical correction around ERA5 | ERA5 |
| `residual_diffusion_deterministic_baseline` | Stage 2 probabilistic refinement | frozen Stage 1 checkpoint |
| `residual_diffusion` | residual-diffusion ablation | ERA5 |
| `conditional_diffusion` | standalone absolute-field diffusion control | none |
| `direct_unet` | direct wind-field control | none |
| `bottleneck_unet_mlp` | joint field and scalar-intensity estimation | none |
| `intensity_correction` | scalar correction from a frozen U-Net field | frozen U-Net checkpoint |
| `intensity_forecast` | six-hour scalar forecast | current intensity and recent history |

Data choices cover GEO–SAR reconstruction, GEO–PMW proxy training, joint
field–intensity training, single-field correction, and intensity forecasting.
List `configs/data/*.yaml` for the current set. Model and data contracts must
match; the common4 smoke example specifies its conditional-diffusion widths.

## Experiment overrides

`experiment=ablations/stage1_peak_aware` applies only the selected Stage 1
objective differences on top of the model/data/trainer groups:

```bash
uv run geo2wf-train \
  model=deterministic_residual \
  experiment=ablations/stage1_peak_aware
```

New ablations should be similarly short instead of copying full configs.

## Historical presets

Complete `configs/config*.yaml` and `configs/v1/*.yaml` files preserve past
runs, PMW candidates, proxy pretraining, and exact checkpoint reproduction.
They remain launchable with `geo2wf-train --config ...`, but cannot be mixed
with Hydra overrides. Prefer adding missing grouped choices when starting a new
experiment.

Notable compatibility presets include:

| Full YAML | Purpose |
|---|---|
| `configs/config_geo_sar_10bands_era5_residual.yaml` | historical Stage 1 |
| `configs/config_geo_sar_10bands_era5_diffusion_residual_deterministic.yaml` | historical Stage 2 |
| `configs/config_geo_sar_10bands_era5_pmw_residual.yaml` | PMW-conditioned Stage 1 candidate |
| `configs/config_geo_sar_10bands_era5_pmw_diffusion_residual_deterministic.yaml` | PMW-conditioned Stage 2 candidate |
| `configs/v1/config_pretrain_geo_pmw*.yaml` | GEO→PMW proxy pretraining |

## Comparability checks

- Keep the export root, filtering, channel order, normalization, and split policy constant.
- Modular data defaults keep test held out; historical presets may merge test into train.
- Stage 2 must use the exact Stage 1 checkpoint recorded in its resolved config/run manifest.
- Residual physical losses are not numerically comparable to normalized diffusion noise MSE.
- Record sampler, reverse-step count, guidance, ensemble size, and validation coverage.
- Compare Stage 2 against its exact frozen baseline and report individual-member and ensemble summaries.

Continue to [configuration](configuration.md), [training](training.md), and
[evaluation](evaluation.md).
