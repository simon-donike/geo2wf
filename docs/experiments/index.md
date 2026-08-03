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

## Modular model choices

| Choice | Scientific role | Baseline |
|---|---|---|
| `deterministic_residual` | Stage 1 physical correction around ERA5 | ERA5 |
| `residual_diffusion_deterministic_baseline` | Stage 2 probabilistic refinement | frozen Stage 1 checkpoint |
| `residual_diffusion` | residual-diffusion ablation | ERA5 |
| `conditional_diffusion` | standalone absolute-field diffusion control | none |

The two checked-in data choices are `geo_sar_common10_era5` and
`geo_sar_common4`. Model/data channel contracts must match; the common4 smoke
example documents its necessary conditional-diffusion width overrides.

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
| `config_geo_sar_10bands_era5_residual.yaml` | historical Stage 1 |
| `config_geo_sar_10bands_era5_diffusion_residual_deterministic.yaml` | historical Stage 2 |
| `config_geo_sar_10bands_era5_pmw_residual.yaml` | PMW-conditioned Stage 1 candidate |
| `config_geo_sar_10bands_era5_pmw_diffusion_residual_deterministic.yaml` | PMW-conditioned Stage 2 candidate |
| `config_pretrain_geo_pmw*.yaml` | GEO→PMW proxy pretraining |

## Comparability checks

- Keep the export root, filtering, channel order, normalization, and split policy constant.
- Modular data defaults keep test held out; historical presets may merge test into train.
- Stage 2 must use the exact Stage 1 checkpoint recorded in its resolved config/run manifest.
- Residual physical losses are not numerically comparable to normalized diffusion noise MSE.
- Record sampler, reverse-step count, guidance, ensemble size, and validation coverage.
- Compare Stage 2 against its exact frozen baseline and inspect individual members as well as ensemble summaries.

Continue to [configuration](configuration.md), [training](training.md), and
[evaluation](evaluation.md).
