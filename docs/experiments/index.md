# Choose an experiment

The modular workflow composes a data choice and model choice. Start with the
maintained field reconstruction:

```bash
uv run geo2wf-train \
  data=geo_sar_common10_era5 \
  model=deterministic_residual
```

## Principal model choices

| Choice | Scientific role | Baseline |
|---|---|---|
| `deterministic_residual` | physical wind-field correction around ERA5 | ERA5 |
| `direct_unet` | direct near-89 GHz PMW brightness-temperature proxy | none |
| `bottleneck_unet_mlp` | joint field and scalar-intensity estimation | none |
| `unet_encoder_mlp_ibtracs` | encoder-only IBTrACS intensity estimation | none |
| `intensity_correction` | scalar correction from a frozen U-Net field | frozen U-Net checkpoint |
| `intensity_forecast` | six-hour scalar forecast | current intensity and recent history |

Data choices cover GEO–SAR reconstruction, GEO–PMW proxy training, joint
field–intensity training, single-field correction, and intensity forecasting.
List `configs/data/*.yaml` for the current set. Model and data contracts must
match; incompatible pairs are rejected by `DataSpec` validation.

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
| `configs/config_geo_sar_10bands_era5_pmw_residual.yaml` | PMW-conditioned Stage 1 candidate |
| `configs/v1/config_pretrain_geo_pmw*.yaml` | GEO→PMW proxy pretraining |

## Comparability checks

- Keep the export root, filtering, channel order, normalization, and split policy constant.
- Modular data defaults keep test held out; historical presets may merge test into train.
- Record validation coverage and the exact comparison baseline.

Continue to [configuration](configuration.md), [training](training.md), and
[evaluation](evaluation.md).
