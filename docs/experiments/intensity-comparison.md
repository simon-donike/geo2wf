# Model presets

Choose a preset for the prediction you need. Each preset can be adapted with
Hydra overrides for data paths, hardware, and training settings.

| Model | With ERA5 | Without ERA5 |
|---|---|---|
| Wind-field U-Net | `intensity_comparison_unet` | `intensity_comparison_unet_no_era5` |
| Joint wind field and maximum wind | `bottleneck_unet_mlp` | `bottleneck_unet_mlp_no_era5` |
| Joint wind field, maximum wind, and radii | `latent_mlp_sar_era5_max_wind_radii` | `latent_mlp_sar_no_era5_max_wind_radii` |
| Scalar wind and radii without SAR supervision | `latent_mlp_no_sar_era5_max_wind_radii` | `latent_mlp_no_sar_no_era5_max_wind_radii` |

```bash
uv run geo2wf-train experiment=bottleneck_unet_mlp \
  data.root=/path/to/geo_sar \
  data.stats_file=/path/to/geo_sar/stats.json
```

The no-ERA5 presets retain the ERA5-available data filter used in the original
experiments. Set `data.require_era5=false` to use observations without an ERA5
companion when the model does not consume ERA5 inputs.

A post-hoc intensity head uses `experiment=unet_intensity_correction` and a
cache of predictions from a frozen field model. See the
[intensity correction guide](../models/intensity-correction.md).
For six-hour scalar forecasts, use `experiment=intensity_forecast_pretrain`;
see the [forecast guide](../models/intensity-forecast.md).

See [configuration](configuration.md) for overrides,
[training](training.md) for checkpoint loading, and
[published results](../results.md) for model comparisons.
