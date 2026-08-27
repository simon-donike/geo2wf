# Archived direct PMW U-Net

!!! archive "Retired experiment family"
    This proxy model is preserved for provenance and is not part of the active
    experiment matrix.

`DirectUNetRegressor` is a deterministic image-to-image control for the
GEO→near-89 GHz passive-microwave proxy task. It predicts brightness
temperature, not surface wind.

## Data and target contract

The maintained experiment uses a one-channel `TB_near89V` target in kelvin.
AMSR2, GMI, SSMIS, ATMS, and MHS source channels are harmonized to that
canonical name by the PMW exporter. Pixels are supervised only where the
regridded swath is valid.

The common10 + ERA5 experiment supplies 23 data-condition channels:

```text
10 GEO bands
 9 ERA5 source + derived fields
 1 distance-to-center field
 3 solar-time fields
──
23 condition channels
```

The model appends the condition-validity mask, so the U-Net receives 24 input
channels. The PMW experiment keeps the full 256 × 256 export instead of the
192 × 192 crop used by the principal GEO–SAR configs.

## Architecture and objective

The backbone reuses the compact residual U-Net architecture from Stage 1:
GroupNorm/SiLU residual blocks, strided-convolution downsampling, bilinear
upsampling, and encoder skips. A sigmoid bounds the normalized output to
`[0,1]`; the loader's target offset and scale then convert it back to kelvin.

Training minimizes masked Huber loss in physical kelvin with a default delta of
2 K. Validation reports MAE, RMSE, bias, and Huber loss in kelvin. Both the
learning-rate scheduler and checkpoint selection monitor `val/rmse_k`.

## Train

```bash
uv run geo2wf-train experiment=geo_pmw_near89_unet
```

The GEO-only comparison uses the same eligible ERA5-filtered cohort but omits
ERA5 from the condition:

```bash
uv run geo2wf-train experiment=geo_pmw_near89_unet_no_era5
```

The direct model validates `target_units == "K"` at startup. It cannot be
pointed at a SAR wind dataset without implementing a different model contract.
For export details and sensor-channel mapping, see [PMW proxy
pretraining](../data/export-geo-pmw.md).
