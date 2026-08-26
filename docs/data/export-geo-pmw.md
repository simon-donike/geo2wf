# PMW proxy pretraining

`geo2wf-export geo-pmw` creates a larger GEO-to-PMW paired dataset with the same one-channel target shape as GEO-to-SAR. The intent is initialization on a related satellite image reconstruction task before SAR fine-tuning.

## Supported PMW channels

All platforms are harmonized to the canonical one-channel target `TB_near89V`; the source channel remains in `pmw_source_channel` provenance.

| Platforms | Source channel | Swath |
|---|---|---|
| AMSR2 GCOM-W1 | `TB_A89.0V` | `S5` |
| GMI GPM | `TB_89.0V` | `S1` |
| SSMIS F16/F17/F18 | `TB_91.665V` | `S4` |
| ATMS NPP/NOAA-20/NOAA-21 | `TB_88.2QV` | `S3` |
| MHS Metop-B/Metop-C/NOAA-19 | `TB_89.0V` | `S1` |

Statistics are pooled under `TB_near89V`, producing one sensor-independent Kelvin scale. Each target remains **one channel**.

## Export

=== "Four GEO bands"

    ```bash
    uv run geo2wf-export geo-pmw \
      --config configs/v1/config_pretrain_geo_pmw.yaml
    ```

=== "Ten GEO bands"

    ```bash
    uv run geo2wf-export geo-pmw \
      --config configs/v1/config_pretrain_geo_pmw_10bands.yaml
    ```

=== "Ten bands + ERA5"

    ```bash
    uv run geo2wf-export geo-pmw \
      --config configs/v1/config_pretrain_geo_pmw_10bands_era5.yaml
    ```

The exporter is PMW-anchored: it finds a closest GEO occurrence for each acceptable PMW observation, builds the same geographic crop contract as the SAR exporter, and writes generic `condition_path` and `target_path` columns. This is why `PairedImageDataset` can load either task without a separate class.

## Direct near-89 GHz regression

Export the 10-band GEO + ERA5 dataset with linearly interpolated PMW supervision:

```bash
uv run geo2wf-export geo-pmw \
  --config configs/export/geo_pmw_near89_common10_era5.yaml
```

The output is a 256 x 256 EPSG:4326 grid at 0.027 degrees per pixel. Linear interpolation is limited to the native swath convex hull and pixels within 1.5 source spacings of a real footprint, so loss never uses extrapolated or gap-bridged values. The loader adds ERA5 wind speed and relative vorticity plus storm-distance and solar-time fields, yielding 23 condition fields. The direct U-Net appends one validity mask internally.

Train the deterministic experiment with:

```bash
uv run geo2wf-train experiment=geo_pmw_near89_unet
```

It predicts bounded normalized brightness temperature, optimizes masked Huber loss in Kelvin, and checkpoints on `val/rmse_k`.
See [Direct PMW U-Net](../models/direct-unet.md) for its exact tensor,
architecture, and unit contract.

## Moving from pretraining to SAR

Transfer is explicit: use `--weights-only-path` so model weights load strictly
while optimizer, scheduler, epoch, and step state start fresh.

```bash
uv run geo2wf-train \
  model=conditional_diffusion \
  --weights-only-path /path/to/pmw-pretraining.ckpt
```

The model architecture and condition/target widths must match exactly.

Evaluate proxy pretraining against training from scratch with identical SAR
data, seed, normalization, and evaluation.
