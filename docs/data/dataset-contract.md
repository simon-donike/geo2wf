# Local data layout and tensors

The current loaders read local raster exports or task-specific caches. The
[Hugging Face release](index.md) stores scientific assets in a catalog layout;
its root is not interchangeable with the export root below. Use the original
export manifests with their referenced files, or the release-specific tooling
documented on the Hub to obtain a compatible view.

## Paired raster export

```text
paired/
├── stats.json
├── train/
│   ├── manifest.csv
│   └── ... GeoTIFFs referenced by the manifest
├── val/
│   ├── manifest.csv
│   └── ...
└── test/
    ├── manifest.csv
    └── ...
```

Manifests identify GEO conditions, SAR targets, optional ERA5/PMW companions,
source times, and IBTrACS centers. The loader accepts generic
`condition_path`/`target_path` and compatible `geo_path`/`sar_path` columns.
GeoTIFFs preserve raw values, band descriptions, CRS, bounds, and validity.
Set `data.root` and `data.stats_file` to the matching export and statistics.
Joint scalar training additionally needs `data.ibtracs_file`.

## Returned sample

| Key | Per-sample shape | Meaning |
|---|---|---|
| `condition` | `[C,H,W]` | Normalized GEO and configured context/features |
| `condition_mask` | `[1,H,W]` | Valid condition/context pixels |
| `target`, `target_physical` | `[T,H,W]` | Normalized target and original physical values |
| `target_mask` | `[1,H,W]` | Observed target footprint |
| `target_norm_offset`, `target_norm_scale` | Broadcastable | Inverse-normalization parameters |
| `condition_bounds`, `target_bounds` | `[4]` | Left, right, bottom, top |
| `center` | `[2]` | IBTrACS latitude/longitude |
| `sample_id`, `meta` | String, mapping | Identity and source provenance |

ERA5 adds normalized/physical `era5_wind_speed` tensors and an
`era5_wind_speed_mask`. Optional PMW supplies `pmw`, `pmw_physical`, `pmw_mask`,
and `pmw_bounds`. The canonical `collate_wind_field_samples` stacks tensors
with a batch dimension while retaining metadata as one mapping per sample.

`DataSpec` exposes ordered channels, units, spatial shape, and companions before
training, so the model can reject incompatible inputs. For the default paired
configuration:

```text
10 GEO + 9 ERA5 + distance + 3 solar = 23 condition channels
ERA5-residual U-Net: 23 + condition mask + ERA5 wind + ERA5 mask = 26 inputs
```

The mask is appended inside the model, not counted in `condition_channels`.

## Normalization and masks

Statistics come from valid training pixels only. The default grouped presets
use robust z-score normalization for conditions (median/IQR scale, clipped at
4 and mapped to [0,1]) and min–max normalization for SAR targets. Preserve the
statistics used by the checkpoint; do not recompute them on evaluation data.

The loader retains physical targets and the affine inverse mapping
\(x = z\,\mathrm{scale} + \mathrm{offset}\). Losses and wind metrics use physical
units. Invalid values are zero-filled after normalization and masked, so
missing pixels are not zero-wind labels. `target_mask` limits SAR supervision;
`era5_wind_speed_mask` separately limits the ERA5 anchor and comparisons.

Aligned rasters are cropped together. Flips also transform ERA5 vector
components and vorticity according to their physical parity. The storm center
comes from IBTrACS metadata, which can differ from the raster crop center.

## Splits and scalar caches

Current grouped configs use `include_test_in_train: false`. Keep storms
disjoint and inspect a historical checkpoint's actual training membership.
`require_era5` filters availability; `use_era5` controls model inputs. Joint
scalar datasets additionally apply target/center eligibility rules.

[Correction](../models/intensity-correction.md) and
[forecast](../models/intensity-forecast.md) datasets use split manifests and
cache metadata instead of the paired-raster contract. Preserve the frozen
producer hashes, target definitions, and feature scaler with each cache.
