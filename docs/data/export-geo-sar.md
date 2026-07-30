# Export GEO–SAR pairs

`scripts/export_geo_sar_geotiffs.py` turns the larger observation manifest and source files into colocated GEO conditions and SAR wind targets.

## Basic export

```bash
uv run python scripts/export_geo_sar_geotiffs.py \
  --config configs/config.yaml
```

The `export` section supplies defaults; explicit CLI flags take precedence. For a safe structural check:

```bash
uv run python scripts/export_geo_sar_geotiffs.py \
  --config configs/config.yaml \
  --limit 2
```

## What happens per pair

1. Read and validate manifest observations.
2. Group records by split and storm.
3. Match each SAR occurrence with its closest GEO occurrence within `closest_match_hours`.
4. Confirm required GEO and SAR channels exist.
5. Select a crop center, build the shared geographic grid, and optionally shift it to include the IBTrACS center.
6. Regrid continuous source channels and construct joint validity.
7. Optionally select a temporally close ERA5 analysis and regrid it.
8. Write condition, context, and target rasters plus one manifest row.
9. Update channel statistics when processing the train split.

Failures caused by missing channels, file I/O, or invalid geometry are recorded and skipped instead of aborting the entire export.

## ERA5-enriched export

```bash
uv run python scripts/export_geo_sar_geotiffs.py \
  --config configs/config_geo_sar_10bands_era5.yaml
```

The configured seven source fields are precipitable water, sea-surface temperature, mean sea-level pressure, 2 m temperature, 2 m dewpoint, and 10 m u/v wind. New exports calculate 10 m wind speed and relative vorticity on the native ERA5 grid before continuous interpolation, preserving more structure than deriving them after nearest-neighbor regridding.

An ERA5 record is rejected when its nearest-time gap exceeds 3.1 hours in the ERA5 presets. The runtime dataset applies the same guard, protecting against stale legacy manifests.

!!! note "Legacy ERA5 exports"
    Existing seven-band exports remain loadable. At runtime the dataset derives wind speed and vorticity; older nearest-neighbor products receive a target-scaled speed and a neutral vorticity fallback as described in the repository README. Re-export for bilinear fields and robust median/IQR statistics.

## Important export keys

| Key | Default | Meaning |
|---|---:|---|
| `geo_channel_set` | `common4` | ABI/AHI-compatible band set |
| `grid_size` | `256` | square output dimension |
| `grid_resolution` | `0.027` | degrees per output pixel |
| `closest_match_hours` | `0.5` | maximum GEO–SAR time gap |
| `center` | `image_center` | initial crop center strategy |
| `shift_center` | `true` | include IBTrACS center if near crop edge |
| `pad` | `8` | source-read padding around the requested grid |
| `include_era5` | `false` | write a context GeoTIFF |
| `limit` | `null` | maximum successful pairs per split |

!!! important "Crop center and storm center are different quantities"
    The `center` configuration key chooses the **raster crop center**. With the
    default `image_center`, `shift_center: true` moves that crop only as far as
    needed to keep the IBTrACS point inside its usable area. The export records
    this potentially shifted crop location as `center_lat`/`center_lon`, while
    preserving the independent track location as
    `ibtracs_center_lat`/`ibtracs_center_lon`. At runtime,
    `batch["center"]` means the latter. Storm metrics therefore remain anchored
    to IBTrACS even when the storm is not at the geometric center of the
    GeoTIFF.

See the complete [configuration reference](../reference/configuration.md).
