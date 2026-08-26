# Export GEO–SAR pairs

`geo2wf-export geo-sar` turns the larger observation manifest and source files into colocated GEO conditions and SAR wind targets.

## Basic export

```bash
uv run geo2wf-export geo-sar \
  --config configs/config.yaml
```

The `export` section supplies defaults; explicit CLI flags take precedence. For a safe structural check:

```bash
uv run geo2wf-export geo-sar \
  --config configs/config.yaml \
  --limit 2
```

## What happens per pair

1. Read and validate manifest observations.
2. Group records by split and storm.
3. Match each SAR occurrence with its closest GEO occurrence within `closest_match_hours`.
4. Optionally match the closest supported PMW swath within `pmw_max_time_gap_hours`.
5. If `include_ibtracs` is enabled, match the nearest IBTrACS track row and
   retain every source field.
6. Confirm required GEO and SAR channels exist.
7. Select a crop center, build the shared geographic grid, and optionally shift it to include the IBTrACS center.
8. Regrid continuous source channels and construct joint validity.
9. Optionally select a temporally close ERA5 analysis and regrid it.
10. Write condition, companion, context, and target rasters plus one manifest row.
11. Update channel statistics when processing the train split.

Failures caused by missing channels, file I/O, or invalid geometry are recorded and skipped instead of aborting the entire export.

## ERA5-enriched export

```bash
uv run geo2wf-export geo-sar \
  --config configs/v1/config_geo_sar_10bands_era5.yaml
```

The configured seven source fields are precipitable water, sea-surface temperature, mean sea-level pressure, 2 m temperature, 2 m dewpoint, and 10 m u/v wind. New exports calculate 10 m wind speed and relative vorticity on the native ERA5 grid before continuous interpolation, preserving more structure than deriving them after nearest-neighbor regridding.

An ERA5 record is rejected when its nearest-time gap exceeds 3.1 hours in the ERA5 presets. The runtime dataset applies the same guard, protecting against stale legacy manifests.

!!! note "Legacy ERA5 exports"
    Existing seven-band exports remain loadable. At runtime the dataset derives wind speed and vorticity; older nearest-neighbor products receive a target-scaled speed and a neutral vorticity fallback as described in [Normalization & masks](normalization.md). Re-export for bilinear fields and robust median/IQR statistics.

## Important export keys

| Key | Default | Meaning |
|---|---:|---|
| `geo_channel_set` | `common4` | ABI/AHI-compatible band set |
| `grid_size` | `256` | square output dimension |
| `grid_resolution` | `0.027` | degrees per output pixel |
| `closest_match_hours` | `0.5` | maximum GEO–SAR time gap |
| `include_pmw` | `false` | export the nearest PMW companion |
| `pmw_max_time_gap_hours` | `12.0` | maximum SAR–PMW time gap |
| `include_ibtracs` | `false` | join the complete nearest IBTrACS record |
| `ibtracs_file` | `<data_root>/IBTrACs/ibtracs.ALL.list.v04r01.csv` | IBTrACS all-basins CSV |
| `ibtracs_max_time_gap_hours` | `6.1` | maximum SAR–IBTrACS time gap |
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

When PMW is available, the exporter writes `<sample_id>_pmw.tif` on exactly the
same grid as GEO and SAR. The manifest records its path, sensor, channel,
timestamp, and signed `pmw_dt_minutes`. A missing or unreadable PMW companion
does not discard an otherwise valid GEO–SAR pair.

When `include_ibtracs` is enabled, the nearest IBTrACS row is expanded into
`ibtracs_<source-column>` manifest columns. Convenience fields include
`ibtracs_msw_kt`, `ibtracs_msw_ms`, `ibtracs_mslp_hpa`, their selected WMO/USA
source columns, and the signed match offset. `ibtracs_schema.json` records
source units for every retained field. The independent
`ibtracs_center_lat`/`ibtracs_center_lon` already present in the source
observation manifest remain available for crop placement and storm geometry
even when the full-row join is disabled.

See the complete [configuration reference](../reference/configuration.md).
