# Export GEO–SAR pairs

Use the [published dataset](index.md) when its processed inputs fit your task.
Rebuilding an export requires an observation manifest and the original source
products; upstream raw training granules are not part of the Hub release.

## Export ten GEO bands with ERA5

```bash
uv run geo2wf-export geo-sar \
  --config configs/config.yaml \
  --data-root /path/to/source-archive \
  --manifest-file /path/to/observation_manifest.csv \
  --geo-channel-set common10 --include-era5 \
  --output-root data/geotiff/geo_sar_10bands_era5 \
  --limit 2
```

`--limit 2` bounds successful pairs per split for an initial check; omit it for
an export of the full supplied manifest. Explicit flags override the `export`
section of the config. Supply local source paths: the checked-in export config
contains historical machine defaults.

The exporter matches each SAR acquisition to the nearest eligible GEO
observation within each storm/split, regrids them onto a shared grid, and
writes raw GeoTIFFs, internal masks, manifests, and training-only statistics.
Missing channels, unreadable files, or invalid geometry are recorded and skipped.

## Settings that define the dataset

| Setting | Current default / behavior |
|---|---|
| `geo_channel_set` | `common4`; explicitly select `common10` for the documented models |
| `grid_size`, `grid_resolution` | 256 pixels, 0.027° spacing in EPSG:4326 |
| `closest_match_hours` | 0.5 h; the paper describes a wider ±90-minute pairing window |
| `include_era5` | Explicit opt-in; source fields plus derived wind speed/vorticity |
| `include_pmw`, `include_ibtracs` | Optional companion and nearest best-track-row export |
| `center`, `shift_center` | Image-centered crop, shifted when needed to include the storm center |

Export defaults therefore do not by themselves recreate the paper's cohort.
Retain the source manifest, matching settings, and normalization statistics for
each version. `center_lat`/`center_lon` describe the raster crop;
`ibtracs_center_lat`/`ibtracs_center_lon` describe the storm and drive runtime
geometry. Scalar training interpolates its own `USA_WIND` targets rather than
using the exporter's nearest-row convenience intensity.

See `uv run python scripts/export_geo_sar_geotiffs.py --help` for flags and the
[local data contract](dataset-contract.md) for loading the result.
