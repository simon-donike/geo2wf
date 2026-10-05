# Public data contract v1

StormSense numerical and index documents have `schema_version: 1`. Standard STAC image metadata instead uses `stac_version: 1.0.0`. JSON numbers are finite; missing scalar output is `null`, never zero or an official wind value. Canonical wind units are m/s, radii km, coordinates WGS84 degrees, and timestamps UTC ISO 8601 ending in `Z`. Identity is the ATCF ID (e.g. `EP152026`), independent of current basin.

## Release layout

```text
latest.json                         mutable pointer, revalidated on every refresh
releases/<UTC version>/catalog.json  immutable summary
releases/<UTC version>/coverage.json immutable accounting report
releases/<UTC version>/evaluation.json  optional immutable validation report
objects/<sha256>.json                immutable storm series, shared across releases
bundles/<sha256>.zip                 immutable daily display-image package
```

`latest.json` contains `version` and `manifest` (`releases/<version>/catalog.json`). The pointer is advanced only after all referenced objects and the catalog upload successfully. Browser selections use storm IDs and timestamps, never array indices or object hashes.

## Catalog

Storm summaries include `peak_official_wind_ms` (maximum available NHC/CPHC reference wind over the storm's lifetime), `peak_category`, and `has_ri`. `has_ri` is true when official tropical-phase fixes show a gain of at least 30 kt over an exact, continuously observed 24-hour window, false when evaluated windows show no qualifying increase, and null when no complete window can be evaluated. Fixes may be at most six hours apart. These fields never use StormSense estimates. Lifetime summaries are saved before local track retention removes older fixes; revised source tracks recalculate them.

| Field | Meaning |
| --- | --- |
| `release`, `generated_at` | Release identity and export time; not advisory freshness |
| `window.start`, `.end` | Inclusive public archive interval; preceding forecast context stays in SQLite |
| `units` | Explicit wind/radius/time units |
| `models.nowcast`, `.forecast` | Pinned IDs, version strings and checkpoint SHA-256 values |
| `source_status.discovery` | `last_attempt`, `last_success`, `error`, `active_count`, snapshot digest |
| `source_status.update` | Last completed finite update and requested time |
| `source_status.backfill` | Last progress, processed/queued counts and completion of that invocation |
| `source_status.history` | Verified discovery window, retrieval time and per-source failures |
| `coverage` | Expected storm-hours, predictions, explained gaps and pending counts |
| `reports` | Coverage and optional forecast-evaluation report locations |
| `storms[]` | Summaries described below |

Each storm summary includes `id`, `name`, `basin` (`AL`, `EP`, `CP`), `active`, `start`, `end`, `peak_category` (−1 depression, 0 tropical storm, 1–5 hurricane categories), `advisory`, `latest_fix`, `latest_prediction`, `metrics`, `change_24h_ms`, `prediction_count`, `gap_count`, `pending_count`, `expected_count`, `record_count`, and `series` (content-addressed relative path). The displayed basin can change while `id` remains stable. `latest_prediction` can be stale; consumers must show its own time. Recent change uses the same nowcast version; `change_24h_reference_kind` identifies retrospective prehistory.

A discovery error preserves the last known active state and freshness. A successful empty `activeStorms` response marks systems inactive. Neither a failed source nor an unpublished release is represented as a successful empty catalog.

## Storm series

The root contains `storm_id`, `track[]`, `records[]`, `forecasts[]`, and `provenance`.

Track fixes retain `time`, `lat`, `lon`, `wind_ms`, `pressure_hpa`, `classification`, `radii_km`, `name`, and source. Repeated ATCF quadrant/threshold rows merge into one fix. Equivalent-area reference radii are `sqrt(mean(quadrant_radius²))`, converted from nautical miles to km. Forecast-track positions do not become observations.

`provenance.track` / `.advisory` identify the retained source snapshot by digest, source URL and initial retrieval timestamp. `track_retrieved_at` and `advisory_retrieved_at` give the storm's latest successful retrievals independently of source advisory times. SQLite retains the snapshot body. Each record points to the snapshot actually used, which can differ from the latest one.

An hourly record has:

| Field | Meaning |
| --- | --- |
| `time` | Requested hourly UTC slot |
| `kind` | `live` or `hindcast`; historical reconstruction never becomes an issued-live prediction |
| `status` | `ready` or `gap` |
| `model_version`, `generated_at` | Exact numerical pipeline/model identity and generation time |
| `metrics` | `vmax_ms`, `rmw_km`, `r34_km`, `r50_km`, `r64_km`, or null for a gap |
| `center` | Position, method, fix time and fix age; null when no authorized center exists |
| `source_snapshot` | Digest of the advisory/BEST metadata used |
| `source_snapshots` | Additional metadata digests when a live center combines a track position with advisory motion |
| `imagery` | Source URL, GOES platform, actual scan start/end, object publication time/ETag/size, valid fraction and measured range bytes; new reads also include retrieval time and grid/channel metadata |
| `reason`, `detail`, `retryable` | Machine-readable gap reason, optional bounded diagnostic and retry eligibility |
| `elapsed_seconds` | Observed completed processing duration, when available |

Center methods are `advisory_fix`, `live_track_fix`, `motion_estimate`, `historical_fix`, `historical_interpolation`. Live centers also identify the position source and motion report time. Historical interpolations can use later/revised BEST fixes and therefore belong only to hindcasts. Successful rows are not replaced by failures. A ready live row is preferred for display when both kinds exist at one time; a live gap does not hide a ready hindcast. The full JSON preserves both records.

Gap reasons include `stale_or_missing_center`, `missing_scan`, `missing_bands`, `missing_quality_flags`, `unexpected_channel_units`, `outside_satellite_coverage`, `poor_coverage`, and `source_unavailable`. Pending work is the absence of a processed record and is counted separately.

## Forecast issue

Each issue contains `storm_id`, `kind`, `anchor_time`, `generated_at`, combined `model_version`, `input_model_version`, `input_times`, `input_vmax_ms`, `experimental: true`, and `predictions[]`. Inputs are ordered t, t−6h, t−12h, all from the same model version. Historical issues use retrospective inputs. Live issues prefer live prehistory and can use retrospective estimates that had already been generated when the issue was produced; `input_kinds` and `input_generated_at` preserve that distinction. Future-generated records are excluded. An issue is absent when any input is missing. Generation time is not the simulated historical issue time.

Each prediction has `lead_hours` (6 or 12), `valid_time`, and `vmax_ms`. The 12-hour value is recursive. There is no confidence interval. The evaluation contract reports live and historical issues separately and identifies the training-excluded subset, reference count, and persistence/trend comparisons.

## Retention and consumers

Keep the rolling 12-calendar-month public window and at least 12 preceding hours locally. Read a consistent SQLite snapshot while exporting. Published releases and storm objects are immutable; consumers should cache them. Revalidate `latest.json`, then resolve its catalog and series relative to the same data root. Do not treat the catalog's export time as the source observation time.

Read-only Worker methods are GET/HEAD. It returns 404 for absent/disallowed objects, 405 for writes, 503 for source errors, and ETag-based 304 responses. Only the new StormSense prefix is accessible.

## Optional display imagery (gibs-geocolor-webp-v1)

New exports and acquisition jobs use even UTC hours. The numerical records and forecast inputs retain their hourly cadence. At an intervening selected hour, the map uses the immediately preceding display slot and labels its actual provider time. A gap at that slot stays a gap; images are never extrapolated from a future frame. Historical releases remain readable.

`series.imagery_bundles[]` adds `{schema_version:1, date, path, sha256, bytes, images}`. Each `path` is a content-addressed `bundles/<sha256>.zip` containing one storm's UTC day; `images` lists its full/preview WebP member paths. The stored ZIP format `geocolor-daily-zip-v1` needs no image decompression: it contains the original WebPs, matching GDAL sidecars, STAC Items, and a `manifest.json` with the storm, date, CRS and complete frame descriptors. Extracting it preserves the original relative paths and GIS metadata. ZIP timestamps and ordering are deterministic. Completed days reuse identical objects; updates create a new immutable current-day object.

The browser preloads each referenced daily package once, validates its SHA-256 and extracts only its WebPs in memory. The existing decoded-image budget still applies. Bundle failures are retryable and do not fall back to hundreds of individual requests. Older releases without bundle descriptors retain individual-image loading.

`series.imagery[]` is independent of `records[].imagery` (the numerical model's original input provenance). Display images never enter inference. The catalog's `imagery` contains a version, source status, coverage counts (`expected`, `ready`, `gaps`, `pending`) and an immutable `objects/<digest>.json` manifest of every referenced display asset. Storm summaries include `imagery_coverage`. Old releases without these fields remain readable.

Each display record includes `storm_id`, hourly `time`, `status`, `reason`, `checked_at`, `version`, `center`, `center_kind`, and `parts`. Ready records additionally include `acquired_at`, `satellite` (East/West), `generated_at` and `metadata` (the STAC Item path). Here **`acquired_at` is the provider-reported GeoColor product timestamp**, not a reconstructed ABI scan start/end. It must fall within `[time - 30 minutes, time]`. Source retrieval and generation times are recorded independently. Center provenance identifies a saved live/hindcast position or retrospective track interpolation; these display assets do not become as-issued predictions.

Only GIBS availability metadata and WMS crops are requested. Older hours get one attempt at the latest reported eligible frame. Recent hours may try up to three frames because newly listed regional imagery can still be empty. Missing centers, no reported frames, empty images, invalid dimensions and source failures remain distinct gap reasons. Raw ABI reconstruction is not a fallback. A gap is explicitly attempted work; a missing row remains pending. Interrupted jobs commit completed hours independently and resume without fetching existing ready assets.

### Georeferencing and migration

Every raster is north-up **EPSG:3857**, with metre coordinates and **PixelIsArea** semantics. All bounds describe **outer pixel edges**. Source WMS BBOX values are rounded to two decimals before both acquisition and transform calculation, so metadata describes the actual requested grid. Full images are 768 pixels high; previews are 256 pixels high. A dateline crop is split into two independently georeferenced parts, with widths proportional to their footprint.

Each part contains:

- `image`, `preview`: immutable relative WebP paths under `imagery/`.
- `sidecar`, `preview_sidecar`: matching `.webp.aux.xml` GDAL PAM metadata. Download each WebP **together with its same-named sidecar** for automatic CRS/transform recognition in GDAL/QGIS.
- `bbox`: `[west, south, east, north]`, canonical WGS84 longitude/latitude degrees within ±180.
- `display_bbox`: the same footprint unwrapped around the storm for continuous antimeridian display. This is a convenience, not the raster's CRS. A future MapLibre image source can use corners `[[west,north],[east,north],[east,south],[west,south]]` from it.
- `sha256`, `bytes`, `preview_bytes`, `source_url`, `retrieved_at`, `valid_fraction`.

The STAC 1.0 Item uses the [Projection Extension v1.1](https://github.com/stac-extensions/projection/tree/v1.1.0). Its geometry is a WGS84 MultiPolygon with separate dateline pieces. Each image and preview asset has `proj:epsg`, `proj:wkt2`, `proj:bbox`, `proj:shape` (`[height,width]`) and `proj:transform` (a row-major affine matrix):

```text
[x_pixel_size, 0, xmin,
 0, -y_pixel_size, ymax,
 0, 0, 1]
```

This maps pixel **corners** to projected coordinates. Pixel `(column,row)` centers use `(column+0.5,row+0.5)`. The PAM sidecar stores equivalent GDAL order `[xmin,x_pixel_size,0,ymax,0,-y_pixel_size]`; no half-pixel offset is added to either transform. Preview transforms preserve exactly the full image's extent at the smaller shape.

STAC `datetime` is the product timestamp; `stormsense:slot_time` is the hourly selection. Metadata includes the full CRS definition, source requests, retrieval timestamps, provider-availability response hashes, image checksums, centre provenance and rendering version. Asset addresses include both pixel checksum and grid, preventing identical pixels at different locations from sharing contradictory sidecars. Georeferencing is not embedded in the WebP bitstream: keep the sidecars and STAC metadata during migration. These are lossy display composites, not calibrated quantitative satellite bands.

### Publication and retention

Publish `imagery/*.webp`, `*.webp.aux.xml`, STAC `*.json` and `bundles/*.zip` before the immutable numerical/index objects and release metadata. Advance `latest.json` last using `rclone copyto`. The read-only Worker serves correct image/XML/JSON MIME types and public CORS for GIS clients. Source processing state lives beside SQLite in `var/stormsense/geocolor/imagery`; no original PNG responses are retained.

Release retention follows the imagery manifest as well as storm-series references. An image, its sidecars and its daily bundle survive while referenced by any retained release; remote deletion also observes the existing 24-hour grace period. Local processing asset cleanup follows retained SQLite visual rows. Never apply a blanket age-based R2 lifecycle to content-addressed images.
