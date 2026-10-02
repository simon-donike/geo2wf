# Public data contract v1

Every document has `schema_version: 1`. JSON numbers are finite; missing scalar output is `null`, never zero or an official wind value. Canonical wind units are m/s, radii km, coordinates WGS84 degrees, and timestamps UTC ISO 8601 ending in `Z`. Identity is the ATCF ID (e.g. `EP152026`), independent of current basin.

## Release layout

```text
latest.json                         mutable pointer, revalidated on every refresh
releases/<UTC version>/catalog.json  immutable summary
releases/<UTC version>/coverage.json immutable accounting report
releases/<UTC version>/evaluation.json  optional immutable validation report
objects/<sha256>.json                immutable storm series, shared across releases
```

`latest.json` contains `version` and `manifest` (`releases/<version>/catalog.json`). The pointer is advanced only after all referenced objects and the catalog upload successfully. Browser selections use storm IDs and timestamps, never array indices or object hashes.

## Catalog

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
