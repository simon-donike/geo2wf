# StormSense implementation evidence — 5 October 2026

The standalone application is hosted at **https://stormsense.hyperalislabs.com/**. It has its own React/Vite build and read-only Worker in the Hyperalis Labs Cloudflare account. The existing documentation and explorer remain independent. See the [runbook](../README.md), [versioned contracts](../CONTRACT.md), and [website deployment verification](website-deployment.json). Local development remains available at http://127.0.0.1:5173/.

## Temporary hourly inference runner — 5 October 2026

The local user timer `stormsense-local.timer` is enabled and publishes at five
minutes past every hour. It reuses the existing Python environment and pinned
models, uses one CPU worker per processing stage, and exits between cycles.
The first full cycle filled 31 missing numerical hours and 17 display-image
slots. Its peak memory was 1.52 GiB. The first timer-triggered cycle completed
successfully in 111 seconds with a 470.9 MiB memory peak and no swap use; this
immediate repeat had no new numerical hours to compute. Its
published release is **`20261005T125717217922Z`**. The final unit has a one-core
CPU quota, a 2 GiB hard memory limit and low CPU/I/O priority.

Hosted checks confirmed that the public pointer matches the local export, with
7,002 predictions, 23 explained gaps and zero pending numerical hours. The user
manager already has lingering enabled, so it survives logout and returns after
reboot; updates still depend on this machine being awake and online. See the
[activation evidence](local-runner.json) and [local runbook](../runner/LOCAL.md).
Permanent hosting remains to be selected. Stop this publisher before enabling
a replacement host or Docker scheduler.

## Daily bundles and official archive metrics — 3 October 2026

Release **`20261003T175702414184Z`** and Worker **`986abea6-cc03-470f-baf9-8c626ebc86b7`** are deployed. The image archive contains **248 daily ZIPs (271.6 MB)**, with **2,680 available frames, 872 explained gaps and zero pending slots**. All 13,780 bundled image and metadata members were checked against their originals; georeferencing, STAC metadata and individual downloadable files are preserved. Numerical records, forecasts and tracks are unchanged.

Nolo's whole-storm preload now makes **23 bundle requests totaling 28.4 MB**, compared with the earlier 1,062 image requests totaling 50.0 MB. Hosted checks verified ZIP checksums, cache misses followed by hits, conditional requests and smooth offline scrubbing after preload. These are measured results, not guaranteed network performance.

Rapid-intensification shading now uses **only official NHC/CPHC winds**. The archive retains its existing columns and adds peak official wind, peak official category and RI status; incomplete reference history is shown as unknown. Scroll-to-zoom is enabled on the existing Leaflet map, with its styling and the single numerical timeline retained.

**235 Python tests, 41 web unit tests and 48 hosted desktop/mobile browser tests passed.** Local verification included 46 browser checks and six final bundle/layout checks. Immutable assets were staged and checked before deployment; `rclone copyto` advanced the StormSense pointer last. See [daily-bundle delivery evidence](daily-bundles.json). Scheduling was inactive at that delivery; the temporary hourly runner is now active as recorded above.

## Earlier rapid intensification and image caching delivery — 3 October

This section records the previous delivery. The daily-bundle delivery above supersedes its request counts and removes the separate model-derived RI shading.

Both storm graphs now shade complete 24-hour windows with a maximum-wind increase of at least 30 kt. Amber identifies the calculation from NHC/CPHC reference winds; teal hatching identifies a separate StormSense estimate. Detection uses original values, excludes forecasts and incomplete windows, and remains fixed when smoothing, units or selected time change. Overlapping windows are merged; keyboard-accessible details list the UTC periods and largest qualifying wind increases. Polo's official interval was verified against its recorded fixes: 20 September 18:00 through 23 September 00:00 UTC, six overlapping windows, maximum increase 85 kt in 24 hours.

Immutable data and image responses now use Cloudflare's Cache API. The hosted probe observed an image `MISS` followed by `HIT`; HEAD and conditional GET also hit cache. The latest-data pointer remains `BYPASS`. Cache hits avoid R2 reads but still invoke the Worker. Whole-storm preloading, individual georeferenced files and the existing numerical release are preserved. Nolo's complete preload contains 1,062 image objects totaling 50.0 MB; shared caching addresses repeat reads without bundling or republishing the archive.

**37 unit tests and 40 desktop/mobile browser tests passed**, with the full browser suite passing locally and on the deployed website. Worker types, the production build and deployment dry run passed. Existing zoom verification now explicitly focuses the selected storm position before measuring scale, avoiding legitimate clipping of track endpoints. See [RI, caching, cost assumptions and current performance evidence](rapid-intensification-caching.json). Earlier delivery results below retain their original measurements.

## Delivered archive

The figures and downloadable bundle below preserve the initial 14:00 UTC delivery snapshot. A later finite cycle published fresh **16:00 UTC** predictions for both active systems in release `20261002T160146269421Z`; its advancing rolling window contains **7,051 predictions, 17 explained gaps and zero pending hours**. The [deployment verification](website-deployment.json) records the current hosted release.

The public window is **2 October 2025, 14:00 UTC through 2 October 2026, 14:00 UTC**. Twelve preceding hours remain in SQLite for forecast context.

| Measure | Result |
| --- | ---: |
| Storms | 38 |
| Expected storm-hours | 7,070 |
| Hours with real predictions | 7,053 |
| Explained gaps | 17 |
| Pending hours | 0 |
| Historical forecast issues in the public window | 6,591 |
| Forecast issues actually generated during live operation | 4 |

[Coverage](coverage.json) accounts for every expected hour, with no historical discovery failures. The gaps are 13 unavailable complete scans, two insufficient-coverage crops, and two recent hours without adequately bracketing historical fixes. Two additional retrospective gap records coexist with successful live predictions at 14:00 UTC, so those hours count as covered. Official intensity was never substituted for missing model output.

The numerical release is about **14 MB uncompressed**, or **1.52 MB compressed**, including provenance, tracks, forecasts and reports. The [local archive bundle](../../../var/stormsense/stormsense-2025-10-02_2026-10-02.tar.gz) contains 42 JSON files and no imagery or checkpoints. Its [checksum and manifest](archive-bundle.json) identify the exact release. Extract it into `apps/stormsense/public/data` to restore the preview.

Release **`20261002T143251092386Z`** was published beneath **`r2:tcd/explorer/stormsense`**. Immutable objects and release metadata were uploaded first; `rclone copyto` advanced this application's pointer last. A downloaded-content check found zero differences, and the remote pointer matches the local pointer. See [publication verification](publication.json). No existing explorer pointer was changed.

## Real data and model checks

[Verification](verification.json) includes real GOES-East, GOES-West and central Pacific samples. All checkpoint/configuration/statistics hashes passed. Comparison with the training loader found a maximum preprocessing difference of **2.51 × 10⁻⁶**, with exact masks and target normalization.

| Probe | Platform | Range bytes | Total CPU run | Inference alone |
| --- | --- | ---: | ---: | ---: |
| Central Pacific | GOES-18 | 20.32 MB | 9.88 s | 0.18 s |
| Atlantic | GOES-19 | 10.88 MB | 10.79 s | 0.15 s |
| Eastern Pacific | GOES-18 | 23.86 MB | 8.83 s | 0.15 s |

These are measured individual probes, not guaranteed throughput. Backfill shares in-memory source reads among storms at the same hour; actual costs depend mainly on network access and runner concurrency. Satellite arrays and reconstructed wind fields were discarded after processing.

The [archive audit](acceptance-audit.json) checked all saved numerical records for finite nonnegative values, scan timing, crop coverage, fresh live centers, forecast input/valid times, absence of future-generated live inputs, source references, public object hashes, and complete coverage accounting.

**Provenance limitation:** 881 early-run predictions predate the explicit per-image retrieval timestamp field. They retain the actual imagery URL, acquisition times, NHC/CPHC snapshot retrieval timestamp and prediction generation timestamp. Their missing image retrieval time is shown as “Not recorded” in the observation details; no timestamp was invented. Later records, including the latest live cycle, include it. Early hindcast issues also predate explicit per-input generation-time fields; they remain clearly retrospective. Every as-issued live forecast records its input modes and generation times.

## Forecast evaluation

The combination remains **experimental**. In this archive, its MAE is worse than persistence at both leads, although better than the recent-trend baseline. No confidence intervals or positive operational skill claims are made.

| Evaluation group | Lead | Cases | Storms | Model MAE | Persistence MAE | Trend MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All retrospective storms | +6 h | 1,087 | 38 | 8.29 m/s | 7.59 m/s | 10.16 m/s |
| All retrospective storms | +12 h | 1,049 | 38 | 9.22 m/s | 7.83 m/s | 13.87 m/s |
| Excluded from nowcast training | +6 h | 986 | 34 | 8.39 m/s | 7.73 m/s | 10.32 m/s |
| Excluded from nowcast training | +12 h | 952 | 34 | 9.23 m/s | 7.90 m/s | 13.99 m/s |

The [evaluation report](evaluation.json) contains exact values, storm membership, hashes, RMSE, bias and storm-macro MAE. Targets are exact valid-time NHC/CPHC BEST fixes, which may be revised. Missing reference times are excluded. Forecast training used IBTrACS 2000–2018. The four live issues had no verifying future observations at delivery, so no live skill result is reported.

## Operation and application checks

A finite live runner cycle successfully discovered both active storms and produced their 14:00 UTC estimates. A repeated update reused those saved estimates and, once historical warm-up was available, issued the current +6/+12 h forecasts. No continuous process or paid inference service was activated.

- **232 Python tests passed**, including the existing repository suite and operational failure/idempotency cases. The added publication regression verifies unchanged bytes with different timestamps and rejects same-sized corruption before pointer advancement.
- **26 web unit tests and 36 desktop/mobile browser tests passed.** The browser suite also passed against the deployed HTTPS website. Tests cover active and quiet periods, source failures, archive filters, forecasts, deep links, downloads, keyboard controls, timeline synchronization and refresh behavior. The production verification identified and fixed weak ETag matching after edge compression.
- A real export arrived in an open storm page with **zero reloads and the selected time preserved**: [refresh evidence](browser-refresh.json).
- Production build, Worker types, deployment dry run, strict documentation build and existing documentation link checks passed.
- A built Python wheel loaded both bootstrapped checkpoints outside the research checkout.

See [check results](checks.json). The Docker configuration was prepared but could not be built because this workstation has no Docker engine.

The deployed UI now includes true-scale wind-radius circles, active-system tracks, official/model categories, NHC threshold lines, optional light chart smoothing and storm age. Rapid scrubbing tests used a 533-hour storm series, retained the track elements, kept the slider's position stable, and saved the selected time on release. Earlier interaction checks measured hosted input-to-next-frame p95 of **24.6 ms desktop / 28.2 ms mobile emulation** on this workstation; this is not a physical-phone benchmark. See [hourly imagery and interaction evidence](hourly-imagery.json) and the [earlier UI verification](ui-enhancements.json).

The custom domain **https://stormsense.hyperalislabs.com/** has valid HTTPS and serves the standalone Worker. Release **`20261002T174218736546Z`** adds **5,335 hourly GeoColor frames**, with **1,733 explained image gaps and zero pending hours**, across the same rolling-year numerical window. Full images, previews, STAC metadata and GDAL georeferencing sidecars total **524.8 MB**; the median full-image part is **60.6 kB**. Every saved WebP checksum and GDAL grid was verified, including actual hosted GOES-East, GOES-West and dateline assets. Independent R2 checks found zero differences across 27,431 image/metadata assets before the pointer advanced. See [hourly imagery evidence](hourly-imagery.json).

Storm pages retain the original single timeline slider, synchronized chart controls, and buffered playback at 2/4/8 hours per second. Category labels sit on the left of the wind graph. Opening a storm downloads its whole image set in the background with selected-hour priority and a visible loaded-image count. Compressed bytes stay available for distant jumps, while a separate 48 MiB decoded cache keeps nearby frames ready. Missing frames clear the old picture while leaving tracks and numerical results usable. WebP files are north-up EPSG:3857 rasters; sidecars preserve pixel grids, and STAC records include geographic footprints, source URLs, product/slot/retrieval times and checksums. Keep each `.webp.aux.xml` alongside its WebP for GIS import. The GIBS product time is a provider-reported display timestamp, separate from the raw ABI scan used for inference. No expensive reconstruction was attempted for unavailable display products.

The active overview still checks NASA GIBS directly every five minutes. The numerical prediction pipeline continues discarding its raw satellite arrays; only the optional compressed display products are retained. The earlier [live-overlay verification](geocolor.json) documents that feature's initial deployment. The `workers.dev` address remains available.

An **82.9 MB** [runner handoff bundle](../../../var/stormsense/runner.tar.gz) contains source, pinned models and a SQLite snapshot with a successful integrity check. [Its manifest](runner-bundle.json) confirms that credentials and imagery are excluded. The [activation runbook](../runner/README.md) is ready for the selected Linux host.

**The website is deployed; permanent inference hosting remains to be selected.** The temporary local hourly runner is active. The Worker has no cron trigger, and the permanent-host example units remain inactive. The website and published archive remain available independently of the local machine; source timestamps show any interruption in updates.

The restored **single timeline** and whole-storm image preloader are deployed and verified. All 531 available frames for Nolo loaded before an offline test of distant slider jumps; full-resolution frames followed in p95 **75.5 ms desktop / 58.1 ms mobile emulation**. The 26 unit tests and 36 hosted browser checks passed. See [image-preloading verification](image-preloading.json).
