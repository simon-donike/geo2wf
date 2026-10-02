# StormSense implementation evidence — 2 October 2026

The standalone application is hosted at **https://stormsense.hyperalislabs.workers.dev/**. It has its own React/Vite build and read-only Worker in the Hyperalis Labs Cloudflare account. The existing documentation and explorer remain independent. See the [runbook](../README.md), [versioned contracts](../CONTRACT.md), and [website deployment verification](website-deployment.json). Local development remains available at http://127.0.0.1:5173/.

## Delivered archive

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

- **222 Python tests passed**, including the existing repository suite and operational failure/idempotency cases.
- **10 web unit tests and 14 desktop/mobile browser tests passed.** The browser suite also passed against the deployed HTTPS website. Tests cover active and quiet periods, source failures, archive filters, forecasts, deep links, downloads, keyboard controls, timeline synchronization and refresh behavior. The production verification identified and fixed weak ETag matching after edge compression.
- A real export arrived in an open storm page with **zero reloads and the selected time preserved**: [refresh evidence](browser-refresh.json).
- Production build, Worker types, deployment dry run, strict documentation build and existing documentation link checks passed.
- A built Python wheel loaded both bootstrapped checkpoints outside the research checkout.

See [check results](checks.json). The Docker configuration was prepared but could not be built because this workstation has no Docker engine.

**The website is deployed. Pending hosting selection:** the portable inference runner and activation of its 15-minute schedule. The Worker has no cron trigger, and the example systemd units have not been installed or enabled. The hosted website and archive remain available independently of the local machine; its catalog will truthfully age until another finite cycle publishes data or continuous inference hosting is selected.
