# StormSense

Independent React/TypeScript/Vite application and NHC/CPHC prediction pipeline. The existing documentation and explorer retain their own build and R2 pointer.

**Hosted website:** https://stormsense.hyperalislabs.com/ — deployed to the Hyperalis Labs Cloudflare account and reading the existing R2 archive. A [temporary local runner](runner/LOCAL.md) updates and publishes hourly at five minutes past the hour while this host is online. Permanent hosting is still to be selected.

The completed 12-month archive, measured data-path checks, forecast evaluation and delivery limitations are recorded in the [implementation report](reports/README.md).

## Local preview

Use Node 22 and Python 3.10 or 3.11. Run Python commands from the repository root:

```bash
uv sync --frozen --group operational --group dev --group docs
uv run --group operational geo2wf-operational bootstrap
uv run --group operational geo2wf-operational discover
uv run --group operational geo2wf-operational --device cpu update
uv run --group operational geo2wf-operational export --output apps/stormsense/public/data
cd apps/stormsense
npm ci
npm run dev
```

The preview is at http://127.0.0.1:5173. `bootstrap` uses the Hugging Face CLI (`hf`, installed separately) to download only the pinned model release assets. Existing matching local research checkpoints can also be used. Checkpoint, configuration, statistics and training-membership checksums are enforced; see [`models.json`](../../src/geo2wf/operational/models.json).

The local database is `var/stormsense/state.sqlite`. Generated data and logs under `var/stormsense` are excluded from Git; delivery reports under `apps/stormsense/reports` are reviewable repository artifacts. The app has its own routes `/`, `/storms/:id`, `/archive`, `/method`, `/about`. Append `?time=<UTC ISO timestamp>&issue=<forecast anchor>` for a shareable selection. Auto-refresh preserves an explicitly selected time. Wind units default to knots; the toggle persists locally. JSON retains canonical m/s and km. All dates are UTC.

The active map includes track history and true-scale R34/R50/R64/RMW rings at the latest estimate. Storm cards show official and model-equivalent wind categories and elapsed days/hours since the first recorded fix. Detail maps update the rings at the selected hour. Intensity charts mark NHC category thresholds in either wind unit. Wind estimate charts default to exponential smoothing with alpha 0.5 (50% newest estimate, 50% previous smoothed value), chosen to reduce wiggles while keeping nominal delay to one hour after comparison against downloaded NHC/CPHC BEST winds across all 38 storms; radii retain the three-hour 60/30/10% filter. Both have an off switch. Interior gaps are linearly interpolated before filtering, with no extrapolation, and original downloads, metric tiles, map radii and issued forecasts remain unchanged. Timeline dragging updates locally and commits its shareable URL on release, keeping track paths and the slider position stable.

Active-storm maps also show approximately **1,800 km GeoColor crops** from [NASA GIBS](https://nasa-gibs.github.io/gibs-api-docs/access-basics/), choosing GOES-East or GOES-West by viewing geometry. “Focus … image” fits the crop to the map; the checkbox and opacity slider control visibility. Daytime imagery uses a natural-colour composite; nighttime uses an infrared blend. The UI labels the actual reported image time and age separately from prediction times. It checks for new frames every five minutes while visible, tries up to three advertised times if a crop is empty, and retains labeled previous imagery if metadata refresh fails. The active overview fetches these latest images directly into the browser. Storm detail pages instead use the saved hourly image archive described below. WMS crops and Leaflet use the same Web Mercator projection, including split crops across the dateline.

After publication, restore the numerical archive into a fresh checkout without rerunning satellite inference:

```bash
rclone copy r2:tcd/explorer/stormsense/objects apps/stormsense/public/data/objects --include '*.json'
rclone copy r2:tcd/explorer/stormsense/releases apps/stormsense/public/data/releases --include '*.json'
rclone copyto r2:tcd/explorer/stormsense/latest.json apps/stormsense/public/data/latest.json
```

## Animated method walkthrough

`/method` is a standalone, lazy-loaded walkthrough that works without the live catalog. A continuous schematic signal loops through the model while the real predictions remain visible. Separate satellite and extra-tensor stacks expand on hover, keyboard focus, or tap, with selectors for all ten infrared bands and the five additional input planes. There is no timeline or explanatory section below the diagram. A comparison toggle retracts the field decoder and switches the displayed scalar predictions to the encoder-only checkpoint. Both modes use the global wind-unit preference. Reduced motion leaves a static diagram with interactive stacks; the signal loop pauses offscreen and in hidden tabs. A compact pause/resume control remains in the diagram header.

The bundled example uses Ian on **27 September 2022 at 11:30:20 UTC**, with actual GOES ABI inputs and predictions from both pinned checkpoints. The selected ABI test raster is reprojected to the storm center and passes the operational crop-validity checks (97.4% coverage). Selection is highest matched best-track intensity among eligible Ian test rasters, then earliest scan and sample ID. Internal feature graphics are schematic; this example is not a skill evaluation or a declaration of the currently deployed model.

Regenerate from the repository root with the existing Python environment, local dataset, IBTrACS file, and the two checkpoints/configurations referenced by the model manifests:

```bash
.venv/bin/python scripts/export_method_example.py
```

The exporter validates checkpoint/configuration/statistics checksums, uses shared operational preprocessing, checks the field's physical conversion and both scalar forward paths, and writes `public/method/`. `example.json` records source hashes, grid bounds, preprocessing, units, and model provenance. The download `wind-field.npy` retains the full 192×192 float32 prediction in m/s, north-up and west-to-east. Channel WebPs use a shared 190–300 K display scale; `wind-field.webp` uses the manifest's wind-speed color scale. `context-tensors.npy` stores the exact five additional normalized input planes (distance, solar-time sine/cosine, solar zenith, validity); their WebP previews use per-channel color scales recorded in the manifest. The exporter also reprojects the matched SAR target to the inference grid, independently checks pixel-center alignment, and verifies the autograd field-loss derivative against the analytical formula. `sar-target.npy`, `field-loss-mask.npy` and `field-gradient.npy` retain numerical values; image and source checksums, observation time, display scales and masking rules are recorded under `training` in the manifest. The small contextual map derives from the site's existing land geometry. These fixed assets ship with the website in both development and production builds; regeneration does not update the operational database, release pointers, or R2.

The adjacent Training tab illustrates prediction, supervision, and gradient updates for each architecture. Its joint objective is `L_field + L_wind + 0.25 L_structure`; encoder fine-tuning uses `L_wind + 0.25 L_structure`. These definitions come from `BottleneckUNetMLPRegressor` and the pinned joint resolved config (field Huber δ=2 m/s, intensity Huber δ=5 m/s, structure Smooth L1 β=20 km), and `historical.training.ScalarAdapter` plus the pinned encoder config. Joint labels are SAR fields and IBTrACS scalars; encoder fine-tuning uses ATCF scalars, with missing structure entries masked. The joint view compares the real matched RADARSAT wind field with the predicted field on the exact same grid and wind-speed color scale. A third map shows the actual masked Huber derivative with respect to predicted wind speed; hatching indicates excluded pixels. Backward pulses trace the decoder, scalar branch, skip connections and shared encoder; these routes are schematic, not parameter-gradient magnitudes. The held-out Ian pair illustrates the objective without updating model weights. Exact definitions are expandable. Tabs support arrow keys, Home and End; motion follows the same pause, visibility and reduced-motion settings as inference.

Run the focused browser checks with `npx playwright test tests/method.spec.ts`. They include desktop/mobile screenshots, unavailable archive data, unit/model switching, hover/tap/keyboard stack expansion, continuously visible predictions, looping flow, and reduced motion.

## Hourly imagery and playback

Storm detail pages keep the original single timeline slider, with play/pause and 2/4/8 hours-per-second playback. Opening a storm preloads its full image history, using eight concurrent downloads and prioritizing the selected hour and nearby previews. Compressed WebP bytes stay in memory for the open storm; distant slider jumps can decode locally without another network request. A separate 48 MiB decoded-image buffer keeps nearby 256 px previews and 768 px full frames ready. Preload progress appears in the existing imagery controls. Changing storms releases the old images; hiding imagery pauses background loading. Playback waits for its current image; scrubbing remains immediate. Unavailable hours hide imagery rather than keeping the previous picture under a new timestamp. A stopped playback commits a shareable time link.

Generate the optional image archive separately from inference:

```bash
uv run --group operational geo2wf-operational imagery --start 2025-10-02T16:01:46Z --end 2026-10-02T16:01:46Z --workers 4
# Reconcile missing display slots across the retained archive, including ended storms:
uv run --group operational geo2wf-operational imagery --workers 2
```

The job uses saved hourly centers or track interpolation. It only calls NASA GIBS; unavailable data is left out with a reason, without a raw-data fallback or model rerun. Use `--limit`, `--storms` or `--retry-gaps` for a bounded retry. Completed frames resume safely. To stop gracefully, create `<db>.stop-imagery`, wait for exit, then remove it before resuming. An update request also drains and stops imagery work so inference can take priority. No new service or scheduler is activated.

**Keep the WebP, `.webp.aux.xml` sidecar and STAC JSON together.** They contain EPSG:3857/WKT, exact pixel transforms, full/preview grids, WGS84 footprints, antimeridian parts, image checksums, product/retrieval/generation times and source URLs. This supports later GIS or map-engine migration independently of Leaflet. See [the imagery contract](CONTRACT.md#optional-hourly-display-imagery-gibs-geocolor-webp-v1).

Processing assets are under `var/stormsense/geocolor/imagery`. Export copies referenced assets into its `imagery/` directory; publication uploads them before the release pointer. Restore a new runner's processing assets with:

```bash
rclone copy r2:tcd/explorer/stormsense/imagery var/stormsense/geocolor/imagery
# Restore imagery for the local website preview as well:
rclone copy r2:tcd/explorer/stormsense/imagery apps/stormsense/public/data/imagery
```

## Default update and catch-up

`geo2wf-operational update` first retrieves current advisories and computes the current live hour. It then refreshes historical tracks across the rolling year plus twelve hours of forecast context and checks every expected storm-hour against SQLite. Interior holes and storms that ended during an outage are included; a newer live record does not hide earlier missing work. Existing predictions and recorded failed attempts for the pinned model are left alone. Only never-attempted slots are queued. Missing hours run satellite retrieval and inference, save independently, and generate retrospective forecasts when context is available. The current live forecast is checked again after catch-up supplies its context. Use `update --workers 2` to reduce catch-up concurrency.

The Python `update` command also builds website WebP images and reconciles the entire retained display-image archive at its existing two-hour cadence, including ended storms and missing local assets. Use `--imagery-workers 2` to control image-fetch concurrency independently of inference. Image catch-up runs even when predictions are already current or the numerical stage fails. The finite runner (`STORMSENSE_PUBLISH=1 apps/stormsense/runner/cycle.sh`) then regenerates evaluation and website exports, and uploads and verifies the release before advancing the R2 pointer. Publication runs even if the local numerical data was already current, so a previously failed upload is repaired on the next successful cycle. `update` saves predictions and images locally; the runner completes export and website publication.

Image catch-up automatically retries source gaps after one hour for frames less than 48 hours old, or after one day for older frames. A missing-center image is retried as soon as a saved prediction or track supplies its center. `imagery --retry-gaps` bypasses these cooldowns; combine it with `--storms STORM_ID` to repair a selected storm. Successful image files are reused. Numerical gaps still require `backfill --retry-gaps`. Unavailable sources remain explained gaps, never fabricated predictions. Interrupted work resumes from committed records; completed results and historical live forecast issues remain immutable. Nothing in this procedure activates a scheduler.

## Finite pipeline commands

```bash
# Discover the preceding calendar year of tracks (explicit dates are optional).
uv run --group operational geo2wf-operational discover-history --start 2025-10-02T00:00:00Z --end 2026-10-02T10:00:00Z

# Resume newest hours first; use --retry-gaps to retry previously recorded gaps.
uv run --group operational geo2wf-operational --device cpu backfill --start 2025-10-02T00:00:00Z --end 2026-10-02T10:00:00Z --skip-discovery --workers 4

# Optional bounds: --limit 48, --storms AL132025 EP152026.
# The backfill includes twelve hours of forecast context before --start.
uv run --group operational geo2wf-operational coverage --start 2025-10-02T00:00:00Z --end 2026-10-02T10:00:00Z
uv run --group operational geo2wf-operational --device cpu verify
uv run --group operational geo2wf-operational evaluate
uv run --group operational geo2wf-operational export --output var/stormsense/export
uv run --group operational geo2wf-operational publish --output var/stormsense/export
uv run --group operational geo2wf-operational retain --output var/stormsense/export
# Review the deletion list, then use retain --apply to enforce local retention.
```

`coverage.json` accounts for every expected hour from the first recorded fix, intersecting the public window. Ended storms stop at their final fix; active systems continue through the requested window's end. Predictions, explained gaps and pending work are separate counts; **pending is never relabeled as a gap**. A complete archive has `pending: 0` and no discovery failures. Track revisions can add hours on a later run. Fixes more than six hours apart do not authorize interpolation. Recent active hours can therefore have valid live estimates but retrospective gaps while the next bracketing BEST fix is still unavailable.

SQLite commits each result independently; interrupted jobs resume without recomputing completed predictions. Ready numerical records and issued forecasts are immutable for a given model version. Use a new pipeline/model version for a changed inference method. Source snapshots contain only NHC/ATCF metadata. Satellite arrays, masks and intermediate wind fields exist in process memory and are discarded. Backfill groups storms by UTC hour to share compressed source strips in memory; those caches close at the end of each batch. HTTP listing requests and range reads have bounded retries. A transient failure remains a retryable gap with its reason.

Only one update/backfill runs for a database. An `update` requests priority: backfill stops submitting work, drains the bounded in-flight queue and yields its lock. Resume the backfill command afterward. To stop backfill without discarding its current work, create `var/stormsense/state.sqlite.stop-backfill`, wait for exit, and remove that file before resuming. A stopped job is not reported complete. Tune `--workers` to available memory and network capacity; CPU inference is supported and GPU use is optional.

## Inputs and prediction methods

- Discovery: [NHC CurrentStorms](https://www.nhc.noaa.gov/CurrentStorms.json), [ATCF current and yearly tracks](https://ftp.nhc.noaa.gov/atcf/). Numbered, advised Atlantic/eastern/central Pacific systems are included; unadvised invests are excluded. IDs survive boundary crossings. The displayed basin follows advisory jurisdiction/location, without renaming an EP-origin system when it reaches the central Pacific.
- Live centers: the latest nonfuture advisory or track position plus fresh reported advisory motion, with position source, fix age and estimation method recorded; no extrapolation beyond six hours. Historical centers interpolate recorded fixes no more than six hours apart and are explicitly retrospective.
- Imagery: public [NOAA GOES ABI](https://registry.opendata.aws/noaa-goes/) Level-2 full-disk multichannel cloud-and-moisture imagery (`ABI-L2-MCMIPF`), Kelvin channels 7–16. Choose the operational East/West platform for the date and the better viewing angle. Use the latest complete scan ending no more than 30 minutes before the hourly slot. Live reads also exclude objects published after the update's retrieval cutoff.
- Geolocation: the source NetCDF projection and fixed-grid coordinates drive the crop; nearest-neighbor reprojection matches the research exporter, with longitude unwrapping around the storm. Decode physical Kelvin values before normalization. DQF 0/1 values are accepted. Require at least 90% valid pixels in the 192×192 crop and a valid central 4×4 patch.
- Nowcast: pinned `latent_sar_no_era5_max_wind_radii`, its robust-zscore statistics (clip 4), 256×256 grid at 0.027°, central 192×192 crop, center-distance channel and three solar channels. Publish the scalar heads: maximum wind, RMW and equivalent-area R34/R50/R64. No official intensity is substituted for missing model output.
- Forecast: pinned `dashboard-mlp` and its saved feature-scaling buffers. Inputs are matching StormSense estimates at t, t−6h and t−12h; +6h is direct and +12h recursive. No official or future observations enter these inputs. Missing warm-up history withholds the issue. Live issues may use already-computed retrospective prehistory, with each input mode and generation time recorded; future-generated estimates are excluded. Retrospective issues remain separate from forecasts issued live.
- Evaluation: compare exact valid-time ATCF reference intensities with the model, persistence and recent-trend baselines. Report +6/+12 h, live/hindcast and storms excluded from nowcast training separately. The report includes sample/storm counts, MAE, RMSE, bias and storm-macro MAE. No uncertainty bands or unsupported skill claims are published.

The evaluation command defaults to the rolling year of forecast anchors. Use `evaluate --start <UTC> --end <UTC>` to reproduce a particular archive report; preceding forecast context remains available as input but is excluded from the scored issue window.

The read-only Worker serves schema-versioned JSON from R2. See [the data contract](CONTRACT.md). Raw numerical model inputs are never published; compact, separately sourced GeoColor display images and georeferencing metadata are published when available.

## Website hosting and deployment

[`wrangler.jsonc`](wrangler.jsonc) configures a separate Workers Static Assets application named `stormsense`, SPA routing and a read-only R2 binding to bucket `tcd`, prefix `explorer/stormsense/`. It has **no cron trigger**. Browser requests use same-origin `/data/`; `VITE_DATA_BASE` can change that root for another read-only host.

The Worker uses Cloudflare's Cache API for immutable images, metadata and releases up to 8 MiB per object, in addition to browser caching. Cache hits skip the R2 binding. `latest.json` always bypasses edge cache; errors and empty HEAD responses are never cached. HEAD and conditional GET requests can reuse an already cached full object. `X-StormSense-Cache` reports `HIT`, `MISS` or `BYPASS`. Cache storage is local to each Cloudflare data center and may be evicted, so this reduces reads without guaranteeing a fixed hit rate. Worker invocations still count toward the account's Workers plan.

Daily ZIPs now preload the open storm with one request per available UTC day. Each package includes full/preview WebPs, their GDAL sidecars, STAC metadata and a portable manifest. Unchanged days reuse identical content-addressed objects; live updates replace only the current day's reference. Individual image/metadata URLs remain available for GIS downloads and older releases. Nolo now needs 23 image-package requests totaling 28,422,791 bytes, compared with 1,062 individual image requests and 50,043,866 bytes in the previous release. At [R2 Standard prices](https://developers.cloudflare.com/r2/pricing/) checked on 3 October 2026, reads include a shared 10-million-operation monthly allowance, then cost $0.36 per million with billing-unit rounding; egress is free. This is a sizing comparison, not an account bill. Browser and edge cache hits reduce actual R2 reads further.

For a data-format deployment, `publish --stage-only` uploads and verifies immutable assets without moving the pointer. Deploy the compatible Worker/UI, then advance the already verified `latest.json` with `rclone copyto --checksum --s3-no-head`. Regular `publish` retains its upload/verify/pointer-last behavior. Never advance a staged pointer before all upload and checksum checks succeed.

```bash
cd apps/stormsense
npm run check:worker
npm run deploy:check   # builds and bundles; does not deploy
# Authorize the Cloudflare account containing the existing tcd bucket.
# Device login also works when the browser is on another machine.
npx wrangler login --device
npx wrangler whoami
npm run deploy
# Then test the actual HTTPS URL printed by Wrangler:
STORMSENSE_BASE_URL=https://<deployed-hostname> npm run test:e2e
```

Use Node 22 (`.nvmrc`); with an older system Node, prefix the commands with `npm exec --yes --package=node@22 --`. The existing rclone credentials authorize R2 data uploads, not Worker deployments. Wrangler therefore needs its own Cloudflare login. The custom domain `stormsense.hyperalislabs.com` is declared in `wrangler.jsonc`; Wrangler provisions its DNS record and HTTPS certificate in the account’s `hyperalislabs.com` zone. The `stormsense.hyperalislabs.workers.dev` address also remains available. If the account has several Workers already, confirm that the new `stormsense` name is unused before the first deploy.

Hosting the website and running inference are separate deployments. The website reads R2 through its binding and stays available while the runner is stopped. Automatic updates currently use the temporary local hourly timer. For permanent hosting, place the CPU runner on an always-on Linux host with persistent storage for SQLite and the pinned models. Stop the temporary publisher before installing the 15-minute timer on that selected host, enable `STORMSENSE_PUBLISH=1`, and configure state backups, retention and failure monitoring. No GPU is required by the verified CPU path. A website deployment alone does not activate prediction updates.

The [hosted-runner handoff](runner/README.md) includes exact activation steps. `python3 apps/stormsense/runner/package.py` creates a portable source/model bundle and a consistent SQLite snapshot without credentials or imagery; the prepared bundle's [checksum and verification](reports/runner-bundle.json) are recorded. Moving to permanent hosting still requires server access or a selected cloud account/budget.

Local preview data in `public/data` must be excluded from production static assets: the production Vite build uses `STORMSENSE_PRODUCTION=1` through the deploy scripts. Data is served by the R2 Worker route instead.

Publication uses the authenticated **rclone** remote `r2:tcd/explorer/stormsense`. It uploads content-addressed storm objects and an immutable UTC release before `rclone copyto` advances this application's `latest.json`. It never touches the existing explorer's pointer. Unchanged storm objects are reused. Local retention keeps a rolling calendar year plus 12 hours of forecast context; release cleanup preserves objects referenced by retained catalogs. Do not use an R2 lifecycle rule that deletes shared objects purely by age.

Publication compares hashes with `--checksum`, including for immutable assets. This permits republishing or restoring equal bytes with different local modification times without attempting unsupported R2 metadata rewrites. Changed immutable content still fails, and a failed upload never advances the pointer. See [rclone's checksum behavior](https://rclone.org/docs/#checksum).

Uploads use `--s3-no-head` because this endpoint returns a version ID but rejects the version-specific HEAD request made by the installed rclone. Publication runs a separate `rclone check --one-way` over imagery, objects and release metadata before advancing the pointer; checksum verification remains required.

[`runner/cycle.sh`](runner/cycle.sh), [`runner/Dockerfile`](runner/Dockerfile) and the example systemd units provide portable, finite entry points. The temporary host currently uses the separate hourly `stormsense-local.timer` described in the [local runbook](runner/LOCAL.md). Stop it before activating a Docker scheduler or permanent-host timer. On a replacement host, start with `STORMSENSE_PUBLISH=0` and enable publication only for the selected runner with the existing authenticated rclone configuration. Every update reconciles missing numerical hours across the rolling year plus twelve hours of forecast context; explicit backfill remains available for a selected range and yields to updates. The runner also reconciles display imagery across the rolling year, including ended storms.

The [full-stack Docker guide](runner/DOCKER.md) covers CPU/GPU training, pinned model downloads, inference scheduling, data export, website hosting and deployment. Root `compose.yaml` defines independent services with persistent mounts. `compose.r2.yaml` mounts the authenticated rclone configuration read-only for publication; the image runs as an unprivileged user. Use `tools operational ...` for the StormSense CLI and `tools train/infer/export/evaluate ...` for research workflows. Container builds are checked by the Docker CI workflow; no scheduler is activated by building an image.

## Verification

```bash
uv run --group operational pytest -q
uv run --group docs mkdocs build --strict
uv run python scripts/check_site_links.py
cd apps/stormsense
npm test
npm run build
npm run check:worker
# Requires an exported real archive with forecasts and an active system's live issue:
npx playwright install chromium
npm run test:e2e
npm run test:export-refresh # exports a new real release while a storm page stays open
```

`geo2wf-operational verify` samples real Atlantic, eastern Pacific and central Pacific recorded centers, checks Kelvin values and preprocessing against the actual training loader using in-memory rasters, runs the pinned checkpoints, and records range bytes/runtime in `var/stormsense/verification.json`. `evaluation.json` and `coverage.json` are regenerated from committed state. Browser tests exercise real forecasts, deep links, downloads and keyboard input on desktop/mobile; quiet/error feeds are explicitly simulated in tests.

If discovery fails, the finite runner exports the recorded failure and last known data, then returns the failed update's exit code. Source freshness always uses the last successful retrieval, independently of the newer export time.

The basemap is the repository's Natural Earth GeoJSON (public domain). The existing ESL logo and brand assets are reused. The Lora web font has a local serif fallback. There is no opening modal.

Remote retention is available with `retain --remote` (dry run) and `retain --remote --apply`. Run export, publication and retention on one runner with the same export directory; the CLI commands share a local publication lock so cleanup cannot remove an export's pending objects. Direct Python API callers should provide the same serialization. Remote cleanup checks the pointer and release set again, protects retained and recent catalogs and their objects, and gives new uploads a 24-hour grace period. No remote cleanup has been scheduled.

Operational satellite transition dates are taken from NOAA's completed transition notices, rather than earlier proposed dates: [GOES-19, 7 April 2025 at 15:10 UTC](https://www.ospo.noaa.gov/data/messages/2025/04/MSG_20250407_1510.html) and [GOES-18, 4 January 2023 at 18:00 UTC](https://www.ospo.noaa.gov/data/messages/2023/01/MSG_20230104_1805.html).

Reproduce the smoothing sweep with Node 22.18+: `node playground/evaluate-smoothing.mjs` from the repository root. The script uses the website filter implementation and downloaded export, and writes `playground/smoothing-results.json` with input hashes, all 459 settings, per-storm scores and leave-one-storm-out selection. The report ranks exact-time pooled wind MAE and includes 6-hour change error and nominal delay. The website uses EMA alpha 0.5, the lowest-MAE tested setting with at most one hour nominal delay; the unconstrained winner is a 17-hour median and is not deployed. This is retrospective tuning, not independent operational validation.

Every map shows subdued dashed NHC/CPHC basin limits, following NOAA's [National Hurricane Operations Plan, section 2.2.1 and Appendix O, Figure O-1](https://www.weather.gov/media/nws/IHC2023/2023_nhop.pdf#page=192): Atlantic north of the equator, eastern North Pacific east of 140°W, and central North Pacific between 140°W and 180°. The meridians end at land as in the provider's schematic; coastlines form the remaining ocean-basin edges. No artificial northern cutoff is drawn. These are storm-feed basin boundaries, not the GOES satellite footprint or guaranteed model coverage. The small map key links to NOAA's definition.
