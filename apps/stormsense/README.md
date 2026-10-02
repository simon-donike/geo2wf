# StormSense

Independent React/TypeScript/Vite application and NHC/CPHC prediction pipeline. The existing documentation and explorer retain their own build and R2 pointer.

**Hosted website:** https://stormsense.hyperalislabs.workers.dev/ — deployed to the Hyperalis Labs Cloudflare account and reading the existing R2 archive. Automatic prediction scheduling remains inactive until its runner host is selected.

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

The local database is `var/stormsense/state.sqlite`. Generated data and logs under `var/stormsense` are excluded from Git; delivery reports under `apps/stormsense/reports` are reviewable repository artifacts. The app has its own routes `/`, `/storms/:id`, `/archive`, `/about`. Append `?time=<UTC ISO timestamp>&issue=<forecast anchor>` for a shareable selection. Auto-refresh preserves an explicitly selected time. Wind units default to knots; the toggle persists locally. JSON retains canonical m/s and km. All dates are UTC.

After publication, restore the numerical archive into a fresh checkout without rerunning satellite inference:

```bash
rclone copy r2:tcd/explorer/stormsense/objects apps/stormsense/public/data/objects --include '*.json'
rclone copy r2:tcd/explorer/stormsense/releases apps/stormsense/public/data/releases --include '*.json'
rclone copyto r2:tcd/explorer/stormsense/latest.json apps/stormsense/public/data/latest.json
```

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

The read-only Worker serves schema-versioned JSON from R2. See [the data contract](CONTRACT.md). Raw imagery is never published.

## Website hosting and deployment

[`wrangler.jsonc`](wrangler.jsonc) configures a separate Workers Static Assets application named `stormsense`, SPA routing and a read-only R2 binding to bucket `tcd`, prefix `explorer/stormsense/`. It has **no cron trigger**. Browser requests use same-origin `/data/`; `VITE_DATA_BASE` can change that root for another read-only host.

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

Use Node 22 (`.nvmrc`); with an older system Node, prefix the commands with `npm exec --yes --package=node@22 --`. The existing rclone credentials authorize R2 data uploads, not Worker deployments. Wrangler therefore needs its own Cloudflare login. The first deployment can use the account's `workers.dev` address; a custom domain can be attached afterward. If the account has several Workers already, confirm that the new `stormsense` name is unused before the first deploy.

Hosting the website and running inference are separate deployments. The website reads R2 through its binding and stays available while the runner is stopped. For automatic updates, place the existing CPU runner on an always-on Linux host with persistent storage for SQLite and the pinned models. Install the 15-minute timer only on that selected host, enable `STORMSENSE_PUBLISH=1`, and configure state backups, retention and failure monitoring. No GPU is required by the verified CPU path. A website deployment alone does not activate prediction updates.

Local preview data in `public/data` must be excluded from production static assets: the production Vite build uses `STORMSENSE_PRODUCTION=1` through the deploy scripts. Data is served by the R2 Worker route instead.

Publication uses the authenticated **rclone** remote `r2:tcd/explorer/stormsense`. It uploads content-addressed storm objects and an immutable UTC release before `rclone copyto` advances this application's `latest.json`. It never touches the existing explorer's pointer. Unchanged storm objects are reused. Local retention keeps a rolling calendar year plus 12 hours of forecast context; release cleanup preserves objects referenced by retained catalogs. Do not use an R2 lifecycle rule that deletes shared objects purely by age.

[`runner/cycle.sh`](runner/cycle.sh), [`runner/Dockerfile`](runner/Dockerfile) and the example systemd units provide portable, finite entry points. Nothing has installed or activated them. Once hosting is selected, schedule `cycle.sh` every 15 minutes: discovery happens each invocation, and hourly slots are idempotent. Start with `STORMSENSE_PUBLISH=0`; enable publication only for the selected runner with the existing authenticated rclone configuration. Backfill is a separate finite job and yields to recent updates.

The container accepts the same CLI arguments. Mount the bootstrapped models at `/app/downloads/models` and persistent state at `/app/var/stormsense`. Publication additionally needs the existing rclone configuration mounted read-only at `/root/.config/rclone`. The image includes rclone; credentials are never part of the build. The Docker configuration has not been built locally because this workstation has no Docker engine.

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
