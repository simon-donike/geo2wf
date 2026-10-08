# Two-hour finetuning migration

The original `models.json` remains the production default until quality gates
pass. The candidate `models-finetuned.json` selects the pinned encoder adapter,
even UTC hours, and a tropical/subtropical classification start gate. Subsequent
weakening does not close that gate. Historical reruns remain hindcasts.

Use the repository Python environment and the authenticated rclone remote.
Do not publish a candidate after a failed comparison. The publisher also
requires a passing full-archive gate matching the candidate model version.

## Automated continuation after a passing pilot

`python -m geo2wf.operational.migration_job --workers 12 --deploy --node-bin
/path/to/node22/bin` performs the remaining rebuild, retries gaps, checks the
full quality gate, deploys the compatible frontend, reconciles and checks again,
verifies image preservation, publishes, switches the existing runner, and
monitors two even-hour cycles. Set `STORMSENSE_MODEL_MANIFEST` to the candidate
manifest before starting. Omit `--deploy` to stop after full-archive validation.

The process must persist after logout. Its progress is recorded in
`var/stormsense-migration/job-status.json`. A failure before cutover leaves
production unchanged; a failure during cutover/monitoring restores the saved
pointer and runner settings. Source edits during the rebuild stop deployment
instead of silently deploying a different implementation. The detailed sequence
below is also the manual recovery procedure.

## Staging and pilot

```bash
export STORMSENSE_MODEL_MANIFEST="$PWD/src/geo2wf/operational/models-finetuned.json"
.venv/bin/python -m geo2wf.operational.rebuild prepare
```

Preparation makes a consistent SQLite backup at
`var/stormsense-migration/state.sqlite`, shares immutable imagery, pins the
pre-migration release locally, and writes `migration.json`. It refuses to replace
an existing staging database. That report fixes the evaluation window and pilot
storm IDs; the first rebuild measured 2,670 eligible slots and 570 pilot slots.

Run `geo2wf.operational.cli --db var/stormsense-migration/state.sqlite
--device cuda backfill --skip-discovery --workers 12 --start START --end END
--storms PILOT_IDS`, using the report's window and IDs. `--start` includes the
existing twelve-hour context extension; the eligibility gate still applies.
Repeat with `--retry-gaps` after resolving transient acquisition failures.
The command is restartable and skips ready candidate rows.

```bash
.venv/bin/python -m geo2wf.operational.rebuild compare
```

The comparison uses identical timestamps and official labels for old/new
predictions and exact forecast anchors. Its exit status is nonzero for missing
coverage, missing comparison labels, or any agreed metric regression. Results
stay local in `pilot-comparison.json`.

Only after the pilot passes, backfill the complete report storm list with the
same window and run `rebuild compare --full`. Keep the original publisher running
through this work. Do not run the staging `update` command while backfill is active.

## Reconciliation and cutover

1. Save the running service configuration and current remote pointer for rollback.
   Pause `stormsense-local.timer` and stop its service. Take a fresh SQLite backup.
2. Run `rebuild reconcile`. It takes both database locks and the production cycle
   lock, merges intervening official observations/images without replacing
   candidate predictions, and invalidates the old full quality gate.
3. Run one staging `update`, then finish any pending candidate slots and repeat
   `rebuild compare --full` over the refreshed window. Stop if the gate fails.
4. Export staging data to `var/stormsense/export` with the candidate manifest.
   Compare the pinned and candidate catalogs: preserve storm tracks and every
   existing in-window imagery record, asset hash, georeferencing sidecar, and
   bundle. Check all image references. The rolling window may age out records;
   their binaries remain protected by the pinned old release.
5. Upload the old release's `pin.json` with `rclone copyto` to the matching
   `r2:tcd/explorer/stormsense/releases/VERSION/pin.json`. Publish the candidate
   with `publish --stage-only`, verify its objects, and then publish normally.
   Publication uploads immutable assets before advancing `latest.json`.
6. Deploy the tested backward-compatible frontend. Configure the runner with
   `STORMSENSE_DB=/work/code/geo2wf/var/stormsense-migration/state.sqlite` and
   `STORMSENSE_MODEL_MANIFEST=/work/code/geo2wf/src/geo2wf/operational/models-finetuned.json`.
   Keep its existing export path, hourly timer, and image-acquisition settings.
7. Resume the timer. Verify two even-hour inference cycles, each with a successful
   export/publication, the candidate model version, fresh eligible predictions,
   and unchanged existing images. Odd-hour cycles still discover and publish.

Do not run imagery deletion during migration. Normal JSON retention keeps seven
recent releases plus any release containing `pin.json`; pinned catalogs protect
their referenced imagery. Remote retention retains its 24-hour grace period.

## Rollback

Stop the publisher, restore the saved runner configuration and original database
selection, and restore the saved remote pointer with `rclone copyto`. Reload
systemd and resume the timer. Both the runner and pointer must be restored;
changing only the pointer would be undone by the next publication. Keep the
staging database and comparison reports for diagnosis.
