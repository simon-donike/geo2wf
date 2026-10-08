# Hosted CPU runner

For Docker, use the [full-stack container guide](DOCKER.md) and root
`compose.yaml`. It includes training, model downloads, inference scheduling,
website serving, deployment tools and persistent mounts. The instructions below
describe the alternative native Python/systemd installation; use one scheduler.

The website is deployed independently at https://stormsense.hyperalislabs.com/.
An hourly user timer is active on the temporary local host, using the existing
Python environment, one worker per processing stage and CPU only. See the
[local runbook](LOCAL.md) for its resource limits, status and stop commands.
Permanent hosting still needs a selected Linux server or cloud account/budget.

## Prepared handoff

Run `python3 apps/stormsense/runner/package.py` from the repository. It creates
`var/stormsense/runner.tar.gz` with the runner source, dependency lock, verified
pinned models, and a consistent SQLite backup of the completed archive. It
excludes credentials, imagery and the local virtualenv. The generated checksum
identifies the exact handoff. Rebuild the package immediately before transfer.

Use a Linux host with Python 3.10 or 3.11, `uv`, `rclone`, and systemd. A starting
allocation is 2 CPU cores, 4 GB memory and 25 GB persistent disk; confirm memory
and runtime on the selected host. The current installed Python environment uses
5.3 GB, while models and numerical state total about 104 MB. No GPU is required.

On the selected host, create a `stormsense` service user and extract the bundle
under `/srv/geo2wf`, owned by that user. As that user:

```bash
cd /srv/geo2wf
uv sync --frozen --python 3.10 --group operational --no-group dev
```

Transfer the existing authenticated rclone configuration securely to the service
user's `~/.config/rclone/rclone.conf` with mode `0600`. Do not put credentials in
the bundle, source control, unit files, logs or chat. Verify that the remote is
named `r2` and can read `r2:tcd/explorer/stormsense/latest.json`.

Restore the saved display assets referenced by the SQLite snapshot before exporting. The compact handoff bundle excludes these WebP files and sidecars, which already live in R2:

```bash
rclone copy r2:tcd/explorer/stormsense/imagery var/stormsense/geocolor/imagery
```

Run one finite cycle as the service user, then confirm the release on the hosted
website:

```bash
STORMSENSE_PUBLISH=1 apps/stormsense/runner/cycle.sh
```

## Activate only on the selected host

These are the permanent-host instructions. Stop `stormsense-local.timer` and
`stormsense-local.service` on the temporary host before activating a replacement.

After the finite cycle passes, install `stormsense.service.example` and
`stormsense.timer.example` into `/etc/systemd/system/` without the `.example`
suffix. Set `Environment=STORMSENSE_PUBLISH=1` in the installed service. Run:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now stormsense.timer
sudo systemctl list-timers stormsense.timer
sudo journalctl -u stormsense.service --since today
```

The timer polls every 15 minutes in UTC. Each invocation is finite. It updates
current live predictions, refreshes historical tracks, and reconciles every
expected numerical hour in the rolling year plus twelve hours of forecast
context. It fills interior holes and includes storms that ended while the runner
was offline, leaving successful numerical records and previously attempted numerical gaps alone.
The Python `update` command also reconciles display imagery across the rolling
year at the existing two-hour cadence, including missing local assets. Image
source gaps retry hourly for frames less than 48 hours old and daily for older
frames; missing-center gaps retry once a center becomes available. Explicit
`imagery --retry-gaps` bypasses cooldowns. Image-source gaps do
not block numerical publication. Evaluation, export and verified pointer-last
publication follow catch-up, even when no new inference was needed.

systemd prevents overlapping runs of the same service; interrupted work resumes
through SQLite. An explicit backfill remains available for selected ranges and
yields to updates. The default update performs its own catch-up automatically.

Record the first successful scheduled run and its public release before marking
the deployment complete. Remove the site's "scheduling not activated" statement
only after that verification. Configure host monitoring for a failed
`stormsense.service`, and alert when the published source freshness exceeds 30
minutes. Use the host provider's persistent-disk backups plus SQLite's backup
API; do not copy an open SQLite database without its journal.

Run `geo2wf-operational retain` to review retention, then schedule
`geo2wf-operational retain --apply` for the local rolling-year window on this
single runner. Remote release cleanup uses `retain --remote --apply` and must
share the runner's export directory/publication lock. Configure those maintenance
jobs and backup retention for the chosen host. Restore the latest consistent
database snapshot and model bundle before restarting a replacement runner.

To stop automatic updates: `sudo systemctl disable --now stormsense.timer`.
The hosted website and published archive remain available.
