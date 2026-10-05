# Temporary hourly runner

StormSense uses this checkout's existing Python environment and pinned models
on the local Linux host until permanent hosting is selected. The user systemd
timer runs at **five minutes past every hour**, with a first run shortly after
activation. It starts a finite update/imagery/evaluation/export/publication cycle;
there is no resident Python process between runs. Publication uses the existing
authenticated rclone remote and verifies immutable uploads before advancing the
StormSense pointer.

`stormsense-local.service` is specific to `/work/code/geo2wf`. It uses CPU only,
one inference worker, one imagery worker, low CPU/I/O priority and a one-core
CPU quota. The combined process group has a 1.5 GiB soft memory limit and a
2 GiB hard limit; these are limits, not reserved idle memory. Systemd and the
existing database lock prevent overlapping cycles.
The 55-minute time limit bounds a stuck cycle; already committed work resumes
on the next run. A cycle that fails is retried at the next scheduled hour.

The current user already has systemd lingering enabled, so the timer remains
available after logout and starts again after reboot. It cannot run while the
machine is powered off, suspended or disconnected. A persistent timer catches a
missed invocation after startup, and the pipeline reconciles missing hours.
The independently hosted website remains available with its last published data.

## Operate

```bash
systemctl --user list-timers stormsense-local.timer
systemctl --user status stormsense-local.service
journalctl --user -u stormsense-local.service --since today

# Run an extra cycle now; systemd does not overlap an already running invocation.
systemctl --user start stormsense-local.service

# Stop scheduling and any current cycle.
systemctl --user disable --now stormsense-local.timer
systemctl --user stop stormsense-local.service

# Resume hourly updates.
systemctl --user enable --now stormsense-local.timer
```

A successful oneshot service becomes `inactive (dead)` between runs; check
`Result=success` and `ExecMainStatus=0`, together with the published data times:

```bash
systemctl --user show stormsense-local.service \
  -p Result -p ExecMainStatus -p MemoryPeak -p CPUUsageNSec
```

If systemd has unloaded the service, its journal retains the completion result
and resource summary.

The timer remains `active (waiting)`. The website's actual source timestamps are
the freshness check; a completed export alone does not guarantee fresh upstream
observations. Investigate a failed service or published data more than two hours
behind its expected update. No external alerting service is configured.

## Installation and removal

The live units are linked from this repository, so moving or removing this
checkout requires updating the units. Editing a linked unit requires a daemon
reload; timer edits also require a timer restart:

```bash
systemd-analyze --user verify apps/stormsense/runner/stormsense-local.service \
  apps/stormsense/runner/stormsense-local.timer
systemctl --user link "$PWD/apps/stormsense/runner/stormsense-local.service" \
  "$PWD/apps/stormsense/runner/stormsense-local.timer"
systemctl --user daemon-reload
systemctl --user enable --now stormsense-local.timer
```

The existing rclone config stays in the user's normal config location. It is
not copied into the repository or service. SQLite remains at
`var/stormsense/state.sqlite`; models remain under `downloads/models`, and
exports/imagery remain under `var/stormsense`. A consistent pre-activation
snapshot is saved at `var/stormsense/backups/before-local-hourly.sqlite`.
This is a one-time snapshot, not an automated backup schedule.

Before migrating, disable this timer and stop its service, then take a new
consistent SQLite snapshot. Start the replacement publisher only after this
one has stopped. Do not enable the Docker scheduler or the permanent-host
15-minute timer alongside this temporary timer. To uninstall the local units
after stopping them:

```bash
systemctl --user disable stormsense-local.timer stormsense-local.service
systemctl --user daemon-reload
```

Verification and measured resource use are recorded in
[`local-runner.json`](../reports/local-runner.json).
