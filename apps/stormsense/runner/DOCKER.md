# Full-stack Docker workflow

The root `compose.yaml` builds the Python research/operational code, training
configs, scripts, Hugging Face CLI, Node 22 website tooling, Wrangler, rclone,
MkDocs and Python tests. Python and npm dependencies use `uv.lock` and
`package-lock.json`. CPU and CUDA images use the same source and PyTorch 2.1.2;
the CPU image omits CUDA libraries. Python remains on the project's supported
3.10 line. Rebuild images to receive base-OS updates; release deployments can
add image digests to pin the base images as well.

The image runs as an unprivileged user. Datasets, downloaded models, checkpoints,
runtime state, exports and credentials are excluded from the build context.
Mounts preserve them across container replacement. Run only one publishing
runner across all hosts. This setup does not change model selection or promote
new training checkpoints into the pinned operational model release.

## Prepare and build

Use a Linux x86-64 Docker host with the Compose v2 and Buildx plugins. On macOS
or ARM machines these images require amd64 emulation; native ARM/GPU support is
not claimed. Run all commands below from the repository root.

```bash
# Do this before Compose creates bind mounts, to preserve your ownership.
mkdir -p downloads data logs inference var/stormsense/export var/docker-artifacts/browser
export LOCAL_UID="$(id -u)"
export LOCAL_GID="$(id -g)"
docker compose build tools web
docker compose run --rm tools help
```

Optional non-secret defaults are in `runner/docker.env.example`; copy it to
the repository root as `.env` and edit it for the host. The UID/GID used to build
the image must be able to read/write the host directories and read any mounted
0600 credential file. Rebuild when changing UID/GID. All services use profiles:
building images starts nothing, and plain `docker compose up` selects no services.

| Service | Purpose |
| --- | --- |
| `tools` | Finite CLI jobs, dataset export, downloads, evaluation, tests, shell |
| `runner` | Sequential full inference cycles with automatic restart |
| `train-cpu` | Finite CPU training jobs |
| `train-gpu` | CUDA training, with one NVIDIA GPU reserved |
| `web` | Production static website serving the local exported archive |
| `web-dev` | Vite preview of the source baked into the image |
| `deploy` | Cloudflare build/deployment tools; defaults to a dry run |
| `browser-tests` | Optional image with Chromium and its system dependencies |

The default Dockerfile target is also the full-stack CPU image:

```bash
docker build -f apps/stormsense/runner/Dockerfile -t geo2wf:cpu .
```

## Bootstrap and run inference

```bash
# Download only the pinned operational model assets into ./downloads/models.
docker compose run --rm tools operational bootstrap

# Finite full cycle: update/catch-up, imagery, evaluation, export. No upload.
docker compose run --rm runner cycle

# Optional local website, reading ./var/stormsense/export at runtime.
docker compose up -d web
# Open http://127.0.0.1:8080

# Activate ongoing local computation. Publication remains off by default.
docker compose up -d runner
docker compose logs -f runner
docker compose stop runner
```

The scheduler starts immediately, then waits 900 seconds **after each completed
cycle**. Failed cycles are logged and retried on the next interval. A database
lock covers the entire cycle, so a manual cycle cannot overlap its scheduled
counterpart (exit 75 means already running). The scheduler forwards shutdown to
the running process group; committed SQLite records survive interruption.
Compose allows two minutes before forced termination. Do not also enable the
example host systemd timer for the same database.

`STORMSENSE_WORKERS`, `STORMSENSE_IMAGERY_WORKERS`,
`STORMSENSE_INTERVAL_SECONDS`, `RUNNER_CPUS` and `RUNNER_MEMORY` are configurable
in `.env`. Defaults are two inference workers, two imagery workers, two CPUs
and a 4 GiB memory cap. These are initial limits, not verified production peak
requirements; measure a complete catch-up on the selected host and adjust.
The last completed cycle's timestamps/exit code are recorded beside SQLite in
`var/stormsense/state.scheduler.json`. Monitor failed cycles, publication/source
freshness and free disk space; an HTTP health check only verifies the web server.

Finite operational commands remain available:

```bash
docker compose run --rm tools operational --device cpu update --workers 2
docker compose run --rm tools operational coverage
docker compose run --rm tools operational evaluate
docker compose run --rm tools operational export --output var/stormsense/export
docker compose run --rm tools operational retain --output var/stormsense/export
```

Use `operational export/evaluate` for StormSense. The short `export/evaluate`
container commands dispatch the research CLI. Other legacy operational commands
such as `bootstrap` and `--device cpu update` are still accepted directly.

## R2 publication

Use the existing authenticated **rclone** config; no AWS CLI is involved. The
optional override mounts the config read-only into the tools/runner containers.
It must be readable by the image's UID. Do not put its contents into `.env`.

```bash
export RCLONE_CONFIG_FILE=/absolute/path/to/rclone.conf
docker compose -f compose.yaml -f compose.r2.yaml run --rm tools \
  rclone lsf r2:tcd/explorer/stormsense --max-depth 1

# Restore saved display assets when migrating an existing database.
docker compose -f compose.yaml -f compose.r2.yaml run --rm tools \
  rclone copy r2:tcd/explorer/stormsense/imagery var/stormsense/geocolor/imagery

# Publish one complete cycle before enabling the publishing scheduler.
docker compose -f compose.yaml -f compose.r2.yaml run --rm \
  -e STORMSENSE_PUBLISH=1 runner cycle

STORMSENSE_PUBLISH=1 docker compose -f compose.yaml -f compose.r2.yaml up -d runner
```

Publishing uses the existing verified immutable uploads and advances
`r2:tcd/explorer/stormsense/latest.json` last via `rclone copyto`. The original
explorer has its own publishing script, also available in the image. To publish
an already generated explorer release mounted under `data/explorer`:

```bash
docker compose -f compose.yaml -f compose.r2.yaml run --rm \
  -e GEO2WF_EXPLORER_DATA_DIR=/app/data/explorer tools \
  bash scripts/sync_explorer_to_r2.sh
```

That script uploads the imagery and manifests under an immutable UTC release
before advancing the original explorer pointer. It is a separate explicit job.

## Data preparation, research inference and training

```bash
docker compose run --rm tools download metadata
docker compose run --rm tools download all --dry-run
docker compose run --rm tools download all
docker compose run --rm tools export geo-sar --help
docker compose run --rm tools infer deterministic-residual --help
docker compose run --rm tools evaluate intensity-comparison --help
```

Downloads persist under `downloads/`. The full release is tens of GB; leave
additional space for derived datasets and checkpoints. The current training
loader needs a paired raster export, not just the downloaded metadata catalog.
See `docs/data/index.md` for the release's matching reproduction source.
Put your export under `data/geo_sar` or mount another host directory at `/app/data`.

```bash
# One CPU training/validation batch against your real export.
docker compose run --rm train-cpu train \
  data=geo_sar_common10_era5 model=deterministic_residual \
  data.root=/app/data/geo_sar data.stats_file=/app/data/geo_sar/stats.json \
  data.loader.num_workers=0 trainer.accelerator=cpu trainer.devices=1 \
  trainer.max_epochs=1 trainer.limit_train_batches=1 trainer.limit_val_batches=1 \
  trainer.enable_checkpointing=false
```

Training logs, manifests and checkpoints persist in `logs/`; research inference
can write to the mounted `inference/` directory. Choose output paths within a
mounted directory for any other scripts. Custom configs can be placed in `data/`
and passed with `--config /app/data/custom.yaml`. W&B is disabled in training
services by default; enable it explicitly and pass `WANDB_API_KEY` from the host
environment with `docker compose run -e WANDB_API_KEY ...`, never a build argument.

GPU training requires a compatible NVIDIA GPU, a working host driver, and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The CUDA image uses PyTorch 2.1.2/CUDA 12.1, preserving the project stack; very
new GPUs may require a separately validated PyTorch upgrade. The image provides
runtime libraries, not a CUDA compiler for custom kernels.

```bash
docker compose build train-gpu
docker compose run --rm train-gpu python -c \
  'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
docker compose run --rm train-gpu train \
  data.root=/app/data/geo_sar data.stats_file=/app/data/geo_sar/stats.json \
  trainer.accelerator=gpu trainer.devices=1

# Resume a checkpoint (replace the run directory).
docker compose run --rm train-gpu train \
  --ckpt-path /app/logs/your-run/checkpoints/last.ckpt \
  data.root=/app/data/geo_sar data.stats_file=/app/data/geo_sar/stats.json \
  trainer.accelerator=gpu trainer.devices=1
```

The image also includes the historical acquisition/training modules and all
helper scripts, callable through `python -m geo2wf.historical.dataset --help`
and `python -m geo2wf.historical.training --help`. Size training hardware for
the chosen model, crop size and batch size; the small CPU runner allocation is
not a training hardware guarantee.

## Website and documentation deployment

The existing public website remains a Cloudflare Worker with static assets and
a read-only R2 binding. Use the full-stack deployment service:

```bash
docker compose run --rm deploy deploy-check
docker compose run --rm tools web-build
# With CLOUDFLARE_API_TOKEN already exported in the host shell:
docker compose run --rm -e CLOUDFLARE_API_TOKEN deploy deploy
```

The checked-in Wrangler config names the existing account/domain. Review it
before targeting a different account. Rclone credentials and Cloudflare Worker
deployment credentials serve different purposes. Deployment does not start
inference. Production builds exclude the local archive, retaining the existing
Worker/R2 data route. Use `tools wrangler ...` for additional deployment commands.
`tools web-build` saves a standalone build under `var/docker-artifacts/website`.

The optional `web` service instead serves its own compiled UI and a read-only
local export, with SPA deep links, immutable asset caching and uncached
`latest.json`. It binds only `127.0.0.1:8080`; put the host's HTTPS reverse proxy
in front of it for public hosting. `web-dev` similarly binds localhost on 5173
and serves baked-in source; rebuild to pick up source edits.

```bash
docker compose run --rm tools docs build --strict --site-dir artifacts/site
docker compose run --rm tools test -q
docker compose run --rm tools bash -c 'cd apps/stormsense && npm test && npm run check:worker'
docker compose build browser-tests
docker compose run --rm browser-tests
```

Browser tests require an exported archive with the real storm fixtures described
in the app README. Their results persist under `var/docker-artifacts/browser`.
Documentation output persists under `var/docker-artifacts/site`. Documentation
publication to GitHub Pages remains the existing `.github/workflows/pages.yml`
workflow; the container can build the same site without GitHub credentials.

## Backup, transfer and validation

Use SQLite's backup API (the existing `runner/package.py` does this), plus model
and asset backups; copying an active SQLite file alone is unsafe. The compact
runner handoff remains a source/model/state bundle, **not** a full Docker build
context. Use the repository to rebuild these images, or transfer built images:

```bash
docker image save geo2wf:cpu geo2wf:web -o var/docker-artifacts/images.tar
# On the destination: docker image load -i images.tar
```

Restore persistent directories and credentials separately. Stop the old
publishing runner before starting its replacement. Retention is still explicit:
review `operational retain` and `operational retain --remote` before `--apply`.

`tests/test_container_runner.py` checks argument forwarding, whole-cycle locking,
worker settings, retry after failure and process-group shutdown. The Docker CI
workflow builds CPU/CUDA images, exercises training and operational tests inside
them, checks the chosen backend, validates deployment/docs builds, and builds
and probes the standalone website. It needs no publication credentials and
does not deploy. GPU computation requires the separate GPU-host check above.

Implementation references: [uv/PyTorch backend selection](https://docs.astral.sh/uv/guides/integration/pytorch/),
[Compose GPU access](https://docs.docker.com/compose/how-tos/gpu-support/),
[Compose secrets](https://docs.docker.com/compose/how-tos/use-secrets/), and
[Wrangler deployment authentication](https://developers.cloudflare.com/workers/ci-cd/external-cicd/github-actions/).
