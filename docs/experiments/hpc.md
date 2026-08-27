# HPC & multi-GPU

## CPU export jobs

Full exports are intended for batch workers, not login nodes:

```bash
qsub scripts/hpc/export_geo_sar_geotiffs_cpu.pbs
```

The PBS file is a site template. Review account, queue, walltime, memory,
source root, output root, and environment activation before submission. Their
legacy script calls remain supported; new launchers should prefer
`geo2wf-export`.

## Composed two-GPU run

```bash
uv sync --frozen
uv run geo2wf-train \
  data=geo_sar_common10_era5 \
  model=deterministic_residual \
  trainer.accelerator=gpu \
  trainer.devices=2 \
  trainer.strategy=ddp_find_unused_parameters_false \
  trainer.precision=16 \
  data.loader.batch_size=2 \
  data.loader.num_workers=8 \
  data.loader.pin_memory=true \
  data.loader.persistent_workers=true
```

The batch size is per process, so this example has global batch size four.
Tune worker count to allocated CPUs and storage behavior; more workers are not
always faster for many raster reads.

## DDP-aware behavior

- The parent creates one run directory and exports `GEO2WF_RUN_DIR`; child ranks reuse it.
- Lightning distributes training samples and synchronizes logged losses.
- Shared metric sums/counts are reduced before epoch values are formed.
- Dataset diagnostic messages and media logging are rank-zero controlled.
- Intensity-balanced sampling controls whether Lightning replaces the train sampler.

## Performance checklist

- Confirm requested devices exist before queue submission.
- Match CPU allocation to `num_workers × processes`.
- Use `pin_memory=true` for CUDA when host memory allows.
- Use `persistent_workers=true` only when `num_workers>0`.
- Keep BLAS thread counts low; runtime defaults are one.
- Reduce validation reconstruction coverage when media generation dominates.
- Record per-device and global batch sizes when comparing learning rates.
- Place exports and tracking caches on filesystems suited to their access pattern.

!!! warning "Automatic accelerator does not create devices"
    `trainer.accelerator=auto trainer.devices=2` still fails on a one-device
    machine. Override `devices=1` locally or request the correct scheduler resources.
