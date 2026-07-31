# HPC & multi-GPU

## CPU export jobs

Full exports are intended for batch workers, not login nodes:

```bash
qsub scripts/hpc/export_geo_sar_geotiffs_cpu.pbs
qsub scripts/hpc/export_geo_pmw_geotiffs_cpu.pbs
```

The scripts activate the project environment, configure paths, and invoke the matching exporter. Review account, queue, walltime, memory, source root, and output root for the target cluster before submitting.

## Prepared two-GPU runs

=== "4 GEO bands"

    ```bash
    uv sync --frozen
    uv run python train.py --config configs/config_geo_sar_2gpu.yaml
    ```

=== "10 GEO bands"

    ```bash
    uv sync --frozen
    uv run python train.py --config configs/config_geo_sar_10bands_2gpu.yaml
    ```

Both use:

```yaml
trainer:
  accelerator: gpu
  devices: 2
  strategy: ddp_find_unused_parameters_false
  precision: 16
```

The per-device batch size is 2, so the global batch size is 4. Eight loader workers are started per DDP process; tune this to CPU allocation and storage behavior rather than assuming more is faster.

## DDP-aware behavior

- The parent process creates one timestamped run directory and exports it through `GEO2WF_RUN_DIR`; child ranks reuse it.
- Lightning distributes training samples and synchronizes logged losses.
- Physical/storm statistic tensors are explicitly gathered and summed before epoch metrics are formed.
- Rank-zero logging is used for dataset filtering messages.
- Fixed reconstruction noise depends on `sample_id`, not process rank.

## Performance checklist

- Set `pin_memory: true` for CUDA training when host memory allows.
- Use `persistent_workers: true` only with `num_workers > 0`.
- Keep BLAS thread environment values low; the scripts default them to 1 to avoid oversubscribing HPC nodes.
- Use DDIM with reduced timesteps when validation sampling dominates wall time.
- Confirm effective batch size before comparing learning rates.
- Store exports and W&B cache on filesystems suited to many raster reads and small metadata writes.

!!! warning "Automatic accelerator with two devices"
    The ERA5 diffusion preset uses `accelerator: auto`, `devices: 2`. On a single-device machine, override it with a local config; Lightning cannot satisfy two devices merely because selection is automatic.
