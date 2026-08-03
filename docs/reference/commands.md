# Commands & environment

## Environment and quality

```bash
uv sync --frozen --group dev --group docs
uv run python -m pytest
uv run mkdocs build --strict
uv run mkdocs serve
```

## Data export

=== "GEO–SAR smoke"

    ```bash
    uv run python scripts/export_geo_sar_geotiffs.py \
      --config configs/config.yaml --limit 2
    ```

=== "GEO–PMW smoke"

    ```bash
    uv run python scripts/export_geo_pmw_geotiffs.py \
      --config configs/config_pretrain_geo_pmw.yaml --limit 2
    ```

=== "ERA5 re-export"

    ```bash
    uv run python scripts/export_geo_sar_geotiffs.py \
      --config configs/config_geo_sar_10bands_era5.yaml
    ```

## Training

```bash
uv run python train.py --config configs/config.yaml
uv run python train.py --config configs/config_pretrain_geo_pmw.yaml
uv run python train.py --config configs/config_geo_sar_10bands_era5.yaml
uv run python train.py --config configs/config_geo_sar_10bands_era5_residual.yaml
uv run python train.py --config configs/config_geo_sar_10bands_era5_pmw_residual.yaml
GEO2WF_BASELINE_CKPT=/path/to/pmw-stage1.ckpt uv run python train.py \
  --config configs/config_geo_sar_10bands_era5_pmw_diffusion_residual_deterministic.yaml
```

Resume or bound validation:

```bash
uv run python train.py --config configs/config.yaml \
  --ckpt-path /path/to/last.ckpt \
  --limit-val-batches 10
```

Evaluate the current and PMW checkpoints separately on the identical one-hour
PMW validation cohort, then apply the stage-specific promotion gates:

```bash
uv run python scripts/evaluate_checkpoint.py \
  --config configs/config_geo_sar_10bands_era5_residual.yaml \
  --checkpoint /path/to/current-stage1.ckpt \
  --output logs/current-stage1-common-pmw.json
uv run python scripts/evaluate_checkpoint.py \
  --config configs/config_geo_sar_10bands_era5_pmw_residual.yaml \
  --checkpoint /path/to/pmw-stage1.ckpt \
  --output logs/pmw-stage1-common-pmw.json
uv run python scripts/compare_pmw_evaluations.py \
  --stage 1 \
  --current logs/current-stage1-common-pmw.json \
  --candidate logs/pmw-stage1-common-pmw.json \
  --output logs/pmw-stage1-promotion.json
```

Use `--stage 2` with the two diffusion reports for Stage 2. The comparator
verifies the ordered evaluation-row fingerprint and full-validation setting,
then applies the documented MAE/peak or probabilistic/CRPS/skill gates. Reports
retain the warning that the current and PMW checkpoints had different training
cohorts.

PMW-aware storm inference writes `pmw-inference-audit.csv` beside each output
summary and omits frames without a supported overpass inside the configured window.

## Batch jobs

```bash
qsub scripts/hpc/export_geo_pmw_geotiffs_cpu.pbs
qsub scripts/hpc/export_geo_sar_geotiffs_cpu.pbs
```

## Environment variables

| Variable | Effect |
|---|---|
| `TCD_DATA_ROOT` | override source archive root and manifest location |
| `GEO_SAR_OUTPUT_ROOT` | redirect conventional SAR exports/data roots |
| `GEO_PMW_OUTPUT_ROOT` | redirect conventional PMW exports/data roots |
| `GEO2WF_RUN_DIR` | inherited DDP run path; normally managed internally |
| `WANDB_DISABLED` | disable W&B logger when true-like |
| `WANDB_MODE` | use `offline` for local-only tracking |
| `WANDB_PROJECT` | override project name |
| `WANDB_DIR`, `WANDB_CACHE_DIR`, `WANDB_CONFIG_DIR` | set by `train.py` under each run |
| `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS` | default to 1 in scripts |
| `MPLCONFIGDIR` | defaults to `/tmp/dif_img_rec_matplotlib` |
