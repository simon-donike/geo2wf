# Commands

Run commands from the repository root after [installation](../getting-started/installation.md).
CLI workflows use local data and checkpoint files; [Hugging Face access](../data/index.md)
is a separate download step.

## Train and export

```bash
uv run geo2wf-train --help
uv run python scripts/export_geo_sar_geotiffs.py --help
uv run python scripts/export_unet_intensity_cache.py --help
uv run python scripts/export_joint_intensity_cache.py --help
uv run python scripts/export_intensity_forecast_cache.py --help
```

See [training](../experiments/training.md) for Hydra overrides and resume flags,
[model presets](../models/index.md#training-presets) for experiment choices,
and [GEO–SAR export](../data/export-geo-sar.md) for source requirements.

## Evaluate local checkpoints

```bash
uv run python scripts/evaluate_intensity_models.py --help
uv run python scripts/evaluate_intensity_correction.py --help
uv run python scripts/evaluate_intensity_forecast.py --help
```

The script help lists workflow-specific arguments; the installed CLI wrappers
provide the workflow selector. The comparison evaluator works with
field, correction, and joint-model artifacts; it does not rerun the removed
experiment orchestration scripts. Metric meanings are documented under
[evaluation](../experiments/evaluation.md).

## Infer over source observations

The field workflow reads an observation manifest and source archive:

```bash
uv run geo2wf-infer deterministic-residual \
  --config /path/to/resolved-config.yaml \
  --checkpoint /path/to/model.ckpt \
  --stats /path/to/paired/stats.json \
  --data-root /path/to/source-archive \
  --manifest /path/to/observation_manifest.csv \
  --reference-root /path/to/reference-series \
  --ibtracs-file /path/to/ibtracs.ALL.list.v04r01.csv \
  --output-root inference/my-run
```

The reference root must contain `<storm_id>/inference-summary.csv` files;
their observation IDs define which source observations are processed.
`--storms` and `--limit` restrict that selection. Scalar workflows consume
their corresponding local caches. For all inference options:

```bash
uv run python scripts/run_storm_unet_inference.py --help
uv run python scripts/run_intensity_correction_inference.py --help
uv run python scripts/run_intensity_forecast_inference.py --help
```

## Check docs

```bash
uv sync --frozen --group docs
uv run mkdocs build --strict
uv run python scripts/check_site_links.py
```
