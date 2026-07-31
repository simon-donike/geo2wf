# Project map

```text
geo2wf/
├── configs/                 experiment presets
├── data/
│   ├── dataset.py           GeoTIFF → normalized sample dictionary
│   └── datamodule.py        split datasets and Lightning loaders
├── scripts/
│   ├── export_geo_sar_geotiffs.py
│   ├── export_geo_pmw_geotiffs.py
│   └── local_env.py
├── src/
│   ├── PixelDiffusion.py    conditional diffusion LightningModule
│   ├── ERA5Residual.py      deterministic physical residual control
│   ├── wind_metrics.py      storm-centric metrics
│   ├── reconstruction_logging.py shared W&B image assembly
│   └── DenoisingDiffusionProcess/
│       ├── forward.py       Gaussian forward process
│       ├── beta_schedules.py
│       ├── samplers/        DDPM and DDIM
│       └── backbones/       ConvNeXt-style U-Net
├── tests/                   unit and learning-behavior tests
├── src/utils/plotting.py        exported-pair visualization
├── train.py                 runtime entry point
├── mkdocs.yml               this documentation site
└── pyproject.toml           package and tool dependencies
```

## Responsibilities by file

| File | Read this when… |
|---|---|
| `train.py` | tracing config-to-Trainer wiring, run directories, W&B, checkpoints |
| `data/dataset.py` | debugging manifest columns, raster masks, normalization, derived ERA5, augmentation |
| `data/datamodule.py` | debugging splits, validation ordering, worker or batch behavior |
| `src/PixelDiffusion.py` | changing diffusion loss, EMA, sampling validation, metrics, images |
| `src/ERA5Residual.py` | changing the deterministic baseline or physical loss |
| `src/reconstruction_logging.py` | changing shared W&B sample assembly or media sizing |
| `src/wind_metrics.py` | changing storm geometry, radial bins, eye gates, metric availability |
| `DenoisingDiffusionProcess.py` | changing reverse-loop orchestration or conditional concatenation |
| `samplers/DDPM.py` / `DDIM.py` | changing posterior coefficients or timestep traversal |
| `scripts/export_*` | changing source pairing, grids, regridding, tags, or statistics |

## Test map

The tests cover beta schedules, the forward process, samplers, reproducible sampling, dataset helpers and ERA5 behavior, normalization/data-learning repairs, residual model learning, pixel-diffusion helpers, and storm metrics. Run all tests after changing any config-facing behavior:

```bash
uv run python -m pytest
```

<div class="grid cards" markdown>

- **[Configuration reference](configuration.md)** — all checked-in YAML keys.
- **[Commands & environment](commands.md)** — copy-ready command index.
- **[Troubleshooting](troubleshooting.md)** — common shape, data, metric, and runtime failures.
- **[Glossary](glossary.md)** — sensor, diffusion, and cyclone terms.

</div>
