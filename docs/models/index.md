# Models and checkpoints

The companion model repository is
[**simon-donike/geo2wf-models** on Hugging Face](https://huggingface.co/simon-donike/geo2wf-models),
paired with the [scientific dataset](https://huggingface.co/datasets/simon-donike/geo2wf-data).
The [paper](../concepts/problem.md) compares field reconstruction, post-hoc
scalar correction, and joint field/scalar learning.

| Model | Inputs | Outputs |
|---|---|---|
| [Field U-Net](era5-residual.md) | GEO, deterministic context, optional ERA5 | Surface wind-speed field; intensity/radii diagnosed from the image |
| [Joint latent MLP](bottleneck-unet-mlp.md) | Same observation inputs | Field plus directly predicted maximum wind and optional radii |
| [Encoder-only MLP](bottleneck-unet-mlp.md#encoder-only-control) | Same observation inputs | Maximum wind and optional radii, without a field decoder or SAR loss |
| [Post-hoc correction](intensity-correction.md) | Frozen U-Net field and current metadata | Corrected maximum wind, optionally scalar radii |
| [Six-hour forecast](intensity-forecast.md) | Current intensity and recent intensity history | +6 h maximum wind; recursive +12 h diagnostic |

SAR is a training target for field models. Best-track intensity and radii
supervise scalar heads. The forecast is a separate downstream task from the
paper's instantaneous reconstruction comparisons.

## Checkpoints

The **2 October 2026** release contains **21 PyTorch Lightning checkpoints**
(1.27 GB of checkpoint files), original configurations, provenance, and a
matching source archive. Checkpoint binaries are not bundled in Git.

From the repository root:

```bash
uv tool install huggingface_hub
python3 scripts/download_artifacts.py models --dry-run
python3 scripts/download_artifacts.py models
```

Files are saved to `downloads/models/`. The script pins a published model
commit and the dataset commit referenced by its `dataset-links.json`.
Use `all` instead of `models` to fetch both releases. See the
[download guide](../data/index.md#browse-and-download) for storage, selective
data downloads, and [using the matching source](../data/index.md#use-the-downloads).

| Release path | Contents |
|---|---|
| `checkpoints/` | Released model weights and pretrained initializers |
| `release/registry.json` | Checkpoint paths, hashes, original configurations, and experiment relationships |
| `dataset-links.json` | Matching data revision, cohorts, normalization, and split hashes |
| `run-provenance/` | Original training and source evidence |
| `code/conference-source.tar.gz` | Matching implementation, catalog loaders, reproduction scripts, and dependency lock |

The [model card](https://huggingface.co/simon-donike/geo2wf-models) lists the
checkpoint groups and validation limits. The earlier viewer's external ViT and
ConvLSTM weights are not included; their available exported results are in the
dataset. Current [StormSense](https://stormsense.hyperalislabs.com/) uses the
released `latent_sar_no_era5_max_wind_radii` nowcast and `dashboard-mlp`
forecast checkpoints, pinned with configuration and asset hashes in
[`models.json`](https://github.com/simon-donike/geo2wf/blob/main/src/geo2wf/operational/models.json).

Use each checkpoint with its matching resolved configuration, channel order,
normalization statistics, and data cohort. Current presets are starting points
for new runs and need not match a historical checkpoint. Record the full Hub
commit for both data and models when reproducing results.

The installed [inference and evaluation commands](../reference/commands.md)
operate on local files. Full-state resume and weights-only initialization are
covered in [training](../experiments/training.md).

## Training presets

Names below are files under `configs/experiment/`, passed as
`experiment=<name>` to `geo2wf-train`.

| Prediction | With ERA5 | Without ERA5 |
|---|---|---|
| Field only | `intensity_comparison_unet` | `intensity_comparison_unet_no_era5` |
| Joint field and intensity | `bottleneck_unet_mlp` | `bottleneck_unet_mlp_no_era5` |
| Joint field, intensity, and radii | `latent_mlp_sar_era5_max_wind_radii` | `latent_mlp_sar_no_era5_max_wind_radii` |
| Scalar intensity and radii, no SAR loss | `latent_mlp_no_sar_era5_max_wind_radii` | `latent_mlp_no_sar_no_era5_max_wind_radii` |

The `latent_mlp_*_max_wind` counterparts disable the radius head. Post-hoc
correction uses `unet_intensity_correction`, with optional `_image_radii` and
`_mlp_radii` variants. Forecast pretraining uses `intensity_forecast_pretrain`.

The no-ERA5 comparison presets keep ERA5 availability filtering to preserve
the matched cohort while withholding ERA5 from the model. For a new study
using observations without ERA5, set `data.require_era5=false` and document the
changed cohort.
