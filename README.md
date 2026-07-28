# geo2wf

Minimum-complexity diffusion baseline for predicting tropical-cyclone SAR wind
fields from geostationary satellite imagery.

This repository is the lightweight counterpart to the larger
`2026-ESL-Tropical-Cyclone-Dynamics` project. The large project builds a
multi-source MOTIF/flow-matching reconstruction system over PMW, IR,
geostationary, SAR, and optional ERA5 observations, with manifest-backed data
selection, source metadata, masking policies, Hydra configs, DDP launchers, and
HPC-oriented training machinery.

`geo2wf` intentionally strips that down to the smallest useful experiment:

```text
geostationary image tensor x  ->  conditional pixel diffusion  ->  SAR wind-field tensor y
```

The goal is to make the geostationary-to-SAR wind-field task easy to reason
about before reintroducing the complexity of multi-source observations,
multi-temporal windows, masking policies, and full tropical-cyclone metadata.

## Data Visualization

![Random GEO-SAR training pairs](resources/geo_sar_random_pairs.png)

Each row shows one random GEO-SAR training pair from the exported GeoTIFF
dataset. The left panel is a geostationary false-color RGB composite using two
IR bands as red/green and water vapor as blue, the middle panel is the paired
SAR wind-speed target, and the red `x` marks the IBTrACS storm center recorded
in the manifest.
The right panel places the GEO and SAR footprints on a simple ocean/land mask,
making the core problem visible: learn a SAR-like wind field from a colocated
geostationary view of the same tropical cyclone.

## W&B Logging Preview

![Example W&B reconstruction logging](resources/wandb_logging_preview_2.png)

Example reconstruction panels logged during training. W&B records validation
reconstructions under `val/reconstruction` and training-sample reconstructions
under `train/reconstruction`.

## Scope

This repo is for paired image-to-image diffusion:

- input `x`: geostationary tropical-cyclone imagery, for example GOES ABI or
  Himawari AHI channels, already cropped/aligned/resampled into a fixed tensor.
- target `y`: SAR-derived near-surface wind-field image, for example C-band SAR
  wind speed or a small multi-channel wind representation, on the same grid as
  `x`.
- model: conditional DDPM-style pixel-space diffusion with a UNet backbone.
- training contract: each dataset item returns a dictionary with `condition`,
  `target`, and `target_mask` tensors shaped `[C, H, W]`.

This is not the high-complexity MOTIF experiment. It does not currently model a
set of heterogeneous source occurrences, observation-time offsets, coordinates,
availability masks, land masks, storm tracks, or per-source characteristic
vectors. Those belong to the larger project and can be pulled in later only if
they improve the simple baseline.

## Current State

What is already wired:

- `train.py`: PyTorch Lightning training entrypoint.
- `configs/config.yaml`: single YAML config for data loader, diffusion model,
  optimizer, trainer, and W&B logging.
- `src/PixelDiffusion.py`: conditional LightningModule around the denoising
  diffusion process.
- `src/DenoisingDiffusionProcess/`: DDPM/DDIM sampling, beta schedules, and UNet
  backbone.
- validation logging: `val/loss`, PSNR, SSIM, L1, and a single `x | pred | y`
  reconstruction panel through W&B.

What is still deliberately minimal:

- `scripts/export_geo_sar_geotiffs.py` creates one-time local GeoTIFF tensors
  from the larger cyclone manifest.
- `scripts/export_geo_pmw_geotiffs.py` creates a larger proxy pretraining
  dataset with the same GEO-to-one-band-target tensor shape.
- `data/dataset.py` reads the exported GeoTIFF manifest, normalizes raw values
  to `[0, 1]`, and returns tensors for training.
- `data/datamodule.py` is shaped for split-based paired datasets.

## Intended Data Contract

`PairedImageDataset` returns:

```python
return {
    "condition": x,
    "target": y,
    "target_mask": mask,
}
```

where:

- `x` is the conditioning geostationary image tensor.
- `y` is the SAR wind-field target tensor.
- `mask` marks valid SAR target pixels.
- image tensors are `torch.float32`, fixed shape `[C, H, W]`, and normalized
  to `[0, 1]` from exported train statistics.

The current model maps inputs from `[0, 1]` to `[-1, 1]` before diffusion and
maps predictions back to `[0, 1]` for visualization and metrics. If the SAR wind
field is represented in physical units or z-scores, update
`PixelDiffusionConditional.input_T()` and `output_T()` accordingly.

Exported real-data layout:

```text
data/geotiff/geo_sar/
  stats.json
  train/
    manifest.csv
    AL012023_sar_geo_20230115070023_abcd1234_geo.tif
    AL012023_sar_geo_20230115070023_abcd1234_sar.tif
  val/
    manifest.csv
  test/
    manifest.csv
```

The PMW pretraining export uses the same layout under `data/geotiff/geo_pmw/`,
with generic manifest columns named `condition_path` and `target_path`. The SAR
manifests keep backward-compatible `geo_path` and `sar_path` columns as well.

The default configs keep the original four-GEO-band datasets. The 10-band
variants use separate folders and config files:

- SAR: `configs/config_geo_sar_10bands.yaml` -> `data/geotiff/geo_sar_10bands`
- SAR 2-GPU: `configs/config_geo_sar_10bands_2gpu.yaml`
- GEO/PMW pretraining: `configs/config_pretrain_geo_pmw_10bands.yaml` -> `data/geotiff/geo_pmw_10bands`
- SAR + ERA5 context: `configs/config_geo_sar_10bands_era5.yaml` -> `data/geotiff/geo_sar_10bands_era5`
- GEO/PMW + ERA5 context: `configs/config_pretrain_geo_pmw_10bands_era5.yaml` -> `data/geotiff/geo_pmw_10bands_era5`

For 10-band proxy pretraining, only the GEO condition expands to ten bands. PMW
targets remain a single selected high-frequency brightness-temperature channel,
matching the SAR target dimensionality.

The ERA5 variants save seven single-level ERA5 fields as a companion
`*_era5.tif` on the same crop/grid as the GEO image: precipitable water, SST,
MSLP, 2m temperature, 2m dewpoint, and 10m u/v wind. The dataset loader reads
that companion file when `context_path` is present, derives 10m wind speed and
10m relative vorticity from u/v wind, and concatenates the ERA5 context with GEO
at load time. With 10 GEO bands this gives `10 + 9 + 1 mask = 20` model input
channels and a one-channel target.

The exporter stores raw physical values in GeoTIFFs with internal masks and
metadata tags. The training dataset handles file-format details and min-max
normalization.

## Model Configuration

The main config is `configs/config.yaml`.

Important channel settings:

```yaml
model:
  in_channels: 5
  out_channels: 1
  unet:
    channels: 6
    out_dim: 1
```

For conditional concatenation, `unet.channels` should usually equal
`model.in_channels + model.out_channels`, and `unet.out_dim` should equal
`model.out_channels`.

For a first small run, keep the default UNet width modest:

```yaml
model:
  unet:
    dim: 48
```

## Training

Create the local UV environment:

```bash
uv sync
```

Run the test suite through UV:

```bash
uv sync --group dev
uv run python -m pytest
```

Export a tiny smoke dataset:

```bash
uv run python scripts/export_geo_sar_geotiffs.py --config configs/config.yaml --limit 2
uv run python scripts/export_geo_pmw_geotiffs.py --config configs/config_pretrain_geo_pmw.yaml --limit 2
```

Run PMW proxy pretraining or SAR fine-tuning:

```bash
uv run python train.py --config configs/config_pretrain_geo_pmw.yaml
uv run python train.py --config configs/config.yaml
```

For the prepared local two-GPU GEO-to-SAR run:

```bash
uv sync --frozen
uv run python train.py --config configs/config_geo_sar_2gpu.yaml
```

The two-GPU config uses DDP without unused-parameter scans, 16-bit mixed precision, a per-GPU batch size of
two (global batch size four), and writes checkpoints below
`logs/checkpoints/geo_sar_2gpu/`.

For full exports on the cluster, submit the CPU PBS scripts instead of running
long jobs on a login node:

```bash
qsub run_scripts/export_geo_pmw_geotiffs_cpu.pbs
qsub run_scripts/export_geo_sar_geotiffs_cpu.pbs
```

To avoid online W&B logging:

```bash
export WANDB_MODE=offline
```

or:

```bash
export WANDB_DISABLED=true
```

## Baseline Development Checklist

1. Run the GeoTIFF exporter with `--limit 2` and inspect the split manifests.
2. Verify dataset tensor shapes: GEO `[4, 256, 256]`, SAR `[1, 256, 256]`.
3. Overfit a tiny subset before scaling up.
4. Compare generated SAR wind fields against targets using image metrics and
   storm-structure diagnostics that matter physically.

## Relationship To The Larger Project

Use `2026-ESL-Tropical-Cyclone-Dynamics` as the reference for:

- data provenance and source naming: `goes_east.ABI`, `goes_west.ABI`,
  `himawari.AHI`, and `sar.cband`.
- manifest-backed tropical-cyclone observation indexing.
- source-specific normalization statistics.
- future integration with PMW, IR, ERA5, track metadata, and multi-temporal
  context windows.

Use this repo when the question is simpler:

> Given a geostationary image crop of a tropical cyclone, can a conditional
> diffusion model generate a plausible SAR-like wind field on the same grid?

That is the baseline this repository is meant to answer.
