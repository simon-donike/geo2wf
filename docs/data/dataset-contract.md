# Dataset contract

`PairedImageDataset` reads one split manifest and returns a dictionary. Fields used by the basic diffusion path are stable; ERA5 and physical-evaluation fields are added when available.

## Core sample

| Key | Shape | Meaning |
|---|---|---|
| `condition` | `[C, H, W]` | normalized GEO, optionally followed by normalized ERA5 context |
| `target` | `[1, H, W]` | normalized SAR wind speed or PMW brightness temperature |
| `condition_mask` | `[1, H, W]` | pixels valid across condition/context |
| `target_mask` | `[1, H, W]` | observed target pixels |
| `target_physical` | `[1, H, W]` | target values in source units, zero outside validity |
| `target_norm_offset` | `[1]` | affine inverse-normalization offset |
| `target_norm_scale` | `[1]` | affine inverse-normalization scale |
| `condition_bounds` | `[4]` | left, right, bottom, top |
| `target_bounds` | `[4]` | left, right, bottom, top |
| `center` | `[2]` | IBTrACS latitude, longitude used by evaluation |
| `sample_id` | string | stable pair identifier |
| `meta` | mapping | source types, sensors, time gap, and channel names |

ERA5-enabled SAR samples also return normalized and physical `era5_wind_speed`, plus `era5_wind_speed_mask`.

!!! important "Center metadata is not a model condition"
    `center` and `target_bounds` establish the coordinate frame for
    storm-centric metrics. Neither model concatenates them to its input
    channels. In particular, `center` is read from
    `ibtracs_center_lat`/`ibtracs_center_lon`, not from the exported raster's
    `center_lat`/`center_lon`. See
    [Evaluation & metrics](../experiments/evaluation.md#eye-center-displacement)
    for the full eye-location calculation.

## Runtime transform order

```mermaid
graph TD
  A[Read raster + internal mask] --> B[Append derived ERA5 speed/vorticity]
  B --> C[Capture physical target and ERA5 baseline]
  C --> D[Normalize each channel]
  D --> E[Replace non-finite values and apply masks]
  E --> F[Concatenate GEO + ERA5]
  F --> G[Resize target/masks]
  G --> H[Physics-aware paired flips]
  H --> I[Return dictionary]
```

Condition rasters are not resized in the current loader; the export is expected to produce the configured grid. Targets and relevant masks/baselines are bilinearly or nearest-neighbor resized to `target_size` as appropriate.

## Condition mask channel

The mask is **not** included in `batch["condition"]`. `PixelDiffusionConditional._prepare_condition()` maps the normalized condition from `[0,1]` to `[-1,1]` and appends the single binary mask. Therefore:

```text
model.in_channels = dataset condition channels + 1
unet.channels      = model.in_channels + model.out_channels
unet.out_dim       = model.out_channels
```

For 10 GEO + 9 ERA5 fields + distance + 3 solar-time fields, these become 23,
24, and 1 respectively.

The deterministic residual model handles masks differently: it receives 23 condition channels, then appends condition mask, explicit ERA5 wind, and ERA5 mask internally, giving its compact U-Net 26 input channels.

## Split behavior

By default, checked-in configs set `include_test_in_train: true`, concatenating test into training while keeping validation separate. This is intentional in the current experiment presets but is not conventional held-out evaluation. Set it to `false` when the test split must remain untouched for final reporting.

Validation is storm-stratified by round-robin reordering, which matters when `limit_val_batches` observes only a prefix.
