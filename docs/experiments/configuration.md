# Configuration guide

One YAML file flows through exporter defaults, runtime data construction, model construction, optimization, trainer behavior, validation, and logging.

## Section map

```yaml
seed: 42                    # reproducibility
export: {}                  # one-time raster preparation
data: {}                    # runtime dataset and loaders
model: {}                   # architecture, diffusion, sampling
optimization: {}            # optimizer, EMA, LR scheduler
trainer: {}                 # Lightning runtime and checkpoints
validation: {}              # reconstruction sampling controls
logging:
  wandb: {}                 # experiment tracking
```

## Channel arithmetic

This is the most important invariant to preserve.

=== "Diffusion"

    ```yaml
    model:
      in_channels: 21       # 10 GEO + 9 ERA5 + distance + condition mask
      out_channels: 1
      unet:
        channels: 22        # prepared condition + noisy target
        out_dim: 1
    ```

=== "Residual diffusion"

    ```yaml
    model:
      type: diffusion_residual
      in_channels: 21       # prepared condition, including distance and mask
      out_channels: 1
      unet:
        channels: 24        # noisy residual + condition + baseline + mask
        out_dim: 1
    ```

=== "Residual"

    ```yaml
    model:
      type: deterministic_residual
      condition_channels: 20  # 10 GEO + 9 ERA5 + distance; masks appended
    ```

A mismatch fails at convolution or explicit input validation; it is not inferred from a batch.

## Export versus data

`export.output_root` is where a future export writes. `data.root` is what training reads. They should normally match, but they are independent so training can point at an existing immutable export.

`TCD_DATA_ROOT` overrides source data discovery. Standard output-root environment overrides are recognized only for roots ending in the conventional `geo_sar`, `geo_sar_10bands`, `geo_pmw`, or `geo_pmw_10bands` names; custom ERA5 roots remain explicit.

## Trainer values with special semantics

`limit_val_batches`
: An integer means that many batches. A float in `[0,1]` means a fraction. The CLI `--limit-val-batches` overrides YAML.

`devices`
: Passed directly to Lightning. `devices: 2` with `accelerator: auto` still requires an environment where Lightning can select two supported devices.

`precision`
: Basic multi-GPU presets use Lightning 1.9’s `16`; ERA5 experiments use 32-bit precision.

`enable_checkpointing`
: Controls whether the callback is registered. The base configs turn it off; production configs turn it on.

## Checkpoint monitor precedence

`trainer.checkpoint.monitor` wins. Otherwise the model’s `checkpoint_monitor` is used, currently `val/eye_structure_score`. Filename, mode, top-k, and `save_last` also come from `trainer.checkpoint` with defaults.

The LR scheduler monitor is independent under `optimization.reduce_lr_on_plateau.monitor`. Ensure the chosen metric is logged in every relevant validation epoch. Eye-structure score is emitted only when all required component metrics are available.

!!! warning "Bounded validation can hide the monitor"
    If too few validation samples contain adequate eye/core coverage, `val/eye_structure_score` may not be produced. Increase reconstruction coverage or choose a reliably available monitor for early debugging.

See [Configuration reference](../reference/configuration.md) for every key used by the checked-in presets.
