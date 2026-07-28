# Conditional diffusion

`PixelDiffusionConditional` wraps `DenoisingDiffusionConditionalProcess` with Lightning training, evaluation, logging, EMA, and deterministic validation latents.

## Channel construction

The core process concatenates the noisy target with the prepared condition:

```text
4-band baseline:    1 noisy target + (4 GEO + 1 mask)       = 6 U-Net channels
10-band baseline:   1 noisy target + (10 GEO + 1 mask)      = 12
10-band + ERA5:     1 noisy target + (10 GEO + 9 ERA5 + 1)  = 21
```

Config invariants:

```yaml
model:
  in_channels: 20       # prepared condition width
  out_channels: 1       # generated target width
  unet:
    channels: 21        # in_channels + out_channels
    out_dim: 1          # out_channels
```

## Backbone

`UnetConvNextBlock` uses:

- a sinusoidal timestep embedding followed by an MLP;
- ConvNeXt-style residual blocks at each scale;
- linear attention at encoder, bottleneck, and decoder stages;
- strided convolutions for downsampling;
- transposed convolutions for upsampling; and
- skip concatenations between matching resolutions.

With `dim: 48` and multipliers `[1,2,4,8]`, feature widths are 48, 96, 192, and 384. The model predicts one noise channel for the one-channel target.

## Training step

1. Read `condition`, `target`, and `target_mask`.
2. If enabled, fill unobserved SAR with weakly weighted ERA5 values.
3. Map external `[0,1]` tensors to diffusion space `[-1,1]`.
4. Append the condition-validity mask.
5. Draw one random timestep per sample.
6. Noise the clean target in closed form.
7. Predict the injected noise and calculate masked MSE.
8. Optimize with AdamW; `ReduceLROnPlateau` watches the configured validation monitor.
9. Update EMA weights after the optimizer step when enabled.

## Exponential moving average

The ERA5 diffusion preset keeps a non-trainable copy of the complete diffusion process:

\[
\theta_{EMA} \leftarrow d\,\theta_{EMA} + (1-d)\,\theta
\]

with `d = 0.999`. Evaluation uses EMA by default. Buffers are copied exactly, and older non-EMA checkpoints can initialize the EMA copy from online weights.

## Validation and inference

Validation first computes the same noise-prediction loss. For the configured number of reconstruction batches it also:

- creates a stable initial latent from `SHA256(validation_seed, sample_id)`;
- runs the full reverse chain;
- maps the output to `[0,1]`;
- calculates normalized PSNR, SSIM, and L1;
- maps back to m/s for physical and storm metrics;
- logs sample range and saturation diagnostics; and
- creates W&B condition/prediction/target panels.

A checkpoint is rejected when its saved diffusion coefficients do not match the configured schedule or timestep count. This prevents silently resuming a linear-schedule model as cosine.

!!! info "What `forward()` returns"
    The public Lightning `forward()` returns values in `[0,1]`. The inner diffusion process returns raw samples in approximately `[-1,1]`; `_predict_batch()` exposes both internally for saturation diagnostics.
