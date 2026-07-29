# Sampling

Training always uses the full configured forward schedule. Sampling chooses how to traverse that learned clock in reverse.

## DDPM

`DDPM_Sampler` visits every training timestep in descending order and samples from the exact posterior. It is ancestral and stochastic except at the final step.

```yaml
model:
  num_timesteps: 1000
  sampling:
    method: ddpm
    # timesteps must equal num_timesteps
```

DDPM is the implicit default when `model.sampling` is absent. It is faithful to the full chain but expensive during validation.

## DDIM

`DDIM_Sampler` builds an endpoint-inclusive, evenly spaced subset of actual training timesteps and stores each scheduled predecessor. The model and sampler therefore agree on timestep embeddings and coefficients.

```yaml
model:
  num_timesteps: 1000
  sampling:
    method: ddim
    timesteps: 100
    eta: 0.0
    clip_sample: true
```

With `eta: 0`, the chain is deterministic given initial noise. Positive eta adds controlled stochasticity. DDIM allows fewer steps, making reconstruction logging much cheaper.

## Clean-sample clipping

Both samplers estimate the clean target at every step. When `clip_sample: true`, that estimate is clamped to `[-1,1]` before calculating the previous sample. This limits runaway errors at low signal-to-noise timesteps and matches the model’s training range.

## Reproducibility

Diffusion validation noise is stable per sample, not merely per batch order. A SHA-256 digest of the validation seed and sample ID seeds a same-device `torch.Generator`. Consequently:

- the same sample receives the same initial latent across epochs;
- storm-stratified ordering does not change the latent;
- global RNG state is not disturbed; and
- DDIM with eta 0 gives directly comparable reconstructions.

DDPM still draws reverse noise. Its sampler accepts an explicit generator, but the current `_predict_batch()` supplies fixed initial noise rather than a reverse-process generator; complete DDPM chains can therefore remain stochastic.

## Schedules

| Schedule | Character | Typical use here |
|---|---|---|
| `linear` | betas rise from 0.0001 to 0.02 | basic 4/10-band presets |
| `cosine` | cumulative signal follows cosine curve | ERA5 diffusion preset |
| `quadratic` | slower early beta growth | available for experiments |
| `sigmoid` | smooth S-shaped beta growth | available for experiments |

Changing a schedule changes the learned problem. Start a fresh run or resume with the checkpoint’s original schedule; code explicitly rejects coefficient mismatches.
