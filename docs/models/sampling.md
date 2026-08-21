# Sampling

Training always uses the full configured forward schedule. Sampling chooses how to traverse that learned clock in reverse.

## DDPM

`DDPM_Sampler` visits every training timestep in descending order and samples from the exact posterior. It is ancestral and stochastic except at the final step.

```yaml
model:
  num_timesteps: 1000
  sampling_method: ddpm
  sampling_timesteps: 1000
```

`sampling_method` defaults to `ddpm` when omitted. The full chain is expensive
during validation.

## DDIM

`DDIM_Sampler` builds an endpoint-inclusive, evenly spaced subset of actual training timesteps and stores each scheduled predecessor. The model and sampler therefore agree on timestep embeddings and coefficients.

```yaml
model:
  num_timesteps: 1000
  sampling_method: ddim
  sampling_timesteps: 100
  sampling_eta: 0.0
  clip_sample: true
```

With `eta: 0`, the chain is deterministic given initial noise. Positive eta adds controlled stochasticity. DDIM allows fewer steps, making reconstruction logging much cheaper.

## Ensembles and classifier-free guidance

Different initial Gaussian latents produce different conditional samples even
when DDIM uses `eta: 0`. `validation_ensemble_size` assigns each sample a stable
latent namespace, making diversity and calibration comparable across epochs.
The model exposes `sample_ensemble(...)`, which returns tensors shaped
`[member, batch, channel, height, width]`.

When condition dropout was enabled during training, `guidance_scale`
combines conditional and zero-condition noise predictions. A value of `1`
performs the original single conditional evaluation. Values above `1` usually
increase fidelity to the baseline and GEO/ERA context at some cost to ensemble
diversity.

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
