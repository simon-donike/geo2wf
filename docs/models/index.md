# Model overview

The principal wind-field system is a two-stage stack:

1. **Stage 1 — deterministic baseline:** learn a physical correction around ERA5.
2. **Stage 2 — residual diffusion:** freeze Stage 1 and sample a signed correction around its output.

[Start with the complete two-stage workflow.](two-stage.md)

| Path | Output | Inputs | Primary use |
|---|---|---|---|
| Two-stage baseline + diffusion | frozen baseline + sampled residual | GEO, ERA5, derived context, masks, Stage 1 field | main reconstruction system |
| Deterministic baseline only | one dense wind field | GEO, ERA5, derived context, masks | interpretable control and Stage 1 checkpoint |
| Residual diffusion on ERA5 | ERA5 + sampled residual | GEO, ERA5, derived context, masks | portable generative ablation without Stage 1 |
| Absolute conditional diffusion | absolute wind sample | GEO, optional ERA5, derived context, masks | standalone research baseline |
| Single-field intensity correction | corrected maximum wind + derived category | one frozen U-Net field, mask, center distance, current metadata | dashboard-scalar estimation |
| Joint U-Net + MLP | wind field and current maximum wind | GEO, optional ERA5, derived context, masks | joint field–intensity estimation |
| Six-hour intensity forecast | maximum wind at +6 h | current correction estimate, IBTrACS winds at −6 h and −12 h | short-range scalar forecast |

```mermaid
flowchart LR
  C[GEO + ERA5 + derived context] --> S1[Stage 1 deterministic baseline]
  S1 --> B[Baseline wind field]
  C --> S2[Stage 2 residual diffusion]
  B --> S2
  S2 --> O[Baseline + sampled correction]
```

Wind-field reconstruction paths share the paired raster contract, physical target conversion, and masks. The intensity correction and forecast models use separate cached scalar contracts.

## Main articles

- **[Two-stage baseline + diffusion](two-stage.md)** — handoff, channel counts, and training order.
- **[Stage 1 deterministic baseline](era5-residual.md)** — residual connection to ERA5, physical Huber loss, and zero-initialized head.
- **[Stage 2 residual diffusion](residual-diffusion.md)** — signed residual transform, frozen baseline, guidance, and ensemble diagnostics.
- **[Single-field intensity correction](intensity-correction.md)** — correct a frozen U-Net field into USA maximum wind and a wind-derived category.
- **[Joint U-Net + MLP](bottleneck-unet-mlp.md)** — estimate the wind field and current IBTrACS intensity from a shared encoder.
- **[Six-hour scalar intensity forecast](intensity-forecast.md)** — forecast maximum wind from the current correction estimate and 12 hours of IBTrACS history.

## Supporting and ablation articles

- **[Standalone conditional diffusion](conditional-diffusion.md)** — generate an absolute field directly from a noise latent and conditions.
- **[Sampling](sampling.md)** — DDPM, DDIM, schedules, clipping, and reproducibility.
- **[Diffusion in one page](../concepts/diffusion-primer.md)** — a short conceptual primer.
