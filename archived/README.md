# Archived experiment code

This tree preserves experiment families that are no longer part of the active
geo2wf experiment matrix. It is reference material, not an importable Python
package or a supported launch surface.

Archived here:

- conditional and residual diffusion models and their compatibility modules;
- the direct GEO-to-PMW proxy model and its paired-data presets;
- the retired decoder-free U-Net encoder/MLP intensity model, data adapter,
  ERA5/no-ERA5 presets, and Lightning module;
- Stage 1/Stage 2 ablation configurations and launchers;
- the matched SAR-versus-IBTrACS target matrix and report builders;
- historical full-YAML and `v1` run presets;
- inference, post-processing, calibration, and tests specific to those tracks.

The active configuration surface remains under `configs/`, active runtime code
under `src/geo2wf/`, and active commands under `scripts/`. Historical files
retain their original internal paths, so commands inside this archive may need
path adjustments and an older checkout to reproduce exactly.

Ignored local artifacts under `logs/`, `wandb/`, `data/`, and `inference/` were
not moved. They may contain large user-owned checkpoints and intermediate data;
the archive covers version-controlled code, configurations, and published
documentation artifacts only.
