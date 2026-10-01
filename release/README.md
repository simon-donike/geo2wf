# Conference release handoff

Branch: `conference-release`, based on `8e26d543ccdf`.
The original checkout and its three uncommitted edits are unchanged.

## Size and inventory

- Original tracked content: 50.59 MiB in 455 files.
- Release tracked content before this small handoff file: 31.15 MiB in 430 files: **38.4% smaller**, including new publication provenance.
- 21 verified local checkpoints: 1.18 GiB, outside Git.
- Compressed bundle: 1.10 GiB, with SHA-256 sidecar and per-file checksums.
- Local bundle: `/work/code/geo2wf/logs/conference-release-artifacts.tar.gz`.
- Unpacked artifacts: `/work/code/geo2wf/logs/conference-release-artifacts/`.

Removed research code, presets, reports, and replaced CSV paths are listed in
[removals.json](removals.json). Earlier research remains in Git history; this
reduces the branch's checked-out contents, not the historical object database.
No local data, old runs, or inference output directories were deleted.

## Reproduction and validation

[registry.json](registry.json) maps the original checkpoints, source revisions,
configs, dataset fingerprints, dependencies, and result reports.
[validation.json](validation.json) records fresh evaluation comparisons and
limitations. See [the reproduction guide](../docs/reproduction.md) for commands.

- 227 tests pass with the reduced dependencies, including checkpoint round trips
  and exact publication-table/provenance tests.
- All 21 checkpoints load and run forward inference.
- Both paper tables reproduce exactly from original full-precision reports.
- Fresh CPU evaluation completed for six architecture checkpoints and eight
  latent checkpoints. Two values cross their final rounding digit; original
  paper values are retained and the differences recorded separately.
- Full saved storm metrics and three figure families reproduce. Fresh inference
  was checked on one real observation per storm in both ERA5 regimes; a full
  dense neural rerun was not performed.
- 150 dashboard MLP forecast rows regenerated.
- Strict docs build and 42-page local-link check pass.
- Browser checks pass for every storm/model, forecasts, imagery layers, timeline,
  NWP, postprocessing, and local/release-pointer data modes, with no browser
  errors or HTTP failures.

## Preserved website and known limits

The stormtracker HTML, JavaScript, styles, deployment workflow, forecast exports,
and R2 asset-first publishing behavior are retained. PMW and forecast support
remain because the dashboard uses them. Archived page URLs have explanatory
replacement pages; the original code is linked through the pre-release commit.

ViT source-checkpoint identity and the external ConvLSTM HPC checkpoint/code are
unavailable locally; their exported dashboard outputs and available provenance
are retained. The earliest field initialization checkpoint has no recoverable
local training config; its exact weights are included. The historical field
used by extra case-study/dashboard correction heads included test observations
in training; its original training and later export configs are both preserved.

No branch push, merge, website deployment, or R2 publication was performed.
