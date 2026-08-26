# Matched IBTrACS versus SAR intensity comparison

This workflow compares scalar supervision from interpolated IBTrACS
`USA_WIND` with a SAR robust peak, defined as the mean of the highest 0.5% of
valid pixels in the resized and center-cropped SAR field. SAR pixel-field
supervision itself is unchanged.

The completed matched-target matrix, the older IBTrACS-only benchmark, and the
Humberto/Kiko/Otis analysis live in the
[published results report](intensity-comparison-results.md).

For every scalar target and ERA5 regime, the workflow compares:

1. a reference-aligned diagnostic from a shared deterministic U-Net field;
2. a learned single-field correction using that diagnostic as its anchor; and
3. the scalar MLP output of a jointly trained U-Net+MLP.

For the IBTrACS target cells it additionally compares a fourth model: the
decoder-free U-Net encoder + MLP trained only against `USA_WIND`. The evaluator
reports this model against both IBTrACS and SAR robust peak, but never labels or
duplicates it as SAR-trained.

The deterministic and joint U-Nets also report SAR field metrics. The
correction model has no field metrics because it emits only a scalar.

## Fair-comparison contract

Every retained sample has a continuous `USA_WIND` target interpolated inside a
three-hour IBTrACS bracket, a finite SAR robust peak, and a valid SAR mask cell
at the storm center after resize and center crop. Both scalar references are
calculated for every retained sample. Target-independent cohort fingerprints
enforce identical sample IDs and split ownership; separate target fingerprints
capture the changed labels.

For each ERA5 regime, one field-only U-Net is trained on this cohort and reused
for the two target variants. IBTrACS corrections use the predicted field
maximum as their anchor; SAR corrections use the predicted field top-0.5% mean.
Full-validation metrics still select checkpoints and control early stopping.

Cache schema v2 records both references, the primary target, correction anchor,
SAR maximum, RI status, 24-hour change, and filtering counts. Schema-v1 caches
remain readable as legacy IBTrACS/max-anchor caches.

Rapid intensification (RI) is diagnostic only. A sample is RI when interpolated
IBTrACS wind has increased by at least 30 kt over the preceding 24 hours. A
missing interpolatable history is non-RI with an undefined change.

## Result products

The generated cohort counts, SAR–IBTrACS divergence statistics, dual-reference
model tables, RI diagnostics, and storm trajectories live in the
[results report](intensity-comparison-results.md). Keeping those values out of
this page makes this document the stable description of the experiment rather
than a mixture of method and one particular run.

The divergence command writes JSON, CSV, and Markdown containing the full
overall/split/RI tables, center-valid rates, signed and absolute-error
quantiles, MAE and bias intervals, and per-sample rows:

```bash
uv run python scripts/analyze_sar_ibtracs_divergence.py \
  --output logs/intensity-comparisons/sar-ibtracs-divergence.json \
  --bootstrap-repetitions 2000 \
  --bootstrap-seed 42
```

## Two-GPU schedule

```mermaid
flowchart LR
  D[Matched dual-reference cohort]
  D -->|GPU 0| J[Train joint U-Net + MLP]
  D -->|GPU 1| U[Train shared field U-Net]
  U -->|GPU 1| C[Export cache v2]
  C -->|GPU 1| R[Train correction]
  J -->|GPU 0, IBTrACS cells| I[Train encoder-only U-Net + MLP]
  J --> E[Dual-reference overall + RI evaluation]
  R --> E
  I --> E
  C --> E
  E --> W[Consolidated files + W&B run]
```

GPU 0 trains the joint model while GPU 1 trains the field U-Net, exports its
cache, and trains the correction. The second target run in each ERA5 regime
reuses the first run's field-U-Net checkpoint.

The workflow explicitly uses `trainer.deterministic=false` because CUDA does
not provide a deterministic backward kernel for reflection padding. The random
seed remains fixed and is recorded with every run.

## Run the complete seed-42 matrix

The matrix runner trains two shared field U-Nets, four joint models, four
correction models, and two IBTrACS-only encoder models. It publishes one grouped W&B evaluation run in project
`geo2wf` with overall and RI tables, per-sample predictions, divergence
tables/plots, configs, checkpoint hashes, and JSON/CSV/Markdown artifacts.

```bash
uv run python scripts/run_matched_intensity_validation_matrix.py \
  --joint-gpu 0 \
  --pipeline-gpu 1 \
  --seed 42 \
  --wandb-project geo2wf
```

The full matrix requires the project's execution host with two visible CUDA
GPUs and valid W&B credentials. In restricted tool sessions, GPU commands and
training processes may need to be launched outside the filesystem sandbox.
The matrix updates the marked section of the results report only after all four
cells and the consolidated evaluation complete successfully.

Use `--smoke-test` for one epoch and one train/validation batch per stage.
`--disable-wandb` runs locally. Epoch counts can be overridden with
`--joint-epochs`, `--unet-epochs`, `--correction-epochs`, and
`--encoder-epochs`.

## Run one target/regime manually

```bash
uv run python scripts/run_intensity_model_comparison.py \
  --joint-gpu 0 \
  --pipeline-gpu 1 \
  --era5 with \
  --intensity-target-source ibtracs \
  --split val

uv run python scripts/run_intensity_model_comparison.py \
  --joint-gpu 0 \
  --pipeline-gpu 1 \
  --era5 without \
  --intensity-target-source sar_robust_peak \
  --split val
```

Both ERA5 regimes require ERA5 availability while selecting the cohort. The
no-ERA5 model omits ERA5 channels and predicts the absolute field, ensuring its
sample IDs and storm splits still match the with-ERA5 regime.

The runner protects Humberto 2025 (`AL082025`), Kiko 2025 (`EP112025`), and
Otis 2023 (`EP182023`) from training by default. It aborts if a protected storm
appears in the source or effective training split and records the audit in
`workflow.json`.

## Reuse checkpoints

Reusing the field U-Net also requires its resolved config because the config
defines the exact data and cache contract:

```bash
uv run python scripts/run_intensity_model_comparison.py \
  --unet-checkpoint /path/to/unet.ckpt \
  --unet-config /path/to/resolved-config.yaml \
  --joint-checkpoint /path/to/joint.ckpt \
  --correction-checkpoint /path/to/correction.ckpt \
  --encoder-checkpoint /path/to/encoder.ckpt \
  --joint-gpu 0 \
  --pipeline-gpu 1 \
  --intensity-target-source ibtracs
```

Reused correction checkpoints must originate from the exact cache identified
by the workflow; tensor compatibility alone is insufficient. Encoder
checkpoints are accepted only for IBTrACS-target workflows.

## Outputs and validation metrics

Each target/regime workflow writes `workflow.json`, stage logs, resolved model
runs, `unet-intensity-cache/`, and `val-comparison.{json,csv,md}`. The matrix
runner additionally writes `matrix-workflow.json`, the divergence artifacts,
`matched-validation.{json,csv,md}`, a per-sample prediction CSV, and a
divergence plot.

Every validation epoch logs under `val_ri/`:

- sample and storm counts;
- MAE, RMSE, bias, storm-macro MAE, category accuracy, macro-F1, and within-one
  accuracy against IBTrACS and the SAR robust peak;
- reference-aligned raw-U-Net baselines; and
- pixel-pooled field MAE, RMSE, and bias for field-producing models.

Distributed validation gathers per-sample rows before computing RI metrics. A
generic dataset with no RI samples logs zero counts and omits undefined rates;
the final schema-v2 evaluator rejects an empty RI cohort.

Validation remains a model-selection diagnostic. Freeze model and
hyperparameter choices before evaluating the held-out test split. One seed is
one trained instance; storm-bootstrap uncertainty does not measure variation
between training seeds.
