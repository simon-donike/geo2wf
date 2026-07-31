# First experiment

This path proves the complete contract with two exported pairs per split and a deliberately bounded training loop.

## 1. Export a tiny GEO–SAR dataset

```bash
uv run python scripts/export_geo_sar_geotiffs.py \
  --config configs/config.yaml \
  --limit 2
```

Expected layout:

```text
data/geotiff/geo_sar/
├── stats.json
├── train/
│   ├── manifest.csv
│   ├── <sample>_geo.tif
│   └── <sample>_sar.tif
├── val/
└── test/
```

The export matches each SAR observation to the closest acceptable GEO observation from the same split, builds a shared 256 × 256 EPSG:4326 grid, regrids channels, writes raw physical values and masks, then calculates statistics from the **training split only**.

## 2. Check one batch

```bash
uv run python - <<'PY'
import yaml
from data import PairedDataModule

with open("configs/config.yaml", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)

dm = PairedDataModule.from_config(config)
dm.setup("fit")
batch = next(iter(dm.train_dataloader()))
for key in ("condition", "target", "condition_mask", "target_mask"):
    print(key, tuple(batch[key].shape), batch[key].dtype)
PY
```

For the 4-band baseline, expect `condition` to have 8 channels: four GEO bands, distance to the IBTrACS center, local-solar-time sine/cosine, and normalized solar-zenith angle. The model appends a one-channel condition-validity mask internally, which is why `model.in_channels` is 9.

## 3. Bound the smoke run

Copy the base config to a local ignored or temporary config and set:

```yaml
trainer:
  max_epochs: 1
  limit_train_batches: 1
  limit_val_batches: 1
  enable_checkpointing: false
logging:
  wandb:
    enabled: false
```

Then launch:

```bash
uv run python train.py --config /path/to/smoke.yaml
```

!!! note "Why a copied config?"
    Keeping smoke-only limits out of `configs/config.yaml` avoids accidentally carrying them into a real experiment. `train.py` also accepts `--limit-val-batches`, but train-batch limiting currently comes from YAML.

## 4. Read the signals

A healthy run should:

- print the seeded Lightning setup and U-Net time-embedding message;
- report `train/loss` and `val/loss` without NaNs;
- perform a reverse-process reconstruction for the configured validation batch;
- report normalized PSNR, SSIM, and L1; and
- report physical metrics when reversible target statistics are available.

Sampling is much more expensive than one training loss step. For a faster exploratory validation path, use a DDIM config with fewer reverse steps; see [Sampling](../models/sampling.md).

## Next

- Inspect [what each batch field means](../data/dataset-contract.md).
- Read the [main two-stage workflow](../models/two-stage.md).
- Learn [how standalone diffusion loss is formed](../models/conditional-diffusion.md).
- Compare [all experiment presets](../experiments/index.md).
