# PMW proxy pretraining

`geo2wf-export geo-pmw` creates a larger GEO-to-PMW paired dataset with the same one-channel target shape as GEO-to-SAR. The intent is initialization on a related satellite image reconstruction task before SAR fine-tuning.

## Supported PMW channels

| Sensor | Selected target | Swath |
|---|---|---|
| AMSR2 on GCOM-W1 | `TB_A89.0V` | `S5` |
| GMI on GPM | `TB_89.0V` | `S1` |
| SSMIS F16/F17/F18 | `TB_91.665V` | `S4` |

Each target remains **one channel**, even in 10-band GEO experiments. That preserves model output dimensionality between proxy pretraining and SAR fine-tuning.

## Export

=== "Four GEO bands"

    ```bash
    uv run geo2wf-export geo-pmw \
      --config configs/config_pretrain_geo_pmw.yaml
    ```

=== "Ten GEO bands"

    ```bash
    uv run geo2wf-export geo-pmw \
      --config configs/config_pretrain_geo_pmw_10bands.yaml
    ```

=== "Ten bands + ERA5"

    ```bash
    uv run geo2wf-export geo-pmw \
      --config configs/config_pretrain_geo_pmw_10bands_era5.yaml
    ```

The exporter is PMW-anchored: it finds a closest GEO occurrence for each acceptable PMW observation, builds the same geographic crop contract as the SAR exporter, and writes generic `condition_path` and `target_path` columns. This is why `PairedImageDataset` can load either task without a separate class.

## Moving from pretraining to SAR

Transfer is explicit: use `--weights-only-path` so model weights load strictly
while optimizer, scheduler, epoch, and step state start fresh.

```bash
uv run geo2wf-train \
  model=conditional_diffusion \
  --weights-only-path /path/to/pmw-pretraining.ckpt
```

The model architecture and condition/target widths must match exactly.

!!! tip "Use proxy data to answer a specific question"
    Compare training from scratch against a controlled initialization with identical SAR data, seed, normalization, and evaluation. More proxy samples alone do not prove better SAR reconstruction.
