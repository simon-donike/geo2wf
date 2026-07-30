# PMW proxy pretraining

`scripts/export_geo_pmw_geotiffs.py` creates a larger GEO-to-PMW paired dataset with the same one-channel target shape as GEO-to-SAR. The intent is initialization on a related satellite image reconstruction task before SAR fine-tuning.

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
    uv run python scripts/export_geo_pmw_geotiffs.py \
      --config configs/config_pretrain_geo_pmw.yaml
    ```

=== "Ten GEO bands"

    ```bash
    uv run python scripts/export_geo_pmw_geotiffs.py \
      --config configs/config_pretrain_geo_pmw_10bands.yaml
    ```

=== "Ten bands + ERA5"

    ```bash
    uv run python scripts/export_geo_pmw_geotiffs.py \
      --config configs/config_pretrain_geo_pmw_10bands_era5.yaml
    ```

The exporter is PMW-anchored: it finds a closest GEO occurrence for each acceptable PMW observation, builds the same geographic crop contract as the SAR exporter, and writes generic `condition_path` and `target_path` columns. This is why `PairedImageDataset` can load either task without a separate class.

## Moving from pretraining to SAR

The checked-in training entry point does not automatically transfer a PMW checkpoint into a SAR run. A compatible transfer requires matching U-Net input/output channel counts and deliberate checkpoint loading. Keep `unet.dim`, `dim_mults`, condition channel count, and one-channel output aligned if transfer is planned.

!!! tip "Use proxy data to answer a specific question"
    Compare training from scratch against a controlled initialization with identical SAR data, seed, normalization, and evaluation. More proxy samples alone do not prove better SAR reconstruction.
