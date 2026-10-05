# Troubleshooting

## Missing manifest, raster, or statistics

Training expects a local export/cache. Check `data.root/<split>/manifest.csv`,
every referenced file, and `data.stats_file`. Downloading only the Hugging Face
indexes does not download imagery, and the Hub catalog root is not the local
export layout. See [dataset access](../data/index.md) and the
[loader contract](../data/dataset-contract.md). `prepare_data()` does not fetch data.

## Channel or checkpoint mismatch

Use the checkpoint's matching configuration, normalization, and band order.
The default ERA5 data has 23 condition channels; the field model appends a
condition mask, ERA5 wind, and ERA5 mask for 26 inputs. Other models assemble
helpers differently. Follow the `DataSpec` error and model guide instead of
changing a channel count just to satisfy a convolution.

## Samples disappear

`require_era5` and the time-gap limit filter missing/stale context. Scalar
workflows also filter target availability and may require a SAR-valid center.
Inspect manifest paths, timestamps, and the selected data config. The number of
source manifest rows need not equal the final loader length.

## Missing structural validation metric

Sparse SAR coverage or a very small validation limit can omit structural
components. Increase validation coverage or use a consistently available
monitor such as `val/loss` during debugging. Some diagnostics are unavailable
by design; they must not be replaced by zeros.

## Slow or hanging data loading

Use zero workers to isolate multiprocessing issues. In the basic paired
config this is `data.loader.num_workers=0`; joint/scalar configs use
`data.num_workers=0`. Disable persistent workers when worker count is zero.
Under DDP, workers and batch size are per process. Reducing reconstruction
media can also shorten validation.

## W&B still starts

Set `WANDB_DISABLED=true` or `logging.wandb.enabled=false`. Offline mode stores
W&B activity locally rather than disabling it.

## Legacy comparison model has no training config

The earlier viewer's ViT and ConvLSTM layers are imported comparison artifacts.
Its **U-Net+MLP** label means post-hoc correction of a frozen field. Current
[StormSense](https://stormsense.hyperalislabs.com/) uses the released joint
GEO-only latent model and scalar forecast MLP. See the
[model inventory](../models/index.md) for checkpoints and trainable implementations.
