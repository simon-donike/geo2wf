# Bottleneck U-Net + MLP

This model learns a SAR wind-field reconstruction and a continuous IBTrACS
maximum-wind estimate in one end-to-end network.

```mermaid
flowchart LR
  X[GEO + ERA5 condition] --> E[Shared U-Net encoder]
  E --> B[Bottleneck feature map]
  B --> D[Skip-connected decoder]
  D --> I[SAR wind field]
  B --> P[Spatial mean + max pooling]
  P --> M[MLP]
  M --> W[IBTrACS USA_WIND in m/s]
```

The scalar target is the continuous IBTrACS `USA_WIND` value converted from
knots to metres per second. Tropical-cyclone categories are not model targets
and do not contribute to the loss.

The data module reuses `PairedImageDataset` for all raster loading,
normalization, masking, and augmentation. It reads `USA_WIND` directly from
IBTrACS and linearly interpolates it at each SAR target timestamp between the
immediately preceding and following valid fixes. The two fixes may be at most
three hours apart. Exact fix timestamps use the recorded value directly.
Category fields, frozen-U-Net caches, and ERA5 are not involved in constructing
the scalar target. Samples outside a valid three-hour bracket are excluded.

With the current local files this retains 842/232/212 train/validation/test
samples when ERA5 is enabled, and 1,237/330/302 when it is disabled. The
remaining records lie outside a valid three-hour `USA_WIND` bracket.

Train with the default local data paths:

```bash
uv run geo2wf-train experiment=bottleneck_unet_mlp
```

Override the paired root and IBTrACS file through environment variables:

```bash
GEO2WF_JOINT_PAIRED_ROOT=/path/to/paired \
GEO2WF_IBTRACS_FILE=/path/to/ibtracs.ALL.list.v04r01.csv \
uv run geo2wf-train experiment=bottleneck_unet_mlp
```

For the GEO-only ablation, use the checked-in preset:

```bash
uv run geo2wf-train experiment=bottleneck_unet_mlp_no_era5
```

This sets `data.use_era5=false` and changes `model.condition_channels` from 23
to 14: ten GEO bands, distance to the storm center, and three solar-time
channels. The paired manifests and IBTrACS interpolation are unchanged. ERA5
rasters, derived ERA5 wind speed/vorticity, and ERA5 companion tensors are not
loaded or passed to either branch.

The objective is

\[
L = w_{image} L_{Huber,image} + w_{intensity} L_{Huber,IBTrACS},
\]

with both weights equal to one by default. Both terms update the shared
encoder. The image term additionally updates the decoder, while the continuous
IBTrACS term updates the bottleneck MLP. Checkpoints are selected by the
combined `val/loss`; the two component losses and image/intensity MAE, RMSE,
and bias are logged separately.
