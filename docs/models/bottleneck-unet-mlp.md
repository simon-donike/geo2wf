# Joint field and latent MLP

`BottleneckUNetMLPRegressor` shares an encoder between a wind-field decoder and an MLP
that predicts continuous IBTrACS maximum wind. Spatial mean and max pooling
connect the bottleneck feature map to the scalar head. This is the joint
representation studied in the [paper](../concepts/problem.md).

The input has 23 condition channels with ERA5 or 14 without it, plus a
condition-validity mask. The decoder predicts a normalized field that is
converted to physical wind using the target statistics. The explicit ERA5
residual addition belongs to the field-only model. The scalar head predicts intensity
directly rather than taking the maximum of the decoded field.

## Targets and loss

The data module interpolates IBTrACS `USA_WIND` to each SAR timestamp only
between valid fixes at most three hours apart; an exact fix uses its recorded
value. Knots are converted to m/s. Categories are derived diagnostics, not
training labels.

The default objective is

\[
L = L_{\mathrm{Huber,field}} + L_{\mathrm{Huber,intensity}},
\]

with transitions of 2 and 5 m/s. Both terms update the encoder; the field term
also updates the decoder. Checkpoints use combined `val/loss`.

## Optional radii supervision

The structure head predicts five nonnegative values in kilometres: eye size,
RMW (called \(R_{\max}\) in the paper), and equivalent-area R34/R50/R64.
Each target has an independent validity mask. A missing eye or radius label
contributes no loss. Radius-enabled presets add masked Huber loss with a
20 km transition and weight 0.25.

Direct head predictions and radii diagnosed from the decoded image are
reported separately; image-diagnosed radii do not contribute to this scalar
loss. The published [supervision ablation](../results.md#latent-supervision-ablation)
tests the contribution of SAR and radius targets.

## Train

```bash
uv run geo2wf-train experiment=latent_mlp_sar_no_era5_max_wind_radii \
  data.root=/path/to/paired \
  data.ibtracs_file=/path/to/ibtracs.ALL.list.v04r01.csv
```

Use `bottleneck_unet_mlp` for the ERA5-conditioned field/intensity model
without radius supervision. The [preset table](index.md#training-presets)
lists the other retained combinations. `data.stats_file` follows `data.root`
in these joint presets. Hardware and logging overrides are described in
[training](../experiments/training.md).

## Encoder-only control

`BottleneckEncoderMLPRegressor` retains the encoder and scalar heads but removes the
decoder and field loss. Select a `latent_mlp_no_sar_*` preset to compare scalar
learning with and without SAR supervision. The matched data cohort can still
require a SAR-valid center; “no SAR” describes the objective, not a claim that
the training samples have no SAR record.
