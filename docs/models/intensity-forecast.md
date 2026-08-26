# Six-hour scalar intensity forecast

This workflow trains a one-step model for maximum sustained wind six hours
after the current anchor. During matched fine-tuning, that anchor is the current
single-field correction estimate; the other history values are IBTrACS
`USA_WIND` at −6 h and −12 h. The model learns a signed change around the
anchor.

```text
forecast(t + 6 h) = max(0, corrected U-Net intensity(t) + learned change)
```

## Export the forecast cache

```bash
uv run geo2wf-export intensity-forecast-cache \
  --ibtracs-file data/IBTrACs/ibtracs.ALL.list.v04r01.csv \
  --intensity-cache-root data/unet_intensity_geostat_nopmw_v2 \
  --intensity-checkpoint /path/to/intensity-correction.ckpt \
  --output-root data/intensity_forecast
```

The exporter creates storm-disjoint historical pretraining splits for
2000–2018 and 2019–2022, plus matched `train`, `val`, and `test` splits. In
historical pretraining the current anchor is IBTrACS itself; in the matched
splits it is the frozen correction-model prediction. The exporter selects one
current field per storm/fix and records source hashes and three
rapid-intensification validation cases in `cache-metadata.json`.

## Train both stages

```bash
uv run geo2wf-train \
  experiment=intensity_forecast_pretrain \
  data.root=data/intensity_forecast

uv run geo2wf-train \
  --weights-only-path /path/to/pretrain.ckpt \
  experiment=intensity_forecast_finetune \
  data.root=data/intensity_forecast
```

The five model inputs are the current anchor, the −6 h and −12 h winds, and
the two consecutive six-hour changes. Validation reports MAE, RMSE, bias,
storm-macro MAE, persistence, and recent-trend baselines.

Each fine-tuning validation epoch also performs a recursive +6 h/+12 h rollout
for `WP282025`, `WP112024`, and `AL092024`. For each storm, initialization is
the latest usable matched sample at or before RI onset. If no pre-onset sample
exists, selection falls back to the earliest usable sample after onset, so the
two-step diagnostic can begin inside the RI period. W&B receives a three-panel
RI plot and a six-row forecast table. The observed +6 h intensity is never fed
into the second forecast step.

## One-step training and +12 h dashboard rollout

`IntensityForecastMLP` is optimized only against a +6 h target. Its
`predict_two_steps(...)` helper feeds the first prediction into the next
history window and applies the same model again, producing recursive +6 h and
+12 h values without using the observed +6 h wind. StormSense's MLP forecast
layer displays this retrospective recursive +12 h diagnostic. Error can
compound across the two steps, so it is not equivalent to a separately trained
12-hour model.

## Evaluate and infer

```bash
uv run geo2wf-evaluate intensity-forecast \
  --cache-root data/intensity_forecast \
  --checkpoint /path/to/forecast.ckpt \
  --split test \
  --output logs/intensity-forecast-evaluation.json

uv run geo2wf-infer intensity-forecast \
  --cache-root data/intensity_forecast \
  --checkpoint /path/to/forecast.ckpt \
  --split test \
  --output inference/intensity-forecast.csv
```

IBTrACS is a retrospective best-track product. Operational use should replace
the historical inputs with real-time advisory or ATCF data and re-evaluate the
distribution shift. The dashboard's separate external ConvLSTM +12 h artifact
is not this MLP and is not implemented in the maintained model package.
