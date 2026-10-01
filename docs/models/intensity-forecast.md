# Six-hour scalar intensity forecast

`IntensityForecastMLP` predicts maximum wind six hours after the current
anchor. Its five inputs are current, −6 h, and −12 h intensity and the two
consecutive six-hour changes. It predicts a signed change added to the current
anchor, with the resulting wind clamped nonnegative.

This is a downstream scalar task, separate from the paper's instantaneous
wind-field and intensity comparisons.

## Data and training

The cache exporter produces historical pretraining splits (2000–2018 train,
2019–2022 validation) and matched train/validation/test records. Historical
anchors are IBTrACS; matched anchors are frozen post-hoc correction predictions.
Cache metadata preserves the feature scaler and producer provenance.

```bash
uv run geo2wf-export intensity-forecast-cache \
  --ibtracs-file /path/to/ibtracs.ALL.list.v04r01.csv \
  --intensity-cache-root /path/to/intensity-cache \
  --intensity-checkpoint /path/to/intensity-correction.ckpt \
  --output-root data/intensity_forecast

uv run geo2wf-train experiment=intensity_forecast_pretrain \
  data.root=data/intensity_forecast
```

The retained preset trains on the historical pretraining splits with a
change-balanced Huber objective. Checkpoints use `val/storm_macro_mae_ms`.

## Evaluate and infer

```bash
uv run geo2wf-evaluate intensity-forecast \
  --cache-root data/intensity_forecast \
  --checkpoint /path/to/forecast.ckpt --split test \
  --output logs/intensity-forecast-evaluation.json
```

Use `geo2wf-infer intensity-forecast` with the same inputs and a CSV output
path for per-record predictions. State whether evaluation uses historical
IBTrACS anchors or matched correction-model anchors.

## StormSense rollout

The dashboard uses an IBTrACS-pretrained checkpoint recursively: the +6 h
prediction becomes an input to the next step, producing +12 h without the
observed +6 h wind. This is not a separately trained 12-hour model. Best-track
inputs make the displayed results retrospective; real-time advisory inputs
would require separate evaluation. The dashboard's external ConvLSTM layer is
a different model whose implementation is outside this package.
