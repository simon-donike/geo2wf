# Post-hoc intensity correction

This model takes one frozen U-Net wind field and current metadata, then adds
a learned signed correction to the field's valid-pixel maximum. A compact
CNN encodes wind, validity, and storm-center distance; an MLP combines those
features with current location/time metadata. The result is clamped
nonnegative. It estimates current intensity from one observation.

The zero-initialized final layer initially reproduces the raw field maximum.
Training uses tropical IBTrACS `USA_WIND` in m/s, with `USA_SSHS` from −1 to 5.
Storm-balanced, capped category-aware weights enter a 5 m/s Huber objective.
Categories are derived from continuous predictions. The default checkpoint
monitor is `val/storm_macro_mae_ms`.

Optional radius presets either retain field-diagnosed radii or enable a
separate masked scalar structure head. Keep the source of each reported radius
explicit. The default `unet_intensity_correction` has no structure loss.

## Prepare a local cache

The cache contains split manifests, frozen field arrays, and
`cache-metadata.json`, which records the producer checkpoint and source hashes.
Generate it from source observations using:

```bash
uv run geo2wf-export intensity-cache \
  --data-root /path/to/archive \
  --manifest /path/to/observation_manifest.csv \
  --ibtracs-file /path/to/ibtracs.ALL.list.v04r01.csv \
  --config /path/to/unet/resolved-config.yaml \
  --checkpoint /path/to/unet.ckpt \
  --stats /path/to/paired/stats.json \
  --output-root data/unet_intensity
```

The frozen producer must match the cache provenance. End-to-end evaluation
also depends on the producer's training membership; a clean downstream split
cannot undo upstream exposure to evaluation storms.

## Train and evaluate

```bash
uv run geo2wf-train experiment=unet_intensity_correction \
  data.root=data/unet_intensity

uv run geo2wf-evaluate intensity-correction \
  --cache-root data/unet_intensity \
  --checkpoint /path/to/intensity.ckpt --split test \
  --output logs/intensity-evaluation.json
```

`geo2wf-infer intensity-correction` accepts the same cache/checkpoint/split
arguments and writes prediction CSV via `--output`. Its outputs include
`raw_unet_max_wind_ms`, `correction_ms`, `output_msw_ms`, and `output_category`.
The earlier three-storm viewer labels this post-hoc product **U-Net+MLP**.
The current [StormSense app](https://stormsense.hyperalislabs.com/) instead
uses the [jointly trained GEO-only latent model](bottleneck-unet-mlp.md).
