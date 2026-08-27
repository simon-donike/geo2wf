# Joint U-Net/latent-MLP structure results

The maximum-wind-only and maximum-wind-plus-radii runs are in progress. This
page will be replaced automatically after both best validation checkpoints are
evaluated once on the held-out test split.

The result builder writes separately labeled latent-MLP and 2D-field radius
metrics, machine-readable JSON and CSV artifacts, and publication-ready plots:

```bash
uv run geo2wf-evaluate latent-structure \
  --max-wind-run logs/latent-structure/max-wind/<completed-run> \
  --radii-run logs/latent-structure/max-wind-radii/<completed-run>
```
