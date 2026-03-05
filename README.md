# Conditional Pixel Diffusion (Minimal)

This repository is intentionally minimal and focused on **conditional pixel-space diffusion**.

## Kept implementation
- `src/PixelDiffusion.py`
- `src/DenoisingDiffusionProcess/`

## Scope
- Conditional pixel diffusion is in scope.
- Unconditional and latent diffusion parts are removed from the active package surface.

## Available notebook
- `03-Conditional-Pixel-Diffusion.ipynb`
- `03-Conditional-Pixel-Diffusion-colab.ipynb`

## Dependencies
Assuming `torch` and `torchvision` are installed:

```bash
pip install pytorch-lightning==1.9.3 einops
```
