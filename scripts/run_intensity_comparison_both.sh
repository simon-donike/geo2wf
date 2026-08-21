#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

env PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/run_intensity_model_comparison.py \
  --joint-gpu 0 \
  --pipeline-gpu 1 \
  --era5 with \
  --split val

env PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/run_intensity_model_comparison.py \
  --joint-gpu 0 \
  --pipeline-gpu 1 \
  --era5 without \
  --split val
