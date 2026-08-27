#!/usr/bin/env bash
set -euo pipefail

lane="${1:?usage: run_latent_mlp_matrix.sh <era5|no-era5|no-sar-era5|no-sar-no-era5> <gpu-index>}"
gpu_index="${2:?usage: run_latent_mlp_matrix.sh <era5|no-era5|no-sar-era5|no-sar-no-era5> <gpu-index>}"

case "${lane}" in
  era5)
    experiments=(
      latent_mlp_sar_era5_max_wind
      latent_mlp_sar_era5_max_wind_radii
      latent_mlp_no_sar_era5_max_wind
      latent_mlp_no_sar_era5_max_wind_radii
    )
    ;;
  no-era5)
    experiments=(
      latent_mlp_sar_no_era5_max_wind
      latent_mlp_sar_no_era5_max_wind_radii
      latent_mlp_no_sar_no_era5_max_wind
      latent_mlp_no_sar_no_era5_max_wind_radii
    )
    ;;
  no-sar-era5)
    experiments=(
      latent_mlp_no_sar_era5_max_wind
      latent_mlp_no_sar_era5_max_wind_radii
    )
    ;;
  no-sar-no-era5)
    experiments=(
      latent_mlp_no_sar_no_era5_max_wind
      latent_mlp_no_sar_no_era5_max_wind_radii
    )
    ;;
  *)
    echo "unknown lane: ${lane}" >&2
    exit 2
    ;;
esac

for experiment in "${experiments[@]}"; do
  echo "[$(date --iso-8601=seconds)] starting ${experiment} on GPU ${gpu_index}"
  CUDA_VISIBLE_DEVICES="${gpu_index}" WANDB_DISABLED=true \
    .venv/bin/geo2wf-train "experiment=${experiment}"
  echo "[$(date --iso-8601=seconds)] completed ${experiment}"
done
