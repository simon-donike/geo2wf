#!/usr/bin/env bash
set -euo pipefail

project_root="/work/code/geo2wf"
python_bin="${project_root}/.venv/bin/python"
output_root="${project_root}/logs/current-experiment-evaluation"

cd "${project_root}"
mkdir -p "${output_root}"

declare -A experiment_roots=(
  [correction_image_radii]="logs/intensity-structure/unet-image-radii"
  [correction_mlp_radii]="logs/intensity-structure/mlp-radii"
  [latent_sar_era5_max_wind]="logs/latent-matrix/sar/era5/max-wind"
  [latent_sar_era5_max_wind_radii]="logs/latent-matrix/sar/era5/max-wind-radii"
  [latent_sar_no_era5_max_wind]="logs/latent-matrix/sar/no-era5/max-wind"
  [latent_sar_no_era5_max_wind_radii]="logs/latent-matrix/sar/no-era5/max-wind-radii"
  [latent_no_sar_era5_max_wind]="logs/latent-matrix/no-sar/era5/max-wind"
  [latent_no_sar_era5_max_wind_radii]="logs/latent-matrix/no-sar/era5/max-wind-radii"
  [latent_no_sar_no_era5_max_wind]="logs/latent-matrix/no-sar/no-era5/max-wind"
  [latent_no_sar_no_era5_max_wind_radii]="logs/latent-matrix/no-sar/no-era5/max-wind-radii"
)

completed_run() {
  local root="$1"
  "${python_bin}" -c '
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
runs = []
for path in root.glob("*/result.json"):
    payload = json.loads(path.read_text())
    if payload.get("status") == "completed":
        runs.append(path.parent)
if runs:
    print(max(runs, key=lambda value: value.name))
' "${root}"
}

declare -A runs=()
while true; do
  waiting=0
  for name in "${!experiment_roots[@]}"; do
    if [[ -n "${runs[${name}]:-}" ]]; then
      continue
    fi
    run_dir="$(completed_run "${experiment_roots[${name}]}")"
    if [[ -n "${run_dir}" ]]; then
      runs["${name}"]="${run_dir}"
      echo "Found completed ${name}: ${run_dir}"
    else
      waiting=$((waiting + 1))
    fi
  done
  if (( waiting == 0 )); then
    break
  fi
  echo "Waiting for ${waiting} experiment(s) at $(date --iso-8601=seconds)"
  sleep 60
done

validation_args=()
for name in "${!runs[@]}"; do
  validation_args+=(--run "${name}=${runs[${name}]}")
done
CUDA_VISIBLE_DEVICES=0 "${python_bin}" scripts/evaluate_current_experiments.py \
  "${validation_args[@]}" \
  --output-dir "${output_root}" \
  --accelerator gpu

with_extra=(
  --extra-run "correction_image_radii=${runs[correction_image_radii]}"
  --extra-run "correction_mlp_radii=${runs[correction_mlp_radii]}"
  --extra-run "latent_sar_era5_max_wind=${runs[latent_sar_era5_max_wind]}"
  --extra-run "latent_sar_era5_max_wind_radii=${runs[latent_sar_era5_max_wind_radii]}"
  --extra-run "latent_no_sar_era5_max_wind=${runs[latent_no_sar_era5_max_wind]}"
  --extra-run "latent_no_sar_era5_max_wind_radii=${runs[latent_no_sar_era5_max_wind_radii]}"
)
without_extra=(
  --extra-run "latent_sar_no_era5_max_wind=${runs[latent_sar_no_era5_max_wind]}"
  --extra-run "latent_sar_no_era5_max_wind_radii=${runs[latent_sar_no_era5_max_wind_radii]}"
  --extra-run "latent_no_sar_no_era5_max_wind=${runs[latent_no_sar_no_era5_max_wind]}"
  --extra-run "latent_no_sar_no_era5_max_wind_radii=${runs[latent_no_sar_no_era5_max_wind_radii]}"
)

CUDA_VISIBLE_DEVICES=0 "${python_bin}" \
  scripts/run_intensity_comparison_storm_inference.py \
  --era5 with \
  --comparison-run logs/intensity-comparisons/20260820T144011Z-with-era5 \
  --output-root "${output_root}/three-storm" \
  --device cuda:0 \
  "${with_extra[@]}" \
  >"${output_root}/three-storm-with-era5.log" 2>&1 &
with_pid=$!

CUDA_VISIBLE_DEVICES=1 "${python_bin}" \
  scripts/run_intensity_comparison_storm_inference.py \
  --era5 without \
  --comparison-run logs/intensity-comparisons/20260820T155344Z-without-era5 \
  --output-root "${output_root}/three-storm" \
  --device cuda:0 \
  "${without_extra[@]}" \
  >"${output_root}/three-storm-without-era5.log" 2>&1 &
without_pid=$!

wait "${with_pid}"
wait "${without_pid}"

test -s "${output_root}/validation-metrics.csv"
test -s "${output_root}/validation-results.json"
test -s "${output_root}/three-storm/with-era5.csv"
test -s "${output_root}/three-storm/with-era5-metrics.csv"
test -s "${output_root}/three-storm/with-era5.json"
test -s "${output_root}/three-storm/without-era5.csv"
test -s "${output_root}/three-storm/without-era5-metrics.csv"
test -s "${output_root}/three-storm/without-era5.json"

echo "Current-experiment evaluation completed at $(date --iso-8601=seconds)"
