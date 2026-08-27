#!/usr/bin/env bash
set -euo pipefail

project_root="/work/code/geo2wf"
python_bin="${project_root}/.venv/bin/python"
output_root="${project_root}/logs/intensity-comparisons/final-three-storm-inference"
parts_root="${output_root}/parts"
log_root="${output_root}/logs"
with_run="${project_root}/logs/intensity-comparisons/20260820T144011Z-with-era5"
without_run="${project_root}/logs/intensity-comparisons/20260820T155344Z-without-era5"
ablation_max_wind="${project_root}/logs/latent-structure/max-wind/20260827-111024_modular/checkpoints/epoch=154-step=22010.ckpt"
ablation_radii="${project_root}/logs/latent-structure/max-wind-radii/20260827-111026_modular/checkpoints/epoch=106-step=15194.ckpt"

mkdir -p "${parts_root}" "${log_root}"
cd "${project_root}"

run_part() {
    local regime="$1"
    local gpu="$2"
    local storm="$3"
    local part_name="$4"
    shift 4

    local label="without-era5"
    local comparison_run="${without_run}"
    if [[ "${regime}" == "with" ]]; then
        label="with-era5"
        comparison_run="${with_run}"
    fi

    local part_root="${parts_root}/${part_name}"
    local csv_path="${part_root}/${label}.csv"
    local json_path="${part_root}/${label}.json"
    if [[ -s "${csv_path}" && -s "${json_path}" ]]; then
        echo "Skipping completed part ${part_name}"
        return
    fi

    echo "Starting ${part_name} on physical GPU ${gpu} at $(date --iso-8601=seconds)"
    local command=(
        "${python_bin}"
        scripts/run_intensity_comparison_storm_inference.py
        --era5 "${regime}"
        --comparison-run "${comparison_run}"
        --storms "${storm}"
        --output-root "${part_root}"
        --device cuda:0
    )
    if [[ "${regime}" == "with" ]]; then
        command+=(
            --ablation-max-wind-checkpoint "${ablation_max_wind}"
            --ablation-radii-checkpoint "${ablation_radii}"
        )
    fi
    command+=("$@")
    CUDA_VISIBLE_DEVICES="${gpu}" "${command[@]}"
    echo "Completed ${part_name} at $(date --iso-8601=seconds)"
}

run_with_era5() {
    run_part with 0 AL082025 with-AL082025
    run_part with 0 EP112025 with-EP112025
    for shard in 0 1 2 3; do
        run_part with 0 EP182023 "with-EP182023-shard${shard}" \
            --shard-index "${shard}" --num-shards 4
    done
}

run_without_era5() {
    run_part without 1 AL082025 without-AL082025
    run_part without 1 EP112025 without-EP112025
    run_part without 1 EP182023 without-EP182023
}

echo "Three-storm inference started at $(date --iso-8601=seconds)"
run_with_era5 >"${log_root}/gpu0-with-era5.log" 2>&1 &
with_pid=$!
run_without_era5 >"${log_root}/gpu1-without-era5.log" 2>&1 &
without_pid=$!

with_status=0
without_status=0
wait "${with_pid}" || with_status=$?
wait "${without_pid}" || without_status=$?
if (( with_status != 0 || without_status != 0 )); then
    echo "Inference failed: with-ERA5=${with_status}, without-ERA5=${without_status}"
    tail -n 80 "${log_root}/gpu0-with-era5.log" || true
    tail -n 80 "${log_root}/gpu1-without-era5.log" || true
    exit 1
fi

"${python_bin}" scripts/combine_intensity_comparison_storm_inference.py \
    --parts-root "${parts_root}" \
    --output-root "${output_root}" \
    --regime both

"${python_bin}" scripts/build_three_storm_nowcast_results.py \
    --with-era5-csv "${output_root}/with-era5.csv" \
    --without-era5-csv "${output_root}/without-era5.csv"

uv run pytest -q tests/test_three_storm_nowcast_results.py
uv run mkdocs build --strict

test -s docs/assets/data/final-results/three-storm-nowcast-predictions.csv
test -s docs/assets/data/final-results/three-storm-nowcast-metrics.csv
test -s docs/assets/data/final-results/three-storm-nowcast.json
test -s docs/assets/images/final-results/three-storm-nowcasts.png
test -s docs/assets/images/final-results/three-storm-nowcasts.pdf

echo "Three-storm inference and publication completed at $(date --iso-8601=seconds)"
