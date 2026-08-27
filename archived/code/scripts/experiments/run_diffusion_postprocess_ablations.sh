#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PYTHON_BIN=${PYTHON_BIN:-"$ROOT_DIR/.venv/bin/python"}
GPU_ID=${GEO2WF_INFERENCE_GPU:-0}
SUITE_ID=${GEO2WF_POSTPROCESS_SUITE_ID:-"diffusion-postprocess-$(date -u +%Y%m%dT%H%M%SZ)"}
SUITE_DIR=${GEO2WF_POSTPROCESS_SUITE_DIR:-"$ROOT_DIR/logs/ablation-suites/$SUITE_ID"}
SOURCE_INFERENCE=${GEO2WF_POSTPROCESS_SOURCE_INFERENCE:-"$ROOT_DIR/logs/ablation-suites/model-c-rerun-20260731T130254Z/inference/guidance_1p2"}
BASELINE_ROOT=${GEO2WF_POSTPROCESS_BASELINE_ROOT:-"$SUITE_DIR/baseline"}
OUTPUT_ROOT=${GEO2WF_POSTPROCESS_OUTPUT_ROOT:-"$SUITE_DIR/postprocess"}
STORMS=${GEO2WF_POSTPROCESS_STORMS:-"AL082025 EP112025"}
BASELINE_CONFIG=${GEO2WF_STAGE1_CONFIG:-"$ROOT_DIR/configs/config_geo_sar_10bands_era5_residual.yaml"}
BASELINE_CHECKPOINT=${GEO2WF_STAGE1_CHECKPOINT:-"$ROOT_DIR/logs/20260730-132206_config_geo_sar_10bands_era5_residual/checkpoints/epoch=038-step=4758.ckpt"}
BASELINE_DEVICE=${GEO2WF_POSTPROCESS_DEVICE:-cuda}

mkdir -p "$SUITE_DIR"
test -x "$PYTHON_BIN"
test -d "$SOURCE_INFERENCE"
test -f "$BASELINE_CONFIG"
test -f "$BASELINE_CHECKPOINT"

"$PYTHON_BIN" - "$SUITE_DIR/suite-manifest.json" "$SUITE_ID" "$SOURCE_INFERENCE" "$BASELINE_ROOT" "$OUTPUT_ROOT" "$STORMS" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path, suite_id, source, baseline, output, storms = sys.argv[1:]
payload = {
    "schema_version": 1,
    "suite_id": suite_id,
    "kind": "diffusion_residual_postprocessing_ablations",
    "status": "running",
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "source_inference": str(Path(source).resolve()),
    "baseline_root": str(Path(baseline).resolve()),
    "output_root": str(Path(output).resolve()),
    "storms": storms.split(),
    "include_test_in_train": True,
    "gain_values": [0.0, 0.25, 0.5, 0.75, 1.0],
    "residual_caps_ms": [8.0, 16.0, 24.0],
    "uncapped_control_included": True,
    "median_kernels": [0, 3],
    "calibration_methods": ["affine", "isotonic"],
    "calibration_predictors": [
        "output_msw_ms_member_median",
        "output_robust_peak_ms_member_median",
    ],
}
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

if [[ ! -f "$BASELINE_ROOT/run-metadata.json" ]]; then
  mkdir -p "$BASELINE_ROOT"
  read -r -a storm_args <<<"$STORMS"
  CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" scripts/save_deterministic_baseline_fields.py \
      --config "$BASELINE_CONFIG" \
      --checkpoint "$BASELINE_CHECKPOINT" \
      --output-root "$BASELINE_ROOT" \
      --storms "${storm_args[@]}" \
      --device "$BASELINE_DEVICE" \
      >"$SUITE_DIR/baseline.log" 2>&1
fi

read -r -a storm_args <<<"$STORMS"
"$PYTHON_BIN" scripts/postprocess_diffusion_ablations.py \
  --input-root "$SOURCE_INFERENCE" \
  --baseline-root "$BASELINE_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --storms "${storm_args[@]}" \
  >"$SUITE_DIR/postprocess.log" 2>&1

calibration_index="$OUTPUT_ROOT/calibration-index.csv"
printf 'variant,metric,method,output_dir,status\n' >"$calibration_index"
while IFS= read -r summary_path; do
  variant_dir=$(dirname "$summary_path")
  variant=$(basename "$variant_dir")
  for metric in msw robust_peak; do
    predictor="output_${metric}_ms_member_median"
    for method in affine isotonic; do
      output_dir="$OUTPUT_ROOT/calibration/$variant/$metric/$method"
      if "$PYTHON_BIN" scripts/calibrate_ensemble_intensity.py \
        "$summary_path" \
        --output-dir "$output_dir" \
        --prediction-column "$predictor" \
        --target-column ibtracs_msw_ms \
        --method "$method" \
        >>"$SUITE_DIR/calibration.log" 2>&1; then
        status=completed
      else
        status=failed
        printf '%s,%s,%s,%s,%s\n' "$variant" "$metric" "$method" "$output_dir" "$status" >>"$calibration_index"
        exit 1
      fi
      printf '%s,%s,%s,%s,%s\n' "$variant" "$metric" "$method" "$output_dir" "$status" >>"$calibration_index"
    done
  done
done < <(find "$OUTPUT_ROOT/variants" -mindepth 2 -maxdepth 2 -name inference-summary.csv -type f | sort)

"$PYTHON_BIN" - "$SUITE_DIR/suite-manifest.json" "$OUTPUT_ROOT" "$calibration_index" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest_path, output_root, calibration_index = map(Path, sys.argv[1:])
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
payload.update(
    {
        "status": "complete",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "postprocess_metadata": str((output_root / "postprocess-metadata.json").resolve()),
        "calibration_index": str(Path(calibration_index).resolve()),
    }
)
manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

printf '%s\n' "$SUITE_DIR"
