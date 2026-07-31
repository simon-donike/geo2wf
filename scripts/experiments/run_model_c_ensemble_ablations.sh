#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PYTHON_BIN=${PYTHON_BIN:-"$ROOT_DIR/.venv/bin/python"}
GPU_ID=${GEO2WF_INFERENCE_GPU:-0}
SUITE_ID=${GEO2WF_ENSEMBLE_SUITE_ID:-"model-c-ensemble-$(date -u +%Y%m%dT%H%M%SZ)"}
SUITE_DIR=${GEO2WF_ENSEMBLE_SUITE_DIR:-"$ROOT_DIR/logs/ablation-suites/$SUITE_ID"}
CHECKPOINT=${GEO2WF_DIFFUSION_CKPT:-"$ROOT_DIR/logs/20260730-150623_config_geo_sar_10bands_era5_diffusion_residual_deterministic/checkpoints/epoch=055-step=6832.ckpt"}
CONFIG=${GEO2WF_DIFFUSION_CONFIG:-"$ROOT_DIR/configs/config_geo_sar_10bands_era5_diffusion_residual_deterministic.yaml"}
STORMS=${GEO2WF_ENSEMBLE_STORMS:-"AL082025 EP112025"}
INFERENCE_DIR="$SUITE_DIR/inference"

mkdir -p "$INFERENCE_DIR"
test -x "$PYTHON_BIN"
test -f "$CHECKPOINT"
test -f "$CONFIG"

"$PYTHON_BIN" - "$SUITE_DIR" "$SUITE_ID" "$GPU_ID" "$CHECKPOINT" "$CONFIG" "$STORMS" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

suite, suite_id, gpu, checkpoint, config, storms = sys.argv[1:]
payload = {
    "schema_version": 1,
    "suite_id": suite_id,
    "kind": "model_c_ensemble_and_calibration_ablations",
    "status": "running",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "suite_dir": str(Path(suite).resolve()),
    "gpu_id": gpu,
    "checkpoint": str(Path(checkpoint).resolve()),
    "config": str(Path(config).resolve()),
    "storms": storms.split(),
    "ensemble_size": 10,
    "guidance_scales": [1.0, 1.2, 1.5],
    "summary_aggregation": "median",
    "member_quantiles": [0.1, 0.9],
    "calibration_methods": ["affine", "isotonic"],
    "calibration_predictors": [
        "output_msw_ms_member_median",
        "output_robust_peak_ms_member_median",
    ],
}
Path(suite, "suite-manifest.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

mark_suite() {
  local status=$1
  "$PYTHON_BIN" - "$SUITE_DIR/suite-manifest.json" "$status" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path, status = Path(sys.argv[1]), sys.argv[2]
payload = json.loads(path.read_text(encoding="utf-8"))
payload["status"] = status
payload["finished_at"] = datetime.now(timezone.utc).isoformat()
path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

collect_results() {
  "$PYTHON_BIN" scripts/collect_ablation_results.py \
    --suite-dir "$SUITE_DIR"
}

mark_inference_failed() {
  "$PYTHON_BIN" - "$INFERENCE_DIR" <<'PYFAIL'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
finished = datetime.now(timezone.utc).isoformat()
for path in root.glob("guidance_*/run-metadata.json"):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") == "complete":
        continue
    payload["status"] = "failed"
    payload["completed_utc"] = finished
    payload["failure_recorded_by"] = "run_model_c_ensemble_ablations.sh"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
PYFAIL
}

on_exit() {
  local exit_code=$?
  trap - EXIT
  collect_results || true
  if [[ $exit_code -ne 0 ]]; then
    mark_inference_failed || true
    mark_suite failed
  fi
  exit "$exit_code"
}
trap on_exit EXIT

read -r -a storm_args <<<"$STORMS"
CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 \
  "$PYTHON_BIN" scripts/run_storm_diffusion_inference.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output-root "$INFERENCE_DIR" \
    --storms "${storm_args[@]}" \
    --ensemble-size 10 \
    --guidance-scales 1.0 1.2 1.5 \
    --summary-aggregation median \
    --member-quantiles 0.1 0.9 \
    --save-member-fields \
    >"$SUITE_DIR/inference.log" 2>&1

for guidance in 1 1p2 1p5; do
  mapfile -t summary_files < <(
    find "$INFERENCE_DIR/guidance_$guidance" \
      -name inference-summary.csv -type f | sort
  )
  if [[ ${#summary_files[@]} -eq 0 ]]; then
    echo "No inference summaries found for guidance $guidance" >&2
    exit 1
  fi

  for method in affine isotonic; do
    for metric in msw robust_peak; do
      predictor="output_${metric}_ms_member_median"
      calibration_dir="$SUITE_DIR/calibration/guidance_$guidance/$metric/$method"
      "$PYTHON_BIN" scripts/calibrate_ensemble_intensity.py \
        "${summary_files[@]}" \
        --output-dir "$calibration_dir" \
        --prediction-column "$predictor" \
        --target-column ibtracs_msw_ms \
        --method "$method" \
        >>"$SUITE_DIR/calibration.log" 2>&1
    done
  done
done

collect_results
mark_suite completed
printf '%s\n' "$SUITE_DIR"
