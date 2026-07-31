#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PYTHON_BIN=${PYTHON_BIN:-"$ROOT_DIR/.venv/bin/python"}
GPU_ID=${GEO2WF_TRAINING_GPU:-1}
SUITE_ID=${GEO2WF_ABLATION_SUITE_ID:-"refinement-$(date -u +%Y%m%dT%H%M%SZ)"}
SUITE_DIR=${GEO2WF_ABLATION_SUITE_DIR:-"$ROOT_DIR/logs/ablation-suites/$SUITE_ID"}
RUNS_DIR="$SUITE_DIR/runs"
EVENTS_PATH="$SUITE_DIR/events.jsonl"
BASELINE_CHECKPOINT=${GEO2WF_INITIAL_BASELINE_CKPT:-"$ROOT_DIR/logs/20260730-132206_config_geo_sar_10bands_era5_residual/checkpoints/epoch=038-step=4758.ckpt"}
WANDB_PROJECT=${GEO2WF_WANDB_PROJECT:-geo2wf-refinement-ablations}
WANDB_MODE=${GEO2WF_WANDB_MODE:-online}

mkdir -p "$RUNS_DIR"
test -x "$PYTHON_BIN"
test -f "$BASELINE_CHECKPOINT"

"$PYTHON_BIN" - "$SUITE_DIR" "$SUITE_ID" "$GPU_ID" "$BASELINE_CHECKPOINT" "$WANDB_MODE" "$WANDB_PROJECT" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

suite, suite_id, gpu, checkpoint, wandb_mode, wandb_project = sys.argv[1:]
payload = {
    "schema_version": 1,
    "suite_id": suite_id,
    "kind": "refinement_training_ablations",
    "status": "running",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "suite_dir": str(Path(suite).resolve()),
    "gpu_id": gpu,
    "initial_baseline_checkpoint": str(Path(checkpoint).resolve()),
    "include_test_in_train": True,
    "wandb": {"mode": wandb_mode, "project": wandb_project},
    "experiments": [
        "stage1_control_finetune",
        "stage1_highwind_only",
        "stage1_peak_only",
        "stage1_radial_only",
        "stage1_exceedance_only",
        "stage1_sampling_only",
        "stage1_peak_aware",
        "stage1_peak_structure_balanced",
        "stage2_anchored_cfg",
        "stage2_weighting_only",
        "stage2_peak_only",
        "stage2_radial_only",
        "stage2_exceedance_only",
        "stage2_multiscale_only",
        "stage2_annular_only",
        "stage2_structured_asinh",
        "stage2_structured_linear",
    ],
}
Path(suite, "suite-manifest.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

record_event() {
  local experiment=$1
  local status=$2
  local run_dir=$3
  "$PYTHON_BIN" - "$EVENTS_PATH" "$experiment" "$status" "$run_dir" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path, experiment, status, run_dir = sys.argv[1:]
record = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "experiment": experiment,
    "status": status,
    "run_dir": str(Path(run_dir).resolve()),
}
with Path(path).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, sort_keys=True) + "\n")
PY
}

collect_results() {
  "$PYTHON_BIN" scripts/collect_ablation_results.py \
    --suite-dir "$SUITE_DIR"
}

mark_suite() {
  local status=$1
  "$PYTHON_BIN" - "$SUITE_DIR/suite-manifest.json" "$status" <<'PYMARK'
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
PYMARK
}

on_exit() {
  local exit_code=$?
  trap - EXIT
  collect_results || true
  if [[ $exit_code -ne 0 ]]; then
    mark_suite failed || true
  fi
  exit "$exit_code"
}
trap on_exit EXIT

run_training() {
  local experiment=$1
  local config_path=$2
  local weights_path=${3:-}
  local baseline_path=${4:-}
  local run_dir="$RUNS_DIR/$experiment"
  mkdir -p "$run_dir"
  if [[ -f "$run_dir/result.json" ]]; then
    local existing_status
    existing_status=$("$PYTHON_BIN" - "$run_dir/result.json" <<'PYSTATUS'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("status", ""))
PYSTATUS
)
    if [[ "$existing_status" == "completed" ]]; then
      record_event "$experiment" skipped_completed "$run_dir"
      return 0
    fi
  fi
  record_event "$experiment" started "$run_dir"
  local command=("$PYTHON_BIN" train.py --config "$config_path")
  if [[ -n "$weights_path" ]]; then
    command+=(--weights-only-path "$weights_path")
  fi

  set +e
  if [[ -n "$baseline_path" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    GEO2WF_RUN_DIR="$run_dir" \
    GEO2WF_BASELINE_CKPT="$baseline_path" \
    WANDB_MODE="$WANDB_MODE" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_RUN_GROUP="${experiment%%_*}" \
    WANDB_NAME="$experiment" \
    PYTHONUNBUFFERED=1 \
      "${command[@]}" >"$run_dir/launcher.log" 2>&1
  else
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    GEO2WF_RUN_DIR="$run_dir" \
    WANDB_MODE="$WANDB_MODE" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_RUN_GROUP="${experiment%%_*}" \
    WANDB_NAME="$experiment" \
    PYTHONUNBUFFERED=1 \
      "${command[@]}" >"$run_dir/launcher.log" 2>&1
  fi
  local exit_code=$?
  set -e

  if [[ $exit_code -eq 0 ]]; then
    record_event "$experiment" completed "$run_dir"
  else
    record_event "$experiment" failed "$run_dir"
  fi
  collect_results
  return "$exit_code"
}

result_checkpoint() {
  local result_path=$1
  "$PYTHON_BIN" - "$result_path" <<'PYCHECKPOINT'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("")
else:
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(payload.get("best_model_path") or payload.get("last_model_path") or "")
PYCHECKPOINT
}

evaluate_stage2() {
  local experiment=$1
  local config_path=$2
  local checkpoint
  checkpoint=$(result_checkpoint "$RUNS_DIR/$experiment/result.json")
  local evaluation="$SUITE_DIR/evaluation/$experiment"
  if [[ -z "$checkpoint" || ! -f "$checkpoint" ]]; then
    record_event "${experiment}_ensemble_evaluation" skipped "$evaluation"
    return 0
  fi

  local inference="$evaluation/inference/guidance_1p2"
  mkdir -p "$inference"
  record_event "${experiment}_ensemble_evaluation" started "$evaluation"
  set +e
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  GEO2WF_BASELINE_CKPT="$STAGE1_CHECKPOINT" \
  PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" scripts/run_storm_diffusion_inference.py \
      --config "$config_path" \
      --checkpoint "$checkpoint" \
      --output-root "$inference" \
      --storms AL082025 EP112025 \
      --ensemble-size 10 \
      --guidance-scale 1.2 \
      --summary-aggregation median \
      --member-quantiles 0.1 0.9 \
      >"$evaluation/inference.log" 2>&1
  local exit_code=$?
  set -e
  if [[ $exit_code -ne 0 ]]; then
    record_event "${experiment}_ensemble_evaluation" failed "$evaluation"
    collect_results
    return "$exit_code"
  fi

  mapfile -t summary_files < <(
    find "$inference" -name inference-summary.csv -type f | sort
  )
  if [[ ${#summary_files[@]} -eq 0 ]]; then
    record_event "${experiment}_ensemble_evaluation" failed "$evaluation"
    return 1
  fi
  for method in affine isotonic; do
    for metric in msw robust_peak; do
      if ! "$PYTHON_BIN" scripts/calibrate_ensemble_intensity.py \
        "${summary_files[@]}" \
        --output-dir "$evaluation/calibration/guidance_1p2/$metric/$method" \
        --prediction-column "output_${metric}_ms_member_median" \
        --target-column ibtracs_msw_ms \
        --method "$method" \
        >>"$evaluation/calibration.log" 2>&1; then
        record_event "${experiment}_ensemble_evaluation" failed "$evaluation"
        collect_results
        return 1
      fi
    done
  done
  record_event "${experiment}_ensemble_evaluation" completed "$evaluation"
  collect_results
}

# Quantify the actual residual tail before testing the linear transform. A
# completed artifact is reusable when a long suite is resumed after interruption.
if [[ ! -s "$SUITE_DIR/residual-distribution-initial.json" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" \
    scripts/analyze_residual_distribution.py \
    --config configs/config_geo_sar_10bands_era5_residual.yaml \
    --checkpoint "$BASELINE_CHECKPOINT" \
    --output "$SUITE_DIR/residual-distribution-initial.json" \
    >"$SUITE_DIR/residual-distribution-initial.log" 2>&1
fi

run_training \
  stage1_control_finetune \
  configs/ablations/config_stage1_control_finetune.yaml \
  "$BASELINE_CHECKPOINT" || true
run_training \
  stage1_highwind_only \
  configs/ablations/config_stage1_highwind_only.yaml \
  "$BASELINE_CHECKPOINT" || true
run_training \
  stage1_peak_only \
  configs/ablations/config_stage1_peak_only.yaml \
  "$BASELINE_CHECKPOINT" || true
run_training \
  stage1_radial_only \
  configs/ablations/config_stage1_radial_only.yaml \
  "$BASELINE_CHECKPOINT" || true
run_training \
  stage1_exceedance_only \
  configs/ablations/config_stage1_exceedance_only.yaml \
  "$BASELINE_CHECKPOINT" || true
run_training \
  stage1_sampling_only \
  configs/ablations/config_stage1_sampling_only.yaml \
  "$BASELINE_CHECKPOINT" || true
run_training \
  stage1_peak_aware \
  configs/ablations/config_stage1_peak_aware.yaml \
  "$BASELINE_CHECKPOINT" || true

if ! run_training \
  stage1_peak_structure_balanced \
  configs/ablations/config_stage1_peak_structure_balanced.yaml \
  "$BASELINE_CHECKPOINT"; then
  echo "The required full Stage 1 experiment failed; see its launcher.log." >&2
  exit 1
fi

STAGE1_RESULT="$RUNS_DIR/stage1_peak_structure_balanced/result.json"
STAGE1_CHECKPOINT=$("$PYTHON_BIN" - "$STAGE1_RESULT" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
print(payload.get("best_model_path") or payload.get("last_model_path") or "")
PY
)
test -n "$STAGE1_CHECKPOINT"
test -f "$STAGE1_CHECKPOINT"

# Re-measure residuals after the peak/structure-aware Stage 1 model, then use
# its q99.9 tail to set the linear-transform clip reproducibly.
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" \
  scripts/analyze_residual_distribution.py \
  --config configs/config_geo_sar_10bands_era5_residual.yaml \
  --checkpoint "$STAGE1_CHECKPOINT" \
  --output "$SUITE_DIR/residual-distribution-stage1-peak-structure.json" \
  >"$SUITE_DIR/residual-distribution-stage1-peak-structure.log" 2>&1

LINEAR_CONFIG="$SUITE_DIR/config_stage2_structured_linear_data_driven.yaml"
"$PYTHON_BIN" - \
  configs/ablations/config_stage2_structured_linear.yaml \
  "$SUITE_DIR/residual-distribution-stage1-peak-structure.json" \
  "$LINEAR_CONFIG" <<'PYCONFIG'
import json
import sys
from pathlib import Path

import yaml

template_path, distribution_path, output_path = map(Path, sys.argv[1:])
config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
distribution = json.loads(distribution_path.read_text(encoding="utf-8"))
clip_ms = float(distribution["recommended_linear_clip_ms"])
config["model"]["residual"]["clip_ms"] = clip_ms
config["ablation_metadata"] = {
    "linear_clip_source": str(distribution_path.resolve()),
    "linear_clip_quantile": "absolute residual q99.9 rounded up to 5 m/s",
    "linear_clip_ms": clip_ms,
}
output_path.write_text(
    yaml.safe_dump(config, sort_keys=False),
    encoding="utf-8",
)
PYCONFIG

run_training \
  stage2_anchored_cfg \
  configs/ablations/config_stage2_anchored_cfg.yaml \
  "" "$STAGE1_CHECKPOINT" || true
run_training \
  stage2_weighting_only \
  configs/ablations/config_stage2_weighting_only.yaml \
  "" "$STAGE1_CHECKPOINT" || true
run_training \
  stage2_peak_only \
  configs/ablations/config_stage2_peak_only.yaml \
  "" "$STAGE1_CHECKPOINT" || true
run_training \
  stage2_radial_only \
  configs/ablations/config_stage2_radial_only.yaml \
  "" "$STAGE1_CHECKPOINT" || true
run_training \
  stage2_exceedance_only \
  configs/ablations/config_stage2_exceedance_only.yaml \
  "" "$STAGE1_CHECKPOINT" || true
run_training \
  stage2_multiscale_only \
  configs/ablations/config_stage2_multiscale_only.yaml \
  "" "$STAGE1_CHECKPOINT" || true
run_training \
  stage2_annular_only \
  configs/ablations/config_stage2_annular_only.yaml \
  "" "$STAGE1_CHECKPOINT" || true
run_training \
  stage2_structured_asinh \
  configs/ablations/config_stage2_structured_asinh.yaml \
  "" "$STAGE1_CHECKPOINT" || true
run_training \
  stage2_structured_linear \
  "$LINEAR_CONFIG" \
  "" "$STAGE1_CHECKPOINT" || true

# Score every successfully trained Stage 2 best checkpoint using the same ten
# stochastic members, seeds, storms, guidance, aggregations, and calibrators.
evaluate_stage2 \
  stage2_anchored_cfg \
  configs/ablations/config_stage2_anchored_cfg.yaml || true
evaluate_stage2 \
  stage2_weighting_only \
  configs/ablations/config_stage2_weighting_only.yaml || true
evaluate_stage2 \
  stage2_peak_only \
  configs/ablations/config_stage2_peak_only.yaml || true
evaluate_stage2 \
  stage2_radial_only \
  configs/ablations/config_stage2_radial_only.yaml || true
evaluate_stage2 \
  stage2_exceedance_only \
  configs/ablations/config_stage2_exceedance_only.yaml || true
evaluate_stage2 \
  stage2_multiscale_only \
  configs/ablations/config_stage2_multiscale_only.yaml || true
evaluate_stage2 \
  stage2_annular_only \
  configs/ablations/config_stage2_annular_only.yaml || true
evaluate_stage2 \
  stage2_structured_asinh \
  configs/ablations/config_stage2_structured_asinh.yaml || true
evaluate_stage2 \
  stage2_structured_linear \
  "$LINEAR_CONFIG" || true

FINAL_STATUS=$("$PYTHON_BIN" - "$EVENTS_PATH" <<'PYSTATUS'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
failed = False
if path.exists():
    for line in path.read_text(encoding="utf-8").splitlines():
        failed = failed or json.loads(line).get("status") == "failed"
print("completed_with_failures" if failed else "completed")
PYSTATUS
)
collect_results
mark_suite "$FINAL_STATUS"
printf '%s\n' "$SUITE_DIR"
