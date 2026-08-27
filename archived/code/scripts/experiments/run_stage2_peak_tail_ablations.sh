#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

PYTHON_BIN=${PYTHON_BIN:-"$ROOT_DIR/.venv/bin/python"}
GPU_ID=${GEO2WF_TRAINING_GPU:-1}
SUITE_ID=${GEO2WF_STAGE2_TAIL_SUITE_ID:-"stage2-peak-tail-$(date -u +%Y%m%dT%H%M%SZ)"}
SUITE_DIR=${GEO2WF_STAGE2_TAIL_SUITE_DIR:-"$ROOT_DIR/logs/ablation-suites/$SUITE_ID"}
RUNS_DIR="$SUITE_DIR/runs"
CONFIGS_DIR="$SUITE_DIR/configs"
EVENTS_PATH="$SUITE_DIR/events.jsonl"
STAGE1_CHECKPOINT=${GEO2WF_STAGE1_CHECKPOINT:-"$ROOT_DIR/logs/ablation-suites/refinement-rerun-20260731T130254Z/runs/stage1_peak_aware/checkpoints/epoch=020-step=5103.ckpt"}
BASE_CONFIG=${GEO2WF_STAGE2_BASE_CONFIG:-"$ROOT_DIR/configs/ablations/config_stage2_structured_asinh.yaml"}
WANDB_PROJECT=${GEO2WF_WANDB_PROJECT:-geo2wf-stage2-peak-tail-ablations}
WANDB_MODE=${GEO2WF_WANDB_MODE:-online}
STORMS=${GEO2WF_ENSEMBLE_STORMS:-"AL082025 EP112025"}

mkdir -p "$RUNS_DIR" "$CONFIGS_DIR"
test -x "$PYTHON_BIN"
test -f "$STAGE1_CHECKPOINT"
test -f "$BASE_CONFIG"

"$PYTHON_BIN" - "$SUITE_DIR" "$SUITE_ID" "$GPU_ID" "$STAGE1_CHECKPOINT" "$BASE_CONFIG" "$WANDB_MODE" "$WANDB_PROJECT" "$STORMS" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

suite, suite_id, gpu, checkpoint, base_config, wandb_mode, wandb_project, storms = sys.argv[1:]
payload = {
    "schema_version": 1,
    "suite_id": suite_id,
    "kind": "stage2_peak_tail_retraining_ablations",
    "status": "running",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "suite_dir": str(Path(suite).resolve()),
    "gpu_id": gpu,
    "stage1_baseline_checkpoint": str(Path(checkpoint).resolve()),
    "base_config": str(Path(base_config).resolve()),
    "include_test_in_train": True,
    "storms": storms.split(),
    "ensemble_size": 10,
    "guidance_scale": 1.2,
    "summary_aggregation": "median",
    "member_quantiles": [0.1, 0.9],
    "calibration_methods": ["affine", "isotonic"],
    "wandb": {"mode": wandb_mode, "project": wandb_project},
    "experiments": [
        "stage2_retrained_structured_asinh",
        "stage2_retrained_peak_weight_0p1",
        "stage2_retrained_peak_weight_0p3",
        "stage2_retrained_tail_bundle",
        "stage2_retrained_tail_bundle_strong",
    ],
}
Path(suite, "suite-manifest.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
  "$PYTHON_BIN" scripts/collect_ablation_results.py --suite-dir "$SUITE_DIR" || true
}

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
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

on_exit() {
  local exit_code=$?
  trap - EXIT
  collect_results
  if [[ $exit_code -ne 0 ]]; then
    mark_suite failed || true
  fi
  exit "$exit_code"
}
trap on_exit EXIT

# Materialize every variant from one frozen template. This keeps the Stage 1
# handoff and all non-tail settings identical across the comparison.
"$PYTHON_BIN" - "$BASE_CONFIG" "$CONFIGS_DIR" "$STAGE1_CHECKPOINT" <<'PY'
import copy
import sys
from pathlib import Path

import yaml

template_path, output_dir, checkpoint = map(Path, sys.argv[1:])
template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
loss_variants = {
    "stage2_retrained_structured_asinh": {
        "id": "stage2_retrained_structured_asinh",
        "design": "corrected_stage1_retrain_control",
        "changes": ["retrained_from_peak_aware_stage1"],
    },
    "stage2_retrained_peak_weight_0p1": {
        "id": "stage2_retrained_peak_weight_0p1",
        "design": "single_factor_ablation",
        "changes": ["retrained_from_peak_aware_stage1", "robust_peak_weight=0.1"],
        "robust_peak_weight": 0.1,
    },
    "stage2_retrained_peak_weight_0p3": {
        "id": "stage2_retrained_peak_weight_0p3",
        "design": "single_factor_ablation",
        "changes": ["retrained_from_peak_aware_stage1", "robust_peak_weight=0.3"],
        "robust_peak_weight": 0.3,
    },
    "stage2_retrained_tail_bundle": {
        "id": "stage2_retrained_tail_bundle",
        "design": "tail_bundle_ablation",
        "changes": [
            "retrained_from_peak_aware_stage1",
            "robust_peak_weight=0.1",
            "high_wind_weight=8.0",
            "exceedance_area_weight=3.0",
        ],
        "robust_peak_weight": 0.1,
        "high_wind_weight": 8.0,
        "exceedance_area_weight": 3.0,
    },
    "stage2_retrained_tail_bundle_strong": {
        "id": "stage2_retrained_tail_bundle_strong",
        "design": "tail_bundle_ablation",
        "changes": [
            "retrained_from_peak_aware_stage1",
            "robust_peak_weight=0.3",
            "high_wind_weight=8.0",
            "exceedance_area_weight=3.0",
        ],
        "robust_peak_weight": 0.3,
        "high_wind_weight": 8.0,
        "exceedance_area_weight": 3.0,
    },
}
output_dir.mkdir(parents=True, exist_ok=True)
for name, settings in loss_variants.items():
    config = copy.deepcopy(template)
    config.setdefault("ablation", {})
    config["ablation"].update(
        {
            "id": settings["id"],
            "family": "stage2_diffusion_peak_tail",
            "reference": "stage2_retrained_structured_asinh",
            "design": settings["design"],
            "changes": settings["changes"],
        }
    )
    config["model"]["residual"]["baseline"]["checkpoint_path"] = str(
        checkpoint.resolve()
    )
    loss = config["model"]["residual"]["loss"]
    for key in ("robust_peak_weight", "high_wind_weight", "exceedance_area_weight"):
        if key in settings:
            loss[key] = settings[key]
    config.setdefault("data", {})["include_test_in_train"] = True
    config.setdefault("validation", {})["log_reconstruction_images"] = True
    config.setdefault("logging", {}).setdefault("wandb", {})["project"] = (
        "geo2wf-stage2-peak-tail-ablations"
    )
    config["logging"]["wandb"]["name"] = name
    (output_dir / f"{name}.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
PY

result_checkpoint() {
  local result_path=$1
  "$PYTHON_BIN" - "$result_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("")
else:
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(payload.get("best_model_path") or payload.get("last_model_path") or "")
PY
}

run_variant() {
  local experiment=$1
  local config_path="$CONFIGS_DIR/$experiment.yaml"
  local run_dir="$RUNS_DIR/$experiment"
  mkdir -p "$run_dir"
  if [[ -f "$run_dir/result.json" ]]; then
    local existing_status
    existing_status=$(
      "$PYTHON_BIN" - "$run_dir/result.json" <<'PY'
import json
import sys
print(json.loads(open(sys.argv[1], encoding="utf-8").read()).get("status", ""))
PY
    )
    if [[ "$existing_status" == "completed" ]]; then
      record_event "$experiment" skipped_completed "$run_dir"
      return 0
    fi
  fi
  record_event "$experiment" started "$run_dir"
  set +e
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
    GEO2WF_RUN_DIR="$run_dir" \
    GEO2WF_BASELINE_CKPT="$STAGE1_CHECKPOINT" \
    WANDB_MODE="$WANDB_MODE" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_RUN_GROUP="stage2_peak_tail" \
    WANDB_NAME="$experiment" \
    PYTHONUNBUFFERED=1 \
      "$PYTHON_BIN" train.py --config "$config_path" >"$run_dir/launcher.log" 2>&1
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

evaluate_variant() {
  local experiment=$1
  local config_path="$CONFIGS_DIR/$experiment.yaml"
  local checkpoint
  checkpoint=$(result_checkpoint "$RUNS_DIR/$experiment/result.json")
  local evaluation="$SUITE_DIR/evaluation/$experiment"
  local inference="$evaluation/inference"
  if [[ -z "$checkpoint" || ! -f "$checkpoint" ]]; then
    record_event "${experiment}_evaluation" skipped "$evaluation"
    return 0
  fi
  mkdir -p "$inference"
  record_event "${experiment}_evaluation" started "$evaluation"
  read -r -a storm_args <<<"$STORMS"
  set +e
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
    GEO2WF_BASELINE_CKPT="$STAGE1_CHECKPOINT" \
    PYTHONUNBUFFERED=1 \
      "$PYTHON_BIN" scripts/run_storm_diffusion_inference.py \
        --config "$config_path" \
        --checkpoint "$checkpoint" \
        --output-root "$inference" \
        --storms "${storm_args[@]}" \
        --ensemble-size 10 \
        --guidance-scale 1.2 \
        --summary-aggregation median \
        --member-quantiles 0.1 0.9 \
        >"$evaluation/inference.log" 2>&1
  local exit_code=$?
  set -e
  if [[ $exit_code -ne 0 ]]; then
    record_event "${experiment}_evaluation" failed "$evaluation"
    collect_results
    return "$exit_code"
  fi
  mapfile -t summary_files < <(find "$inference" -name inference-summary.csv -type f | sort)
  if [[ ${#summary_files[@]} -eq 0 ]]; then
    record_event "${experiment}_evaluation" failed "$evaluation"
    return 1
  fi
  for method in affine isotonic; do
    for metric in msw robust_peak; do
      if ! "$PYTHON_BIN" scripts/calibrate_ensemble_intensity.py \
        "${summary_files[@]}" \
        --output-dir "$evaluation/calibration/$metric/$method" \
        --prediction-column "output_${metric}_ms_member_median" \
        --target-column ibtracs_msw_ms \
        --method "$method" \
        >>"$evaluation/calibration.log" 2>&1; then
        record_event "${experiment}_evaluation" failed "$evaluation"
        collect_results
        return 1
      fi
    done
  done
  record_event "${experiment}_evaluation" completed "$evaluation"
  collect_results
}

variants=(
  stage2_retrained_structured_asinh
  stage2_retrained_peak_weight_0p1
  stage2_retrained_peak_weight_0p3
  stage2_retrained_tail_bundle
  stage2_retrained_tail_bundle_strong
)

# Run sequentially so a long run cannot evict another process from the GPU;
# failures are recorded and the remaining ablations continue.
for experiment in "${variants[@]}"; do
  run_variant "$experiment" || true
done
for experiment in "${variants[@]}"; do
  evaluate_variant "$experiment" || true
done

FINAL_STATUS=$(
  "$PYTHON_BIN" - "$EVENTS_PATH" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
failed = any(
    json.loads(line).get("status") == "failed"
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
)
print("completed_with_failures" if failed else "completed")
PY
)
collect_results
mark_suite "$FINAL_STATUS"
printf '%s\n' "$SUITE_DIR"
