#!/usr/bin/env bash
set -euo pipefail
runner_root="${STORMSENSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$runner_root"
runner_python="${STORMSENSE_PYTHON:-$runner_root/.venv/bin/python}"
runner_device="${STORMSENSE_DEVICE:-cpu}"
runner_db="${STORMSENSE_DB:-var/stormsense/state.sqlite}"
runner_export="${STORMSENSE_EXPORT:-var/stormsense/export}"
runner_models="${STORMSENSE_MODELS:-downloads/models}"
command_args=(-m geo2wf.operational.cli --db "$runner_db" --model-root "$runner_models" --device "$runner_device")
update_status=0
"$runner_python" "${command_args[@]}" update || update_status=$?
# Optional imagery never blocks numerical publication. Revisit recent gaps as
# GIBS finishes producing its delayed display product. This installs no timer.
imagery_status=0
"$runner_python" "${command_args[@]}" imagery --recent-hours 48 --active-only --workers 2 || imagery_status=$?
# Export the recorded discovery failure as well as successes, preserving the
# previous numerical data and the last successful source retrieval time.
"$runner_python" "${command_args[@]}" evaluate
"$runner_python" "${command_args[@]}" export --output "$runner_export"
if [[ "${STORMSENSE_PUBLISH:-0}" == "1" ]]; then
  "$runner_python" "${command_args[@]}" publish --output "$runner_export"
fi
if [[ "$update_status" != "0" ]]; then exit "$update_status"; fi
exit "$imagery_status"
