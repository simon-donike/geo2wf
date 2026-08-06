#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="${GEO2WF_EXPLORER_DATA_DIR:-${repo_dir}/docs/explorer}"
remote="${R2_REMOTE:-r2:tcd/explorer}"
version="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"

if ! command -v rclone >/dev/null 2>&1; then
  echo "error: rclone is required" >&2
  exit 1
fi
if [[ ! "${version}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "error: version may contain only letters, numbers, dot, underscore, and dash" >&2
  exit 1
fi
for required in storm-data.json geo sar pmw forecasts; do
  if [[ ! -e "${source_dir}/${required}" ]]; then
    echo "error: missing ${source_dir}/${required}" >&2
    exit 1
  fi
done

destination="${remote}/releases/${version}"
echo "Uploading explorer data to ${destination}"

# Release paths are immutable. Upload every asset before publishing latest.json.
rclone copy "${source_dir}/geo" "${destination}/geo" --include "*.webp"
rclone copy "${source_dir}/sar" "${destination}/sar" --include "*.png"
rclone copy "${source_dir}/pmw" "${destination}/pmw" --include "*.png"
rclone copy "${source_dir}/forecasts" "${destination}/forecasts" --include "*.json"
rclone copyto "${source_dir}/storm-data.json" "${destination}/storm-data.json"

pointer_file="$(mktemp)"
trap "rm -f ${pointer_file}" EXIT
printf "{\"version\":\"%s\",\"manifest\":\"releases/%s/storm-data.json\"}\n" \
  "${version}" "${version}" > "${pointer_file}"
rclone copyto "${pointer_file}" "${remote}/latest.json"

echo "Published ${version}"
echo "Release pointer: ${remote}/latest.json"
