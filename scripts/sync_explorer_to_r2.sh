#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="${GEO2WF_EXPLORER_DATA_DIR:-${repo_dir}/docs/explorer}"
bucket="${R2_BUCKET:-tcd}"
prefix="${R2_PREFIX:-explorer}"
endpoint="${R2_ENDPOINT:-https://9d9a9d8281d0329e2a5a36456ee7a9ff.r2.cloudflarestorage.com}"
version="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"

if ! command -v aws >/dev/null 2>&1; then
  echo "error: aws CLI is required" >&2
  exit 1
fi
if [[ ! "${version}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "error: version may contain only letters, numbers, dot, underscore, and dash" >&2
  exit 1
fi
for required in storm-data.json geo sar; do
  if [[ ! -e "${source_dir}/${required}" ]]; then
    echo "error: missing ${source_dir}/${required}" >&2
    exit 1
  fi
done

destination="s3://${bucket}/${prefix}/releases/${version}"
echo "Uploading explorer data to ${destination}"

# Release paths are immutable. Upload all assets before publishing latest.json.
aws s3 sync "${source_dir}/geo/" "${destination}/geo/" \
  --endpoint-url "${endpoint}" --exclude "*" --include "*.webp" \
  --cache-control "public,max-age=31536000,immutable"
aws s3 sync "${source_dir}/sar/" "${destination}/sar/" \
  --endpoint-url "${endpoint}" --exclude "*" --include "*.png" \
  --cache-control "public,max-age=31536000,immutable"
aws s3 cp "${source_dir}/storm-data.json" "${destination}/storm-data.json" \
  --endpoint-url "${endpoint}" --content-type "application/json" \
  --cache-control "public,max-age=31536000,immutable"

pointer_file="$(mktemp)"
trap 'rm -f "${pointer_file}"' EXIT
printf '{"version":"%s","manifest":"releases/%s/storm-data.json"}\n' \
  "${version}" "${version}" > "${pointer_file}"
aws s3 cp "${pointer_file}" "s3://${bucket}/${prefix}/latest.json" \
  --endpoint-url "${endpoint}" --content-type "application/json" \
  --cache-control "no-cache"

echo "Published ${version}"
echo "Public pointer: ${R2_PUBLIC_BASE_URL:-<your R2 public domain>}/${prefix}/latest.json"
