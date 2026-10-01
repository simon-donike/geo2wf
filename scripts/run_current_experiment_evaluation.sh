#!/usr/bin/env bash
# Fixed publication cohort and checkpoints; no latest-run discovery or polling.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
exec uv run python scripts/conference_release.py evaluate-latent "$@"
