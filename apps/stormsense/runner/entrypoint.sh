#!/usr/bin/env bash
set -euo pipefail
cd "${STORMSENSE_ROOT:-/app}"
case "${1:-help}" in
  help)
    cat <<'EOF'
geo2wf full-stack container

  operational [args]  StormSense bootstrap/update/backfill/export/publish/retain
  cycle               One update, imagery, evaluation, export and optional publish
  schedule            Repeat finite cycles (STORMSENSE_INTERVAL_SECONDS=900)
  train [args]        Training with Hydra overrides or --config
  infer [args]        Research storm inference
  evaluate [args]     Research evaluation
  export [args]       Dataset preparation/export
  download [args]     Download pinned data/model releases using hf
  web-dev             Website development server on port 5173
  web-build           Build production website assets into artifacts/website
  deploy-check        Build and dry-run the Cloudflare Worker deployment
  deploy              Build and deploy the Cloudflare website/Worker
  docs [args]         MkDocs (e.g. docs build --strict --site-dir artifacts/site)
  test [args]         Python tests
  bash / python / hf / rclone / npm / wrangler ...   Run tools directly

Default startup only displays this help. Publication and deployment are explicit.
Legacy operational commands (e.g. bootstrap, --device cpu update) still work.
EOF
    ;;
  cycle)
    shift
    if [[ $# -ne 0 ]]; then
      echo "cycle takes no arguments; configure it with STORMSENSE_* variables" >&2
      exit 2
    fi
    exec bash apps/stormsense/runner/cycle.sh ;;
  schedule) shift; exec python apps/stormsense/runner/schedule.py "$@" ;;
  operational) shift; exec geo2wf-operational "$@" ;;
  train|infer|evaluate|export)
    workflow="$1"; shift; exec "geo2wf-$workflow" "$@" ;;
  download) shift; exec python scripts/download_artifacts.py "$@" ;;
  docs) shift; exec mkdocs "$@" ;;
  test) shift; exec python -m pytest "$@" ;;
  web-dev) shift; cd apps/stormsense; exec npm run dev -- --host 0.0.0.0 "$@" ;;
  web-build) shift; cd apps/stormsense; export STORMSENSE_PRODUCTION=1; exec npm run build -- --outDir ../../artifacts/website "$@" ;;
  deploy-check|deploy)
    workflow="$1"; shift; cd apps/stormsense
    if [[ "$workflow" == deploy-check ]]; then workflow=deploy:check; fi
    exec npm run "$workflow" -- "$@" ;;
  wrangler) shift; cd apps/stormsense; exec ./node_modules/.bin/wrangler "$@" ;;
  bootstrap|discover|discover-history|update|backfill|coverage|verify|imagery|publish|retain|--*)
    exec geo2wf-operational "$@" ;;
  *) exec "$@" ;;
esac
