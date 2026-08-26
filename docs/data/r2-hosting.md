# Host explorer data on Cloudflare R2

The explorer can keep its frontend on GitHub Pages while R2 serves the generated JSON and image overlays. Releases are immutable and `latest.json` is published last, so the site never selects a partially uploaded release.

## One-time Cloudflare setup

Uploads use the authenticated rclone remote `r2:tcd/explorer`. The authenticated API endpoint is not the URL the browser should fetch.

In the R2 settings for the `tcd` bucket:

1. Create an R2 API token with Object Read & Write access to `tcd`.
2. Add a custom domain for production, or temporarily enable the `r2.dev` development URL.
3. Add this CORS policy, replacing the origin if the documentation domain changes:

```json
[
  {
    "AllowedOrigins": ["https://tcd.hyperalis.com"],
    "AllowedMethods": ["GET", "HEAD"],
    "AllowedHeaders": ["*"],
    "MaxAgeSeconds": 86400
  }
]
```

Configure the `r2` rclone remote locally with credentials that can write to the `tcd` bucket. Do not commit its configuration. Verify access with `rclone lsd r2:tcd/explorer`.

## Publish a release

Regenerate the browser data, then sync it. An explicit version makes rollback and auditing straightforward:

```bash
uv run python scripts/export_storm_explorer_data.py
./scripts/sync_explorer_to_r2.sh 2026-07-30
```

This publishes the GEO, SAR, and PMW overlays, lazy-loaded per-storm forecast JSON, `storm-data.json`, and its flat observation-level `storm-data.csv` view under `explorer/releases/2026-07-30/`, then advances `explorer/latest.json`. Forecast files are uploaded before the manifest that references them, and `latest.json` is advanced last. Omit the version to use a UTC timestamp. Old releases remain available for rollback.

Point the website at one or more public release-pointer URLs in
`docs/explorer/data-config.js`. They are tried in order, which permits a
production endpoint followed by an optional fallback:

```javascript
window.GEO2WF_EXPLORER_RELEASE_URLS = [
  "https://data.example.org/explorer/latest.json",
];
```

Commit and deploy that change once the public domain and CORS policy are active. To roll back without rebuilding the website, upload a `latest.json` whose `manifest` refers to an older release.

## Optional automation

For CI, provide an rclone configuration containing the `r2` remote, then run the same script after data generation. Large inference inputs are not in this repository, so automatic export belongs on the machine or workflow that produces a completed inference version.
