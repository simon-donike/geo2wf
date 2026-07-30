# Host explorer data on Cloudflare R2

The explorer can keep its frontend on GitHub Pages while R2 serves the generated JSON and image overlays. Releases are immutable and `latest.json` is published last, so the site never selects a partially uploaded release.

## One-time Cloudflare setup

`https://9d9a9d8281d0329e2a5a36456ee7a9ff.r2.cloudflarestorage.com` is the authenticated S3 API endpoint used for uploads. It is **not** the URL the browser should fetch.

In the R2 settings for the `tcd` bucket:

1. Create an R2 API token with Object Read & Write access to `tcd`.
2. Add a custom domain for production, or temporarily enable the `r2.dev` development URL.
3. Add this CORS policy, replacing the origin if the documentation domain changes:

```json
[
  {
    "AllowedOrigins": ["https://simon-donike.github.io"],
    "AllowedMethods": ["GET", "HEAD"],
    "AllowedHeaders": ["*"],
    "MaxAgeSeconds": 86400
  }
]
```

Configure AWS CLI credentials locally. Do not commit them:

```bash
export AWS_ACCESS_KEY_ID="<R2 access key ID>"
export AWS_SECRET_ACCESS_KEY="<R2 secret access key>"
export AWS_DEFAULT_REGION="auto"
```

## Publish a release

Regenerate the browser data, then sync it. An explicit version makes rollback and auditing straightforward:

```bash
uv run python scripts/export_storm_explorer_data.py
R2_PUBLIC_BASE_URL=https://data.example.org \
  ./scripts/sync_explorer_to_r2.sh 2026-07-30
```

This publishes `explorer/releases/2026-07-30/` and then `explorer/latest.json`. Running the same command again synchronizes missing or changed files. Omit the version to use a UTC timestamp. Old releases remain available for rollback.

Point the website at the public URL in `docs/explorer/data-config.js`:

```javascript
window.GEO2WF_EXPLORER_RELEASE_URL =
  "https://data.example.org/explorer/latest.json";
```

Commit and deploy that change once the public domain and CORS policy are active. To roll back without rebuilding the website, upload a `latest.json` whose `manifest` refers to an older release.

## Optional automation

For CI, store `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `R2_ENDPOINT`, and `R2_PUBLIC_BASE_URL` as secrets, then run the same script after data generation. Large inference inputs are not in this repository, so automatic export belongs on the machine or workflow that produces a completed inference version.
