// Keep this empty to use data bundled with the documentation.
// URLs are tried in order; the r2.dev URL is a development fallback.
// The *.r2.cloudflarestorage.com endpoint is only for authenticated uploads.
const explorerPreviewHost =
  location.hostname === "localhost" ||
  location.hostname === "127.0.0.1" ||
  location.hostname.endsWith(".trycloudflare.com");

window.GEO2WF_EXPLORER_RELEASE_URLS = explorerPreviewHost ? [] : [
  "https://data.tcd.hyperalislabs.com/explorer/latest.json",
  "https://pub-35b383ddf0c1402c9edd4b9f100619d8.r2.dev/explorer/latest.json",
];
