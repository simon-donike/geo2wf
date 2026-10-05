// Read-only probe of a deployed Worker; local Vite does not implement edge cache.
import { request, devices, expect } from "@playwright/test";
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";

const baseURL = process.env.STORMSENSE_BASE_URL;
assert(baseURL, "Set STORMSENSE_BASE_URL to the deployed website");
const client = await request.newContext({
  baseURL,
  userAgent: devices["Desktop Chrome"].userAgent,
});
const cacheStatus = (response) => response.headers()["x-stormsense-cache"];
try {
  const latest = await client.get("/data/latest.json");
  assert.equal(latest.status(), 200);
  assert.equal(cacheStatus(latest), "BYPASS");
  const pointer = await latest.json();
  const catalog = await (await client.get("/data/" + pointer.manifest)).json();
  const storm = catalog.storms.find((s) => s.id === "EP152026");
  const series = await (await client.get("/data/" + storm.series)).json();
  const frame = series.imagery.find((frame) => frame.status === "ready");
  const bundle = series.imagery_bundles?.[0];
  const path = "/data/" + (bundle?.path || frame.parts[0].image);
  const first = await client.get(path);
  assert.equal(first.status(), 200);
  const bytes = await first.body();
  if (bundle) {
    assert.equal(first.headers()["content-type"], "application/zip");
    assert.equal(bytes.length, bundle.bytes);
    assert.equal(createHash("sha256").update(bytes).digest("hex"), bundle.sha256);
  }
  const statuses = [cacheStatus(first)];
  await expect
    .poll(
      async () => {
        const response = await client.get(path + "?cache-probe=1");
        assert.equal(response.status(), 200);
        assert.deepEqual(await response.body(), bytes);
        statuses.push(cacheStatus(response));
        return cacheStatus(response);
      },
      { timeout: 10000 },
    )
    .toBe("HIT");
  const head = await client.head(path);
  assert.equal(head.status(), 200);
  assert.equal(cacheStatus(head), "HIT");
  assert.equal((await head.body()).length, 0);
  const conditional = await client.get(path, {
    headers: { "If-None-Match": first.headers().etag },
  });
  assert.equal(conditional.status(), 304);
  assert.equal(cacheStatus(conditional), "HIT");
  const pointerAgain = await client.get("/data/latest.json");
  assert.equal(cacheStatus(pointerAgain), "BYPASS");
  assert.deepEqual(await pointerAgain.json(), pointer);
  assert.equal((await client.get("/data/secret.json")).status(), 404);
  assert.equal((await client.post(path)).status(), 405);
  const report = {
    checked_at: new Date().toISOString(),
    url: baseURL,
    release: pointer.version,
    asset: path,
    daily_bundle: Boolean(bundle),
    bytes: bytes.length,
    cache_statuses: statuses,
    query_strings_share_cache: true,
    head_cache: cacheStatus(head),
    conditional_status: conditional.status(),
    conditional_cache: cacheStatus(conditional),
    latest_pointer_cache: cacheStatus(pointerAgain),
    writes_rejected: true,
  };
  if (process.env.STORMSENSE_CACHE_REPORT) {
    await writeFile(
      process.env.STORMSENSE_CACHE_REPORT,
      JSON.stringify(report, null, 2) + "\n",
    );
  }
  console.log(JSON.stringify(report));
} finally {
  await client.dispose();
}
