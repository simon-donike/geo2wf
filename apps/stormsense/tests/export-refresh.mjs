// Real local pipeline-to-browser check. Requires an exported numerical archive.
import { chromium, expect } from "@playwright/test";
import { spawn } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";

const root = fileURLToPath(new URL("../../../", import.meta.url));
const output = resolve(root, "apps/stormsense/public/data");
const read = (path) => JSON.parse(readFileSync(resolve(output, path), "utf8"));
const before = read("latest.json");
const catalog = read(before.manifest);
const candidates = catalog.storms.filter((s) => s.prediction_count > 12);
const storm = candidates.find((s) => s.pending_count > 0) || candidates[0];
assert(storm, "Export real storm estimates before running this check");

const browser = await chromium.launch();
try {
  const page = await browser.newPage();
  await page.clock.install();
  const base = process.env.STORMSENSE_PREVIEW_URL || "http://127.0.0.1:5173";
  const url = `${base}/storms/${storm.id}?time=${encodeURIComponent(storm.latest_prediction.time)}`;
  await page.goto(url);
  await expect(
    page.getByRole("slider", { name: "Storm timeline" }),
  ).toBeVisible();
  const selected = page.locator(".chart-panel .panel-heading p");
  const initialTime = await selected.innerText();
  let navigations = 0;
  page.on("framenavigated", (frame) => {
    if (frame === page.mainFrame()) navigations++;
  });
  await new Promise((fulfill, reject) => {
    const child = spawn(
      process.env.STORMSENSE_PYTHON || resolve(root, ".venv/bin/python"),
      [
        "-m",
        "geo2wf.operational.cli",
        "--db",
        process.env.STORMSENSE_DB || "var/stormsense/state.sqlite",
        "export",
        "--start",
        catalog.window.start,
        "--end",
        catalog.window.end,
        "--output",
        output,
      ],
      { cwd: root, stdio: ["ignore", "ignore", "pipe"] },
    );
    let diagnostic = "";
    child.stderr.on("data", (data) => {
      diagnostic = (diagnostic + data).slice(-4000);
    });
    child.on("error", reject);
    child.on("close", (code) =>
      code === 0
        ? fulfill()
        : reject(Error(diagnostic || `Export exited ${code}`)),
    );
  });
  const after = read("latest.json");
  assert.notEqual(after.version, before.version);
  const refreshed = page.waitForResponse((r) =>
    r.url().endsWith(`/data/${after.manifest}`),
  );
  await page.clock.fastForward(61000);
  assert.equal((await refreshed).status(), 200);
  await expect(selected).toHaveText(initialTime);
  assert.equal(page.url(), url);
  assert.equal(navigations, 0);
  const result = {
    before_release: before.version,
    after_release: after.version,
    storm_id: storm.id,
    selected_time: storm.latest_prediction.time,
    preserved_selection: true,
    page_reloads: 0,
    real_export: true,
  };
  writeFileSync(
    resolve(root, "var/stormsense/browser-refresh.json"),
    JSON.stringify(result, null, 2) + "\n",
  );
  console.log(JSON.stringify(result));
} finally {
  await browser.close();
}
