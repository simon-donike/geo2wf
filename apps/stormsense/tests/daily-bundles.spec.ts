import { test, expect, type APIRequestContext } from "@playwright/test";
import type { Catalog, Series } from "../src/types";
import { wind } from "../src/data";
import { writeFile } from "node:fs/promises";

async function data(request: APIRequestContext) {
  const pointer = await (await request.get("/data/latest.json")).json();
  const catalog: Catalog = await (
    await request.get("/data/" + pointer.manifest)
  ).json();
  return catalog;
}

test("daily packs replace individual requests and retain correct imagery between image slots", async ({
  page,
  request,
}, testInfo) => {
  const catalog = await data(request);
  const storm = catalog.storms.find((s) => s.id === "EP152026")!;
  const series: Series = await (
    await request.get("/data/" + storm.series)
  ).json();
  const frames = series.imagery!.filter((f) => f.status === "ready");
  expect(frames.every((f) => new Date(f.time).getUTCHours() % 2 === 0)).toBe(
    true,
  );
  expect(series.imagery_bundles!.length).toBeGreaterThan(0);
  const downloads: string[] = [];
  page.on("request", (request) => {
    const path = new URL(request.url()).pathname;
    if (path.endsWith(".zip") || path.endsWith(".webp")) downloads.push(path);
  });
  const start = Date.now();
  await page.goto(`/storms/${storm.id}`);
  await expect(page.locator(".hourly-imagery")).toHaveAttribute(
    "data-preload-ready",
    String(frames.length),
    { timeout: 60000 },
  );
  const elapsed = Date.now() - start;
  expect(downloads.filter((path) => path.endsWith(".webp"))).toEqual([]);
  expect(downloads).toHaveLength(series.imagery_bundles!.length);
  expect(new Set(downloads).size).toBe(downloads.length);
  const frame = frames.at(-10)!;
  const next = new Date(Date.parse(frame.time) + 3600000).toISOString();
  await page.goto(`/storms/${storm.id}?time=${encodeURIComponent(next)}`);
  await expect(page.locator(".hourly-satellite-crop").first()).toHaveAttribute(
    "data-slot-time",
    frame.time,
  );
  await expect(page.locator(".hourly-imagery time")).toHaveAttribute(
    "datetime",
    frame.acquired_at!,
  );
  await writeFile(
    testInfo.outputPath("daily-bundles.json"),
    JSON.stringify(
      {
        storm_id: storm.id,
        project: testInfo.project.name,
        frames: frames.length,
        bundle_requests: series.imagery_bundles!.length,
        individual_image_requests: 0,
        bundle_bytes: series.imagery_bundles!.reduce((n, b) => n + b.bytes, 0),
        preload_elapsed_ms: elapsed,
      },
      null,
      2,
    ),
  );
});

test("archive displays official lifetime peak wind, category and RI on desktop and mobile", async ({
  page,
  request,
}, testInfo) => {
  const catalog = await data(request);
  await page.goto("/archive");
  for (const storm of catalog.storms) {
    const row = page.locator(`a.archive-row[href='/storms/${storm.id}']`);
    await expect(row.locator(".archive-peak-wind")).toContainText(
      wind(storm.peak_official_wind_ms, "kt"),
    );
    await expect(row.locator(".archive-ri")).toHaveText(
      storm.has_ri == null ? "—" : storm.has_ri ? "Yes" : "No",
    );
    if (storm.peak_category! > 0)
      await expect(row.locator(".archive-peak-category")).toHaveText(
        `Category ${storm.peak_category}`,
      );
  }
  const polo = catalog.storms.find((s) => s.id === "EP172026")!;
  expect(polo.has_ri).toBe(true);
  expect(polo.peak_category).toBe(5);
  await page
    .getByRole("button", { name: "Wind unit: kt. Switch units." })
    .click();
  await expect(
    page.locator(`a.archive-row[href='/storms/${polo.id}'] .archive-peak-wind`),
  ).toContainText(wind(polo.peak_official_wind_ms, "m/s"));
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.locator(".archive-table").focus();
  if (testInfo.project.name === "mobile") {
    await page.keyboard.press("End");
    await page
      .locator(".archive-table")
      .evaluate((el) => (el.scrollLeft = el.scrollWidth));
    await expect(
      page
        .locator(".table-heading")
        .getByText("Rapid intensification", { exact: true }),
    ).toBeVisible();
  }
  await page.screenshot({
    path: `test-results/archive-columns-${testInfo.project.name}.png`,
    fullPage: true,
  });
});

test("map wheel zoom changes geographic scale without changing the map styling", async ({
  page,
}) => {
  await page.goto("/storms/EP172026?time=2026-09-21T12%3A00%3A00Z");
  await page.getByRole("button", { name: "Focus image", exact: true }).click();
  const ring = page.locator(".radius-r34_km");
  const width = await ring.evaluate(
    (el) => (el as SVGGraphicsElement).getBBox().width,
  );
  const map = page.getByLabel("Storm track map", { exact: true });
  const color = await map.evaluate(
    (el) => getComputedStyle(el).backgroundColor,
  );
  const bounds = (await map.boundingBox())!;
  await page.mouse.move(
    bounds.x + bounds.width / 2,
    bounds.y + bounds.height / 2,
  );
  await page.mouse.wheel(0, -120);
  await expect
    .poll(() =>
      ring.evaluate((el) => (el as SVGGraphicsElement).getBBox().width),
    )
    .toBeGreaterThan(width * 1.7);
  expect(await map.evaluate((el) => getComputedStyle(el).backgroundColor)).toBe(
    color,
  );
  await expect(page.locator(".storm-track").first()).toBeAttached();
});
