import { expect, test } from "@playwright/test";
import { writeFile } from "node:fs/promises";
import { preferredRecords, stamp } from "../src/data";
import {
  officialCategory,
  RADII,
  stormAge,
  windCategory,
} from "../src/presentation";
import type { Catalog, Series } from "../src/types";

async function catalogFrom(request: any): Promise<Catalog> {
  const pointer = await (await request.get("/data/latest.json")).json();
  return (await request.get("/data/" + pointer.manifest)).json();
}

test("active map shows tracks and scaled radii, with categories and storm age", async ({
  page,
  request,
}) => {
  const catalog = await catalogFrom(request);
  const active = catalog.storms.filter((s) => s.active);
  const now = Date.parse(catalog.generated_at);
  await page.clock.install({ time: new Date(now) });
  await page.goto("/");
  await expect(page.locator(".storm-track").first()).toBeAttached();
  await expect(page.locator(".latest-storm-position")).toHaveCount(
    active.length,
  );
  const expectedRings = active.reduce(
    (count, storm) =>
      count +
      RADII.filter((r) => (storm.latest_prediction?.metrics?.[r.key] || 0) > 0)
        .length,
    0,
  );
  await expect(page.locator(".wind-radius")).toHaveCount(expectedRings);
  for (const storm of active) {
    const card = page.locator(`.storm-card[href="/storms/${storm.id}"]`);
    await expect(card.locator(".classification")).toHaveText(
      officialCategory(storm.latest_fix),
    );
    await expect(card.locator(".storm-age")).toContainText(
      stormAge(storm.start, now),
    );
    if (storm.metrics)
      await expect(card.locator(".wind-category")).toContainText(
        windCategory(storm.metrics.vmax_ms),
      );
  }
});

test("rings follow selected raw radii and geographic zoom; thresholds and smoothing switch correctly", async ({
  page,
  request,
}, testInfo) => {
  const catalog = await catalogFrom(request);
  const storm = catalog.storms.find(
    (s) => s.peak_category === 5 && s.prediction_count > 24,
  )!;
  const series: Series = await (
    await request.get("/data/" + storm.series)
  ).json();
  const record = preferredRecords(series.records).find(
    (r) => r.metrics && r.center,
  )!;
  await page.goto(
    `/storms/${storm.id}?time=${encodeURIComponent(record.time)}`,
  );
  for (const metric of RADII) {
    await expect(page.locator(`.radius-${metric.key}`)).toHaveAttribute(
      "data-radius-km",
      String(record.metrics![metric.key]),
    );
  }
  // Center the selected hour before measuring zoom. With the full track in
  // view, zooming its midpoint can legitimately clip an early storm position.
  await page.getByRole("button", { name: "Focus image", exact: true }).click();
  const ring = page.locator(".radius-r34_km");
  const before = await ring.evaluate(
    (element) => (element as SVGGraphicsElement).getBBox().width,
  );
  await page.locator(".leaflet-control-zoom-in").click();
  await expect
    .poll(async () =>
      ring.evaluate(
        (element) => (element as SVGGraphicsElement).getBBox().width,
      ),
    )
    .toBeGreaterThan(before * 1.7);
  const categoryLabels = page.locator(".chart-panel svg");
  await expect(
    categoryLabels.getByText("Cat 1 · 64 kt", { exact: true }),
  ).toBeVisible();
  await expect(
    categoryLabels.getByText("Cat 5 · 137 kt", { exact: true }),
  ).toBeVisible();
  for (const label of await categoryLabels
    .locator(".category-axis-label")
    .all()) {
    await expect(label).toHaveAttribute("text-anchor", "start");
    expect(Number(await label.getAttribute("x"))).toBeLessThan(100);
  }
  const modelPath = page.locator(".chart-panel svg path[stroke='#40c7bd']");
  const smoothed = await modelPath.getAttribute("d");
  const metrics = await page.locator(".metric-strip").allTextContents();
  const smoothing = page.getByRole("checkbox", {
    name: "Smooth estimates",
  });
  await smoothing.uncheck();
  await expect(modelPath).not.toHaveAttribute("d", smoothed!);
  expect(await page.locator(".metric-strip").allTextContents()).toEqual(
    metrics,
  );
  await expect(ring).toHaveAttribute(
    "data-radius-km",
    String(record.metrics!.r34_km),
  );
  await page
    .getByRole("button", { name: "Wind unit: kt. Switch units." })
    .click();
  await expect(
    categoryLabels.getByText("Cat 1 · 32.9 m/s", { exact: true }),
  ).toBeVisible();
  await expect(
    categoryLabels.getByText("Cat 5 · 70.5 m/s", { exact: true }),
  ).toBeVisible();
  await page.reload();
  await expect(smoothing).not.toBeChecked();
  const downloadEvent = page.waitForEvent("download");
  await page.getByRole("button", { name: "JSON", exact: true }).click();
  const download = await downloadEvent;
  const stream = await download.createReadStream();
  let content = "";
  for await (const chunk of stream!) content += chunk.toString();
  expect(JSON.parse(content).records).toEqual(series.records);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  expect(
    await page
      .locator(".track-context")
      .evaluate(
        (element) =>
          element.getBoundingClientRect().bottom <=
          element.closest(".detail-map-shell")!.getBoundingClientRect().bottom,
      ),
  ).toBe(true);
  await page.screenshot({
    path: `test-results/radii-categories-${testInfo.project.name}.png`,
    fullPage: true,
  });
});

test("rapid scrubbing keeps tracks and slider position stable, commits one shareable time on release", async ({
  page,
  request,
}, testInfo) => {
  const catalog = await catalogFrom(request);
  const storm = [...catalog.storms].sort(
    (a, b) => b.prediction_count - a.prediction_count,
  )[0];
  const series: Series = await (
    await request.get("/data/" + storm.series)
  ).json();
  const records = preferredRecords(series.records);
  await page.goto(`/storms/${storm.id}`);
  const slider = page.getByRole("slider", { name: "Storm timeline" });
  await expect(slider).toHaveValue(String(records.length - 1));
  await expect(page.locator(".storm-track").first()).toBeAttached();
  await slider.scrollIntoViewIfNeeded();
  const beforeURL = page.url();
  const measurement = await slider.evaluate(
    async (element, selectedTimes) => {
      const input = element as HTMLInputElement;
      const track = document.querySelector(".storm-track");
      const initialY = input.getBoundingClientRect().top;
      const setter = Object.getOwnPropertyDescriptor(
        HTMLInputElement.prototype,
        "value",
      )!.set!;
      const samples: number[] = [];
      let positionShift = 0;
      let synchronized = true;
      for (const { index, label } of selectedTimes) {
        const started = performance.now();
        setter.call(input, String(index));
        input.dispatchEvent(new Event("input", { bubbles: true }));
        await new Promise<void>((resolve) =>
          requestAnimationFrame(() => resolve()),
        );
        samples.push(performance.now() - started);
        synchronized &&=
          document.querySelector(".timeline-panel strong")?.textContent ===
          label;
        positionShift = Math.max(
          positionShift,
          Math.abs(input.getBoundingClientRect().top - initialY),
        );
      }
      return {
        samples,
        synchronized,
        positionShift,
        sameTrack: track === document.querySelector(".storm-track"),
      };
    },
    Array.from({ length: 50 }, (_, i) => {
      const index = Math.round((i * (records.length - 1)) / 60);
      return { index, label: stamp(records[index].time) };
    }),
  );
  expect(measurement.synchronized).toBe(true);
  expect(measurement.sameTrack).toBe(true);
  expect(measurement.positionShift).toBeLessThan(2);
  expect(page.url()).toBe(beforeURL);
  const sorted = [...measurement.samples].sort((a, b) => a - b);
  const p95 = sorted[Math.floor(sorted.length * 0.95)];
  expect(
    p95,
    "95% of input-to-next-frame updates stay below 100 ms",
  ).toBeLessThan(100);
  const reportPath = testInfo.outputPath("timeline-responsiveness.json");
  await writeFile(
    reportPath,
    JSON.stringify(
      {
        storm_id: storm.id,
        records: records.length,
        project: testInfo.project.name,
        p95_ms: p95,
        max_ms: Math.max(...sorted),
        ...measurement,
      },
      null,
      2,
    ),
  );
  await testInfo.attach("timeline-responsiveness.json", {
    path: reportPath,
    contentType: "application/json",
  });
  await slider.dispatchEvent("pointerup");
  const selection = await slider.inputValue();
  await expect
    .poll(() => new URL(page.url()).searchParams.get("time"))
    .toBe(records[Number(selection)].time);
  await page.reload();
  await expect(slider).toHaveValue(selection);
  // Exercise native dragging as well as repeated input events, including release outside the control.
  await slider.scrollIntoViewIfNeeded();
  const bounds = (await slider.boundingBox())!;
  await page.mouse.move(
    bounds.x + bounds.width * 0.75,
    bounds.y + bounds.height / 2,
  );
  await page.mouse.down();
  await page.mouse.move(
    bounds.x + bounds.width * 0.25,
    bounds.y + bounds.height / 2,
    { steps: 20 },
  );
  await page.mouse.move(bounds.x + bounds.width * 0.25, bounds.y - 40);
  await page.mouse.up();
  const dragged = await slider.inputValue();
  expect(dragged).not.toBe(selection);
  await expect
    .poll(() => new URL(page.url()).searchParams.get("time"))
    .toBe(records[Number(dragged)].time);
});
