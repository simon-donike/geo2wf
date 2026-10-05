import { expect, test, type APIRequestContext } from "@playwright/test";
import type { Catalog, Series } from "../src/types";
import { preferredRecords } from "../src/data";
import { writeFile } from "node:fs/promises";
const imageSlot = (time: string) =>
  new Date(Math.floor(Date.parse(time) / 7200000) * 7200000)
    .toISOString()
    .replace(".000Z", "Z");

async function archive(request: APIRequestContext) {
  const pointer = await (await request.get("/data/latest.json")).json();
  const catalog: Catalog = await (
    await request.get("/data/" + pointer.manifest)
  ).json();
  const storm = catalog.storms.find((s) => s.id === "EP152026")!;
  const series: Series = await (
    await request.get("/data/" + storm.series)
  ).json();
  const records = preferredRecords(series.records);
  const frames = series.imagery!.filter(
    (f) => f.status === "ready" && records.some((r) => r.time === f.time),
  );
  return { storm, series, records, frames };
}

test("archived WebP images match the selected hour and expose portable georeferencing", async ({
  page,
  request,
}, testInfo) => {
  const { storm, frames } = await archive(request);
  const frame = frames.at(-12)!;
  await page.goto(`/storms/${storm.id}?time=${encodeURIComponent(frame.time)}`);
  const image = page.locator(".hourly-satellite-crop").first();
  await expect(image).toHaveAttribute("data-slot-time", frame.time);
  await expect(image).toHaveAttribute("data-resolution", "full");
  await expect(
    page.getByRole("slider", { name: "Image timeline", exact: true }),
  ).toHaveCount(0);
  await expect(
    page.getByRole("slider", { name: "Storm timeline", exact: true }),
  ).toHaveCount(1);
  await expect(page.locator(".map-time-controls")).toHaveCount(0);
  await expect(image).toHaveAttribute("data-image-time", frame.acquired_at!);
  await expect(page.locator(".hourly-imagery time")).toHaveAttribute(
    "datetime",
    frame.acquired_at!,
  );
  await page.getByRole("button", { name: "Focus image", exact: true }).click();
  const item = await (await request.get("/data/" + frame.metadata)).json();
  expect(item.stac_version).toBe("1.0.0");
  expect(item.properties["stormsense:slot_time"]).toBe(frame.time);
  const asset = item.assets["visual-0"];
  expect(asset["proj:epsg"]).toBe(3857);
  expect(asset["proj:transform"]).toHaveLength(9);
  const sidecar = await request.get("/data/" + frame.parts[0].sidecar);
  expect(await sidecar.text()).toContain("GeoTransform");
  const screenshot = `test-results/hourly-image-${testInfo.project.name}.png`;
  await page.locator(".detail-map-shell").screenshot({ path: screenshot });
  await page
    .getByRole("checkbox", { name: "Satellite imagery", exact: true })
    .uncheck();
  await expect(page.locator(".hourly-satellite-crop")).toHaveCount(0);
  await page.reload();
  await expect(
    page.getByRole("checkbox", { name: "Satellite imagery", exact: true }),
  ).not.toBeChecked();
});

test("playback buffers images, synchronizes time, and pauses for manual scrubbing", async ({
  page,
  request,
}) => {
  const { storm, frames, records } = await archive(request);
  const frame = frames.at(-20)!;
  await page.goto(`/storms/${storm.id}?time=${encodeURIComponent(frame.time)}`);
  await expect(page.locator(".hourly-satellite-crop").first()).toHaveAttribute(
    "data-slot-time",
    frame.time,
  );
  const slider = page.getByRole("slider", {
    name: "Storm timeline",
    exact: true,
  });
  const initial = Number(await slider.inputValue());
  await page
    .getByRole("combobox", { name: "Timeline playback speed" })
    .selectOption("8");
  await page
    .getByRole("button", { name: "Play timeline", exact: true })
    .click();
  await expect
    .poll(async () => Number(await slider.inputValue()), { timeout: 15000 })
    .toBeGreaterThan(initial + 3);
  await page
    .getByRole("button", { name: "Pause timeline", exact: true })
    .click();
  const current = Number(await slider.inputValue());
  await expect(page.locator(".hourly-satellite-crop").first()).toHaveAttribute(
    "data-slot-time",
    imageSlot(records[current].time),
  );
  await expect
    .poll(() => new URL(page.url()).searchParams.get("time"))
    .toBe(records[current].time);
  expect(
    Number(
      await page.locator(".hourly-imagery").getAttribute("data-cache-bytes"),
    ),
  ).toBeLessThan(48 * 1024 * 1024);
  await page
    .getByRole("button", { name: "Play timeline", exact: true })
    .click();
  await slider.focus();
  await page.keyboard.press("ArrowLeft");
  await expect(
    page.getByRole("button", { name: "Play timeline", exact: true }),
  ).toBeVisible();
  const selected = Number(await slider.inputValue());
  await expect(page.locator(".hourly-satellite-crop").first()).toHaveAttribute(
    "data-slot-time",
    imageSlot(records[selected].time),
  );
});

test("a failed or unavailable image never leaves another hour's picture on the map", async ({
  page,
  request,
}) => {
  const { storm, series, frames, records } = await archive(request);
  const previous = frames.at(-12)!,
    target = frames.at(-11)!;
  // Deterministic provider gap while preserving the real numerical estimates.
  const modified = {
    ...series,
    imagery: series.imagery!.map((f) =>
      f.time === target.time
        ? { ...f, status: "gap", reason: "empty_provider_image", parts: [] }
        : f,
    ),
  };
  await page.route("**/data/" + storm.series, (route) =>
    route.fulfill({ json: modified }),
  );
  await page.goto(
    `/storms/${storm.id}?time=${encodeURIComponent(previous.time)}`,
  );
  await expect(page.locator(".hourly-satellite-crop").first()).toHaveAttribute(
    "data-slot-time",
    previous.time,
  );
  const slider = page.getByRole("slider", {
    name: "Storm timeline",
    exact: true,
  });
  await slider.fill(String(records.findIndex((r) => r.time === target.time)));
  await expect(page.locator(".hourly-satellite-crop")).toHaveCount(0);
  await expect(page.locator(".hourly-imagery .satellite-status")).toContainText(
    "Image unavailable for this hour",
  );
  await expect(page.locator(".selected-storm-position")).toBeAttached();
  await expect(slider).toBeEnabled();
});

test("image download failure can be retried without disabling the timeline", async ({
  page,
  request,
}) => {
  const { storm, frames } = await archive(request);
  const frame = frames.at(-15)!;
  await page.route("**/data/bundles/*.zip", (route) =>
    route.fulfill({ status: 503, body: "Unavailable" }),
  );
  await page.goto(`/storms/${storm.id}?time=${encodeURIComponent(frame.time)}`);
  await expect(
    page.getByRole("button", { name: "Retry image", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("slider", { name: "Storm timeline", exact: true }),
  ).toBeEnabled();
  await page.unroute("**/data/bundles/*.zip");
  await page.getByRole("button", { name: "Retry image", exact: true }).click();
  await expect(page.locator(".hourly-satellite-crop").first()).toHaveAttribute(
    "data-slot-time",
    frame.time,
  );
});

test("all storm images preload and distant slider jumps work without a network", async ({
  page,
  request,
}, testInfo) => {
  test.setTimeout(90000);
  const { storm, frames, records } = await archive(request);
  await page.goto(`/storms/${storm.id}`);
  const imagery = page.locator(".hourly-imagery");
  await expect(imagery).toHaveAttribute(
    "data-preload-ready",
    String(frames.length),
    { timeout: 70000 },
  );
  await expect(imagery).toContainText(
    `All ${frames.length.toLocaleString()} images loaded`,
  );
  // A fresh image at a far-away hour must come from retained bytes, even when
  // the network disappears. This catches a prefetch that only warmed HTTP cache.
  await page.context().setOffline(true);
  const samples = await page
    .getByRole("slider", { name: "Storm timeline", exact: true })
    .evaluate(
      async (element, jumps) => {
        const input = element as HTMLInputElement;
        const setter = Object.getOwnPropertyDescriptor(
          HTMLInputElement.prototype,
          "value",
        )!.set!;
        const durations: number[] = [];
        for (const jump of jumps) {
          const start = performance.now();
          setter.call(input, String(jump.index));
          input.dispatchEvent(new Event("input", { bubbles: true }));
          await new Promise<void>((resolve, reject) => {
            const check = () => {
              const image = document.querySelector(".hourly-satellite-crop");
              if (
                image?.getAttribute("data-slot-time") === jump.time &&
                image.getAttribute("data-resolution") === "full"
              )
                resolve();
              else if (performance.now() - start > 4000)
                reject(Error("Preloaded image did not display offline"));
              else requestAnimationFrame(check);
            };
            requestAnimationFrame(check);
          });
          durations.push(performance.now() - start);
        }
        return durations;
      },
      Array.from({ length: 12 }, (_, i) => {
        const frame = frames[Math.floor((((i * 7) % 12) / 12) * frames.length)];
        return {
          time: frame.time,
          index: records.findIndex((r) => r.time === frame.time),
        };
      }),
    );
  const sorted = [...samples].sort((a, b) => a - b);
  const p95 = sorted[Math.floor(sorted.length * 0.95)];
  expect(
    p95,
    "Preloaded full images follow distant scrubbing within 150 ms",
  ).toBeLessThan(150);
  expect(
    Number(await imagery.getAttribute("data-cache-bytes")),
  ).toBeLessThanOrEqual(48 * 1024 * 1024);
  await writeFile(
    testInfo.outputPath("preloaded-image-scrubbing.json"),
    JSON.stringify(
      {
        storm_id: storm.id,
        images: frames.length,
        project: testInfo.project.name,
        offline: true,
        p95_ms: p95,
        samples,
      },
      null,
      2,
    ),
  );
});
