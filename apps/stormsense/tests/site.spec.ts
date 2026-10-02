import { test, expect } from "@playwright/test";
import type { Catalog, Series } from "../src/types";
async function data(request: any) {
  const pointer = await (await request.get("/data/latest.json")).json();
  const catalog: Catalog = await (
    await request.get("/data/" + pointer.manifest)
  ).json();
  return { pointer, catalog };
}
test("real active overview, basin filtering, units, keyboard and responsive layout", async ({
  page,
  request,
}, testInfo) => {
  const { catalog } = await data(request);
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "A clearer view of the storm." }),
  ).toBeVisible();
  await expect(page.locator(".storm-card")).toHaveCount(
    catalog.storms.filter((s) => s.active).length,
  );
  await page
    .getByRole("button", { name: "Central Pacific", exact: true })
    .click();
  await expect(page.locator(".storm-card")).toHaveCount(
    catalog.storms.filter((s) => s.active && s.basin === "CP").length,
  );
  await page
    .getByRole("button", { name: "Wind unit: kt. Switch units." })
    .click();
  await expect(
    page.getByRole("button", { name: "Wind unit: m/s. Switch units." }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.goto("/");
  await page.keyboard.press("Tab");
  await expect(
    page.getByRole("link", { name: "Skip to content" }),
  ).toBeFocused();
  await page.locator("body").click({ position: { x: 5, y: 200 } });
  await page.screenshot({
    path: `test-results/overview-${testInfo.project.name}.png`,
    fullPage: true,
  });
});
test("real archive search and filters", async ({ page, request }) => {
  const { catalog } = await data(request);
  await page.goto("/archive");
  await page
    .getByRole("textbox", { name: "Search storms" })
    .fill(catalog.storms[0].id);
  await expect(page.locator("a.archive-row")).toHaveCount(1);
  await page
    .getByRole("textbox", { name: "Search storms" })
    .fill("no-such-storm");
  await expect(
    page.getByRole("heading", { name: "No matching storms" }),
  ).toBeVisible();
  await page.getByRole("textbox", { name: "Search storms" }).fill("");
  await page.getByLabel("Peak official category").selectOption("5");
  await expect(page.locator("a.archive-row")).toHaveCount(
    catalog.storms.filter((s) => s.peak_category === 5).length,
  );
});
test("real historical forecasts, timeline deep link and downloads", async ({
  page,
  request,
}, testInfo) => {
  const { catalog } = await data(request);
  let chosen: Series | undefined;
  for (const storm of catalog.storms.filter((s) => s.prediction_count > 12)) {
    const series: Series = await (
      await request.get("/data/" + storm.series)
    ).json();
    if (series.forecasts.length) {
      chosen = series;
      break;
    }
  }
  expect(
    chosen,
    "Export at least one real historical forecast before browser testing",
  ).toBeTruthy();
  const series = chosen!;
  const issue = series.forecasts[0];
  await page.goto(
    `/storms/${series.storm_id}?time=${encodeURIComponent(issue.anchor_time)}&issue=${encodeURIComponent(issue.anchor_time)}`,
  );
  await expect(page.getByText("+6 hours", { exact: true })).toBeVisible();
  await expect(page.getByText("+12 hours", { exact: true })).toBeVisible();
  await expect(page.getByLabel("Forecast issue")).toHaveValue(
    `${issue.kind}|${issue.anchor_time}`,
  );
  const timeline = page.getByRole("slider", { name: "Storm timeline" });
  const initial = await timeline.inputValue();
  await timeline.focus();
  await page.keyboard.press("ArrowLeft");
  await expect(timeline).not.toHaveValue(initial);
  expect(new URL(page.url()).searchParams.has("issue")).toBe(false);
  const link = page.url();
  const selection = await timeline.inputValue();
  await page.reload();
  await expect(timeline).toHaveValue(selection);
  expect(page.url()).toBe(link);
  const chartTime = page.locator(".chart-panel .panel-heading p");
  const selectedTime = await chartTime.innerText();
  await page.getByRole("img", { name: /Maximum sustained wind/ }).hover();
  await expect(chartTime).not.toHaveText(selectedTime);
  await page.mouse.move(0, 0);
  await expect(chartTime).toHaveText(selectedTime);
  const downloadEvent = page.waitForEvent("download");
  await page.getByRole("button", { name: "CSV", exact: true }).click();
  expect((await downloadEvent).suggestedFilename()).toBe(
    `${series.storm_id}.csv`,
  );
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: `test-results/detail-${testInfo.project.name}.png`,
    fullPage: true,
  });
});
test("real live forecast and selection survive a refreshed catalog", async ({
  page,
  request,
}) => {
  const { pointer, catalog } = await data(request);
  const storm = catalog.storms.find(
    (s) => s.active && s.latest_prediction?.kind === "live",
  );
  expect(
    storm,
    "Run and export a real live update before this check",
  ).toBeTruthy();
  const series: Series = await (
    await request.get("/data/" + storm!.series)
  ).json();
  const issue = series.forecasts.find(
    (f) =>
      f.kind === "live" && f.anchor_time === storm!.latest_prediction!.time,
  );
  expect(
    issue,
    "Compute the required StormSense history before issuing a live forecast",
  ).toBeTruthy();
  await page.clock.install();
  await page.goto(`/storms/${storm!.id}`);
  await expect(page.getByLabel("Forecast issue")).toHaveValue(
    `live|${issue!.anchor_time}`,
  );
  await expect(page.getByText("Live issue", { exact: false })).toBeVisible();
  const timeline = page.getByRole("slider", { name: "Storm timeline" });
  await timeline.focus();
  const latest = await timeline.inputValue();
  await page.keyboard.press("ArrowLeft");
  await page.clock.runFor(100);
  await expect(timeline).not.toHaveValue(latest);
  const selected = await timeline.inputValue();
  const link = page.url();
  await page.route("**/data/" + pointer.manifest, (route) =>
    route.fulfill({
      json: {
        ...catalog,
        storms: catalog.storms.map((s) =>
          s.id === storm!.id ? { ...s, name: `${s.name} refreshed` } : s,
        ),
      },
    }),
  );
  await page.clock.fastForward(61_000);
  await expect(
    page.getByRole("heading", { name: new RegExp(`${storm!.name} refreshed`) }),
  ).toBeVisible();
  await expect(timeline).toHaveValue(selected);
  expect(page.url()).toBe(link);
});
test("quiet period and source failure remain distinct", async ({
  page,
  request,
}) => {
  const { pointer, catalog } = await data(request);
  const quiet = {
    ...catalog,
    storms: catalog.storms.map((s) => ({ ...s, active: false })),
  };
  await page.route("**/data/" + pointer.manifest, (r) =>
    r.fulfill({ json: quiet }),
  );
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "A quieter ocean" }),
  ).toBeVisible();
  await page.unroute("**/data/" + pointer.manifest);
  await page.route("**/data/latest.json", (r) =>
    r.fulfill({ status: 503, json: { error: "offline" } }),
  );
  await page.reload();
  await expect(page.getByRole("alert")).toContainText("503");
  await expect(
    page.getByRole("heading", { name: "A quieter ocean" }),
  ).toHaveCount(0);
});

test("an advisory failure is not presented as a quiet ocean", async ({
  page,
  request,
}) => {
  const { pointer, catalog } = await data(request);
  const failed = {
    ...catalog,
    storms: [],
    source_status: {
      ...catalog.source_status,
      discovery: { error: "NHC feed unavailable", last_success: null },
    },
  };
  await page.route("**/data/" + pointer.manifest, (r) =>
    r.fulfill({ json: failed }),
  );
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Storm status unavailable" }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "A quieter ocean" }),
  ).toHaveCount(0);
  await page.unroute("**/data/" + pointer.manifest);
  await page.route("**/data/" + pointer.manifest, (r) =>
    r.fulfill({
      json: {
        ...failed,
        source_status: { ...failed.source_status, discovery: null },
      },
    }),
  );
  await page.reload();
  await expect(
    page.getByRole("heading", { name: "Storm status unavailable" }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "A quieter ocean" }),
  ).toHaveCount(0);
});

test("freshness keeps aging when repeated browser refreshes fail", async ({
  page,
  request,
}) => {
  const { catalog } = await data(request);
  const retrieved = catalog.source_status.discovery?.last_success;
  expect(retrieved).toBeTruthy();
  await page.clock.install({
    time: new Date(Date.parse(retrieved!) + 10 * 60000),
  });
  await page.goto("/");
  const freshness = page
    .locator(".freshness")
    .filter({ hasText: "NHC/CPHC feed" });
  await expect(freshness).toContainText("10m ago");
  await page.route("**/data/latest.json", (r) =>
    r.fulfill({ status: 503, json: { error: "offline" } }),
  );
  await page.clock.fastForward(61000);
  await expect(page.getByRole("alert")).toContainText("503");
  await expect(freshness).toContainText("11m ago");
  await page.clock.fastForward(61000);
  await expect(freshness).toContainText("12m ago");
});
