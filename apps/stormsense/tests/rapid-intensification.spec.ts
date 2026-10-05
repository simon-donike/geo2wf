import { test, expect, type APIRequestContext } from "@playwright/test";
import type { Catalog, Series } from "../src/types";

async function polo(request: APIRequestContext) {
  const pointer = await (await request.get("/data/latest.json")).json();
  const catalog: Catalog = await (
    await request.get("/data/" + pointer.manifest)
  ).json();
  const storm = catalog.storms.find((s) => s.id === "EP172026")!;
  expect(storm).toBeTruthy();
  const series: Series = await (
    await request.get("/data/" + storm.series)
  ).json();
  return { storm, series };
}

test("real Polo RI periods match on both charts and survive smoothing, units and scrubbing", async ({
  page,
  request,
}, testInfo) => {
  const { storm } = await polo(request);
  await page.goto(`/storms/${storm.id}`);
  const official = page.locator('.ri-band[data-source="official"]');
  await expect(official).toHaveCount(2);
  // From Polo's recorded 24-hour wind increases: six overlapping qualifying
  // windows, with the largest increase 85 kt. The same interval appears twice.
  for (const band of await official.all()) {
    await expect(band).toHaveAttribute(
      "data-start",
      "2026-09-20T18:00:00.000Z",
    );
    await expect(band).toHaveAttribute("data-end", "2026-09-23T00:00:00.000Z");
    await expect(band.locator("title")).toContainText("+85.0 kt");
  }
  await expect(page.locator('.ri-band[data-source="model"]')).toHaveCount(0);
  const periods = () =>
    page
      .locator(".ri-band")
      .evaluateAll((bands) =>
        bands.map((band) => [
          band.getAttribute("data-source"),
          band.getAttribute("data-start"),
          band.getAttribute("data-end"),
        ]),
      );
  const before = await periods();
  await page
    .getByRole("checkbox", { name: /^(Light smoothing|Smooth estimates)/ })
    .uncheck();
  await page
    .getByRole("button", { name: "Wind unit: kt. Switch units." })
    .click();
  const slider = page.getByRole("slider", { name: "Storm timeline" });
  await expect(slider).toHaveCount(1);
  await slider.focus();
  await page.keyboard.press("Home");
  expect(await periods()).toEqual(before);
  const details = page.locator(".chart-panel .ri-key summary");
  await details.focus();
  await page.keyboard.press("Enter");
  await expect(page.locator(".chart-panel .ri-key ul")).toBeVisible();
  await expect(page.locator(".chart-panel .ri-key ul")).toContainText(
    "NHC/CPHC reference",
  );
  await page.keyboard.press("Enter");
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.locator(".chart-panel").screenshot({
    path: `test-results/rapid-intensification-${testInfo.project.name}.png`,
  });
});

test("steady official winds stay unshaded despite model intensification and forecasts", async ({
  page,
  request,
}) => {
  const { storm, series } = await polo(request);
  for (const fix of series.track) fix.wind_ms = 30;
  // Keep the real model RI signal: it must not influence shading.
  for (const forecast of series.forecasts) {
    forecast.input_vmax_ms = [30, 30, 30];
    forecast.predictions.forEach(
      (prediction) =>
        (prediction.vmax_ms = prediction.lead_hours === 6 ? 60 : 90),
    );
  }
  await page.route("**/data/" + storm.series, (route) =>
    route.fulfill({ json: series }),
  );
  const issue = series.forecasts[0];
  await page.goto(
    `/storms/${storm.id}?time=${encodeURIComponent(issue.anchor_time)}&issue=${encodeURIComponent(issue.anchor_time)}`,
  );
  await expect(
    page.getByRole("img", { name: /Maximum sustained wind/ }),
  ).toBeVisible();
  await expect(page.locator(".chart-panel svg")).toBeVisible();
  await expect(
    page.getByText("Experimental forecast", { exact: true }),
  ).toHaveCount(0);
  await expect(page.locator(".ri-band")).toHaveCount(0);
  await expect(page.locator(".ri-key")).toHaveCount(0);
});
