import { test, expect } from "@playwright/test";

test("provider boundaries and source key appear on overview and storm maps", async ({ page }, testInfo) => {
  for (const route of ["/", "/storms/EP152026"]) {
    await page.goto(route);
    const key = page.locator(".provider-coverage-key");
    await expect(key).toHaveText("NHC/CPHC coverage outline");
    await expect(key).toHaveAttribute("href", /2023_nhop\.pdf#page=192$/);
    const pane = page.locator(".leaflet-coverage-pane");
    await expect(pane).toHaveCSS("pointer-events", "none");
    await expect(pane).toHaveCSS("z-index", "350");
    await expect(pane.locator(".provider-coverage-boundary")).toHaveCount(3);
    await expect(pane.locator('[data-boundary="Central Pacific western limit · 180°"]')).toHaveAttribute("stroke-dasharray", "5 7");
    await expect(pane.locator('[data-boundary*="division"]')).toHaveCount(0);
    await expect(page.locator(".leaflet-container")).toBeVisible();
    if (route === "/") {
      const zoomOut = page.locator(".leaflet-control-zoom-out");
      while ((await zoomOut.getAttribute("aria-disabled")) !== "true") {
        await zoomOut.click();
        await page.waitForTimeout(300);
      }
      await page.locator(".leaflet-container").screenshot({ path: testInfo.outputPath("coverage.png") });
    }
  }
});
