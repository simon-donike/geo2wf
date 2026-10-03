import { expect, test } from "@playwright/test";
import type { Catalog } from "../src/types";

async function getCatalog(request: any): Promise<Catalog> {
  const pointer = await (await request.get("/data/latest.json")).json();
  return (await request.get("/data/" + pointer.manifest)).json();
}

test("live overview GeoColor crops are georeferenced, labeled and configurable", async ({
  page,
  request,
}, testInfo) => {
  test.setTimeout(90000);
  const catalog = await getCatalog(request);
  const storm = catalog.storms.find(
    (s) => s.active && s.latest_prediction?.metrics,
  )!;
  await page.goto(`/`);
  const stormImages = page.locator(
    `img.satellite-crop[data-storm-id="${storm.id}"]`,
  );
  const image = stormImages.first();
  await expect(image).toBeVisible({ timeout: 45000 });
  await expect
    .poll(
      () =>
        image.evaluate(
          (e) =>
            (e as HTMLImageElement).complete &&
            (e as HTMLImageElement).naturalWidth > 0,
        ),
      { timeout: 45000 },
    )
    .toBe(true);
  await expect
    .poll(() => image.evaluate((e) => getComputedStyle(e).opacity))
    .toBe("1");
  const imageTime = (await image.getAttribute("data-image-time"))!;
  await expect(page.locator(".satellite-status time").first()).toHaveAttribute(
    "datetime",
    imageTime,
  );
  const src = new URL((await image.getAttribute("src"))!);
  expect(src.searchParams.get("SRS")).toBe("EPSG:3857");
  expect(src.searchParams.get("TIME")).toBe(imageTime);
  const pixels = await image.evaluate((element) => {
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 32;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(element as HTMLImageElement, 0, 0, 32, 32);
    const data = ctx.getImageData(0, 0, 32, 32).data;
    return {
      opaque: Array.from(data).filter((value, i) => i % 4 === 3 && value > 0)
        .length,
      colors: new Set(Array.from(data).filter((_, i) => i % 4 !== 3)).size,
    };
  });
  expect(pixels.opaque).toBeGreaterThan(100);
  expect(pixels.colors).toBeGreaterThan(30);
  // Initial track acquisition fits the overview once; focus after it settles.
  await expect(page.locator(".storm-track").first()).toBeAttached();
  await page.getByRole("button", { name: `Focus ${storm.name} image` }).click();
  // A dateline crop has two fragments; compare the complete footprint.
  await expect
    .poll(() =>
      stormImages.evaluateAll((images) =>
        images.reduce(
          (width, image) => width + image.getBoundingClientRect().width,
          0,
        ),
      ),
    )
    .toBeGreaterThan(150);
  await page
    .getByRole("slider", { name: "Satellite imagery opacity" })
    .fill("0.5");
  await expect(page.locator(".leaflet-satellite-pane")).toHaveCSS(
    "opacity",
    "0.5",
  );
  await expect(page.locator(".wind-radius").first()).toBeAttached();
  await page
    .getByRole("slider", { name: "Satellite imagery opacity" })
    .fill("0.85");
  await page
    .locator(".map-shell")
    .screenshot({ path: `test-results/geocolor-${testInfo.project.name}.png` });
  await page.getByRole("checkbox", { name: "Latest GeoColor" }).uncheck();
  await expect(page.locator("img.satellite-crop")).toHaveCount(0);
  await page.reload();
  await expect(
    page.getByRole("checkbox", { name: "Latest GeoColor" }),
  ).not.toBeChecked();
  const historical = catalog.storms.find((s) => !s.active)!;
  await page.goto(`/storms/${historical.id}`);
  await expect(
    page.getByRole("heading", { name: new RegExp(historical.name) }),
  ).toBeVisible();
  await expect(
    page.getByRole("checkbox", { name: "Latest GeoColor" }),
  ).toHaveCount(0);
});

test("empty newly advertised crops fall back through three reported times", async ({
  page,
  request,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const catalog = await getCatalog(request);
  const storm = catalog.storms.find((s) => s.active)!;
  const end = Math.floor(Date.now() / 600000) * 600000 - 600000;
  const times = [0, 1, 2].map((i) =>
    new Date(end - i * 600000).toISOString().replace(".000Z", "Z"),
  );
  const requested = new Set<string>();
  await page.route("https://gibs.earthdata.nasa.gov/**", (route) => {
    const url = new URL(route.request().url());
    if (url.pathname.endsWith(".xml"))
      return route.fulfill({
        contentType: "application/xml",
        body: `<Domains><DimensionDomain><Domain>${times.join(",")}</Domain></DimensionDomain></Domains>`,
      });
    requested.add(url.searchParams.get("TIME")!);
    return route.fulfill({
      contentType: "image/png",
      // Transparent no-data image, as occasionally served for a new GIBS frame.
      body: Buffer.from(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABpfZFQAAAAABJRU5ErkJggg==",
        "base64",
      ),
    });
  });
  await page.goto(`/`);
  await expect.poll(() => requested.size).toBe(3);
  expect([...requested]).toEqual(times);
  await expect(page.locator(".satellite-status")).toContainText(
    "Image unavailable",
  );
  await expect(page.locator("img.satellite-crop")).toHaveCount(0);
  await expect(page.locator(".storm-track").first()).toBeAttached();
  await expect(
    page.getByRole("button", { name: "Retry imagery" }),
  ).toBeEnabled();
  await page.locator(".leaflet-control-zoom-in").click();
  await expect(page.locator(".satellite-status")).toContainText(
    "Image unavailable",
  );
  expect(errors).toEqual([]);
});

test("imagery failure leaves tracks usable and can be retried", async ({
  page,
  request,
}) => {
  const catalog = await getCatalog(request);
  const storm = catalog.storms.find((s) => s.active)!;
  await page.route("https://gibs.earthdata.nasa.gov/**", (route) =>
    route.fulfill({ status: 503, body: "Satellite unavailable" }),
  );
  await page.goto(`/`);
  await expect(page.locator(".satellite-status")).toContainText(
    "Imagery unavailable",
  );
  await expect(page.locator(".storm-track").first()).toBeAttached();
  await expect(
    page.getByRole("link", { name: "Archive", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Retry imagery" }),
  ).toBeEnabled();
  await page.getByRole("checkbox", { name: "Latest GeoColor" }).uncheck();
  await expect(page.locator(".satellite-status")).toContainText(
    "Satellite imagery is off",
  );
});
