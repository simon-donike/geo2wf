import { readFileSync } from "node:fs";
import { test, expect } from "@playwright/test";
import { SHOWCASE_STEPS, showcaseUrl } from "../src/showcaseSteps";

test.beforeEach(async ({ page, baseURL }) => {
  const origin = new URL(baseURL!).origin;
  await page.route("**/*", (route) => {
    const url = new URL(route.request().url());
    return url.origin !== origin || url.pathname.startsWith("/data/")
      ? route.abort()
      : route.continue();
  });
});

async function ready(page: import("@playwright/test").Page) {
  await expect(page.locator(".showcase-controls")).toHaveAttribute("data-ready", "true", { timeout: 30000 });
}
async function chapter(page: import("@playwright/test").Page, id: string) {
  const destination = SHOWCASE_STEPS.findIndex(step => step.id === id);
  let current = SHOWCASE_STEPS.findIndex(step => step.id === new URL(page.url()).searchParams.get("tour"));
  while (current !== destination) {
    await ready(page);
    await page.getByRole("button", { name: current < destination ? "Next" : "Back", exact: true }).click();
    current += current < destination ? 1 : -1;
    await expect(page.locator(".showcase-controls")).toHaveAttribute("data-step", SHOWCASE_STEPS[current].id);
  }
  await ready(page);
}

test("live overview reports unavailability; saved stops work with archive blocked", async ({
  page,
}, info) => {
  const errors: string[] = [];
  const liveRequests: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("request", (request) => {
    if (new URL(request.url()).pathname.startsWith("/data/"))
      liveRequests.push(request.url());
  });
  if (info.project.name === "desktop")
    await page.setViewportSize({ width: 1366, height: 768 });
  await page.goto("/method");
  await page.getByRole("button", { name: "Start guided tour" }).click();
  await ready(page);
  const controls = page.getByRole("complementary", {
    name: "Showcase controls",
  });
  await expect(controls.locator("details")).toHaveCount(0);
  await expect(page.getByText("Rehearsal notes", { exact: true })).toHaveCount(0);
  await expect(page).toHaveURL(/\/\?showcase=v1&tour=live/);
  await expect(controls.getByRole("button")).toHaveCount(3);
  await expect(controls.getByRole("combobox")).toHaveCount(0);
  for (const [i, step] of SHOWCASE_STEPS.entries()) {
    await ready(page);
    await expect(
      controls.getByRole("heading", { name: step.title, exact: true }),
    ).toBeVisible();
    const target = page.locator(
      `[data-tour="${step.anchor}"][data-tour-highlight="true"]`,
    );
    await expect(target).toBeVisible();
    if (step.id === "reference") {
      await expect(target).toHaveAttribute("open", "");
      await expect(target.getByRole("heading", { name: "NHC/CPHC reference analysis" })).toBeVisible();
      await expect(target.getByRole("heading", { name: "StormSense inference" })).toBeVisible();
      await expect(target).toContainText("dropsondes");
      await expect(target).toContainText("No aircraft winds");
      await expect(target.locator(".reference-images img")).toHaveCount(4);
      await expect.poll(() => target.locator(".reference-images img").evaluateAll(
        images => images.every(image => (image as HTMLImageElement).complete && (image as HTMLImageElement).naturalWidth > 0),
      )).toBe(true);
    }
    await expect(page.locator(".showcase-highlight")).toHaveAttribute("data-moving", "false");
    if (step.view)
      await expect(
        page.getByRole("tab", { name: new RegExp(`^${step.view}$`, "i") }),
      ).toHaveAttribute("aria-selected", "true");
    const layout = await page.evaluate(() => {
      const content = document.getElementById("showcase-content")!.getBoundingClientRect();
      const card = document.querySelector(".showcase-controls")!.getBoundingClientRect();
      const frame = document.querySelector(".showcase-highlight")!.getBoundingClientRect();
      const clip = document.querySelector(".showcase-highlight-layer")!.getBoundingClientRect();
      const bottom = Math.min(frame.bottom, clip.bottom);
      return { floating: content.bottom > card.top, inView: card.top >= 0 && card.bottom <= innerHeight && card.left >= 0 && card.right <= innerWidth,
        obscures: frame.left < card.right && frame.right > card.left && frame.top < card.bottom && bottom > card.top,
        overflow: document.documentElement.scrollWidth > innerWidth };
    });
    expect(layout.floating).toBe(true);
    expect(layout.inView).toBe(true);
    expect(layout.obscures).toBe(false);
    expect(layout.overflow).toBe(false);
    if ([1, 4, 7].includes(i))
      await page.screenshot({
        path: `test-results/showcase-${step.id}-${info.project.name}.png`,
      });
    if (i < SHOWCASE_STEPS.length - 1)
      await controls.getByRole("button", { name: "Next", exact: true }).click();
  }
  await page
    .getByRole("button", { name: "Play timeline", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Pause timeline", exact: true }),
  ).toBeVisible();
  await controls.getByRole("button", { name: "Back", exact: true }).click();
  await expect(
    page.getByRole("tab", { name: "Pipeline", exact: true }),
  ).toHaveAttribute("aria-selected", "true");
  await controls.getByRole("button", { name: "Next", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Play timeline", exact: true }),
  ).toBeVisible();
  expect(liveRequests).toHaveLength(1);
  expect(liveRequests[0]).toContain("/data/latest.json");
  expect(errors).toEqual([]);
});

test("deep links, refresh, keyboard chapters and free exploration", async ({
  page,
}) => {
  await page.goto("/method?showcase=v1&tour=supervision");
  await ready(page);
  await expect(
    page.getByRole("tab", { name: "Training", exact: true }),
  ).toHaveAttribute("aria-selected", "true");
  await page.reload();
  await ready(page);
  await expect(
    page.getByRole("tab", { name: "Training", exact: true }),
  ).toHaveAttribute("aria-selected", "true");
  await expect(page.getByRole("complementary", { name: "Showcase controls" }).getByRole("button", { name: /pause|resume/i })).toHaveCount(0);
  const training = page.getByRole("tab", { name: "Training", exact: true });
  await training.focus();
  await page.keyboard.press("End");
  await expect(
    page.getByRole("tab", { name: "Pipeline", exact: true }),
  ).toBeFocused();
  await page.keyboard.press("ArrowRight");
  await expect(
    page.getByRole("tab", { name: "Inference", exact: true }),
  ).toBeFocused();
  await page
    .getByRole("button", { name: "Encoder-only scalars", exact: true })
    .click();
  await chapter(page, "outputs");
  await page.getByRole("button", { name: "Next", exact: true }).click();
  await expect(training).toHaveAttribute("aria-selected", "true");
  await expect(
    page.getByRole("button", { name: "Joint field + scalars", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await chapter(page, "storm");
  const slider = page.getByRole("slider", {
    name: "Storm timeline",
    exact: true,
  });
  await expect(slider).toHaveValue("201");
  await slider.focus();
  await page.keyboard.press("ArrowRight");
  await expect(slider).toHaveValue("202");
  await page.getByRole("button", { name: "Next", exact: true }).click();
  await page.getByRole("button", { name: "Back", exact: true }).click();
  await expect(slider).toHaveValue("201");
  await page.getByRole("link", { name: "Method", exact: true }).click();
  await expect(page).toHaveURL(/method\?showcase=v1/);
  await expect(page.locator("[data-tour-highlight]")).toHaveCount(0);
  await page.getByRole("button", { name: "Next", exact: true }).click();
  await page.getByRole("button", { name: "Back", exact: true }).click();
  await expect(slider).toHaveValue("201");
});

test("exit restores the previous location and saved preferences", async ({
  page,
}) => {
  await page.goto("/method?view=training&model=encoder");
  await page.evaluate(() => {
    localStorage.setItem("stormsense-unit", "m/s");
    localStorage.setItem("stormsense-smoothing", "on");
    localStorage.setItem("stormsense-satellite", "off");
  });
  await page.reload();
  await page.getByRole("button", { name: "Start guided tour" }).click();
  await ready(page);
  await expect(
    page.getByRole("button", { name: "Wind unit: kt. Switch units." }),
  ).toBeAttached();
  await page.getByRole("button", { name: "Exit", exact: true }).click();
  await expect(page).toHaveURL(/\/method\?view=training&model=encoder$/);
  await expect(
    page.getByRole("button", { name: "Wind unit: m/s. Switch units." }),
  ).toBeVisible();
  expect(
    await page.evaluate(() =>
      ["stormsense-unit", "stormsense-smoothing", "stormsense-satellite"].map(
        (k) => localStorage.getItem(k),
      ),
    ),
  ).toEqual(["m/s", "on", "off"]);
});

test("a missing asset prevents readiness and can be retried", async ({
  page,
}) => {
  let fail = true;
  await page.route("**/showcase/v1/packs/*.zip", (route) =>
    fail ? route.fulfill({ status: 404, body: "missing" }) : route.continue(),
  );
  await page.goto(showcaseUrl(SHOWCASE_STEPS[1]));
  await expect(page.getByRole("alert")).toContainText("Could not prepare");
  await expect(
    page.getByRole("button", { name: "Next", exact: true }),
  ).toBeDisabled();
  fail = false;
  await page.getByRole("button", { name: "Retry preparation" }).click();
  await ready(page);
  await expect(
    page.getByRole("slider", { name: "Storm timeline", exact: true }),
  ).toBeAttached();
});

test("a same-sized corrupted download pack is rejected before readiness", async ({ page }) => {
  let corrupt = true;
  await page.route("**/showcase/v1/packs/*.zip", async (route) => {
    if (!corrupt) return route.continue();
    const response = await route.fetch();
    const bytes = await response.body();
    bytes[0] ^= 1;
    await route.fulfill({ response, body: bytes });
  });
  await page.goto(showcaseUrl(SHOWCASE_STEPS[1]));
  await expect(page.getByRole("alert")).toContainText("checksum mismatch");
  await expect(page.getByRole("button", { name: "Next", exact: true })).toBeDisabled();
  corrupt = false;
  await page.getByRole("button", { name: "Retry preparation" }).click();
  await ready(page);
});

test("prepared imagery and model views survive a network loss, with reduced motion", async ({
  page,
  context,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(showcaseUrl(SHOWCASE_STEPS[1]));
  await ready(page);
  await expect(page.locator(".hourly-satellite-crop")).toHaveCount(1);
  await context.setOffline(true);
  await chapter(page, "inputs");
  await expect(page.locator(".method-theater")).toHaveAttribute(
    "data-motion",
    "reduced",
  );
  await expect
    .poll(() =>
      page
        .locator(".method-stack img")
        .evaluateAll((images) =>
          images.every(
            (img) =>
              (img as HTMLImageElement).complete &&
              (img as HTMLImageElement).naturalWidth > 0,
          ),
        ),
    )
    .toBe(true);
  await chapter(page, "pipeline");
  await expect(page.locator(".pipeline-number i").first()).toHaveCSS(
    "animation-name",
    "none",
  );
  await chapter(page, "interpretation");
  await expect(page.locator(".hourly-satellite-crop")).toHaveCount(1);
  await expect(page.locator(".hourly-imagery")).toHaveAttribute(
    "data-preload-ready",
    "292",
  );
  // Both ends of the full history remain navigable after network loss.
  const slider = page.getByRole("slider", { name: "Storm timeline", exact: true });
  await slider.focus();
  await page.keyboard.press("Home");
  await expect(slider).toHaveValue("0");
  await expect(page).toHaveURL(/time=2026-09-10T12%3A00%3A00Z/);
  await expect(page.locator(".hourly-satellite-crop")).toHaveCount(1);
  await page.keyboard.press("End");
  await expect(slider).toHaveValue("291");
  await expect(page).toHaveURL(/time=2026-10-04T18%3A00%3A00Z/);
  await expect(page.locator(".hourly-satellite-crop")).not.toHaveCount(0);
});

test("one highlight frame animates between targets and follows scrolling", async ({ page }) => {
  await page.goto("/method?showcase=v1&tour=inputs");
  await ready(page);
  const highlight = page.locator(".showcase-highlight");
  await expect(highlight).toHaveAttribute("data-moving", "false");
  await highlight.evaluate(el => { el.setAttribute("data-persistent-test", "same-frame"); });
  const intermediate = await page.evaluate(async () => {
    const next = [...document.querySelectorAll<HTMLButtonElement>(".showcase-buttons button")].find(b => b.textContent === "Next")!;
    next.click();
    await new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));
    const frame = document.querySelector<HTMLElement>(".showcase-highlight")!;
    const target = document.querySelector<HTMLElement>('[data-tour="network"]')!;
    const a = frame.getBoundingClientRect(), b = target.getBoundingClientRect();
    return { moving: frame.dataset.moving, delta: Math.abs(a.x - b.x + 8) + Math.abs(a.y - b.y + 8) + Math.abs(a.width - b.width - 16) + Math.abs(a.height - b.height - 16) };
  });
  expect(intermediate.moving).toBe("true");
  expect(intermediate.delta).toBeGreaterThan(1);
  await expect(highlight).toHaveAttribute("data-moving", "false");
  await expect(highlight).toHaveAttribute("data-persistent-test", "same-frame");
  const alignment = () => page.evaluate(() => {
    const a = document.querySelector(".showcase-highlight")!.getBoundingClientRect();
    const b = document.querySelector('[data-tour="network"]')!.getBoundingClientRect();
    return Math.abs(a.x - b.x + 8) + Math.abs(a.y - b.y + 8) + Math.abs(a.width - b.width - 16) + Math.abs(a.height - b.height - 16);
  });
  await expect.poll(alignment).toBeLessThan(1);
  await page.locator("#showcase-content").evaluate(el => { el.scrollTop += 80; });
  await expect.poll(alignment).toBeLessThan(1);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.getByRole("button", { name: "Next", exact: true }).click();
  await expect(highlight).toHaveAttribute("data-moving", "false");
  await expect(page.locator('[data-tour="outputs"]')).toHaveAttribute("data-tour-highlight", "true");
});


test("the introduction uses live catalog refreshes before switching to the saved example", async ({ page }) => {
  const root = new URL("../public/showcase/v1/", import.meta.url);
  const read = (path: string) => JSON.parse(readFileSync(new URL(path, root), "utf8"));
  const catalog = read(read("latest.json").manifest);
  let refreshes = 0;
  await page.route("**/data/latest.json", route => {
    refreshes++;
    return route.fulfill({ json: { schema_version: 1, manifest: "releases/live-test/catalog.json" } });
  });
  await page.route("**/data/releases/live-test/catalog.json", route => route.fulfill({ json: catalog }));
  await page.clock.install();
  await page.goto("/?showcase=v1&tour=live");
  await expect(page.getByRole("heading", { name: "Satellite-based estimates of tropical cyclone winds" })).toBeVisible();
  await expect(page.locator(".showcase-record-label")).toHaveCount(0);
  await expect(page.locator(".showcase-controls")).toHaveAttribute("data-assets-ready", "true", { timeout: 30000 });
  const before = refreshes;
  await page.clock.fastForward(5 * 60 * 1000 + 1000);
  await expect.poll(() => refreshes).toBeGreaterThan(before);
  await page.getByRole("button", { name: "Next", exact: true }).click();
  await ready(page);
  await expect(page.locator(".showcase-record-label")).toContainText("Historical hindcast");
  await expect(page.getByRole("slider", { name: "Storm timeline", exact: true })).toHaveAttribute("max", "291");
  const after = refreshes;
  await page.clock.fastForward(5 * 60 * 1000 + 1000);
  expect(refreshes).toBe(after);
});
