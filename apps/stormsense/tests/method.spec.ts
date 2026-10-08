import { test, expect } from "@playwright/test";
import { readFileSync } from "node:fs";
import type { MethodExample } from "../src/methodExample";

const example: MethodExample = JSON.parse(
  readFileSync(
    new URL("../public/method/example.json", import.meta.url),
    "utf8",
  ),
);
test.beforeEach(async ({ page }) => {
  await page.route("**/data/**", (route) => route.abort());
});

test("predictions are immediate, stable and independent of archive availability", async ({
  page,
}, testInfo) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const archiveRequests: string[] = [];
  page.on("request", (request) => {
    if (request.url().includes("/data/")) archiveRequests.push(request.url());
  });
  await page.goto("/method");
  await expect(
    page.getByRole("heading", {
      name: "From satellite imagery to surface winds.",
    }),
  ).toBeVisible();
  await expect(page.getByTestId("method-vmax")).toHaveText(
    (example.models.joint.scalars.vmax_ms / 0.514444).toFixed(0),
  );
  const field = page.getByRole("img", {
    name: "Actual dense surface wind-speed prediction for Ian from the joint model",
  });
  await expect(field).toBeVisible();
  await expect(field).toHaveCSS("clip-path", "none");
  await expect
    .poll(() =>
      field.evaluate(
        (img: HTMLImageElement) => img.complete && img.naturalWidth === 192,
      ),
    )
    .toBe(true);
  await expect(page.getByRole("slider")).toHaveCount(0);
  await expect(page.locator(".method-page > section")).toHaveCount(1);
  await expect(
    page.locator(
      ".method-explanation, .method-learning, .method-provenance, .method-stages, .method-controls",
    ),
  ).toHaveCount(0);
  await expect(page.getByRole("alert")).toHaveCount(0);
  expect(archiveRequests).toEqual([]);
  await page
    .getByRole("button", { name: "Encoder-only scalars", exact: true })
    .click();
  await expect(page.getByTestId("method-vmax")).toHaveText(
    (example.models.encoder.scalars.vmax_ms / 0.514444).toFixed(0),
  );
  await expect(field).toHaveCount(0);
  await page
    .getByRole("button", { name: "Wind unit: kt. Switch units." })
    .click();
  await expect(page.getByTestId("method-vmax")).toHaveText(
    example.models.encoder.scalars.vmax_ms.toFixed(0),
  );
  await page
    .getByRole("button", { name: "Joint field + scalars", exact: true })
    .click();
  await expect(page.getByTestId("method-vmax")).toHaveText(
    example.models.joint.scalars.vmax_ms.toFixed(0),
  );
  await expect(field).toBeVisible();
  await page.screenshot({
    path: `test-results/method-steady-${testInfo.project.name}.png`,
    fullPage: true,
  });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  expect(errors).toEqual([]);
});

test("both real tensor stacks expand on hover, keyboard focus and tap", async ({
  page,
}, testInfo) => {
  await page.goto("/method");
  for (const label of ["Satellite bands", "Extra tensors"]) {
    const stack = page.getByRole("button", { name: `${label} image stack` });
    await expect(stack).toHaveAttribute("aria-expanded", "false");
    if (testInfo.project.name === "desktop") {
      await stack.hover();
      await expect(stack).toHaveAttribute("aria-expanded", "true");
      await page.mouse.move(0, 0);
      await expect(stack).toHaveAttribute("aria-expanded", "false");
    } else {
      await stack.tap();
      await expect(stack).toHaveAttribute("aria-expanded", "true");
      await stack.tap();
      await expect(stack).toHaveAttribute("aria-expanded", "false");
    }
    await page.keyboard.press("Tab");
    await stack.focus();
    await expect(stack).toHaveAttribute("aria-expanded", "true");
    await page.screenshot({
      path: `test-results/method-stack-${label.split(" ")[0]}-${testInfo.project.name}.png`,
      fullPage: true,
    });
    await page.keyboard.press("Escape");
    await expect(stack).toHaveAttribute("aria-expanded", "false");
  }
  await page.getByRole("button", { name: "View CMI_C09", exact: true }).click();
  await expect(
    page.getByRole("img", {
      name: "Actual CMI_C09 infrared crop for Hurricane Ian",
    }),
  ).toBeVisible();
  for (const layer of example.context_channels) {
    await page
      .getByRole("button", { name: `View ${layer.label}`, exact: true })
      .click();
    const image = page.getByRole("img", {
      name: `Actual ${layer.label} input tensor for Hurricane Ian`,
    });
    await expect(image).toBeVisible();
    await expect
      .poll(() =>
        image.evaluate(
          (img: HTMLImageElement) => img.complete && img.naturalWidth === 192,
        ),
      )
      .toBe(true);
  }
  await expect(page.getByTestId("method-vmax")).toHaveText(
    (example.models.joint.scalars.vmax_ms / 0.514444).toFixed(0),
  );
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
});

test("flow repeats while predictions stay visible, and pauses in a hidden tab", async ({
  page,
}) => {
  await page.goto("/method");
  const theater = page.getByRole("region", {
    name: "Interactive model walkthrough",
  });
  await expect(theater).toHaveAttribute("data-motion", "running");
  const flow = page.locator(".method-flow-lit").first();
  const flowTime = () =>
    flow.evaluate((el) => Number(el.getAnimations()[0]?.currentTime || 0));
  await expect.poll(flowTime, { timeout: 12000 }).toBeGreaterThan(5500);
  await expect(page.getByTestId("method-vmax")).toHaveText(
    (example.models.joint.scalars.vmax_ms / 0.514444).toFixed(0),
  );
  await expect(page.locator(".method-field-image > img")).toHaveCSS(
    "clip-path",
    "none",
  );
  await page.getByRole("button", { name: "Pause flow", exact: true }).click();
  await expect(theater).toHaveAttribute("data-motion", "paused");
  await expect(flow).toHaveCSS("animation-play-state", "paused");
  await page.getByRole("button", { name: "Resume flow", exact: true }).click();
  await page.evaluate(() => {
    Object.defineProperty(document, "hidden", {
      configurable: true,
      value: true,
    });
    document.dispatchEvent(new Event("visibilitychange"));
  });
  await expect(theater).toHaveAttribute("data-motion", "paused");
  await page.evaluate(() => {
    Reflect.deleteProperty(document, "hidden");
    document.dispatchEvent(new Event("visibilitychange"));
  });
  await expect(theater).toHaveAttribute("data-motion", "running");
});

test("reduced motion retains predictions and interactive stacks", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/method");
  await expect(page.locator(".method-theater")).toHaveAttribute(
    "data-motion",
    "reduced",
  );
  await expect(page.locator(".method-flow-lit").first()).toHaveCSS(
    "animation-name",
    "none",
  );
  await expect(page.getByTestId("method-vmax")).toHaveText(
    (example.models.joint.scalars.vmax_ms / 0.514444).toFixed(0),
  );
  await page.getByRole("button", { name: "Extra tensors image stack" }).click();
  await expect(
    page.getByRole("button", { name: "Extra tensors image stack" }),
  ).toHaveAttribute("aria-expanded", "true");
  await page.getByRole("link", { name: "About", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "The storm record is not available" }),
  ).toBeVisible();
  await page.getByRole("link", { name: "Method", exact: true }).click();
  await expect(
    page.getByRole("heading", {
      name: "From satellite imagery to surface winds.",
    }),
  ).toBeVisible();
});

test("training tab shows the correct objective and update path for each model", async ({
  page,
}, testInfo) => {
  await page.goto("/method");
  const inference = page.getByRole("tab", { name: "Inference", exact: true });
  const training = page.getByRole("tab", { name: "Training", exact: true });
  await expect(inference).toHaveAttribute("aria-selected", "true");
  await training.click();
  const panel = page.getByRole("tabpanel", { name: "Training", exact: true });
  await expect(panel).toBeVisible();
  await expect(page.getByTestId("method-vmax")).not.toBeVisible();
  const sar = panel.getByRole("img", {
    name: "Real SAR-derived wind-speed field for Hurricane Ian, aligned to the prediction grid",
  });
  await expect(sar).toBeVisible();
  const rasters = panel.locator(".method-training-raster img");
  await expect(rasters).toHaveCount(3);
  for (const img of await rasters.all()) {
    await expect
      .poll(() =>
        img.evaluate(
          (el: HTMLImageElement) =>
            el.complete && el.naturalWidth === 192 && el.naturalHeight === 192,
        ),
      )
      .toBe(true);
  }
  await expect(panel.locator(".method-backward-flow")).toHaveCount(6);
  await expect
    .poll(() =>
      panel
        .locator(".method-backward-pulse")
        .first()
        .evaluate((el) => Number(el.getAnimations()[0]?.currentTime || 0)),
    )
    .toBeGreaterThan(250);
  const scale = panel.locator(".field-loss .method-scale");
  await expect(scale).toContainText("kt");
  await page
    .getByRole("button", { name: "Wind unit: kt. Switch units." })
    .click();
  await expect(scale).toContainText("m/s");
  await page
    .getByRole("button", { name: "Wind unit: m/s. Switch units." })
    .click();
  await panel.locator(".method-backpropagation").evaluate((el) =>
    el.getAnimations().forEach((animation) => {
      animation.currentTime = 3500;
      animation.pause();
    }),
  );
  await expect(panel.locator(".method-training-equation")).toHaveAttribute(
    "aria-label",
    "Total loss equals field loss plus wind loss plus 0.25 times structure loss",
  );
  await page.screenshot({
    path: `test-results/method-training-joint-${testInfo.project.name}.png`,
    fullPage: true,
  });
  await page
    .getByRole("button", { name: "Encoder-only scalars", exact: true })
    .click();
  await expect(sar).toHaveCount(0);
  await expect(panel.locator(".method-backward-flow")).toHaveCount(2);
  await expect(panel.locator(".method-training-equation")).toHaveAttribute(
    "aria-label",
    "Total loss equals wind loss plus 0.25 times structure loss",
  );
  await panel.getByText("Exact loss definitions", { exact: true }).click();
  await expect(
    panel.getByText(/ATCF fine-tuning dataset has no eye-size labels/),
  ).toBeVisible();
  await page.getByRole("button", { name: "Pause flow", exact: true }).click();
  await expect(panel.locator(".method-gradient-return i")).toHaveCSS(
    "animation-play-state",
    "paused",
  );
  await expect(panel.locator(".method-backward-pulse").first()).toHaveCSS(
    "animation-play-state",
    "paused",
  );
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: `test-results/method-training-encoder-${testInfo.project.name}.png`,
    fullPage: true,
  });
  await training.focus();
  await page.keyboard.press("ArrowLeft");
  await expect(inference).toBeFocused();
  await expect(page.getByTestId("method-vmax")).toHaveText(
    (example.models.encoder.scalars.vmax_ms / 0.514444).toFixed(0),
  );
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.keyboard.press("ArrowRight");
  await expect(training).toBeFocused();
  await expect(panel.locator(".method-gradient-return i")).toHaveCSS(
    "animation-name",
    "none",
  );
  await expect(panel.locator(".method-backward-pulse").first()).toHaveCSS(
    "animation-name",
    "none",
  );
  await expect(
    page.getByRole("button", { name: "Pause flow", exact: true }),
  ).toHaveCount(0);
});
