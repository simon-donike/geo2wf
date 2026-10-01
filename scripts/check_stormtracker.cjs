// Optional browser integration check. Requires Playwright and Chromium.
// STORMSENSE_URL defaults to the local MkDocs build served on port 8765.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.STORMSENSE_URL || 'http://127.0.0.1:8765';
(async () => {
  const browser = await chromium.launch({headless: true,
    ...(process.env.CHROMIUM_PATH ? {executablePath: process.env.CHROMIUM_PATH} : {})});
  const errors = [], failures = [];
  try {
    for (const releaseMode of [false, true]) {
      const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
      page.on('pageerror', e => errors.push(e.message));
      page.on('response', r => { if (r.status() >= 400) failures.push(`${r.status()} ${r.url()}`); });
      if (releaseMode) {
        // Exercise the production pointer/manifest/asset resolver with a local fixture.
        await page.route('**/explorer/data-config.js', route => route.fulfill({contentType: 'application/javascript', body: `window.GEO2WF_EXPLORER_RELEASE_URLS=[${JSON.stringify(base + '/fixture/latest.json')}];`}));
        await page.route('**/fixture/latest.json', route => route.fulfill({json: {version: 'smoke', manifest: 'releases/smoke/storm-data.json'}}));
        await page.route('**/fixture/releases/smoke/**', route => {
          const relative = new URL(route.request().url()).pathname.split('/fixture/releases/smoke/')[1];
          return route.fulfill({path: path.join(__dirname, '../site/explorer', relative)});
        });
      }
      await page.goto(`${base}/explorer/dashboard.html`);
      await page.waitForSelector('#stormSelect');
      if (await page.locator('#welcomeBanner').isVisible()) await page.locator('#dismissWelcome').click();
      assert.equal(await page.locator('#stormSelect option').count(), 3);
      for (const storm of ['AL082025', 'EP112025', 'EP182023']) {
        await page.locator('#stormSelect').selectOption(storm);
        await page.locator('#nowcastMode').click();
        for (const model of ['vit', 'unet', 'unet_mlp']) {
          await page.locator('#modelSelector').selectOption(model);
          assert.ok(await page.locator('#charts svg').count() > 0);
        }
        await page.locator('#timeSlider').evaluate(el => {el.value = Math.floor(Number(el.max)/2); el.dispatchEvent(new Event('input', {bubbles:true}));});
        await page.locator('#forecastMode').click();
        for (const model of ['convlstm', 'mlp']) {
          await page.locator('#forecastModelSelector').selectOption(model);
          await page.waitForFunction(() => !document.querySelector('#charts .forecast-placeholder'));
          assert.equal(await page.locator('#forecastNotice').isVisible(), false);
          assert.ok(await page.locator('#charts svg').count() > 0);
        }
      }
      await page.locator('#nowcastMode').click();
      await page.locator('label').filter({has: page.locator('#postProcessing')}).click();
      await page.locator('label').filter({has: page.locator('#showNwp')}).click();
      const layers = page.locator('.imagery-layer-toggle');
      assert.ok(await layers.count() >= 3);
      for (let i=0; i<await layers.count(); i++) {
        await layers.nth(i).click();
        await layers.nth(i).click();
      }
      await page.locator('#playButton').click();
      await page.waitForTimeout(200);
      await page.locator('#playButton').click();
      await page.waitForTimeout(1000);
      fs.mkdirSync(path.join(__dirname, '../build/conference'), {recursive:true});
      await page.screenshot({path:path.join(__dirname, `../build/conference/stormtracker-${releaseMode?'release':'local'}.png`)});
      await page.close();
    }
    assert.deepEqual(errors, []);
    assert.deepEqual(failures, []);
    console.log('PASS: three storms; all nowcast/forecast models; timelines; map layers; NWP; postprocessing; local and release-pointer data; no browser errors or HTTP failures.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode=1; });
