import { expect, test } from "@playwright/test";
import type { Catalog, Series } from "../src/types";

test("two-hour migration retains pre-onset images and labels their prediction scope", async ({ page, request }) => {
  const pointer = await (await request.get('/data/latest.json')).json();
  const catalog: Catalog = await (await request.get('/data/' + pointer.manifest)).json();
  const storm = catalog.storms.find(s => s.id === 'EP152026')!;
  const series: Series = await (await request.get('/data/' + storm.series)).json();
  const frames = series.imagery!.filter(f => f.status === 'ready');
  const before = frames[0];
  const onset = frames[Math.min(5, frames.length - 1)].time;
  series.prediction_schedule = {cadence_hours: 2, start_gate: true, eligible_start: onset};
  series.records = series.records.filter(r => r.time >= onset && new Date(r.time).getUTCHours() % 2 === 0);
  storm.prediction_schedule = series.prediction_schedule;
  catalog.prediction_cadence_hours = 2;
  await page.route('**/data/' + storm.series, route => route.fulfill({json: series}));
  await page.route('**/data/' + pointer.manifest, route => route.fulfill({json: catalog}));
  await page.goto(`/storms/${storm.id}?time=${encodeURIComponent(before.time)}`);
  await expect(page.getByText('Before tropical/subtropical classification · imagery only')).toBeVisible();
  await expect(page.locator('.hourly-satellite-crop').first()).toHaveAttribute('data-slot-time', before.time);
  await expect(page.getByRole('slider', {name: 'Storm timeline', exact: true})).toHaveCount(1);
  await expect(page.getByText(/estimates · every 2 hours/)).toBeVisible();
  await page.goto(`/storms/${storm.id}?time=${encodeURIComponent(onset)}`);
  await expect(page.getByText('Before tropical/subtropical classification · imagery only')).toHaveCount(0);
});
