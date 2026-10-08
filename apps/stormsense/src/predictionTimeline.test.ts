import { expect, it } from "vitest";
import { predictionTimeline } from "./predictionTimeline";
import { interpolatePoints, smoothPoints } from "./smoothing";
import type { Series } from "./types";

it("keeps pre-eligibility imagery without inventing predictions", () => {
  const series = {storm_id: "AL012026", records: [],
    prediction_schedule: {start_gate: true, cadence_hours: 2, eligible_start: "2026-09-01T04:00:00Z"},
    imagery: [{time: "2026-09-01T02:00:00Z"}, {time: "2026-09-01T04:00:00Z"}],
  } as unknown as Series;
  const rows = predictionTimeline(series);
  expect(rows.map(r => r.reason)).toEqual(["outside_prediction_scope", "prediction_unavailable"]);
  expect(rows.every(r => r.metrics === null)).toBe(true);
  expect(series.records).toHaveLength(0);
});

it("interpolates two-hour predictions only inside observed endpoints", () => {
  const points = [null, 10, 20, null].map((value, i) => ({time: `2026-09-01T0${i*2}:00:00Z`, value}));
  expect(interpolatePoints(points).map(p => p.value)).toEqual([null, null, 10, 15, 20, null, null]);
  expect(smoothPoints(points)[0].value).toBeNull();
  expect(smoothPoints(points).at(-1)?.value).toBeNull();
  expect(points.map(p => p.value)).toEqual([null, 10, 20, null]);
});
