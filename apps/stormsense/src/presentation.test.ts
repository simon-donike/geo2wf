import { describe, expect, it } from "vitest";
import { KNOT } from "./data";
import {
  nearestIndex,
  officialCategory,
  smoothPoints,
  interpolatePoints,
  stormAge,
  windCategory,
} from "./presentation";
import type { Fix } from "./types";

describe("storm presentation", () => {
  it("uses the NHC boundaries without turning non-tropical systems into hurricanes", () => {
    for (const [knots, name] of [
      [33, "Tropical depression"],
      [34, "Tropical storm"],
      [63.9, "Tropical storm"],
      [64, "Category 1"],
      [83, "Category 2"],
      [96, "Category 3"],
      [113, "Category 4"],
      [137, "Category 5"],
    ] as const)
      expect(windCategory(knots * KNOT)).toBe(name);
    expect(officialCategory({ classification: "EX", wind_ms: 70 } as Fix)).toBe(
      "Post-tropical",
    );
    expect(
      officialCategory({ classification: "HU", wind_ms: 83 * KNOT } as Fix),
    ).toBe("Category 2 hurricane");
    expect(windCategory(null)).toBe("Unavailable");
  });
  it("shows elapsed storm days and hours", () => {
    expect(
      stormAge("2026-09-01T00:00:00Z", Date.parse("2026-09-03T10:59:00Z")),
    ).toBe("2 days 10 hours");
    expect(
      stormAge("2026-09-01T00:00:00Z", Date.parse("2026-09-02T01:00:00Z")),
    ).toBe("1 day 1 hour");
  });
  it("uses exponential smoothing without using future values or changing the original records", () => {
    const points = [10, 20, 40, 100].map((value, i) => ({
      time: `2026-09-01T0${i}:00:00Z`,
      value,
    }));
    const result = smoothPoints(points);
    expect(result[0].value).toBe(10);
    expect(result[1].value).toBe(15);
    expect(result[2].value).toBe(27.5);
    expect(result[3].value).toBe(63.75);
    expect(smoothPoints(points.slice(0, 3))).toEqual(result.slice(0, 3));
    expect(points.map((p) => p.value)).toEqual([10, 20, 40, 100]);
  });
  it("interpolates bounded gaps without extrapolating", () => {
    const result = interpolatePoints([
      { time: "2026-09-01T00:00:00Z", value: 10 },
      { time: "2026-09-01T01:00:00Z", value: null },
      { time: "2026-09-01T02:00:00Z", value: 80 },
      { time: "2026-09-01T04:00:00Z", value: 40 },
    ]);
    expect(result.map((p) => p.value)).toEqual([10, 45, 80, 60, 40]);
  });
  it("preserves unbounded gaps and follows a rising wind with about one hour of lag", () => {
    const points = Array.from({ length: 20 }, (_, i) => ({
      time: new Date(Date.UTC(2026, 8, 1, i)).toISOString(), value: i,
    }));
    expect(smoothPoints(points).at(-1)!.value).toBeCloseTo(18, 4);
    const missing = [{ ...points[0], value: null }, points[1], { ...points[2], value: null }];
    expect(interpolatePoints(missing).map(p => p.value)).toEqual([null, 1, null]);
  });
  it("finds the nearest hour at boundaries and defaults to the latest observation", () => {
    const times = [0, 3600000, 7200000];
    expect(nearestIndex(times, null)).toBe(2);
    expect(nearestIndex(times, new Date(-1).toISOString())).toBe(0);
    expect(nearestIndex(times, new Date(3599999).toISOString())).toBe(1);
    expect(nearestIndex(times, new Date(1800000).toISOString())).toBe(0);
    expect(nearestIndex(times, new Date(10000000).toISOString())).toBe(2);
  });
});
