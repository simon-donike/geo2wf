import { describe, expect, it } from "vitest";
import { KNOT } from "./data";
import { rapidIntensification } from "./rapidIntensification";

const time = (hour: number) =>
  new Date(Date.UTC(2026, 8, 1, hour)).toISOString();
const points = (hours = 24, cadence = 1, gainKnots = 30) =>
  Array.from({ length: hours / cadence + 1 }, (_, index) => ({
    time: time(index * cadence),
    value: (40 + (index * cadence * gainKnots) / 24) * KNOT,
    version: "v1",
  }));

describe("rapid intensification", () => {
  it("includes exactly 30 kt in 24 hours, excludes just below it", () => {
    const result = rapidIntensification(points(), "model", 1);
    expect(result).toHaveLength(1);
    expect(result[0]).toMatchObject({
      source: "model",
      start: time(0),
      end: time(24),
      windows: 1,
    });
    expect(result[0].maxChangeMs).toBeCloseTo(30 * KNOT);
    expect(rapidIntensification(points(24, 1, 29.999), "model", 1)).toEqual([]);
  });
  it("requires exact endpoints and does not extrapolate a short window", () => {
    expect(rapidIntensification(points(23, 1, 60), "model", 1)).toEqual([]);
    expect(
      rapidIntensification(
        [
          { time: time(0), value: 10 },
          { time: time(25), value: 50 },
        ],
        "official",
        6,
      ),
    ).toEqual([]);
  });
  it("rejects absent hours, explicit gaps and invalid values", () => {
    expect(
      rapidIntensification(
        points().filter((_, i) => i !== 12),
        "model",
        1,
      ),
    ).toEqual([]);
    for (const value of [null, NaN, Infinity, -1]) {
      const input = points().map((point, i) =>
        i === 12 ? { ...point, value } : point,
      );
      expect(rapidIntensification(input, "model", 1)).toEqual([]);
    }
  });
  it("handles six-hour reference fixes separately, without bridging missed fixes", () => {
    expect(rapidIntensification(points(24, 6), "official", 6)[0].source).toBe(
      "official",
    );
    expect(
      rapidIntensification(
        points(24, 6).filter((_, i) => i !== 2),
        "official",
        6,
      ),
    ).toEqual([]);
    expect(rapidIntensification(points(24, 6), "model", 1)).toEqual([]);
  });
  it("merges overlapping windows without double-darkening the chart", () => {
    const result = rapidIntensification(points(48), "model", 1);
    expect(result).toHaveLength(1);
    expect(result[0]).toMatchObject({
      start: time(0),
      end: time(48),
      windows: 25,
    });
    expect(result[0].maxChangeMs).toBeCloseTo(30 * KNOT);
  });
  it("starts a new event after a gap or model change", () => {
    const input = points(72).map((point, i) => ({
      ...point,
      value: i === 36 ? null : point.value,
    }));
    expect(
      rapidIntensification(input, "model", 1).map(({ start, end }) => ({
        start,
        end,
      })),
    ).toEqual([
      { start: time(0), end: time(35) },
      { start: time(37), end: time(72) },
    ]);
    const versionChange = points().map((point, i) => ({
      ...point,
      version: i >= 12 ? "v2" : "v1",
    }));
    expect(rapidIntensification(versionChange, "model", 1)).toEqual([]);
  });
  it("sorts and deduplicates fixes, rejecting ambiguous duplicate values", () => {
    const input = points(24, 6);
    expect(
      rapidIntensification([...input, input[2]].reverse(), "official", 6),
    ).toHaveLength(1);
    expect(
      rapidIntensification(
        [...input, { ...input[2], value: 1 }],
        "official",
        6,
      ),
    ).toEqual([]);
  });
});
