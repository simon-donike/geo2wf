import { describe, it, expect } from "vitest";
import { age, csv, isStale, preferredRecords, wind } from "./data";
import type { RecordHour, Series } from "./types";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { Chart } from "./Chart";
const row = (kind: "live" | "hindcast", status: "gap" | "ready") =>
  ({
    time: "2026-09-01T00:00:00Z",
    kind,
    status,
    metrics:
      status === "ready"
        ? { vmax_ms: 25, rmw_km: 20, r34_km: 100, r50_km: 40, r64_km: 10 }
        : null,
    reason: status === "gap" ? "missing_scan" : null,
  }) as RecordHour;
describe("public data behavior", () => {
  it("keeps isolated observations visible on either side of a gap", () => {
    const markup = renderToStaticMarkup(
      createElement(Chart, {
        title: "Wind",
        unit: "m/s",
        lines: [
          {
            label: "Estimate",
            color: "teal",
            points: [
              { time: "2026-09-01T00:00:00Z", value: 30 },
              { time: "2026-09-01T01:00:00Z", value: null },
              { time: "2026-09-01T02:00:00Z", value: 40 },
            ],
          },
        ],
      }),
    );
    expect(markup.match(/<circle /g)).toHaveLength(2);
    expect(markup).not.toContain("NaN");
  });
  it("uses a ready hindcast when the live slot is a gap", () =>
    expect(
      preferredRecords([row("live", "gap"), row("hindcast", "ready")])[0].kind,
    ).toBe("hindcast"));
  it("prefers a live estimate to a retrospective estimate", () =>
    expect(
      preferredRecords([row("live", "ready"), row("hindcast", "ready")])[0]
        .kind,
    ).toBe("live"));
  it("reports missing and stale timestamps accurately", () => {
    expect(isStale(null)).toBe(true);
    expect(isStale("2026-01-01", 3, Date.parse("2026-01-02"))).toBe(true);
    expect(age(undefined)).toBe("Awaiting data");
  });
  it("converts wind units while retaining canonical CSV units", () => {
    expect(wind(25.7222, "kt")).toBe("50");
    const result = csv({ records: [row("hindcast", "ready")] } as Series);
    expect(result).toContain("vmax_ms");
    expect(result).toContain('"25"');
  });
});
