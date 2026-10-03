import { describe, expect, it } from "vitest";
import {
  domainURL,
  latestProductTime,
  satelliteCrops,
  satelliteSide,
} from "./satellite";

describe("GeoColor map imagery", () => {
  const now = Date.parse("2026-10-02T16:25:00Z");
  it("uses reported available times and excludes future or stale frames", () => {
    expect(
      latestProductTime(
        "<Domains><Domain>2026-10-02T15:10:00Z/2026-10-02T15:40:00Z/PT10M,2026-10-02T16:00:00Z</Domain></Domains>",
        now,
      ),
    ).toBe("2026-10-02T16:00:00Z");
    expect(
      latestProductTime(
        "<Domain>2026-10-02T16:00:00Z/2026-10-02T17:00:00Z/PT10M</Domain>",
        now,
      ),
    ).toBe("2026-10-02T16:20:00Z");
    for (const xml of [
      "<Exception>offline</Exception>",
      "<Domain>2026-10-02T17:00:00Z</Domain>",
      "<Domain>2026-10-02T10:00:00Z</Domain>",
    ])
      expect(() => latestProductTime(xml, now)).toThrow("No recent");
  });
  it("selects East/West by viewing geometry and shares five-minute metadata URLs", () => {
    expect(satelliteSide(-65)).toBe("East");
    expect(satelliteSide(-110)).toBe("West");
    expect(satelliteSide(-166)).toBe("West");
    expect(satelliteSide(178)).toBe("West");
    expect(domainURL("West", now + 60000)).toBe(domainURL("West", now));
    expect(domainURL("West", now)).toContain(
      "2026-10-02T13:25:00Z--2026-10-02T16:25:00Z",
    );
  });
  it("maps the WMS crop to the same Web Mercator bounds as the Leaflet image", () => {
    const crop = satelliteCrops(
      19.3,
      -110.9,
      "West",
      "2026-10-02T15:40:00Z",
    )[0];
    const url = new URL(crop.url);
    expect(url.searchParams.get("SRS")).toBe("EPSG:3857");
    expect(url.searchParams.get("TIME")).toBe("2026-10-02T15:40:00Z");
    const bbox = url.searchParams.get("BBOX")!.split(",").map(Number);
    const project = ([lat, lon]: number[]) => [
      (6378137 * lon * Math.PI) / 180,
      6378137 * Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360)),
    ];
    [...project(crop.bounds[0]), ...project(crop.bounds[1])].forEach(
      (value, i) => expect(value).toBeCloseTo(bbox[i], 1),
    );
    expect(crop.bounds[0][0]).toBeLessThan(19.3);
    expect(crop.bounds[1][0]).toBeGreaterThan(19.3);
    expect((bbox[2] - bbox[0]) * Math.cos((19.3 * Math.PI) / 180)).toBeCloseTo(
      1800000,
      1,
    );
  });
  it("splits dateline crops without stretching or reversing them", () => {
    for (const lon of [-177, 178]) {
      const crops = satelliteCrops(20, lon, "West", "2026-10-02T15:40:00Z");
      expect(crops).toHaveLength(2);
      expect(crops[0].bounds[1][1]).toBeCloseTo(crops[1].bounds[0][1], 8);
      expect(
        crops.reduce(
          (sum, c) => sum + Number(new URL(c.url).searchParams.get("WIDTH")),
          0,
        ),
      ).toBe(1024);
      crops.forEach((c) => {
        const bbox = new URL(c.url).searchParams
          .get("BBOX")!
          .split(",")
          .map(Number);
        expect(bbox[0]).toBeGreaterThanOrEqual(-20037508.35);
        expect(bbox[2]).toBeLessThanOrEqual(20037508.35);
        expect(bbox[0]).toBeLessThan(bbox[2]);
      });
    }
  });
});
