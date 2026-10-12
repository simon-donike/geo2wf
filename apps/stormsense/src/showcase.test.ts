import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { unzipSync } from "fflate";
import manifest from "../public/showcase/v1/manifest.json";
import { SHOWCASE_STEPS, showcaseUrl } from "./showcaseSteps";

describe("versioned showcase delivery", () => {
  it("ships every pinned asset with its expected size and checksum", () => {
    for (const file of [...manifest.files, ...manifest.packs]) {
      const bytes = readFileSync(
        new URL(`../public${file.url}`, import.meta.url),
      );
      expect(bytes.length, file.url).toBe(file.bytes);
      expect(createHash("sha256").update(bytes).digest("hex"), file.url).toBe(
        file.sha256,
      );
    }
  });
  it("preserves Nolo’s full published history, schedule, gaps and provenance", () => {
    const root = new URL("../public/showcase/v1/", import.meta.url);
    const read = (path: string) =>
      JSON.parse(readFileSync(new URL(path, root), "utf8"));
    const catalog = read(read("latest.json").manifest);
    const series = read(catalog.storms[0].series);
    expect(catalog.storms).toHaveLength(1);
    expect(catalog.storms[0].active).toBe(false);
    expect(series.records).toHaveLength(253);
    expect(series.imagery).toHaveLength(292);
    expect(series.track).toHaveLength(98);
    expect(series.forecasts).toHaveLength(244);
    expect(manifest.window).toEqual({ start: "2026-09-10T12:00:00Z", end: "2026-10-04T18:00:00Z" });
    expect(series.imagery[0].time).toBe(manifest.window.start);
    expect(series.imagery.at(-1).time).toBe(manifest.window.end);
    expect(series.track[0].time).toBe(manifest.window.start);
    expect(series.track.at(-1).time).toBe(manifest.window.end);
    expect(series.prediction_schedule).toEqual(manifest.source_storm.prediction_schedule);
    expect(series.records[0].time).toBe(series.prediction_schedule.eligible_start);
    expect(catalog.coverage).toEqual({expected: 253, predictions: 252, gaps: 1, pending: 0});
    expect(series.records.filter((r: { status: string }) => r.status === "gap")).toHaveLength(1);
    expect({ ...catalog.storms[0], series: manifest.source_storm.series }).toEqual(manifest.source_storm);
    expect(
      series.records.every(
        (r: { kind: string; status: string; time: string }) =>
          r.kind === "hindcast" &&
          r.time >= manifest.window.start &&
          r.time <= manifest.window.end,
      ),
    ).toBe(true);
    expect(
      series.records.find(
        (r: { time: string }) => r.time === manifest.recorded_prediction.time,
      ),
    ).toEqual(manifest.recorded_prediction);
  });
  it("delivery packs contain each local asset exactly once with identical bytes", () => {
    const seen: string[] = [];
    for (const pack of manifest.packs) {
      const entries = unzipSync(readFileSync(new URL(`../public${pack.url}`, import.meta.url)));
      const files = manifest.files.slice(pack.start, pack.start + pack.count);
      expect(Object.keys(entries)).toHaveLength(files.length);
      for (const file of files) {
        const bytes = entries[file.url.replace("/showcase/v1/", "")];
        expect(bytes.length).toBe(file.bytes);
        expect(createHash("sha256").update(bytes).digest("hex"), file.url).toBe(file.sha256);
        seen.push(file.url);
      }
    }
    expect(seen).toEqual(manifest.files.filter(f => f.url.startsWith("/showcase/v1/")).map(f => f.url));
    expect(new Set(seen).size).toBe(seen.length);
  });
  it("gives each stop a reproducible local location", () => {
    expect(new Set(SHOWCASE_STEPS.map((s) => s.id)).size).toBe(10);
    expect(SHOWCASE_STEPS[0].page).toBe("overview");
    expect(SHOWCASE_STEPS.some(step => step.id === "learning")).toBe(false);
    expect(new URL(showcaseUrl(SHOWCASE_STEPS[0]), "https://local.invalid").pathname).toBe("/");
    for (const step of SHOWCASE_STEPS) {
      const url = new URL(showcaseUrl(step), "https://local.invalid");
      expect(url.searchParams.get("showcase")).toBe("v1");
      expect(url.searchParams.get("tour")).toBe(step.id);
      if (step.page === "storm")
        expect(url.searchParams.get("time")).toBe(step.time);
    }
  });
});
