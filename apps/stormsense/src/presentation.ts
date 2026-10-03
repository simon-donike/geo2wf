import { className, KNOT } from "./data";
import type { Fix, Metrics } from "./types";
import type { Point } from "./Chart";

// NHC Saffir–Simpson thresholds, in knots; canonical values remain in m/s.
export const WIND_BANDS = [
  { knots: 34, label: "Tropical storm", short: "TS", color: "#7fc8bc" },
  { knots: 64, label: "Category 1", short: "Cat 1", color: "#e0d291" },
  { knots: 83, label: "Category 2", short: "Cat 2", color: "#e7b77d" },
  { knots: 96, label: "Category 3", short: "Cat 3", color: "#ed936f" },
  { knots: 113, label: "Category 4", short: "Cat 4", color: "#df7c89" },
  { knots: 137, label: "Category 5", short: "Cat 5", color: "#c898d4" },
];
export function windCategory(value?: number | null) {
  if (value == null || !Number.isFinite(value)) return "Unavailable";
  return (
    WIND_BANDS.slice()
      .reverse()
      .find((band) => value + 1e-8 >= band.knots * KNOT)?.label ??
    "Tropical depression"
  );
}
export function officialCategory(fix?: Fix | null) {
  const category = windCategory(fix?.wind_ms);
  return fix?.classification === "HU" && category.startsWith("Category")
    ? `${category} hurricane`
    : className(fix?.classification);
}
export function stormAge(start: string, now = Date.now()) {
  const elapsed = now - Date.parse(start);
  if (!Number.isFinite(elapsed)) return "Unknown";
  const hours = Math.max(0, Math.floor(elapsed / 3600000));
  const days = Math.floor(hours / 24),
    rest = hours % 24;
  return `${days ? `${days} ${days === 1 ? "day" : "days"} ` : ""}${rest} ${rest === 1 ? "hour" : "hours"}`;
}
export const RADII: {
  key: keyof Metrics;
  label: string;
  color: string;
  dashed?: boolean;
}[] = [
  { key: "r34_km", label: "R34", color: "#40c7bd" },
  { key: "r50_km", label: "R50", color: "#c8c397" },
  { key: "r64_km", label: "R64", color: "#e9ac7a" },
  { key: "rmw_km", label: "RMW", color: "#e8e2d7", dashed: true },
];

/** Light causal display filter: 60% current, 30% previous, 10% two hours ago.
 * Gaps reset the window; no future values or missing estimates are introduced.
 */
export function smoothPoints(points: Point[]): Point[] {
  let history: Point[] = [];
  return points.map((point) => {
    if (point.value == null || !Number.isFinite(point.value)) {
      history = [];
      return point;
    }
    if (
      history.length &&
      Date.parse(point.time) - Date.parse(history.at(-1)!.time) > 1.5 * 3600000
    )
      history = [];
    history.push(point);
    history = history.slice(-3);
    let sum = 0,
      weight = 0;
    history
      .slice()
      .reverse()
      .forEach((p, i) => {
        const w = [0.6, 0.3, 0.1][i];
        sum += p.value! * w;
        weight += w;
      });
    return { time: point.time, value: sum / weight };
  });
}

export function nearestIndex(times: number[], requested?: string | null) {
  if (!times.length) return 0;
  const target = requested ? Date.parse(requested) : NaN;
  if (!Number.isFinite(target)) return times.length - 1;
  let low = 0,
    high = times.length - 1;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (times[middle] < target) low = middle + 1;
    else high = middle;
  }
  return low > 0 && target - times[low - 1] <= times[low] - target
    ? low - 1
    : low;
}
