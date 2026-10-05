import { KNOT } from "./data";

export type RISource = "official" | "model";
export interface RIInterval {
  source: RISource;
  start: string;
  end: string;
  maxChangeMs: number;
  windows: number;
}
interface WindPoint {
  time: string;
  value: number | null; // Unsmoothed maximum sustained wind, in m/s.
  version?: string;
}
const HOUR = 3_600_000;

/** Union of complete, exact 24-hour windows gaining at least 30 kt.
 * Missing/invalid values, cadence gaps and model changes break continuity.
 * Call with observations/estimates only, never forecast valid-time points.
 */
export function rapidIntensification(
  points: WindPoint[],
  source: RISource,
  maxGapHours: number,
): RIInterval[] {
  const unique = new Map<number, WindPoint>();
  for (const point of points) {
    const time = Date.parse(point.time);
    if (!Number.isFinite(time)) continue;
    const prior = unique.get(time);
    // Conflicting duplicate fixes are ambiguous, not extra evidence of RI.
    unique.set(
      time,
      prior && (prior.value !== point.value || prior.version !== point.version)
        ? { ...point, value: null }
        : point,
    );
  }
  const ordered = [...unique].sort(([a], [b]) => a - b);
  const history = new Map<number, { value: number; segment: number }>();
  const intervals: RIInterval[] = [];
  let segment = 0;
  let previous: { time: number; version?: string } | undefined;
  for (const [time, point] of ordered) {
    if (
      point.value == null ||
      !Number.isFinite(point.value) ||
      point.value < 0
    ) {
      segment++;
      previous = undefined;
      continue;
    }
    if (
      previous &&
      (time - previous.time > maxGapHours * HOUR ||
        point.version !== previous.version)
    )
      segment++;
    const before = history.get(time - 24 * HOUR);
    const change = before ? point.value - before.value : 0;
    if (before?.segment === segment && change >= 30 * KNOT - 1e-8) {
      const start = new Date(time - 24 * HOUR).toISOString();
      const end = new Date(time).toISOString();
      const last = intervals.at(-1);
      if (last && Date.parse(last.end) >= time - 24 * HOUR) {
        last.end = end;
        last.maxChangeMs = Math.max(last.maxChangeMs, change);
        last.windows++;
      } else {
        intervals.push({ source, start, end, maxChangeMs: change, windows: 1 });
      }
    }
    history.set(time, { value: point.value, segment });
    previous = { time, version: point.version };
  }
  return intervals;
}
