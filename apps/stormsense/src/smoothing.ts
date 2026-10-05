/** Causal hourly filters shared by the website and playground evaluation. */
export type Smoothing = {
  style: "raw" | "legacy" | "mean" | "median" | "linear" | "gaussian" | "ema";
  parameter: number;
};
type Sample = { time: string; value: number | null };
export const WIND_SMOOTHING: Smoothing = { style: "ema", parameter: 0.5 };
export const STRUCTURE_SMOOTHING: Smoothing = { style: "legacy", parameter: 3 };

/** Fill only bounded gaps, including absent hourly timestamps. No extrapolation. */
export function interpolatePoints(points: Sample[]): Sample[] {
  const expanded: Sample[] = [];
  for (const point of points) {
    if (expanded.length) {
      for (let t = Date.parse(expanded.at(-1)!.time) + 3600000; t < Date.parse(point.time); t += 3600000)
        expanded.push({ time: new Date(t).toISOString(), value: null });
    }
    expanded.push({ ...point });
  }
  let left = -1;
  for (let right = 0; right < expanded.length; right++) {
    if (expanded[right].value == null || !Number.isFinite(expanded[right].value)) continue;
    if (left >= 0 && right > left + 1) {
      const a = expanded[left], b = expanded[right];
      for (let i = left + 1; i < right; i++) {
        const fraction = (Date.parse(expanded[i].time) - Date.parse(a.time)) / (Date.parse(b.time) - Date.parse(a.time));
        expanded[i].value = a.value! + fraction * (b.value! - a.value!);
      }
    }
    left = right;
  }
  return expanded;
}

export function smoothPoints(points: Sample[], config = WIND_SMOOTHING): Sample[] {
  let history: Sample[] = [];
  let previous: number | null = null;
  return interpolatePoints(points).map((point) => {
    if (point.value == null || !Number.isFinite(point.value)) {
      history = [];
      previous = null;
      return point;
    }
    if (history.length && Date.parse(point.time) - Date.parse(history.at(-1)!.time) > 1.5 * 3600000) {
      history = [];
      previous = null;
    }
    history.push(point);
    const size = config.style === "legacy" ? 3 : config.style === "gaussian" ? Math.ceil(4 * config.parameter) + 1 : config.style === "ema" || config.style === "raw" ? 1 : config.parameter;
    history = history.slice(-size);
    let value = point.value;
    if (config.style === "ema") {
      value = previous == null ? value : config.parameter * value + (1 - config.parameter) * previous;
    } else if (config.style === "median") {
      const values = history.map(p => p.value!).sort((a, b) => a - b);
      const mid = Math.floor(values.length / 2);
      value = values.length % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2;
    } else if (config.style !== "raw") {
      let total = 0, weights = 0;
      history.slice().reverse().forEach((p, i) => {
        const weight = config.style === "legacy" ? [0.6, 0.3, 0.1][i] : config.style === "linear" ? config.parameter - i : config.style === "gaussian" ? Math.exp(-0.5 * (i / config.parameter) ** 2) : 1;
        total += p.value! * weight;
        weights += weight;
      });
      value = total / weights;
    }
    previous = value;
    return { time: point.time, value };
  });
}
