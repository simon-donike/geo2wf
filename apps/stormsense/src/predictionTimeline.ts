import type { RecordHour, Series } from "./types";
import { preferredRecords } from "./data";

/** Keep display imagery navigable even where the prediction start gate is closed. */
export function predictionTimeline(series: Series | null): RecordHour[] {
  if (!series) return [];
  const records = preferredRecords(series.records);
  const present = new Set(records.map((r) => r.time));
  for (const frame of series.imagery ?? []) {
    if (present.has(frame.time)) continue;
    const schedule = series.prediction_schedule;
    const excluded = schedule?.start_gate &&
      (!schedule.eligible_start || frame.time < schedule.eligible_start);
    records.push({storm_id: series.storm_id, time: frame.time, kind: "hindcast",
      model_version: "", generated_at: "", status: "gap", metrics: null,
      center: null, reason: excluded ? "outside_prediction_scope" : "prediction_unavailable"});
  }
  return records.sort((a, b) => a.time.localeCompare(b.time));
}
