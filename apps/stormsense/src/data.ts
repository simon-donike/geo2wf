import type { Catalog, RecordHour, Series } from "./types";
export const BASINS = {
  AL: "Atlantic",
  EP: "Eastern Pacific",
  CP: "Central Pacific",
};
export const KNOT = 0.514444;
export const stamp = (time: string, compact = false) =>
  new Date(time).toLocaleString("en-GB", {
    timeZone: "UTC",
    day: "2-digit",
    month: "short",
    ...(compact
      ? {}
      : { year: "numeric", hour: "2-digit", minute: "2-digit", hour12: false }),
  }) + (compact ? "" : " UTC");
export function age(time?: string | null, now = Date.now()) {
  if (!time) return "Awaiting data";
  const minutes = Math.max(0, Math.floor((now - Date.parse(time)) / 60000));
  return minutes < 60
    ? `${minutes}m ago`
    : minutes < 1440
      ? `${Math.floor(minutes / 60)}h ago`
      : `${Math.floor(minutes / 1440)}d ago`;
}
export const isStale = (time?: string | null, hours = 3, now = Date.now()) =>
  !time || now - Date.parse(time) > hours * 3600000;
export const wind = (value: number | null | undefined, unit: "kt" | "m/s") =>
  value == null ? "—" : (value / (unit === "kt" ? KNOT : 1)).toFixed(0);
export const className = (value?: string) =>
  ({
    HU: "Hurricane",
    TS: "Tropical storm",
    TD: "Tropical depression",
    SS: "Subtropical storm",
    SD: "Subtropical depression",
    EX: "Post-tropical",
    DB: "Disturbance",
  })[value || ""] ||
  value ||
  "System";
const root = (import.meta.env?.VITE_DATA_BASE || "/data/").replace(/\/?$/, "/");
export const dataUrl = (path: string) => root + path;
async function read<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(dataUrl(path), {
    signal,
    cache: path === "latest.json" ? "no-store" : "default",
  });
  if (!response.ok)
    throw Error(
      response.status === 404
        ? "Data has not been published yet."
        : `Data could not be loaded (${response.status}).`,
    );
  const value = await response.json();
  if (!value || value.schema_version !== 1)
    throw Error("This data release uses an unsupported format.");
  return value as T;
}
export async function getCatalog(signal?: AbortSignal) {
  const pointer = await read<{ schema_version: 1; manifest: string }>(
    "latest.json",
    signal,
  );
  if (!/^releases\/[A-Za-z0-9_-]+\/catalog\.json$/.test(pointer.manifest))
    throw Error("Invalid data release.");
  const catalog = await read<Catalog>(pointer.manifest, signal);
  if (
    !Array.isArray(catalog.storms) ||
    !catalog.coverage ||
    !catalog.window ||
    !catalog.models?.nowcast ||
    !catalog.models?.forecast ||
    !catalog.source_status ||
    catalog.storms.some((s) => !s.id || !(s.basin in BASINS) || !s.series)
  )
    throw Error("Invalid storm catalog.");
  return catalog;
}
export async function getSeries(path: string, signal?: AbortSignal) {
  if (!/^objects\/[a-f0-9]{64}\.json$/.test(path))
    throw Error("Invalid storm series.");
  const series = await read<Series>(path, signal);
  const keys = ["vmax_ms", "rmw_km", "r34_km", "r50_km", "r64_km"] as const;
  if (
    !Array.isArray(series.records) ||
    !Array.isArray(series.forecasts) ||
    !Array.isArray(series.track) ||
    series.records.some(
      (r) =>
        !Number.isFinite(Date.parse(r.time)) ||
        (r.status === "ready" &&
          (!r.metrics ||
            keys.some((key) => !Number.isFinite(r.metrics![key])))),
    )
  )
    throw Error("Invalid storm history.");
  return series;
}
export function preferredRecords(records: RecordHour[]) {
  const unique = new Map<string, RecordHour>();
  for (const r of records) {
    const prior = unique.get(r.time);
    if (
      !prior ||
      (r.status === "ready" && (prior.status !== "ready" || r.kind === "live"))
    )
      unique.set(r.time, r);
  }
  return [...unique.values()].sort((a, b) => a.time.localeCompare(b.time));
}
export function download(
  name: string,
  body: string,
  type = "application/json",
) {
  const url = URL.createObjectURL(new Blob([body], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function csv(series: Series) {
  const fields = [
    "record_type",
    "time",
    "kind",
    "status",
    "model_version",
    "generated_at",
    "valid_time",
    "lead_hours",
    "vmax_ms",
    "rmw_km",
    "r34_km",
    "r50_km",
    "r64_km",
    "reason",
  ];
  const rows: (string | number | null | undefined)[][] = series.records.map(
    (r) => [
      "estimate",
      r.time,
      r.kind,
      r.status,
      r.model_version,
      r.generated_at,
      r.time,
      0,
      r.metrics?.vmax_ms,
      r.metrics?.rmw_km,
      r.metrics?.r34_km,
      r.metrics?.r50_km,
      r.metrics?.r64_km,
      r.reason,
    ],
  );
  for (const f of series.forecasts ?? [])
    for (const p of f.predictions)
      rows.push([
        "forecast",
        f.anchor_time,
        f.kind,
        "ready",
        f.model_version,
        f.generated_at,
        p.valid_time,
        p.lead_hours,
        p.vmax_ms,
      ]);
  for (const f of series.track ?? [])
    rows.push([
      "official",
      f.time,
      "reference",
      "reference",
      "",
      "",
      f.time,
      0,
      f.wind_ms,
      f.radii_km?.rmw,
      f.radii_km?.r34,
      f.radii_km?.r50,
      f.radii_km?.r64,
    ]);
  return [
    fields.join(","),
    ...rows.map((row) =>
      fields
        .map((_, i) => `"${String(row[i] ?? "").replaceAll('"', '""')}"`)
        .join(","),
    ),
  ].join("\n");
}
