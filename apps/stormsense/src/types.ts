export type Basin = "AL" | "EP" | "CP";
export type Kind = "live" | "hindcast";
export interface Metrics {
  vmax_ms: number;
  rmw_km: number;
  r34_km: number;
  r50_km: number;
  r64_km: number;
}
export interface Fix {
  time: string;
  lat: number;
  lon: number;
  wind_ms: number | null;
  classification: string;
  radii_km?: Record<string, number>;
  source: string;
  url?: string;
}
export interface RecordHour {
  storm_id: string;
  time: string;
  kind: Kind;
  model_version: string;
  generated_at: string;
  status: "ready" | "gap";
  metrics: Metrics | null;
  reason: string | null;
  center: {
    lat: number;
    lon: number;
    method: string;
    fix_time: string;
    age_hours: number;
  } | null;
  imagery?: {
    end: string;
    retrieved_at?: string;
    satellite: number;
    valid_fraction: number;
    url: string;
  };
}
export interface Forecast {
  storm_id: string;
  anchor_time: string;
  generated_at: string;
  kind: Kind;
  model_version: string;
  input_times: string[];
  input_kinds?: Kind[];
  input_generated_at?: string[];
  input_vmax_ms: number[];
  experimental: true;
  predictions: { lead_hours: number; valid_time: string; vmax_ms: number }[];
}
export interface Storm {
  id: string;
  name: string;
  basin: Basin;
  active: boolean;
  start: string;
  end: string;
  peak_category: number | null;
  advisory: Fix | null;
  latest_fix: Fix | null;
  latest_prediction: RecordHour | null;
  metrics: Metrics | null;
  change_24h_ms: number | null;
  change_24h_reference_kind?: Kind | null;
  prediction_count: number;
  gap_count: number;
  record_count: number;
  expected_count: number;
  pending_count: number;
  series: string;
}
export interface Series {
  schema_version: 1;
  storm_id: string;
  track: Fix[];
  records: RecordHour[];
  forecasts: Forecast[];
  imagery?: ImageHour[];
}
export interface ImageHour {
  storm_id: string;
  time: string;
  status: "ready" | "gap";
  reason: string | null;
  version: string;
  acquired_at?: string;
  checked_at: string;
  satellite?: "East" | "West";
  metadata?: string;
  parts: {
    image: string;
    preview: string;
    sidecar: string;
    preview_sidecar: string;
    bbox: [number, number, number, number];
    display_bbox: [number, number, number, number];
    sha256: string;
    bytes: number;
    preview_bytes: number;
  }[];
}
export interface Catalog {
  schema_version: 1;
  release: string;
  generated_at: string;
  window: { start: string; end: string };
  reports?: { coverage: string; evaluation: string | null };
  models: Record<string, { id: string; version: string; sha256: string }>;
  source_status: {
    discovery: { last_success?: string; error?: string | null } | null;
    update: { last_completed?: string } | null;
    backfill: { complete?: boolean } | null;
  };
  coverage: {
    expected: number;
    predictions: number;
    gaps: number;
    pending: number;
  };
  storms: Storm[];
}
