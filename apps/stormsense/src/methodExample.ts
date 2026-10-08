export type MethodMode = "joint" | "encoder";
export interface MethodScalars {
  vmax_ms: number;
  rmw_km: number;
  r34_km: number;
  r50_km: number;
  r64_km: number;
}
export interface MethodExample {
  schema_version: number;
  storm: { id: string; name: string };
  sample_id: string;
  time: string;
  center: { lat: number; lon: number };
  track: { time: string; lat: number; lon: number }[];
  map_bounds: number[];
  channels: { id: string; image: string; sha256: string }[];
  context_channels: {
    id: string;
    label: string;
    short_label: string;
    image: string;
    sha256: string;
    display_min: number;
    display_max: number;
  }[];
  models: Record<
    MethodMode,
    { id: string; sha256: string; scalars: MethodScalars }
  >;
  field: {
    image: string;
    values: string;
    min_ms: number;
    max_ms: number;
    palette: string[];
  };
  grid: {
    width: number;
    height: number;
    resolution_degrees: number;
    bounds: number[];
    crs: string;
    valid_fraction: number;
  };
  provenance: {
    source: string;
    source_sha256: string;
    stats_sha256: string;
    ibtracs_sha256: string;
    selection: string;
    preprocessing: string;
  };
}
