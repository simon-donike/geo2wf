import type { Storm } from "./types";

export type SatelliteSide = "East" | "West";
export const GIBS = "https://gibs.earthdata.nasa.gov";
export const CROP_KM = 1800;
const R = 6378137,
  WORLD = Math.PI * R;
export const satelliteSide = (lon: number): SatelliteSide =>
  Math.cos(((lon + 137) * Math.PI) / 180) >
  Math.cos(((lon + 75.2) * Math.PI) / 180)
    ? "West"
    : "East";
export const latestCenter = (storm: Storm) =>
  storm.latest_prediction?.center || storm.latest_fix;

export function domainURL(side: SatelliteSide, now = Date.now()) {
  // Rounded endpoints make requests cacheable across visitors for five minutes.
  const end = Math.floor(now / 300000) * 300000;
  const iso = (value: number) =>
    new Date(value).toISOString().replace(".000Z", "Z");
  return `${GIBS}/wmts/epsg3857/best/1.0.0/GOES-${side}_ABI_GeoColor/default/GoogleMapsCompatible_Level7/all/${iso(end - 3 * 3600000)}--${iso(end)}.xml`;
}

/** Select a reported product time, never a guessed/default or future frame. */
export function availableProductTimes(xml: string, now = Date.now()) {
  const domain = xml.match(/<Domain>\s*([^<]*)\s*<\/Domain>/)?.[1];
  const times = (domain || "")
    .split(",")
    .flatMap((value) => {
      const [startText, endText, period] = value.trim().split("/");
      const start = Date.parse(startText);
      if (!Number.isFinite(start) || start > now) return [];
      if (!endText) return [start];
      const end = Date.parse(endText);
      const minutes = Number(period?.match(/^PT(\d+)M$/)?.[1]);
      if (!Number.isFinite(end) || end < start || !minutes) return [];
      const step = minutes * 60000;
      const last =
        start + Math.floor((Math.min(end, now) - start) / step) * step;
      return Array.from(
        { length: Math.min(3, Math.floor((last - start) / step) + 1) },
        (_, i) => last - i * step,
      );
    })
    .filter((time) => now - time <= 3 * 3600000);
  if (!times.length) throw Error("No recent GeoColor frame is available.");
  return [...new Set(times)]
    .sort((a, b) => b - a)
    .slice(0, 3)
    .map((time) => new Date(time).toISOString().replace(".000Z", "Z"));
}
export const latestProductTime = (xml: string, now = Date.now()) =>
  availableProductTimes(xml, now)[0];

export interface SatelliteCrop {
  url: string;
  bounds: [[number, number], [number, number]];
}

/** WMS and Leaflet both use EPSG:3857; geographic bounds locate the same pixels.
 * Split at the dateline, retaining continuous display longitudes on the map.
 */
export function satelliteCrops(
  lat: number,
  lon: number,
  side: SatelliteSide,
  time: string,
): SatelliteCrop[] {
  if (
    !Number.isFinite(lat) ||
    !Number.isFinite(lon) ||
    Math.abs(lat) > 75 ||
    Math.abs(lon) > 180
  )
    return [];
  const x = (R * lon * Math.PI) / 180;
  const y = R * Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360));
  const half = (CROP_KM * 500) / Math.cos((lat * Math.PI) / 180);
  const south = Math.max(-WORLD, y - half),
    north = Math.min(WORLD, y + half);
  const unproject = (px: number, py: number): [number, number] => [
    ((2 * Math.atan(Math.exp(py / R)) - Math.PI / 2) * 180) / Math.PI,
    ((px / R) * 180) / Math.PI,
  ];
  const parts: SatelliteCrop[] = [];
  for (
    let world = Math.floor((x - half + WORLD) / (2 * WORLD));
    world <= Math.floor((x + half + WORLD) / (2 * WORLD));
    world++
  ) {
    const left = Math.max(x - half, -WORLD + world * 2 * WORLD);
    const right = Math.min(x + half, WORLD + world * 2 * WORLD);
    if (right <= left) continue;
    const params = new URLSearchParams({
      SERVICE: "WMS",
      VERSION: "1.1.1",
      REQUEST: "GetMap",
      LAYERS: `GOES-${side}_ABI_GeoColor`,
      STYLES: "",
      SRS: "EPSG:3857",
      BBOX: [left - world * 2 * WORLD, south, right - world * 2 * WORLD, north]
        .map((v) => v.toFixed(2))
        .join(","),
      WIDTH: String(
        Math.max(1, Math.round((1024 * (right - left)) / (half * 2))),
      ),
      HEIGHT: "1024",
      FORMAT: "image/png",
      TRANSPARENT: "TRUE",
      TIME: time,
    });
    parts.push({
      url: `${GIBS}/wms/epsg3857/best/wms.cgi?${params}`,
      bounds: [unproject(left, south), unproject(right, north)],
    });
  }
  return parts;
}
