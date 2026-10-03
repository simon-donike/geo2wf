import { memo, useEffect, useMemo, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import type { GeoJsonObject } from "geojson";
import type { Fix, Metrics, Storm, ImageHour } from "./types";
import { RADII } from "./presentation";
import { SatelliteOverlay } from "./SatelliteOverlay";
import { HourlyImagery } from "./HourlyImagery";

const NO_STORMS: Storm[] = [],
  NO_TRACK: Fix[] = [];
const NO_TRACKS: Record<string, Fix[]> = {};
type Position = { lat: number; lon: number };

function drawTrack(group: L.LayerGroup, fixes: Fix[]) {
  let section: L.LatLngTuple[] = [];
  const add = () => {
    if (section.length > 1)
      L.polyline(section, {
        color: "#38bfbd",
        weight: 2,
        opacity: 0.75,
        interactive: false,
        className: "storm-track",
      }).addTo(group);
  };
  fixes.forEach((fix, index) => {
    if (section.length && Math.abs(fix.lon - section.at(-1)![1]) > 180) {
      add();
      section = [];
    }
    section.push([fix.lat, fix.lon]);
    if (index % 4 === 0)
      L.circleMarker([fix.lat, fix.lon], {
        radius: 2.5,
        color: "#80d4ce",
        weight: 1,
        fillOpacity: 0.8,
        interactive: false,
      }).addTo(group);
  });
  add();
}

function ring(
  group: L.LayerGroup,
  position: Position,
  metric: (typeof RADII)[number],
  km: number,
) {
  const circle = L.circle([position.lat, position.lon], {
    radius: km * 1000,
    color: metric.color,
    weight: 1.4,
    opacity: 0.9,
    fillColor: metric.color,
    fillOpacity: metric.key === "r34_km" ? 0.045 : 0,
    dashArray: metric.dashed ? "4 4" : undefined,
    interactive: false,
    className: `wind-radius radius-${metric.key}`,
  }).addTo(group);
  circle.getElement()?.setAttribute("data-radius-km", String(km));
  return circle;
}

export const MapView = memo(function MapView({
  storms = NO_STORMS,
  track = NO_TRACK,
  tracks = NO_TRACKS,
  selected,
  radii,
  onSelect,
  detail = false,
  liveStorm,
  latestImagery = true,
  imagery,
  imageryTime,
  onImageryReady,
}: {
  storms?: Storm[];
  track?: Fix[];
  tracks?: Record<string, Fix[]>;
  selected?: Position | null;
  radii?: Metrics | null;
  onSelect?: (id: string) => void;
  detail?: boolean;
  liveStorm?: Storm;
  latestImagery?: boolean;
  imagery?: ImageHour[];
  imageryTime?: string;
  onImageryReady?: (ready: boolean) => void;
}) {
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<L.Map | null>(null);
  const [mapInstance, setMapInstance] = useState<L.Map | null>(null);
  const imageryStorms = useMemo(
    () =>
      detail
        ? liveStorm
          ? [liveStorm]
          : NO_STORMS
        : storms.filter((s) => s.active),
    [detail, liveStorm, storms],
  );
  const tracksLayer = useRef<L.LayerGroup | null>(null);
  const positionsLayer = useRef<L.LayerGroup | null>(null);
  const selectionLayer = useRef<L.LayerGroup | null>(null);
  const dot = useRef<L.CircleMarker | null>(null);
  const circles = useRef(new Map<string, L.Circle>());
  const fit = useRef("");
  const select = useRef(onSelect);
  select.current = onSelect;

  useEffect(() => {
    if (!container.current) return;
    const instance = L.map(container.current, {
      zoomControl: false,
      attributionControl: false,
      minZoom: 2,
      maxZoom: 9,
      scrollWheelZoom: false,
    }).setView([23, -112], 3);
    map.current = instance;
    fit.current = "";
    L.control.zoom({ position: "bottomright" }).addTo(instance);
    L.control
      .scale({ position: "bottomleft", imperial: false })
      .addTo(instance);
    L.control
      .attribution({ position: "bottomleft", prefix: false })
      .addAttribution("Natural Earth · NHC/CPHC")
      .addTo(instance);
    instance.createPane("land");
    instance.getPane("land")!.style.zIndex = "250";
    instance.createPane("satellite");
    instance.getPane("satellite")!.style.zIndex = "300";
    instance.getPane("satellite")!.style.pointerEvents = "none";
    setMapInstance(instance);
    const background = L.layerGroup().addTo(instance);
    for (let lon = -180; lon <= 180; lon += 20)
      L.polyline(
        [
          [-80, lon],
          [80, lon],
        ],
        {
          color: "#333e50",
          weight: 0.5,
          opacity: 0.4,
          interactive: false,
        },
      ).addTo(background);
    for (let lat = -60; lat <= 80; lat += 20)
      L.polyline(
        [
          [lat, -220],
          [lat, 180],
        ],
        {
          color: "#333e50",
          weight: 0.5,
          opacity: 0.4,
          interactive: false,
        },
      ).addTo(background);
    const controller = new AbortController();
    fetch("/land.geojson", { signal: controller.signal })
      .then((r) => r.json())
      .then((land: GeoJsonObject) =>
        L.geoJSON(land, {
          pane: "land",
          style: {
            color: "#424d59",
            fillColor: "#273140",
            fillOpacity: 1,
            weight: 0.7,
          },
          interactive: false,
        }).addTo(background),
      )
      .catch(() => {});
    tracksLayer.current = L.layerGroup().addTo(instance);
    positionsLayer.current = L.layerGroup().addTo(instance);
    selectionLayer.current = L.layerGroup().addTo(instance);
    const observer = new ResizeObserver(() => instance.invalidateSize());
    observer.observe(container.current);
    return () => {
      controller.abort();
      observer.disconnect();
      instance.remove();
      map.current = null;
      dot.current = null;
      circles.current.clear();
    };
  }, []);

  // Track paths persist during scrubbing; only changed track data rebuilds them.
  useEffect(() => {
    const layer = tracksLayer.current;
    if (!layer) return;
    layer.clearLayers();
    if (detail) drawTrack(layer, track);
    else
      storms.forEach((storm) => drawTrack(layer, tracks[storm.id] || NO_TRACK));
  }, [detail, track, storms, tracks]);

  useEffect(() => {
    const layer = positionsLayer.current;
    if (!layer) return;
    layer.clearLayers();
    for (const storm of storms) {
      const prediction = storm.latest_prediction;
      const position = prediction?.center || storm.latest_fix;
      if (!position) continue;
      if (prediction?.center && prediction.metrics)
        RADII.forEach((metric) => {
          const km = prediction.metrics![metric.key];
          if (Number.isFinite(km) && km > 0) ring(layer, position, metric, km);
        });
      const marker = L.circleMarker([position.lat, position.lon], {
        radius: 7,
        color: "#faf9f5",
        weight: 2,
        fillColor:
          storm.latest_fix?.classification === "HU" ? "#eead77" : "#42c6bb",
        fillOpacity: 1,
        className: "latest-storm-position",
      }).addTo(layer);
      const label = document.createElement("span");
      label.textContent = storm.name;
      marker.bindTooltip(label, {
        permanent: true,
        direction: "right",
        offset: [10, 0],
        className: "storm-label",
      });
      marker.on("click", () => select.current?.(storm.id));
    }
  }, [storms]);

  useEffect(() => {
    const layer = selectionLayer.current;
    if (!layer) return;
    if (!selected) {
      layer.clearLayers();
      dot.current = null;
      circles.current.clear();
      return;
    }
    const position: L.LatLngTuple = [selected.lat, selected.lon];
    for (const metric of RADII) {
      const km = radii?.[metric.key];
      let circle = circles.current.get(metric.key);
      if (km == null || !Number.isFinite(km) || km <= 0) {
        if (circle) layer.removeLayer(circle);
        circles.current.delete(metric.key);
      } else if (circle) {
        circle.setLatLng(position).setRadius(km * 1000);
        circle.getElement()?.setAttribute("data-radius-km", String(km));
      } else {
        circle = ring(layer, selected, metric, km);
        circles.current.set(metric.key, circle);
      }
    }
    if (dot.current) dot.current.setLatLng(position).bringToFront();
    else
      dot.current = L.circleMarker(position, {
        radius: 6,
        color: "#faf9f5",
        weight: 2,
        fillColor: "#019fa2",
        fillOpacity: 1,
        className: "selected-storm-position",
        interactive: false,
      }).addTo(layer);
  }, [selected, radii]);

  useEffect(() => {
    const instance = map.current;
    if (!instance) return;
    const key = detail ? "detail" : storms.map((s) => s.id).join(",");
    const ready = detail ? track.length > 0 : storms.every((s) => tracks[s.id]);
    const fitKey = `${key}:${ready ? "tracks" : "positions"}`;
    if (fitKey === fit.current) return;
    const positions: L.LatLngTuple[] = detail
      ? track.map((fix) => [fix.lat, fix.lon])
      : storms.flatMap((storm) => {
          const fixes = tracks[storm.id] || NO_TRACK;
          const latest = storm.latest_prediction?.center || storm.latest_fix;
          return [
            ...fixes.map((f) => [f.lat, f.lon] as L.LatLngTuple),
            ...(latest ? [[latest.lat, latest.lon] as L.LatLngTuple] : []),
          ];
        });
    if (detail && selected) positions.push([selected.lat, selected.lon]);
    if (positions.length) {
      instance.fitBounds(L.latLngBounds(positions), {
        padding: detail ? [45, 45] : [60, 65],
        maxZoom: detail ? 5 : 4,
        animate: false,
      });
      fit.current = fitKey;
    }
  }, [detail, track, tracks, storms, selected]);

  return (
    <>
      <div
        className={`map-canvas ${detail ? "detail-map" : ""}`}
        ref={container}
        aria-label={
          detail ? "Storm track map" : "Map of active tropical cyclones"
        }
      />
      {imagery !== undefined ? (
        <HourlyImagery
          map={mapInstance}
          frames={imagery}
          time={imageryTime}
          onReady={onImageryReady}
        />
      ) : (
        imageryStorms.length > 0 && (
          <SatelliteOverlay
            map={mapInstance}
            storms={imageryStorms}
            latest={latestImagery}
          />
        )
      )}
    </>
  );
});

export function RadiusLegend() {
  return (
    <div className="radius-legend" aria-label="Wind radius legend">
      {RADII.map((r) => (
        <span key={r.key}>
          <i
            style={{
              borderColor: r.color,
              borderStyle: r.dashed ? "dashed" : "solid",
            }}
          />
          {r.label}
        </span>
      ))}
      <small>StormSense radii · to scale</small>
    </div>
  );
}
