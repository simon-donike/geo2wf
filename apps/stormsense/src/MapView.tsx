import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import type { GeoJsonObject } from "geojson";
import type { Fix, Storm } from "./types";

export function MapView({
  storms = [],
  track = [],
  selected,
  onSelect,
  detail = false,
}: {
  storms?: Storm[];
  track?: Fix[];
  selected?: { lat: number; lon: number } | null;
  onSelect?: (id: string) => void;
  detail?: boolean;
}) {
  const container = useRef<HTMLDivElement>(null),
    map = useRef<L.Map | null>(null),
    layer = useRef<L.LayerGroup | null>(null),
    fit = useRef("");
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
      .attribution({ position: "bottomleft", prefix: false })
      .addAttribution("Natural Earth · NHC/CPHC")
      .addTo(instance);
    instance.createPane("land");
    instance.getPane("land")!.style.zIndex = "250";
    const background = L.layerGroup().addTo(instance);
    for (let lon = -180; lon <= 180; lon += 20)
      L.polyline(
        [
          [-80, lon],
          [80, lon],
        ],
        { color: "#333e50", weight: 0.5, opacity: 0.4, interactive: false },
      ).addTo(background);
    for (let lat = -60; lat <= 80; lat += 20)
      L.polyline(
        [
          [lat, -220],
          [lat, 180],
        ],
        { color: "#333e50", weight: 0.5, opacity: 0.4, interactive: false },
      ).addTo(background);
    const controller = new AbortController();
    fetch("/land.geojson", { signal: controller.signal })
      .then((r) => r.json())
      .then((land: GeoJsonObject) => {
        L.geoJSON(land, {
          pane: "land",
          style: {
            color: "#424d59",
            fillColor: "#273140",
            fillOpacity: 1,
            weight: 0.7,
          },
          interactive: false,
        }).addTo(background);
      })
      .catch(() => {});
    layer.current = L.layerGroup().addTo(instance);
    const observer = new ResizeObserver(() => instance.invalidateSize());
    observer.observe(container.current);
    return () => {
      controller.abort();
      observer.disconnect();
      instance.remove();
      map.current = null;
    };
  }, []);
  useEffect(() => {
    const instance = map.current,
      group = layer.current;
    if (!instance || !group) return;
    group.clearLayers();
    const positions: L.LatLngExpression[] = [];
    if (track.length) {
      const points = track.map((f) => [f.lat, f.lon] as [number, number]);
      // Break tracks at the antimeridian instead of drawing across the entire map.
      let section: [number, number][] = [];
      for (const point of points) {
        if (
          section.length &&
          Math.abs(point[1] - section[section.length - 1][1]) > 180
        ) {
          L.polyline(section, {
            color: "#38bfbd",
            weight: 2,
            opacity: 0.7,
          }).addTo(group);
          section = [];
        }
        section.push(point);
      }
      L.polyline(section, { color: "#38bfbd", weight: 2, opacity: 0.7 }).addTo(
        group,
      );
      track.forEach((f, i) => {
        if (i % 4 === 0)
          L.circleMarker([f.lat, f.lon], {
            radius: 2.5,
            color: "#80d4ce",
            weight: 1,
            fillOpacity: 0.8,
          }).addTo(group);
      });
      positions.push(...points);
    }
    for (const storm of storms) {
      const fix = storm.latest_fix;
      if (!fix) continue;
      const marker = L.circleMarker([fix.lat, fix.lon], {
        radius: 8,
        color: "#131b2c",
        weight: 3,
        fillColor: fix.classification === "HU" ? "#eead77" : "#42c6bb",
        fillOpacity: 1,
      }).addTo(group);
      const label = document.createElement("span");
      label.textContent = storm.name;
      marker.bindTooltip(label, {
        permanent: true,
        direction: "right",
        offset: [10, 0],
        className: "storm-label",
      });
      marker.on("click", () => onSelect?.(storm.id));
      positions.push([fix.lat, fix.lon]);
    }
    if (selected) {
      L.circleMarker([selected.lat, selected.lon], {
        radius: 17,
        color: "#46d0c5",
        weight: 1,
        fillColor: "#46d0c5",
        fillOpacity: 0.08,
      }).addTo(group);
      L.circleMarker([selected.lat, selected.lon], {
        radius: 6,
        color: "#faf9f5",
        weight: 2,
        fillColor: "#019fa2",
        fillOpacity: 1,
      }).addTo(group);
    }
    const key = detail
      ? JSON.stringify(track.map((f) => [f.lat, f.lon]))
      : storms.map((s) => s.id).join(",");
    if (key !== fit.current && positions.length) {
      instance.fitBounds(L.latLngBounds(positions), {
        padding: detail ? [45, 45] : [85, 90],
        maxZoom: detail ? 5 : 4,
        animate: false,
      });
      fit.current = key;
    }
  }, [storms, track, selected, onSelect, detail]);
  return (
    <div
      className={`map-canvas ${detail ? "detail-map" : ""}`}
      ref={container}
      aria-label={
        detail ? "Storm track map" : "Map of active tropical cyclones"
      }
    />
  );
}
