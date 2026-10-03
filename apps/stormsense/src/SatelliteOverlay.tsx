import { useEffect, useMemo, useState } from "react";
import L from "leaflet";
import type { Storm } from "./types";
import { age, stamp } from "./data";
import {
  CROP_KM,
  domainURL,
  latestCenter,
  availableProductTimes,
  satelliteCrops,
  satelliteSide,
  type SatelliteSide,
} from "./satellite";

interface Frame {
  side: SatelliteSide;
  times: string[];
}
export function SatelliteOverlay({
  map,
  storms,
  latest = true,
}: {
  map: L.Map | null;
  storms: Storm[];
  latest?: boolean;
}) {
  const [enabled, setEnabled] = useState(
    () => localStorage.getItem("stormsense-satellite") !== "off",
  );
  const [opacity, setOpacity] = useState(0.85);
  const [frames, setFrames] = useState<Frame[]>([]);
  const [fallback, setFallback] = useState<
    Record<string, { frame: string; index: number }>
  >({});
  const [errors, setErrors] = useState<SatelliteSide[]>([]);
  const [loads, setLoads] = useState<
    Record<string, "loading" | "ready" | "error">
  >({});
  const [revision, setRevision] = useState(0);
  const [now, setNow] = useState(Date.now);
  const targetKey = storms
    .map((s) => {
      const center = latestCenter(s);
      return `${s.id}:${center?.lat}:${center?.lon}`;
    })
    .join("|");
  const targets = useMemo(
    () =>
      storms.flatMap((storm) => {
        const center = latestCenter(storm);
        return center
          ? [
              {
                id: storm.id,
                name: storm.name,
                ...center,
                side: satelliteSide(center.lon),
              },
            ]
          : [];
      }),
    [targetKey],
  );
  const sidesKey = [...new Set(targets.map((t) => t.side))].sort().join(",");
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 60000);
    return () => clearInterval(timer);
  }, []);
  useEffect(() => {
    if (!enabled || !latest || !sidesKey) return;
    const controller = new AbortController();
    let busy = false;
    const refresh = async () => {
      if (busy) return;
      busy = true;
      const sides = sidesKey.split(",") as SatelliteSide[];
      const results = await Promise.allSettled(
        sides.map(async (side) => {
          const response = await fetch(domainURL(side), {
            signal: AbortSignal.any([
              controller.signal,
              AbortSignal.timeout(15000),
            ]),
          });
          if (!response.ok) throw Error(`Imagery metadata ${response.status}`);
          return { side, times: availableProductTimes(await response.text()) };
        }),
      );
      if (!controller.signal.aborted) {
        setFrames((previous) =>
          sides.flatMap((side, index) => {
            const result = results[index];
            const frame =
              result.status === "fulfilled"
                ? result.value
                : previous.find((f) => f.side === side);
            return frame ? [frame] : [];
          }),
        );
        setErrors(
          sides.filter((_, index) => results[index].status === "rejected"),
        );
        setNow(Date.now());
      }
      busy = false;
    };
    void refresh();
    const timer = setInterval(() => {
      if (!document.hidden) void refresh();
    }, 300000);
    return () => {
      controller.abort();
      clearInterval(timer);
    };
  }, [enabled, latest, sidesKey, revision]);
  const frameKey = frames
    .map((f) => `${f.side}:${f.times.join(",")}`)
    .join("|");
  const selectedFrame = (id: string, side: SatelliteSide) => {
    const frame = frames.find((f) => f.side === side);
    if (!frame) return undefined;
    const index =
      fallback[id]?.frame === frame.times[0] ? fallback[id].index : 0;
    return {
      side,
      time: frame.times[index],
      index,
      first: frame.times[0],
      canRetry: index + 1 < frame.times.length,
    };
  };
  const crops = useMemo(
    () =>
      targets.flatMap((target) => {
        const frame = selectedFrame(target.id, target.side);
        return frame
          ? satelliteCrops(target.lat, target.lon, frame.side, frame.time).map(
              (crop, i) => ({
                ...crop,
                id: `${target.id}:${i}`,
                time: frame.time,
                stormId: target.id,
                frame,
              }),
            )
          : [];
      }),
    [targetKey, frameKey, fallback],
  );
  // Keep imagery independent from selected-hour rendering and map zoom/panning.
  useEffect(() => {
    if (!map || !enabled || !latest) return;
    const group = L.layerGroup().addTo(map);
    let disposed = false;
    setLoads(Object.fromEntries(crops.map((crop) => [crop.id, "loading"])));
    const layers = crops.map((crop) => {
      const layer = L.imageOverlay(crop.url, crop.bounds, {
        pane: "satellite",
        opacity: 0,
        interactive: false,
        crossOrigin: "anonymous",
        className: "satellite-crop",
        alt: `GOES GeoColor image ${stamp(crop.time)}`,
      });
      const update = (status: "ready" | "error") => {
        if (disposed) return;
        if (status === "ready") layer.setOpacity(1);
        else {
          group.removeLayer(layer);
          if (crop.frame.canRetry)
            setFallback((prior) => ({
              ...prior,
              [crop.stormId]: {
                frame: crop.frame.first,
                index: crop.frame.index + 1,
              },
            }));
        }
        setLoads((prior) => ({ ...prior, [crop.id]: status }));
      };
      layer.on("load", () => {
        // A newly advertised GIBS time can return a transparent no-data PNG
        // while regional tiles are still arriving. Never label it a loaded image.
        const canvas = document.createElement("canvas");
        canvas.width = canvas.height = 32;
        const context = canvas.getContext("2d");
        if (!context) {
          update("error");
          return;
        }
        try {
          context.drawImage(layer.getElement()!, 0, 0, 32, 32);
          const pixels = context.getImageData(0, 0, 32, 32).data;
          update(
            pixels.some((alpha, i) => i % 4 === 3 && alpha > 0)
              ? "ready"
              : "error",
          );
        } catch {
          update("error");
        }
      });
      layer.on("error", () => update("error"));
      layer.addTo(group);
      layer.getElement()?.setAttribute("data-image-time", crop.time);
      layer.getElement()?.setAttribute("data-storm-id", crop.stormId);
      return layer;
    });
    const timeout = setTimeout(() => {
      layers.forEach((layer, i) => {
        if (
          !layer.getElement()?.complete ||
          !layer.getElement()?.naturalWidth
        ) {
          group.removeLayer(layer);
          layer.off();
          setLoads((prior) => ({ ...prior, [crops[i].id]: "error" }));
        }
      });
    }, 30000);
    return () => {
      disposed = true;
      clearTimeout(timeout);
      // Leaflet's remove event detaches its map zoom/view listeners. Keep that
      // internal handler until removal; clearing it first leaves stale layers.
      group.remove();
      layers.forEach((l) => l.off());
    };
  }, [map, crops, enabled, latest, revision]);
  useEffect(() => {
    if (!map) return;
    map.getPane("satellite")!.style.opacity = String(opacity);
  }, [map, opacity]);
  if (!storms.length) return null;
  const failedImages = Object.values(loads).some((s) => s === "error");
  return (
    <div className="satellite-controls">
      <div className="satellite-toolbar">
        <label>
          <input
            type="checkbox"
            checked={enabled}
            onChange={(event) => {
              setEnabled(event.target.checked);
              localStorage.setItem(
                "stormsense-satellite",
                event.target.checked ? "on" : "off",
              );
            }}
          />{" "}
          Latest GeoColor
        </label>
        <label className="satellite-opacity">
          Opacity
          <input
            aria-label="Satellite imagery opacity"
            type="range"
            min="0.2"
            max="1"
            step="0.05"
            value={opacity}
            disabled={!enabled || !latest}
            onChange={(event) => setOpacity(Number(event.target.value))}
          />
        </label>
      </div>
      <div className="satellite-focus">
        {targets.map((target) => (
          <button
            key={target.id}
            disabled={!enabled || !latest || !map || !crops.length}
            onClick={() => {
              const bounds = L.latLngBounds(
                crops
                  .filter((crop) => crop.stormId === target.id)
                  .flatMap((crop) => crop.bounds),
              );
              if (bounds.isValid())
                map?.fitBounds(bounds, {
                  padding: [12, 12],
                  maxZoom: 7,
                  animate: false,
                });
            }}
          >
            Focus {target.name} image
          </button>
        ))}
      </div>
      <div className="satellite-status" role="status">
        {!enabled ? (
          "Satellite imagery is off."
        ) : !latest ? (
          "Latest imagery is hidden while browsing earlier hours."
        ) : targets.length === 0 ? (
          "A current storm position is needed for imagery."
        ) : (
          <>
            {targets.map((target) => {
              const frame = selectedFrame(target.id, target.side);
              const states = Object.entries(loads)
                .filter(([id]) => id.startsWith(target.id + ":"))
                .map(([, status]) => status);
              return (
                <span
                  key={target.id}
                  className={
                    frame && now - Date.parse(frame.time) > 90 * 60000
                      ? "satellite-delayed"
                      : ""
                  }
                >
                  {target.name} · GOES-{target.side} ·{" "}
                  {frame ? (
                    <time dateTime={frame.time}>
                      {stamp(frame.time)} · {age(frame.time, now)}
                    </time>
                  ) : errors.includes(target.side) ? (
                    "Imagery unavailable"
                  ) : (
                    "Checking latest image…"
                  )}
                  {frame && states.includes("loading") ? " · Loading…" : ""}
                  {frame && states.includes("error")
                    ? " · Image unavailable"
                    : ""}
                  {frame && errors.includes(target.side)
                    ? " · Refresh failed; previous image time shown"
                    : ""}
                </span>
              );
            })}
            {(errors.length > 0 || failedImages) && (
              <button
                onClick={() => {
                  setFallback({});
                  setRevision((r) => r + 1);
                }}
              >
                Retry imagery
              </button>
            )}
          </>
        )}
      </div>
      <p>
        ~{CROP_KM.toLocaleString()} km crop · Colour by day, infrared blend at
        night.
        <br />
        <a
          href="https://worldview.earthdata.nasa.gov/"
          target="_blank"
          rel="noreferrer"
        >
          NASA GIBS / NOAA / CIRA ↗
        </a>{" "}
        · Image time is separate from the estimate.
      </p>
    </div>
  );
}
