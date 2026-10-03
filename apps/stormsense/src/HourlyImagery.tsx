import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import L from "leaflet";
import type { ImageHour } from "./types";
import { dataUrl, stamp } from "./data";
import { ImageCache, decodeBlob } from "./imageCache";
import { ImageArchive } from "./imageArchive";

export function HourlyImagery({
  map,
  frames,
  time,
  onReady,
}: {
  map: L.Map | null;
  frames: ImageHour[];
  time?: string;
  onReady?: (ready: boolean) => void;
}) {
  const [enabled, setEnabled] = useState(
    () => localStorage.getItem("stormsense-satellite") !== "off",
  );
  const [opacity, setOpacity] = useState(0.85);
  const [, render] = useState(0);
  const cache = useRef<ImageCache | null>(null);
  const archive = useRef<ImageArchive | null>(null);
  const group = useRef<L.LayerGroup | null>(null);
  const readyFrames = useMemo(
    () => frames.filter((f) => f.status === "ready"),
    [frames],
  );
  const lookup = useMemo(
    () => new Map(frames.map((f) => [f.time, f])),
    [frames],
  );
  const frame = time ? lookup.get(time) : undefined;
  useEffect(() => {
    let update: number | null = null;
    const notify = () => {
      if (update === null)
        update = requestAnimationFrame(() => {
          update = null;
          render((n) => n + 1);
        });
    };
    const downloads = new ImageArchive(notify);
    const instance = new ImageCache(
      () => render((n) => n + 1),
      async (url, signal) =>
        decodeBlob(await downloads.read(url, signal), signal),
    );
    archive.current = downloads;
    cache.current = instance;
    render((n) => n + 1);
    return () => {
      group.current?.remove();
      group.current = null;
      instance.close();
      downloads.close();
      if (update !== null) cancelAnimationFrame(update);
      cache.current = null;
      archive.current = null;
    };
  }, []);
  const allImages = useMemo(
    () => [
      ...readyFrames.flatMap((f) => f.parts.map((p) => dataUrl(p.preview))),
      ...readyFrames.flatMap((f) => f.parts.map((p) => dataUrl(p.image))),
    ],
    [readyFrames],
  );
  const wanted = useMemo(() => {
    if (!enabled || !time) return [];
    const nearest = [...readyFrames]
      .sort(
        (a, b) =>
          Math.abs(Date.parse(a.time) - Date.parse(time)) -
          Math.abs(Date.parse(b.time) - Date.parse(time)),
      )
      .slice(0, 25);
    // Decode the selected neighbourhood first. Every other image downloads to
    // the compressed archive, so later jumps need only a small local decode.
    return nearest.flatMap((f, i) =>
      f.parts.flatMap((p) =>
        i < 9 ? [dataUrl(p.preview), dataUrl(p.image)] : [dataUrl(p.preview)],
      ),
    );
  }, [readyFrames, time, enabled]);
  useEffect(() => {
    archive.current?.set(allImages, wanted, !enabled);
    cache.current?.window(wanted);
  }, [allImages, wanted, enabled]);
  const preloaded = readyFrames.filter((f) =>
    f.parts.every((p) => archive.current?.state(dataUrl(p.image)) === "ready"),
  ).length;
  const loaded =
    frame?.status === "ready"
      ? frame.parts.map((part) => {
          const full = cache.current?.get(dataUrl(part.image));
          const preview = cache.current?.get(dataUrl(part.preview));
          return {
            part,
            entry: full?.state === "ready" ? full : preview,
            resolution: full?.state === "ready" ? "full" : "preview",
            failed: full?.state === "error" && preview?.state === "error",
            fullFailed: full?.state === "error",
          };
        })
      : [];
  const complete =
    loaded.length > 0 && loaded.every((p) => p.entry?.state === "ready");
  const failed = loaded.some((p) => p.failed);
  const fullFailed = loaded.some((p) => p.fullFailed);
  const ready = !enabled || frame?.status !== "ready" || complete || failed;
  useEffect(() => {
    onReady?.(ready);
  }, [ready, time, onReady]);
  const displayKey =
    enabled && complete
      ? `${frame?.time}|${loaded.map((p) => p.resolution).join(",")}`
      : "";
  useLayoutEffect(() => {
    // No stale picture at a different selected time, including gaps or a cold
    // jump. All frames are decoded before the map gets their image elements.
    group.current?.remove();
    group.current = null;
    if (!map || !displayKey || !frame) return;
    const layer = L.layerGroup().addTo(map);
    group.current = layer;
    loaded.forEach(({ part, entry, resolution }) => {
      const [west, south, east, north] = part.display_bbox;
      const img = entry!.decoded!.image;
      img.setAttribute("data-slot-time", frame.time);
      img.setAttribute("data-image-time", frame.acquired_at!);
      img.setAttribute("data-resolution", resolution);
      L.imageOverlay(
        img,
        [
          [south, west],
          [north, east],
        ],
        {
          pane: "satellite",
          interactive: false,
          className: "satellite-crop hourly-satellite-crop",
          alt: `GOES GeoColor ${stamp(frame.acquired_at!)}`,
        },
      ).addTo(layer);
    });
    return () => {
      layer.remove();
      if (group.current === layer) group.current = null;
    };
  }, [map, displayKey, frame]);
  useEffect(() => {
    if (map) map.getPane("satellite")!.style.opacity = String(opacity);
  }, [map, opacity]);
  const focus = () => {
    if (!frame) return;
    const bounds = L.latLngBounds(
      frame.parts.flatMap((p) => {
        const [w, s, e, n] = p.display_bbox;
        return [
          [s, w],
          [n, e],
        ] as L.LatLngTuple[];
      }),
    );
    if (bounds.isValid())
      map?.fitBounds(bounds, { padding: [12, 12], maxZoom: 7, animate: false });
  };
  return (
    <div
      className="satellite-controls hourly-imagery"
      data-buffer-ready={ready}
      data-cache-bytes={cache.current?.decodedBytes ?? 0}
      data-compressed-bytes={archive.current?.compressedBytes ?? 0}
      data-preload-ready={preloaded}
      data-preload-total={readyFrames.length}
    >
      <div className="satellite-toolbar">
        <label>
          <input
            type="checkbox"
            checked={enabled}
            onChange={(e) => {
              setEnabled(e.target.checked);
              localStorage.setItem(
                "stormsense-satellite",
                e.target.checked ? "on" : "off",
              );
            }}
          />
          Satellite imagery
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
            disabled={!enabled}
            onChange={(e) => setOpacity(Number(e.target.value))}
          />
        </label>
      </div>
      <div className="satellite-focus">
        <button
          onClick={focus}
          disabled={!enabled || frame?.status !== "ready"}
        >
          Focus image
        </button>
        {frame?.metadata && (
          <a href={dataUrl(frame.metadata)} target="_blank" rel="noreferrer">
            Georeferencing & source ↗
          </a>
        )}
      </div>
      <div className="satellite-status" role="status">
        {!enabled ? (
          "Satellite imagery is off."
        ) : frame?.status === "ready" ? (
          <>
            <span>
              GOES-{frame.satellite} · Image{" "}
              <time dateTime={frame.acquired_at}>
                {stamp(frame.acquired_at!)}
              </time>
            </span>
            <span>
              {failed
                ? "Image could not be loaded."
                : !complete
                  ? "Loading selected image…"
                  : loaded.some((p) => p.resolution === "preview")
                    ? fullFailed
                      ? "Preview · full resolution unavailable"
                      : "Preview · sharpening…"
                    : "Hourly image · full resolution"}
            </span>
            {(failed || fullFailed) && (
              <button
                onClick={() => {
                  archive.current?.retry();
                  cache.current?.retry();
                }}
              >
                Retry image
              </button>
            )}
          </>
        ) : (
          <span>
            Image unavailable for this hour
            {frame?.reason
              ? ` · ${frame.reason.replaceAll("_", " ")}`
              : " · not processed yet"}
            .
          </span>
        )}
      </div>
      <p>
        ~1,800 km crop ·{" "}
        {preloaded === readyFrames.length
          ? `All ${preloaded.toLocaleString()} images loaded`
          : `${preloaded.toLocaleString()} / ${readyFrames.length.toLocaleString()} images loaded`}
        <br />
        NASA GIBS / NOAA / CIRA · Colour by day, infrared blend at night.
      </p>
    </div>
  );
}
