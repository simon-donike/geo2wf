import { memo, useEffect, useId, useMemo, useRef } from "react";
import { KNOT, stamp } from "./data";
import type { RIInterval } from "./rapidIntensification";
export interface Point {
  time: string;
  value: number | null;
}
export interface Line {
  label: string;
  color: string;
  points: Point[];
  dashed?: boolean;
  gapHours?: number;
}
function prepare(lines: Line[]) {
  const all = lines
    .flatMap((l) => l.points)
    .filter((p) => p.value != null && Number.isFinite(p.value));
  if (!all.length) return null;
  const start = Math.min(...all.map((p) => Date.parse(p.time))),
    rawEnd = Math.max(...all.map((p) => Date.parse(p.time))),
    end = Math.max(rawEnd, start + 3600000);
  const top = Math.max(
    20,
    Math.ceil(Math.max(...all.map((p) => p.value!)) / 20) * 20,
  );
  const x = (t: string) => 50 + ((Date.parse(t) - start) / (end - start)) * 670,
    y = (v: number) => 208 - (v / top) * 174;
  const path = (line: Line) => {
    let previous: number | undefined;
    return line.points
      .map((p) => {
        if (p.value == null) {
          previous = undefined;
          return "";
        }
        const time = Date.parse(p.time);
        const cmd =
          previous === undefined ||
          time - previous > (line.gapHours ?? 1.5) * 3600000
            ? "M"
            : "L";
        previous = time;
        return `${cmd}${x(p.time).toFixed(2)},${y(p.value).toFixed(2)}`;
      })
      .join(" ");
  };
  const paths = lines.map((line) => ({ ...line, path: path(line) }));
  const dots = lines.flatMap((line) =>
    line.points.flatMap((point, index) => {
      const connected = (other?: Point) =>
        other?.value != null &&
        Math.abs(Date.parse(other.time) - Date.parse(point.time)) <=
          (line.gapHours ?? 1.5) * 3600000;
      return point.value != null &&
        !connected(line.points[index - 1]) &&
        !connected(line.points[index + 1])
        ? [
            {
              key: line.label + point.time,
              x: x(point.time),
              y: y(point.value),
              color: line.color,
            },
          ]
        : [];
    }),
  );
  return { start, end, top, x, y, paths, dots };
}
export const Chart = memo(function Chart({
  lines,
  selected,
  onTime,
  onSelect,
  unit,
  title,
  thresholds = [],
  rapidIntervals = [],
}: {
  lines: Line[];
  selected?: string;
  onTime?: (time: string | null) => void;
  onSelect?: (time: string) => void;
  unit: string;
  title: string;
  thresholds?: { value: number; label: string; color: string }[];
  rapidIntervals?: RIInterval[];
}) {
  const clip = useId().replaceAll(":", "");
  const geometry = useMemo(() => prepare(lines), [lines]);
  const frame = useRef<number | null>(null),
    hoverTime = useRef<string | null>(null);
  useEffect(
    () => () => {
      if (frame.current != null) cancelAnimationFrame(frame.current);
    },
    [],
  );
  if (!geometry)
    return (
      <div className="chart-empty">No values available for this period.</div>
    );
  const { start, end, top, x, y, paths, dots } = geometry;
  const cursor = selected ? x(selected) : null;
  const visibleIntervals = rapidIntervals.filter(
    (interval) =>
      interval.source === "official" &&
      Date.parse(interval.end) > start &&
      Date.parse(interval.start) < end,
  );
  const intervalDescription = (interval: RIInterval) =>
    `${interval.source === "official" ? "NHC/CPHC reference" : "StormSense estimate"}: ${stamp(interval.start)} to ${stamp(interval.end)}. Largest 24-hour increase: +${(interval.maxChangeMs / (unit === "m/s" ? 1 : KNOT)).toFixed(1)} ${unit === "m/s" ? "m/s" : "kt"}. ${interval.windows} qualifying 24-hour window${interval.windows === 1 ? "" : "s"}.`;
  return (
    <div className="chart-wrap">
      <svg
        viewBox="0 0 750 252"
        role="img"
        aria-label={`${title}, ${unit}`}
        onClick={(event) => {
          if (!onSelect) return;
          const bounds = event.currentTarget.getBoundingClientRect();
          const px = ((event.clientX - bounds.left) / bounds.width) * 750;
          onSelect(
            new Date(
              start + Math.max(0, Math.min(1, (px - 50) / 670)) * (end - start),
            ).toISOString(),
          );
        }}
        onPointerLeave={() => {
          if (frame.current != null) cancelAnimationFrame(frame.current);
          frame.current = null;
          onTime?.(null);
        }}
        onPointerMove={(event) => {
          if (!onTime) return;
          const bounds = event.currentTarget.getBoundingClientRect();
          const px = ((event.clientX - bounds.left) / bounds.width) * 750;
          const t =
            start + Math.max(0, Math.min(1, (px - 50) / 670)) * (end - start);
          hoverTime.current = new Date(
            Math.round(t / 3600000) * 3600000,
          ).toISOString();
          if (frame.current == null)
            frame.current = requestAnimationFrame(() => {
              frame.current = null;
              onTime(hoverTime.current);
            });
        }}
      >
        <defs>
          <clipPath id={clip}>
            <rect x="49" y="25" width="673" height="185" />
          </clipPath>
          <pattern
            id={`${clip}-ri`}
            width="7"
            height="7"
            patternUnits="userSpaceOnUse"
            patternTransform="rotate(35)"
          >
            <rect width="7" height="7" fill="#40c7bd" fillOpacity="0.07" />
            <line
              x1="0"
              y1="0"
              x2="0"
              y2="7"
              stroke="#40c7bd"
              strokeOpacity="0.25"
              strokeWidth="2"
            />
          </pattern>
        </defs>
        <g clipPath={`url(#${clip})`} className="ri-bands">
          {visibleIntervals.map((interval) => (
            <rect
              key={`${interval.source}-${interval.start}`}
              className="ri-band"
              data-source={interval.source}
              data-start={interval.start}
              data-end={interval.end}
              x={x(interval.start)}
              y="27"
              width={x(interval.end) - x(interval.start)}
              height="181"
              fill={
                interval.source === "official" ? "#e9ac7a" : `url(#${clip}-ri)`
              }
              fillOpacity={interval.source === "official" ? 0.16 : 1}
            >
              <title>{intervalDescription(interval)}</title>
            </rect>
          ))}
        </g>
        {[0, 1, 2, 3, 4].map((i) => (
          <g key={i}>
            <line
              x1="50"
              x2="720"
              y1={y((top * i) / 4)}
              y2={y((top * i) / 4)}
              stroke="#343849"
              strokeDasharray={i ? "3 5" : undefined}
            />
            <text
              x="38"
              y={y((top * i) / 4) + 4}
              textAnchor="end"
              className="axis-label"
            >
              {Math.round((top * i) / 4)}
            </text>
          </g>
        ))}
        {[0, 1, 2, 3, 4].map((i) => (
          <text
            key={i}
            x={50 + (670 * i) / 4}
            y="232"
            textAnchor={i === 0 ? "start" : i === 4 ? "end" : "middle"}
            className="axis-label"
          >
            {new Date(start + ((end - start) * i) / 4).toLocaleDateString(
              "en-GB",
              { timeZone: "UTC", day: "numeric", month: "short" },
            )}
          </text>
        ))}
        <text x="50" y="17" className="axis-label">
          {unit}
        </text>
        {thresholds
          .filter((t) => t.value > 0 && t.value <= top)
          .map((t) => (
            <g
              key={t.label}
              className="category-threshold"
              data-value={t.value}
            >
              <line
                x1="50"
                x2="720"
                y1={y(t.value)}
                y2={y(t.value)}
                stroke={t.color}
                strokeOpacity="0.55"
                strokeDasharray="6 5"
              />
              <text
                x="56"
                y={y(t.value) - 4}
                textAnchor="start"
                fill={t.color}
                className="category-axis-label"
                stroke="#171c2f"
                strokeWidth="3"
                paintOrder="stroke"
              >
                {t.label}
              </text>
            </g>
          ))}
        <g clipPath={`url(#${clip})`}>
          {paths.map((line) => (
            <path
              key={line.label}
              d={line.path}
              stroke={line.color}
              strokeWidth="2.2"
              strokeDasharray={line.dashed ? "5 5" : undefined}
              fill="none"
              vectorEffect="non-scaling-stroke"
            />
          ))}
          {cursor !== null && cursor >= 50 && cursor <= 720 && (
            <line
              x1={cursor}
              x2={cursor}
              y1="27"
              y2="208"
              stroke="#a8aeb8"
              strokeWidth="1"
              strokeDasharray="3 4"
            />
          )}
          {dots.map((dot) => (
            <circle
              key={dot.key}
              cx={dot.x}
              cy={dot.y}
              r="3"
              fill={dot.color}
            />
          ))}
        </g>
      </svg>
      <div className="chart-legend">
        {lines.map((line) => (
          <span key={line.label}>
            <i style={{ background: line.color }} />
            {line.label}
          </span>
        ))}
      </div>
      {thresholds.length > 0 && (
        <div className="category-key" aria-label="Wind category thresholds">
          {thresholds
            .filter((t) => t.value > 0 && t.value <= top)
            .map((t) => (
              <span key={t.label} style={{ color: t.color }}>
                <i />
                {t.label}
              </span>
            ))}
        </div>
      )}
      {visibleIntervals.length > 0 && (
        <div className="ri-key" aria-label="Rapid intensification shading">
          <details>
            <summary>
              <span className="ri-definition">
                Rapid intensification · ≥30 kt / 24 h
              </span>
              {(["official", "model"] as const)
                .filter((source) =>
                  visibleIntervals.some((interval) => interval.source === source),
                )
                .map((source) => (
                  <span key={source}>
                    <i className={`ri-swatch ri-${source}`} />
                    {source === "official"
                      ? "NHC/CPHC winds"
                      : "StormSense estimate"}
                  </span>
                ))}
              <span className="ri-toggle">Shaded periods</span>
            </summary>
            <div className="ri-details">
              <p>
                Overlapping qualifying 24-hour windows are joined. Calculated from
                original winds; forecasts and incomplete windows are excluded.
                Shading on the radii chart marks the same intensity events.
              </p>
              <ul>
                {visibleIntervals.map((interval) => (
                  <li key={`${interval.source}-${interval.start}`}>
                    {intervalDescription(interval)}
                  </li>
                ))}
              </ul>
            </div>
          </details>
        </div>
      )}
    </div>
  );
});
