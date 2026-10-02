import { useId } from "react";
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
export function Chart({
  lines,
  selected,
  onTime,
  onSelect,
  unit,
  title,
}: {
  lines: Line[];
  selected?: string;
  onTime?: (time: string | null) => void;
  onSelect?: (time: string) => void;
  unit: string;
  title: string;
}) {
  const clip = useId().replaceAll(":", "");
  const all = lines
    .flatMap((l) => l.points)
    .filter((p) => p.value != null && Number.isFinite(p.value));
  if (!all.length)
    return (
      <div className="chart-empty">No values available for this period.</div>
    );
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
  const cursor = selected ? x(selected) : null;
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
        onPointerLeave={() => onTime?.(null)}
        onPointerMove={(event) => {
          if (!onTime) return;
          const bounds = event.currentTarget.getBoundingClientRect();
          const px = ((event.clientX - bounds.left) / bounds.width) * 750;
          const t =
            start + Math.max(0, Math.min(1, (px - 50) / 670)) * (end - start);
          onTime(new Date(t).toISOString());
        }}
      >
        <defs>
          <clipPath id={clip}>
            <rect x="49" y="25" width="673" height="185" />
          </clipPath>
        </defs>
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
        <g clipPath={`url(#${clip})`}>
          {lines.map((line) => (
            <path
              key={line.label}
              d={path(line)}
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
          {lines.flatMap((line) =>
            line.points.map((point, index) => {
              const connected = (other?: Point) =>
                other?.value != null &&
                Math.abs(Date.parse(other.time) - Date.parse(point.time)) <=
                  (line.gapHours ?? 1.5) * 3600000;
              return point.value != null &&
                !connected(line.points[index - 1]) &&
                !connected(line.points[index + 1]) ? (
                <circle
                  key={line.label + point.time}
                  cx={x(point.time)}
                  cy={y(point.value)}
                  r="3"
                  fill={line.color}
                />
              ) : null;
            }),
          )}
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
    </div>
  );
}
