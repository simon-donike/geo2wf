import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import { createRoot } from "react-dom/client";
import {
  BrowserRouter,
  Link,
  NavLink,
  Route,
  Routes,
  useNavigate,
  useParams,
  useSearchParams,
} from "react-router-dom";
import { MapView, RadiusLegend } from "./MapView";
import { Chart, type Line } from "./Chart";
import { rapidIntensification } from "./rapidIntensification";
import {
  age,
  BASINS,
  csv,
  dataUrl,
  download,
  getCatalog,
  getSeries,
  isStale,
  KNOT,
  preferredRecords,
  stamp,
  wind,
} from "./data";
import type { Basin, Catalog, Fix, Series, Storm } from "./types";
import {
  nearestIndex,
  officialCategory,
  RADII,
  smoothPoints,
  interpolatePoints,
  STRUCTURE_SMOOTHING,
  stormAge,
  windCategory,
  WIND_BANDS,
} from "./presentation";
import "./styles.css";

const UnitContext = createContext<{ unit: "kt" | "m/s"; toggle: () => void }>({
  unit: "kt",
  toggle: () => {},
});
const useUnit = () => useContext(UnitContext);
function Icon({
  name,
}: {
  name: "arrow" | "storm" | "download" | "map" | "clock" | "search";
}) {
  const paths = {
    arrow: "M5 12h14m-6-6 6 6-6 6",
    storm: "M12 8a4 4 0 1 1-4 4m0 0C1 7 8 1 14 3m2 9c7 5 0 11-6 9",
    download: "M12 3v12m-5-5 5 5 5-5M5 17v4h14v-4",
    map: "m3 5 6-2 6 2 6-2v16l-6 2-6-2-6 2zm6-2v16m6-14v16",
    clock: "M12 7v5l3 2M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0",
    search: "m16 16 5 5M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0",
  };
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={paths[name]} />
    </svg>
  );
}
function Freshness({ time, label }: { time?: string | null; label: string }) {
  const [now, setNow] = useState(Date.now);
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 60000);
    return () => clearInterval(timer);
  }, []);
  const maxAge = label === "NHC/CPHC feed" ? 0.5 : label === "Advisory" ? 6 : 2;
  return (
    <span
      className={`freshness ${isStale(time, maxAge, now) ? "stale" : ""}`}
      title={time ? stamp(time) : undefined}
    >
      <i />
      {label} · {age(time, now)}
    </span>
  );
}
function BasinTabs({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="basin-tabs" role="group" aria-label="Filter by basin">
      {[["all", "All basins"], ...Object.entries(BASINS)].map(([id, label]) => (
        <button
          key={id}
          className={value === id ? "selected" : ""}
          aria-pressed={value === id}
          onClick={() => onChange(id)}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
function Empty({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="empty">
      <Icon name="storm" />
      <h3>{title}</h3>
      <p>{children}</p>
    </div>
  );
}
function StormAge({ start }: { start: string }) {
  const [now, setNow] = useState(Date.now);
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 60000);
    return () => clearInterval(timer);
  }, []);
  return (
    <span
      className="storm-age"
      title={`Since first NHC/CPHC track fix: ${stamp(start)}`}
    >
      <Icon name="clock" /> Age · {stormAge(start, now)}
    </span>
  );
}
function StormCard({ storm }: { storm: Storm }) {
  const { unit } = useUnit();
  return (
    <Link to={`/storms/${storm.id}`} className="storm-card">
      <div className="card-top">
        <span className="eyebrow">{BASINS[storm.basin]}</span>
        <Icon name="arrow" />
      </div>
      <div className="storm-name">
        <img className="storm-logo" src="/brand/mark.svg" alt="" />
        <h3>{storm.name}</h3>
        <span
          className={`classification ${storm.latest_fix?.classification === "HU" ? "hurricane" : ""}`}
        >
          {officialCategory(storm.latest_fix)}
        </span>
      </div>
      <StormAge start={storm.start} />
      <div className="card-metrics">
        <div>
          <small>StormSense intensity</small>
          <strong>
            {wind(storm.metrics?.vmax_ms, unit)} <em>{unit}</em>
          </strong>
          <span className="wind-category">
            {storm.metrics
              ? `${windCategory(storm.metrics.vmax_ms)} equivalent`
              : "Category unavailable"}
          </span>
        </div>
        <div>
          <small
            title={
              storm.change_24h_reference_kind === "hindcast"
                ? "Compared with a retrospective StormSense estimate"
                : undefined
            }
          >
            24-hour change
            {storm.change_24h_reference_kind === "hindcast"
              ? " · retrospective"
              : ""}
          </small>
          <strong className="change">
            {storm.change_24h_ms == null
              ? "—"
              : `${storm.change_24h_ms >= 0 ? "+" : ""}${wind(storm.change_24h_ms, unit)}`}{" "}
            <em>{unit}</em>
          </strong>
        </div>
      </div>
      <div className="card-footer">
        <div>
          <Freshness time={storm.latest_prediction?.time} label="Estimate" />
          <br />
          <Freshness time={storm.advisory?.time} label="Advisory" />
        </div>
        <span>Explore storm ↗</span>
      </div>
    </Link>
  );
}
function Overview({ catalog }: { catalog: Catalog }) {
  const navigate = useNavigate();
  const [basin, setBasin] = useState("all");
  const active = useMemo(
    () => catalog.storms.filter((s) => s.active),
    [catalog.storms],
  );
  const storms = useMemo(
    () => active.filter((s) => basin === "all" || s.basin === basin),
    [active, basin],
  );
  const [tracks, setTracks] = useState<Record<string, Fix[]>>({});
  const [trackError, setTrackError] = useState(false);
  const [trackRetry, setTrackRetry] = useState(0);
  const trackKey = active
    .map((s) => `${s.id}:${s.series}:${s.advisory?.time}`)
    .join("|");
  useEffect(() => {
    const controller = new AbortController();
    setTrackError(false);
    void Promise.allSettled(
      active.map(async (storm) => {
        const series = await getSeries(storm.series, controller.signal);
        const fixes = series.track;
        const advisory = storm.advisory;
        return [
          storm.id,
          advisory && advisory.time > (fixes.at(-1)?.time ?? "")
            ? [...fixes, advisory]
            : fixes,
        ] as const;
      }),
    ).then((results) => {
      if (controller.signal.aborted) return;
      setTracks(
        Object.fromEntries(
          results.flatMap((r) => (r.status === "fulfilled" ? [r.value] : [])),
        ),
      );
      setTrackError(results.some((r) => r.status === "rejected"));
    });
    return () => controller.abort();
  }, [trackKey, catalog.release, trackRetry]);
  const openStorm = useCallback(
    (id: string) => navigate(`/storms/${id}`),
    [navigate],
  );
  const unconfirmed =
    !catalog.source_status.discovery?.last_success ||
    Boolean(catalog.source_status.discovery?.error);
  const ready = catalog.coverage.predictions;
  return (
    <>
      <section className="page-heading">
        <div>
          <div className="eyebrow">
            <span className="tiny-line" />
            Tropical cyclone intelligence
          </div>
          <h1>
            A clearer view
            <br />
            of the storm.
          </h1>
          <p>
            From satellite observations to storm-scale understanding.
            <br className="desktop" /> Follow evolving winds across the Atlantic
            and Pacific.
          </p>
        </div>
        <div className="overview-totals">
          <div>
            <strong>
              {unconfirmed ? "—" : String(active.length).padStart(2, "0")}
            </strong>
            <span>Active systems</span>
          </div>
          <div>
            <strong>
              12<span>mo</span>
            </strong>
            <span>Storm archive</span>
          </div>
          <div>
            <strong>
              1<span>hr</span>
            </strong>
            <span>Target update interval</span>
          </div>
        </div>
      </section>
      <section className="overview-section">
        <div className="section-heading">
          <h2>
            Across the basins{" "}
            <span className="count">
              {unconfirmed ? "Unconfirmed" : `${active.length} active`}
            </span>
          </h2>
          <Freshness
            time={catalog.source_status.discovery?.last_success}
            label="NHC/CPHC feed"
          />
        </div>
        {catalog.source_status.discovery?.error && (
          <p className="notice" role="alert">
            The storm feed could not be refreshed. Positions and active status
            reflect the last successful retrieval.
          </p>
        )}
        <BasinTabs value={basin} onChange={setBasin} />
        <div className="overview-grid">
          <div className="map-shell">
            <div className="map-top">
              <span>
                <Icon name="map" /> Storm positions
              </span>
              <span>NHC / CPHC</span>
            </div>
            <MapView storms={storms} tracks={tracks} onSelect={openStorm} />
            <RadiusLegend />
            {trackError && (
              <p className="map-note" role="status">
                Some tracks could not be loaded.{" "}
                <button onClick={() => setTrackRetry((n) => n + 1)}>
                  Retry tracks
                </button>
              </p>
            )}
            <div className="map-bottom">
              <span>
                <i className="dot warm" />
                Hurricane <i className="dot teal" />
                Other active system
              </span>
              <span>Rings follow latest estimates</span>
            </div>
          </div>
          <div className="storm-list">
            {storms.length ? (
              storms.map((s) => <StormCard key={s.id} storm={s} />)
            ) : (
              <Empty
                title={
                  unconfirmed ? "Storm status unavailable" : "A quieter ocean"
                }
              >
                {unconfirmed
                  ? "Current activity has not been confirmed by a successful NHC/CPHC update."
                  : `No active systems in ${basin === "all" ? "these basins" : BASINS[basin as Basin]}.`}{" "}
                Explore the archive to follow earlier storms.
              </Empty>
            )}
          </div>
        </div>
      </section>
      <section className="archive-feature">
        <div>
          <span className="eyebrow">Every storm has a story</span>
          <h2>Look back. See the evolution.</h2>
          <p>
            Explore a year of storm tracks, hourly wind estimates, and
            experimental intensity forecasts.
          </p>
          <Link to="/archive" className="button">
            Explore the archive <Icon name="arrow" />
          </Link>
        </div>
        <div className="archive-stat">
          <strong>{catalog.storms.length}</strong>
          <span>storms in the archive</span>
        </div>
        <div className="archive-stat">
          <strong>{ready.toLocaleString()}</strong>
          <span>hourly estimates computed</span>
        </div>
      </section>
      <div className="subtle-note">
        <Icon name="clock" /> Updates run hourly while the temporary runner is
        online. Source times show the freshness of the latest completed runs.
      </div>
      {catalog.coverage.pending > 0 && (
        <p className="notice">
          Archive processing is underway:{" "}
          {catalog.coverage.pending.toLocaleString()} storm-hours remain.
          Available results can already be explored.
        </p>
      )}
    </>
  );
}
function Archive({ catalog }: { catalog: Catalog }) {
  const [query, setQuery] = useState(""),
    [basin, setBasin] = useState("all"),
    [category, setCategory] = useState("all"),
    [from, setFrom] = useState(""),
    [to, setTo] = useState("");
  const { unit } = useUnit();
  const storms = catalog.storms.filter(
    (s) =>
      (basin === "all" || s.basin === basin) &&
      `${s.name} ${s.id}`.toLowerCase().includes(query.toLowerCase()) &&
      (category === "all" || s.peak_category === Number(category)) &&
      (!from || s.end >= from) &&
      (!to || s.start < `${to}T23:59:59Z`),
  );
  return (
    <>
      <section className="page-heading compact">
        <div>
          <div className="eyebrow">The past twelve months</div>
          <h1>A record of every evolution.</h1>
          <p>Trace a storm from its first fix to its final chapter.</p>
        </div>
        <span className="date-range">
          {stamp(catalog.window.start, true)} —{" "}
          {stamp(catalog.window.end, true)}
        </span>
      </section>
      <BasinTabs value={basin} onChange={setBasin} />
      <div className="archive-filters">
        <label className="search-field">
          <Icon name="search" />
          <input
            aria-label="Search storms"
            placeholder="Search by name or storm ID"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </label>
        <label>
          Peak official category
          <select
            value={category}
            onChange={(e) => setCategory(e.target.value)}
          >
            <option value="all">All categories</option>
            <option value="-1">Tropical depression</option>
            <option value="0">Tropical storm</option>
            {[1, 2, 3, 4, 5].map((i) => (
              <option key={i} value={i}>
                Category {i}
              </option>
            ))}
          </select>
        </label>
        <label>
          From
          <input
            aria-label="From date"
            type="date"
            value={from}
            onChange={(e) => setFrom(e.target.value)}
          />
        </label>
        <label>
          To
          <input
            aria-label="To date"
            type="date"
            value={to}
            onChange={(e) => setTo(e.target.value)}
          />
        </label>
      </div>
      <p className="result-count">
        {storms.length} storms · Historical estimates use retrospective track
        centers.
      </p>
      <div
        className="archive-table"
        tabIndex={0}
        aria-label="Storm archive table"
      >
        <div className="archive-row table-heading">
          <span>Storm</span>
          <span>Basin</span>
          <span>Period · UTC</span>
          <span>Latest estimate</span>
          <span>Max wind · NHC/CPHC</span>
          <span>Max category</span>
          <span>Rapid intensification</span>
          <span>Available / gaps</span>
          <span />
        </div>
        {storms.map((s) => (
          <Link className="archive-row" key={s.id} to={`/storms/${s.id}`}>
            <span className="archive-name">
              <strong>{s.name}</strong>
              <small>
                {s.id} {s.active && <b>ACTIVE</b>}
              </small>
              <small className="mobile-coverage">
                {s.prediction_count} estimates · {s.gap_count} gaps ·{" "}
                {s.pending_count} pending
              </small>
            </span>
            <span>{BASINS[s.basin]}</span>
            <span>
              {stamp(s.start, true)} — {stamp(s.end, true)}
            </span>
            <span>
              {wind(s.metrics?.vmax_ms, unit)} <small>{unit}</small>
            </span>
            <span className="archive-peak-wind" data-label="Max wind">
              {wind(s.peak_official_wind_ms, unit)} <small>{unit}</small>
            </span>
            <span className="archive-peak-category" data-label="Max category">
              {s.peak_category == null
                ? "—"
                : s.peak_category > 0
                  ? `Category ${s.peak_category}`
                  : s.peak_category === 0
                    ? "Tropical storm"
                    : "Tropical depression"}
            </span>
            <span
              className="archive-ri"
              data-label="RI"
              title={
                s.has_ri == null
                  ? "Insufficient official reference history"
                  : s.has_ri
                    ? "Rapid intensification detected in NHC/CPHC winds"
                    : "No rapid intensification detected in available NHC/CPHC winds"
              }
            >
              {s.has_ri == null ? "—" : s.has_ri ? "Yes" : "No"}
            </span>
            <span className="coverage-cell">
              <span>
                {s.prediction_count}{" "}
                <small>
                  / {s.gap_count} gaps · {s.pending_count} pending
                </small>
              </span>
              <i
                style={
                  {
                    "--progress": `${s.expected_count ? (s.prediction_count / s.expected_count) * 100 : 0}%`,
                  } as React.CSSProperties
                }
              />
            </span>
            <Icon name="arrow" />
          </Link>
        ))}
      </div>
      {!storms.length && (
        <Empty title="No matching storms">
          Try another name, basin, category, or date range.
        </Empty>
      )}
    </>
  );
}
function Detail({ catalog }: { catalog: Catalog }) {
  const { id } = useParams();
  const storm = catalog.storms.find((s) => s.id === id);
  const [loadedSeries, setSeries] = useState<Series | null>(null),
    [error, setError] = useState(""),
    [retry, setRetry] = useState(0);
  const [params, setParams] = useSearchParams();
  const series = loadedSeries?.storm_id === id ? loadedSeries : null;
  const references = useMemo(() => {
    const fixes = series?.track ?? [];
    const advisory = storm?.advisory;
    return advisory && advisory.time > (fixes.at(-1)?.time ?? "")
      ? [...fixes, advisory]
      : fixes;
  }, [series, storm?.advisory]);
  const { unit } = useUnit();
  useEffect(() => {
    if (!storm) return;
    const controller = new AbortController();
    setError("");
    getSeries(storm.series, controller.signal)
      .then(setSeries)
      .catch((e) => {
        if (!controller.signal.aborted) setError(String(e.message));
      });
    return () => controller.abort();
  }, [storm?.series, retry]);
  const records = useMemo(
    () => preferredRecords(series?.records ?? []),
    [series],
  );
  const requested = params.get("time");
  const issue = params.get("issue"),
    issueKind = params.get("issue_kind");
  const [selection, setSelection] = useState<{
    time: string | null;
    issue?: string | null;
    kind?: string | null;
  }>(() => ({ time: requested, issue, kind: issueKind }));
  useEffect(() => {
    setSelection({ time: requested, issue, kind: issueKind });
  }, [requested, issue, issueKind]);
  const [hover, setHover] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(4);
  const [imageReady, setImageReady] = useState(true);
  useEffect(() => {
    setPlaying(false);
  }, [id]);
  const [smoothing, setSmoothing] = useState(
    () => localStorage.getItem("stormsense-smoothing") !== "off",
  );
  const times = useMemo(
    () => records.map((r) => Date.parse(r.time)),
    [records],
  );
  const index = nearestIndex(times, selection.time);
  const recordLookup = useMemo(
    () => new Map(series?.records.map((r) => [`${r.kind}|${r.time}`, r])),
    [series],
  );
  const alternate = selection.kind
    ? recordLookup.get(`${selection.kind}|${records[index]?.time}`)
    : undefined;
  const selected = alternate?.status === "ready" ? alternate : records[index];
  const commitTime = useCallback(
    (time: string) => {
      if (
        params.get("time") === time &&
        !params.has("issue") &&
        !params.has("issue_kind")
      )
        return;
      setParams(
        (prior) => {
          const next = new URLSearchParams(prior);
          next.set("time", time);
          next.delete("issue");
          next.delete("issue_kind");
          return next;
        },
        { replace: true, preventScrollReset: true },
      );
    },
    [params, setParams],
  );
  const selectTime = useCallback(
    (time: string) => {
      setPlaying(false);
      const selectedTime = records[nearestIndex(times, time)]?.time;
      if (!selectedTime) return;
      setSelection({ time: selectedTime });
      setHover(null);
      commitTime(selectedTime);
    },
    [records, times, commitTime],
  );
  const scrub = (value: string) => {
    setPlaying(false);
    const time = records[Number(value)]?.time;
    if (time) {
      setSelection({ time });
      setHover(null);
    }
  };
  const finishScrub = (value: string) => {
    const time = records[Number(value)]?.time;
    if (time) commitTime(time);
  };
  const togglePlayback = () => {
    if (playing) {
      setPlaying(false);
      if (selected) commitTime(selected.time);
    } else {
      if (index >= records.length - 1) setSelection({ time: records[0].time });
      setPlaying(true);
    }
  };
  useEffect(() => {
    if (!playing || !imageReady || !records.length) return;
    const timer = setTimeout(() => {
      if (index >= records.length - 1) {
        setPlaying(false);
        commitTime(records[index].time);
      } else {
        setSelection({ time: records[index + 1].time });
        setHover(null);
      }
    }, 1000 / playbackRate);
    return () => clearTimeout(timer);
  }, [playing, imageReady, index, records, playbackRate, commitTime]);
  useEffect(() => {
    const pauseWhenHidden = () => {
      if (document.hidden) {
        setPlaying(false);
        if (selected) commitTime(selected.time);
      }
    };
    document.addEventListener("visibilitychange", pauseWhenHidden);
    return () =>
      document.removeEventListener("visibilitychange", pauseWhenHidden);
  }, [selected, commitTime]);
  const factor = unit === "kt" ? 1 / KNOT : 1;
  const rapidIntervals = useMemo(
    () => [
      ...rapidIntensification(
        references.map((fix) => ({
          time: fix.time,
          value: ["TD", "TS", "HU"].includes(fix.classification)
            ? fix.wind_ms
            : null,
        })),
        "official",
        6,
      ),
    ],
    [references],
  );
  const intensity = useMemo(() => {
    const points = records.map((r) => ({
      time: r.time,
      value: r.metrics ? r.metrics.vmax_ms * factor : null,
    }));
    const lines: Line[] = [
      {
        label: smoothing ? "StormSense · smoothed" : "StormSense estimate",
        color: "#40c7bd",
        points: smoothing ? smoothPoints(points) : interpolatePoints(points),
      },
      {
        label: "NHC/CPHC reference",
        color: "#e8e2d7",
        dashed: true,
        gapHours: 6,
        points: references.map((f) => ({
          time: f.time,
          value: f.wind_ms == null ? null : f.wind_ms * factor,
        })),
      },
    ];
    return lines;
  }, [records, references, factor, smoothing]);
  const structure = useMemo(
    () =>
      RADII.map(({ key, label, color }) => {
        const points = records.map((r) => ({
          time: r.time,
          value: r.metrics?.[key] ?? null,
        }));
        return {
          label,
          color,
          points: smoothing
            ? smoothPoints(points, STRUCTURE_SMOOTHING)
            : interpolatePoints(points),
        };
      }),
    [records, smoothing],
  );
  const thresholds = useMemo(
    () =>
      WIND_BANDS.map((b) => ({
        value: b.knots * KNOT * factor,
        color: b.color,
        label: `${b.short} · ${unit === "kt" ? b.knots : (b.knots * KNOT).toFixed(1)} ${unit}`,
      })),
    [factor, unit],
  );
  const official = references
    .filter((f) => f.time <= (selected?.time || storm?.end || ""))
    .at(-1);
  const metrics = selected?.metrics;
  if (!storm)
    return (
      <Empty title="Storm not found">
        This storm is outside the current archive.{" "}
        <Link to="/archive">Browse available storms.</Link>
      </Empty>
    );
  return (
    <>
      <Link to={storm.active ? "/" : "/archive"} className="back-link">
        ← {storm.active ? "All active systems" : "Storm archive"}
      </Link>
      <section className="storm-heading">
        <div>
          <div className="eyebrow">
            {BASINS[storm.basin]} / {storm.id}{" "}
            <span className="status-label">
              {storm.active ? "ACTIVE SYSTEM" : "ARCHIVED SYSTEM"}
            </span>
          </div>
          <h1>
            {storm.name}
            <span className="classification">
              {officialCategory(official || storm.latest_fix)}
            </span>
          </h1>
          <p>
            {stamp(storm.start, true)} — {stamp(storm.end, true)} ·{" "}
            {storm.prediction_count.toLocaleString()} hourly estimates
          </p>
        </div>
        <div className="download-actions">
          <button
            disabled={!series}
            onClick={() =>
              series && download(`${storm.id}.csv`, csv(series), "text/csv")
            }
          >
            <Icon name="download" /> CSV
          </button>
          <button
            disabled={!series}
            onClick={() =>
              series &&
              download(`${storm.id}.json`, JSON.stringify(series, null, 2))
            }
          >
            JSON
          </button>
        </div>
      </section>
      {error && (
        <p role="alert" className="notice">
          {error} <button onClick={() => setRetry((x) => x + 1)}>Retry</button>
        </p>
      )}
      {!series && !error && (
        <p role="status" className="loading-state">
          Loading storm history…
        </p>
      )}
      {series && (
        <>
          <div className="metric-strip">
            <div>
              <span>StormSense intensity</span>
              <strong>
                {wind(metrics?.vmax_ms, unit)} <small>{unit}</small>
              </strong>
              <span className="wind-category">
                {metrics
                  ? `${windCategory(metrics.vmax_ms)} equivalent`
                  : "Category unavailable"}
              </span>
              <em>
                {selected?.kind === "live"
                  ? "Issued live estimate"
                  : "Retrospective estimate"}
              </em>
            </div>
            <div>
              <span>NHC/CPHC intensity</span>
              <strong>
                {wind(official?.wind_ms, unit)} <small>{unit}</small>
              </strong>
              <span className="wind-category">
                {official ? officialCategory(official) : "Category unavailable"}
              </span>
              <em>
                {official ? stamp(official.time) : "No reference available"}
              </em>
            </div>
            <div>
              <span>Radius of max. wind</span>
              <strong>
                {metrics ? metrics.rmw_km.toFixed(0) : "—"} <small>km</small>
              </strong>
              <em>Model structure estimate</em>
            </div>
            <div>
              <span>Gale-force radius · R34</span>
              <strong>
                {metrics ? metrics.r34_km.toFixed(0) : "—"} <small>km</small>
              </strong>
              <em>Equivalent-area radius</em>
            </div>
          </div>
          <section className="timeline-panel">
            <div>
              <span className="eyebrow">Explore the timeline</span>
              <strong>
                {selected ? stamp(selected.time) : "No processed hours"}
              </strong>
              {selected?.status === "gap" && (
                <span className="gap-label">
                  Data gap · {selected.reason?.replaceAll("_", " ")}
                </span>
              )}
            </div>
            <div className="playback-controls">
              <button
                disabled={records.length < 2}
                aria-label={playing ? "Pause timeline" : "Play timeline"}
                onClick={togglePlayback}
              >
                {playing ? "Ⅱ Pause" : "▶ Play"}
              </button>
              <label>
                Speed{" "}
                <select
                  aria-label="Timeline playback speed"
                  value={playbackRate}
                  onChange={(e) => setPlaybackRate(Number(e.target.value))}
                >
                  <option value="2">2 hours / sec</option>
                  <option value="4">4 hours / sec</option>
                  <option value="8">8 hours / sec</option>
                </select>
              </label>
              <span role="status">
                {playing
                  ? imageReady
                    ? "Playing"
                    : "Buffering image…"
                  : "Drag to explore"}
              </span>
            </div>
            <input
              aria-label="Storm timeline"
              type="range"
              min="0"
              max={Math.max(0, records.length - 1)}
              value={index}
              disabled={!records.length}
              aria-valuetext={
                selected ? stamp(selected.time) : "No processed hours"
              }
              onChange={(e) => scrub(e.target.value)}
              onPointerUp={(e) => finishScrub(e.currentTarget.value)}
              onPointerDown={(e) =>
                e.currentTarget.setPointerCapture(e.pointerId)
              }
              onPointerCancel={(e) => finishScrub(e.currentTarget.value)}
              onKeyUp={(e) => finishScrub(e.currentTarget.value)}
              onBlur={(e) => finishScrub(e.currentTarget.value)}
            />
            <footer>
              <span>{stamp(storm.start, true)}</span>
              <button
                onClick={() => {
                  setPlaying(false);
                  setSelection({ time: null });
                  setParams({}, { replace: true, preventScrollReset: true });
                  setHover(null);
                }}
              >
                Latest observation ↗
              </button>
              <span>{stamp(storm.end, true)}</span>
            </footer>
          </section>
          <div className="detail-grid">
            <div className="dashboard-column">
            <section className="map-shell detail-map-shell">
              <div className="map-top">
                <span>
                  <Icon name="map" /> The storm’s path
                </span>
                <span>
                  {selected?.center?.method === "motion_estimate"
                    ? "Estimated center"
                    : "Track history"}
                </span>
              </div>
              <MapView
                detail
                track={references}
                selected={selected ? selected.center : storm.latest_fix}
                radii={metrics}
                liveStorm={storm.active ? storm : undefined}
                latestImagery={selected?.time === storm.latest_prediction?.time}
                imagery={series.imagery}
                imageryBundles={series.imagery_bundles}
                imageryTime={selected?.time}
                onImageryReady={setImageReady}
              />
              <RadiusLegend />
              <div className="track-context">
                <span>Selected position</span>
                <strong>
                  {selected?.center
                    ? `${selected.center.lat.toFixed(1)}° N · ${Math.abs(selected.center.lon).toFixed(1)}° ${selected.center.lon < 0 ? "W" : "E"}`
                    : "Unavailable"}
                </strong>
                <small>
                  {selected?.imagery
                    ? `GOES-${selected.imagery.satellite} · ${stamp(selected.imagery.end)}`
                    : "Satellite scan unavailable"}
                </small>
              </div>
            </section>
            </div>
            <div className="dashboard-column">
            <section className="chart-panel">
              <div className="panel-heading">
                <div className="eyebrow">Intensity through time</div>
                <h2>Following the wind</h2>
                <p>
                  {hover
                    ? stamp(hover)
                    : selected
                      ? stamp(selected.time)
                      : "No predictions yet"}
                </p>
              </div>
              <Chart
                title="Maximum sustained wind"
                unit={unit}
                lines={intensity}
                thresholds={thresholds}
                rapidIntervals={rapidIntervals}
                selected={hover || selected?.time}
                onTime={setHover}
                onSelect={selectTime}
              />
              <label
                className="smoothing-control"
                title="A trailing three-hour display filter. Original estimates remain in downloads."
              >
                <input
                  type="checkbox"
                  checked={smoothing}
                  onChange={(e) => {
                    setSmoothing(e.target.checked);
                    localStorage.setItem(
                      "stormsense-smoothing",
                      e.target.checked ? "on" : "off",
                    );
                  }}
                />{" "}
                Smooth estimates
              </label>

            </section>          <section className="structure-panel">
            <div className="panel-heading">
              <div className="eyebrow">Storm structure</div>
              <h2>How far the winds reach</h2>
              <p>
                Model estimates of equivalent-area wind radii
                {smoothing ? ", lightly smoothed" : ""}. Interior gaps are
                linearly interpolated.
              </p>
            </div>
            <Chart
              title="Wind radii"
              unit="km"
              selected={selected?.time}
              lines={structure}
              rapidIntervals={rapidIntervals}
              onSelect={selectTime}
            />
          </section>
            </div>
          </div>
          <details className="provenance">
            <summary>About this observation & data quality</summary>
            <p>
              Selected hour: {selected ? stamp(selected.time) : "Unavailable"}
              <br />
              Generated:{" "}
              {selected ? stamp(selected.generated_at) : "Unavailable"}
              <br />
              Center fix:{" "}
              {selected?.center
                ? `${stamp(selected.center.fix_time)} · ${selected.center.age_hours.toFixed(1)} hours old · ${selected.center.method.replaceAll("_", " ")}`
                : "Unavailable"}
              <br />
              Satellite acquisition:{" "}
              {selected?.imagery
                ? `${stamp(selected.imagery.end)} · ${(100 * selected.imagery.valid_fraction).toFixed(1)}% valid coverage`
                : "Unavailable"}
              {selected?.imagery && (
                <>
                  <br />
                  Satellite retrieval:{" "}
                  {selected.imagery.retrieved_at
                    ? stamp(selected.imagery.retrieved_at)
                    : "Not recorded for this early-run estimate"}
                </>
              )}
            </p>
            <p>Model: {selected?.model_version}</p>
          </details>
          <p className="subtle-note">
            {selected?.kind === "live"
              ? "This estimate was generated during a live update."
              : "This is a historical hindcast using retrospective storm centers."}{" "}
            Model estimates are distinct from
            NHC/CPHC advisories. <Link to="/about">Read the methods ↗</Link>
          </p>
        </>
      )}
    </>
  );
}
function About({ catalog }: { catalog: Catalog }) {
  return (
    <div className="about-page">
      <div className="eyebrow">Observations into understanding</div>
      <h1>What you’re looking at.</h1>
      <p className="intro">
        StormSense follows tropical cyclones using geostationary satellite
        observations, translating cloud patterns into estimates of surface wind
        intensity and storm size.
      </p>
      <div className="method-grid">
        <article>
          <span className="method-number">01</span>
          <h2>A satellite view</h2>
          <p>
            Ten infrared GOES ABI channels are sampled around the NHC/CPHC storm
            center. Each hourly slot uses a complete scan from the preceding 30
            minutes. Source imagery is read temporarily and discarded.
          </p>
        </article>
        <article>
          <span className="method-number">02</span>
          <h2>A model estimate</h2>
          <p>
            One GEO-only joint model estimates maximum sustained wind, the
            radius of maximum wind, and equivalent-area radii for 34, 50, and
            64-knot winds. It does not require SAR or ERA5 during inference.
          </p>
        </article>
        <article>
          <span className="method-number">03</span>
          <h2>A look ahead</h2>
          <p>
            The experimental forecast model uses StormSense intensity at the
            current hour, −6 hours, and −12 hours. It predicts +6 hours
            directly, then +12 hours recursively. It never receives future
            observations. A forecast is withheld when its input history is
            incomplete.
          </p>
        </article>
        <article>
          <span className="method-number">04</span>
          <h2>An honest record</h2>
          <p>
            Live centers use the latest position and fresh reported motion, for
            up to six hours. Historical centers are interpolated from
            retrospective tracks. Hindcasts are labeled separately from live
            estimates, and missing inputs stay visible as gaps.
          </p>
        </article>
      </div>
      <section className="methods-detail">
        <h2>Sources, versions & availability</h2>
        <p>
          NHC/CPHC reference winds come from advisories and ATCF track files,
          which may later be revised. These are shown independently of
          StormSense predictions. Equivalent-area radii describe a circle with
          the same area as the wind footprint; they are not quadrant-specific
          warning boundaries.
        </p>
        <p>
          Map rings show these model radii at their actual geographic scale.
          Category lines use NHC wind thresholds; a StormSense category is a
          wind-speed equivalent, separate from the official classification.
          Storm age starts at the first recorded NHC/CPHC track fix.
        </p>
        <p>
          Wind charts use exponential smoothing: 50% of the newest estimate and
          50% of the previous smoothed value. This reduces wiggles with
          approximately one hour of lag. Radii retain the three-hour 60/30/10%
          weighted average. Interior gaps are linearly interpolated between
          available estimates before filtering; leading and trailing gaps remain
          empty. Historical interpolation uses both endpoints. Turn smoothing
          off above the intensity chart to see the unfiltered, interpolated
          curves. Metric tiles, map rings, downloads, official observations, and
          issued forecasts use the original values.
        </p>
        <p>
          Rapid-intensification shading marks a wind increase of at least 30 kt
          (15.4 m/s) over exactly 24 hours. Amber uses NHC/CPHC reference winds
          during tropical phases. These are calculated indicators, not issued
          NHC event labels. Overlapping qualifying windows are joined. Original,
          unsmoothed values are used, with no interpolation across missing fixes
          and no model or forecast inputs. Reference fixes must be at most six
          hours apart. The radii chart repeats these intensity periods for
          context.
        </p>
        <p>
          Storm pages show saved GeoColor images from GOES-East or GOES-West
          through NASA GIBS, covering roughly 1,800 km around each recorded
          position. Play or drag the timeline to explore them. All available
          images for the open storm preload in the background, with the selected
          hour and nearby frames first. Small previews sharpen as full images
          arrive. Missing imagery remains an explicit gap; predictions are
          independent. The view uses colour by day and an infrared blend at
          night. Its displayed provider product time is separate from the scan
          used for inference and can precede the selected hour by up to 30
          minutes. Compact WebP images retain geospatial sidecars and source
          metadata. Only this easy-to-obtain display product is archived; raw
          satellite data is discarded. The active overview separately checks for
          the latest available image every five minutes while visible.
        </p>
        <p>
          Updates run hourly while the temporary runner is online. The interface
          reports the actual age of each completed run and source observation;
          interruptions remain visible through those timestamps.
        </p>
        <dl>
          <dt>Nowcast</dt>
          <dd>{catalog.models.nowcast.version}</dd>
          <dt>Forecast</dt>
          <dd>{catalog.models.forecast.version}</dd>
          <dt>Data release</dt>
          <dd>{catalog.release}</dd>
          <dt>Archive coverage</dt>
          <dd>
            {catalog.coverage.predictions.toLocaleString()} estimates ·{" "}
            {catalog.coverage.gaps.toLocaleString()} recorded gaps ·{" "}
            {catalog.coverage.pending.toLocaleString()} pending
          </dd>
        </dl>
        <p>
          The forecast chain is experimental. Evaluation compares it with
          persistence and recent-trend baselines, with training-excluded storms
          reported separately. No confidence interval or operational skill is
          implied.
        </p>
        <div className="source-links">
          <a href={dataUrl(`releases/${catalog.release}/coverage.json`)}>
            Coverage report ↗
          </a>
          {catalog.reports?.evaluation && (
            <a href={dataUrl(catalog.reports.evaluation)}>
              Forecast evaluation ↗
            </a>
          )}
          <a href="https://www.nhc.noaa.gov/" target="_blank" rel="noreferrer">
            NHC / CPHC ↗
          </a>
          <a
            href="https://registry.opendata.aws/noaa-goes/"
            target="_blank"
            rel="noreferrer"
          >
            NOAA GOES data ↗
          </a>
          <a href="https://tcd.hyperalis.com/" target="_blank" rel="noreferrer">
            Research documentation ↗
          </a>
        </div>
      </section>
    </div>
  );
}
function App() {
  const [catalog, setCatalog] = useState<Catalog | null>(null),
    [error, setError] = useState(""),
    [loading, setLoading] = useState(true),
    [revision, setRevision] = useState(0);
  const [unit, setUnit] = useState<"kt" | "m/s">(() =>
    localStorage.getItem("stormsense-unit") === "m/s" ? "m/s" : "kt",
  );
  useEffect(() => {
    const controller = new AbortController();
    let busy = false;
    const refresh = async () => {
      if (busy) return;
      busy = true;
      try {
        setCatalog(await getCatalog(controller.signal));
        setError("");
      } catch (e) {
        if (!controller.signal.aborted) setError((e as Error).message);
      } finally {
        busy = false;
        setLoading(false);
      }
    };
    void refresh();
    const timer = setInterval(() => {
      if (!document.hidden) void refresh();
    }, 60000);
    return () => {
      controller.abort();
      clearInterval(timer);
    };
  }, [revision]);
  const toggle = () =>
    setUnit((prior) => {
      const next = prior === "kt" ? "m/s" : "kt";
      localStorage.setItem("stormsense-unit", next);
      return next;
    });
  return (
    <UnitContext.Provider value={{ unit, toggle }}>
      <a href="#main" className="skip-link">
        Skip to content
      </a>
      <header className="topbar">
        <Link className="brand" to="/">
          <img className="brand-icon" src="/brand/mark.svg" alt="" />
          <strong>
            StormSense<span>EARTH SYSTEMS LAB</span>
          </strong>
        </Link>
        <nav aria-label="Main navigation">
          <NavLink to="/" end>
            Active storms
          </NavLink>
          <NavLink to="/archive">Archive</NavLink>
          <NavLink to="/about">About & methods</NavLink>
        </nav>
        <div className="header-tools">
          <button
            onClick={toggle}
            aria-label={`Wind unit: ${unit}. Switch units.`}
          >
            {unit}
            <span> ⇄</span>
          </button>
          <a href="https://eslab.ai/" aria-label="Earth Systems Lab">
            <img src="/brand/esl.png" alt="Earth Systems Lab" />
          </a>
        </div>
      </header>
      <main id="main">
        {error && (
          <div className="notice" role="alert">
            {error} {catalog && "Showing the last loaded release."}{" "}
            <button onClick={() => setRevision((n) => n + 1)}>Retry</button>
          </div>
        )}
        {loading && !catalog ? (
          <div className="loading-state" role="status">
            <span className="loader" />
            Connecting to the storm record…
          </div>
        ) : catalog ? (
          <Routes>
            <Route path="/" element={<Overview catalog={catalog} />} />
            <Route path="/archive" element={<Archive catalog={catalog} />} />
            <Route
              path="/storms/:id"
              element={<Detail key={location.pathname} catalog={catalog} />}
            />
            <Route path="/about" element={<About catalog={catalog} />} />
            <Route
              path="*"
              element={
                <Empty title="Page not found">
                  <Link to="/">Return to active storms.</Link>
                </Empty>
              }
            />
          </Routes>
        ) : (
          <Empty title="The storm record is not available">
            Start a data update and export a release, or retry once the data
            service is available.
          </Empty>
        )}
      </main>
      <footer className="site-footer">
        <div>
          <Icon name="storm" />
          <span>
            StormSense <small>Satellite insight. Storm by storm.</small>
          </span>
        </div>
        <span>NHC / CPHC coverage · All times UTC</span>
        <a href="https://tcd.hyperalis.com/">
          A Tropical Cyclone Dynamics research project ↗
        </a>
      </footer>
    </UnitContext.Provider>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>,
);
