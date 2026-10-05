"""Reference-only storm summaries; no model output enters these calculations."""

from datetime import timedelta
import math

from .common import KNOT, category, iso, utc


def official_summary(track, advisory=None):
    fixes = {utc(f["time"]): dict(f) for f in track}
    if advisory and (not fixes or utc(advisory["time"]) > max(fixes)):
        fixes[utc(advisory["time"])] = advisory
    winds = [
        f["wind_ms"]
        for f in fixes.values()
        if f.get("wind_ms") is not None
        and math.isfinite(f["wind_ms"])
        and f["wind_ms"] >= 0
    ]
    history, events = {}, []
    segment, evaluable = 0, 0
    previous = None
    for time, fix in sorted(fixes.items()):
        wind = fix.get("wind_ms")
        if (
            fix.get("classification") not in {"TD", "TS", "HU"}
            or wind is None
            or not math.isfinite(wind)
            or wind < 0
        ):
            segment += 1
            previous = None
            continue
        if previous is not None and time - previous > timedelta(hours=6):
            segment += 1
        begin = time - timedelta(hours=24)
        before = history.get(begin)
        if before and before[1] == segment:
            evaluable += 1
            change = wind - before[0]
            if change >= 30 * KNOT - 1e-8:
                if events and utc(events[-1]["end"]) >= begin:
                    events[-1]["end"] = iso(time)
                    events[-1]["windows"] += 1
                    events[-1]["max_change_ms"] = max(
                        events[-1]["max_change_ms"], change
                    )
                else:
                    events.append(
                        {
                            "start": iso(begin),
                            "end": iso(time),
                            "max_change_ms": change,
                            "windows": 1,
                        }
                    )
        history[time] = (wind, segment)
        previous = time
    peak = max(winds) if winds else None
    return {
        "peak_wind_ms": peak,
        "peak_category": category(peak) if peak is not None else None,
        "has_ri": True if events else False if evaluable else None,
        "ri_events": events,
    }
