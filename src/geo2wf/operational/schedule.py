"""Shared prediction schedule. Legacy manifests retain their hourly policy."""

from datetime import timedelta

from .common import hours, iso, utc
from .models import MANIFEST


def cadence():
    return MANIFEST.get("prediction_policy", {}).get("cadence_hours", 1)


def qualified(value):
    value = str(value or "").strip().upper().replace("-", " ").replace("_", " ")
    return value in {
        "TD",
        "TS",
        "HU",
        "SD",
        "SS",
        "TROPICAL DEPRESSION",
        "TROPICAL STORM",
        "HURRICANE",
        "SUBTROPICAL DEPRESSION",
        "SUBTROPICAL STORM",
    }


def eligibility_start(storm, live=False):
    if not MANIFEST.get("prediction_policy", {}).get("classification_start_gate"):
        return storm["start"]
    if live:
        return storm.get("prediction_qualified_at")
    times = [
        f["time"] for f in storm.get("track", []) if qualified(f.get("classification"))
    ]
    return min(times) if times else None


def slots(storm, start, end, live=False):
    onset = eligibility_start(storm, live)
    if onset is None:
        return
    begin = max(utc(start), utc(storm["start"]), utc(onset))
    finish = min(utc(end), utc(end) if storm.get("active") else utc(storm["end"]))
    for at in hours(begin, finish):
        if at >= begin and at.hour % cadence() == 0:
            yield at


def metadata(storm):
    starts = [
        s for s in (eligibility_start(storm), eligibility_start(storm, live=True)) if s
    ]
    onset = min(starts) if starts else None
    if onset:
        at = utc(onset)
        rounded = at.replace(minute=0, second=0, microsecond=0)
        while rounded < at or rounded.hour % cadence():
            rounded += timedelta(hours=1)
        onset = iso(rounded)
    return {
        "cadence_hours": cadence(),
        "eligible_start": onset,
        "start_gate": bool(
            MANIFEST.get("prediction_policy", {}).get("classification_start_gate")
        ),
    }


def visible_slots(storm, start, end):
    """Include observed live eligibility while best tracks are still catching up."""
    return sorted(
        set(slots(storm, start, end)) | set(slots(storm, start, end, live=True))
    )
