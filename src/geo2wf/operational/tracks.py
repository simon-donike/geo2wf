"""NHC/CPHC discovery and ATCF parsing. Storm IDs are never derived from longitude."""

from __future__ import annotations

import gzip
import json
import math
import re
from bisect import bisect_left
from datetime import datetime, timedelta

from .common import KNOT, UTC, fetch, finite, iso, utc

CURRENT = "https://www.nhc.noaa.gov/CurrentStorms.json"
ATCF = "https://ftp.nhc.noaa.gov/atcf"
STORM_ID = re.compile(r"^(AL|EP|CP)(0[1-9]|[1-4][0-9])(20\d{2})$")


def coordinate(value):
    text = str(value).strip().upper()
    if not re.fullmatch(r"\d+[NSEW]", text):
        raise ValueError(f"invalid ATCF coordinate {text!r}")
    result = int(text[:-1]) / 10 * (-1 if text[-1] in "SW" else 1)
    if abs(result) > (90 if text[-1] in "NS" else 180):
        raise ValueError("coordinate out of range")
    return result


def parse_atcf(body: bytes | str, storm_id: str):
    storm_id = storm_id.upper()
    if not STORM_ID.fullmatch(storm_id):
        raise ValueError("unsupported storm ID")
    if isinstance(body, bytes):
        body = gzip.decompress(body) if body.startswith(b"\x1f\x8b") else body
        body = body.decode("utf-8")
    fixes = {}
    for line in body.splitlines():
        f = [part.strip() for part in line.split(",")]
        if len(f) < 11 or f[4] != "BEST":
            continue
        try:
            stamp = iso(datetime.strptime(f[2], "%Y%m%d%H").replace(tzinfo=UTC))
            lat, lon = coordinate(f[6]), coordinate(f[7])
        except ValueError:
            continue
        # B-decks repeat each fix for different wind thresholds; preserve one position.
        wind = finite(f[8])
        record = fixes.setdefault(
            stamp,
            {
                "time": stamp,
                "lat": lat,
                "lon": lon,
                "wind_ms": wind * KNOT if wind is not None and wind > 0 else None,
                "pressure_hpa": finite(f[9]) if finite(f[9]) not in (None, 0) else None,
                "classification": f[10],
                "name": f[27].title() if len(f) > 27 else storm_id,
                "radii_km": {},
                "source": "NHC/CPHC ATCF BEST",
            },
        )
        if len(f) >= 17 and f[11] in {"34", "50", "64"} and f[12] == "NEQ":
            quadrants = [finite(x) for x in f[13:17]]
            if all(x is not None and x >= 0 for x in quadrants) and any(
                x > 0 for x in quadrants
            ):
                record["radii_km"]["r" + f[11]] = (
                    math.sqrt(sum(x * x for x in quadrants) / 4) * 1.852
                )
        if len(f) > 19 and finite(f[19]) not in (None, 0):
            record["radii_km"]["rmw"] = float(f[19]) * 1.852
    return sorted(fixes.values(), key=lambda row: row["time"])


def parse_current(body):
    data = json.loads(body)
    if not isinstance(data, dict) or not isinstance(data.get("activeStorms"), list):
        raise ValueError("NHC response has no activeStorms array")
    result = []
    for item in data["activeStorms"]:
        sid = str(item.get("id", "")).upper()
        if not STORM_ID.fullmatch(sid):
            continue
        stamp = iso(item["lastUpdate"])
        lat, lon = finite(item.get("latitudeNumeric")), finite(
            item.get("longitudeNumeric")
        )
        if lat is None or lon is None or abs(lat) > 90 or abs(lon) > 180:
            raise ValueError(f"NHC has invalid coordinates for {sid}")
        wind = finite(item.get("intensity"))
        basin = str(item.get("binNumber", ""))[:2].upper()
        if basin not in {"AL", "EP", "CP"}:
            basin = "AL" if sid.startswith("AL") else ("CP" if lon < -140 else "EP")
        result.append(
            {
                "id": sid,
                "name": item.get("name", sid),
                "basin": basin,
                "advisory": {
                    "time": stamp,
                    "lat": lat,
                    "lon": lon,
                    "wind_ms": wind * KNOT if wind is not None else None,
                    "classification": item.get("classification", ""),
                    "motion_direction": finite(item.get("movementDir")),
                    "motion_speed_kt": finite(item.get("movementSpeed")),
                    "url": (item.get("publicAdvisory") or {}).get("url"),
                    "source": "NHC/CPHC advisory",
                },
            }
        )
    return result


def track_urls(year, now=None):
    directory = f"{ATCF}/btk/" if year == utc(now).year else f"{ATCF}/archive/{year}/"
    body = fetch(directory).decode()
    names = sorted(
        set(
            re.findall(
                r'href="(b(?:al|ep|cp)\d{2}' + str(year) + r'\.dat(?:\.gz)?)"',
                body,
                re.I,
            )
        )
    )
    return [
        (name[1:9].upper(), directory + name)
        for name in names
        if STORM_ID.fullmatch(name[1:9].upper())
    ]


def historical_center(fixes, at):
    at = utc(at)
    times = [utc(f["time"]) for f in fixes]
    index = bisect_left(times, at)
    if index < len(times) and times[index] == at:
        f = fixes[index]
        return {
            "lat": f["lat"],
            "lon": f["lon"],
            "method": "historical_fix",
            "fix_time": f["time"],
            "age_hours": 0.0,
        }
    if index == 0 or index == len(times):
        return None
    left, right = fixes[index - 1], fixes[index]
    gap = (times[index] - times[index - 1]).total_seconds() / 3600
    if gap > 6:
        return None
    weight = (at - times[index - 1]).total_seconds() / (gap * 3600)
    delta = ((right["lon"] - left["lon"] + 180) % 360) - 180
    return {
        "lat": left["lat"] + weight * (right["lat"] - left["lat"]),
        "lon": (left["lon"] + weight * delta + 180) % 360 - 180,
        "method": "historical_interpolation",
        "fix_time": left["time"],
        "age_hours": gap * weight,
    }


def live_center(advisory, at):
    age = (utc(at) - utc(advisory["time"])).total_seconds() / 3600
    if age < 0 or age > 6:
        return None
    lat, lon = advisory["lat"], advisory["lon"]
    if age > 0:
        direction, speed = advisory.get("motion_direction"), advisory.get(
            "motion_speed_kt"
        )
        if speed == 0 and direction is None:
            direction = 0
        if (
            direction is None
            or speed is None
            or speed < 0
            or speed > 100
            or not 0 <= direction <= 360
        ):
            return None
        bearing = math.radians(direction)
        distance = speed * 1.852 * age / 6371
        phi, lam = math.radians(lat), math.radians(lon)
        phi2 = math.asin(
            math.sin(phi) * math.cos(distance)
            + math.cos(phi) * math.sin(distance) * math.cos(bearing)
        )
        lam2 = lam + math.atan2(
            math.sin(bearing) * math.sin(distance) * math.cos(phi),
            math.cos(distance) - math.sin(phi) * math.sin(phi2),
        )
        lat, lon = math.degrees(phi2), (math.degrees(lam2) + 180) % 360 - 180
    return {
        "lat": lat,
        "lon": lon,
        "method": "motion_estimate" if age else "advisory_fix",
        "fix_time": advisory["time"],
        "age_hours": age,
    }


def latest_live_center(advisory, fixes, at):
    """Use the latest non-future position and only a fresh reported motion."""
    at = utc(at)
    eligible = [(fix, False) for fix in fixes if utc(fix["time"]) <= at]
    if utc(advisory["time"]) <= at:
        eligible.append((advisory, True))
    if not eligible:
        return None
    fix, is_advisory = max(eligible, key=lambda item: (utc(item[0]["time"]), item[1]))
    motion_age = (at - utc(advisory["time"])).total_seconds() / 3600
    current = dict(fix)
    if not is_advisory and 0 <= motion_age <= 6:
        current.update(
            motion_direction=advisory.get("motion_direction"),
            motion_speed_kt=advisory.get("motion_speed_kt"),
        )
    center = live_center(current, at)
    if center:
        center["position_source"] = "advisory" if is_advisory else "track"
        center["motion_time"] = advisory["time"] if center["age_hours"] else None
        if not is_advisory and not center["age_hours"]:
            center["method"] = "live_track_fix"
    return center
