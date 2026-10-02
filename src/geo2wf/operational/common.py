"""Small, deterministic helpers shared by operational components."""

from __future__ import annotations

import calendar
import hashlib
import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import time

UTC = timezone.utc
KNOT = 0.514444


def utc(value=None):
    if value is None:
        return datetime.now(UTC)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("timestamps must include a timezone")
        return value.astimezone(UTC)
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return (
        result.replace(tzinfo=UTC) if result.tzinfo is None else result.astimezone(UTC)
    )


def iso(value=None):
    return utc(value).isoformat(timespec="seconds").replace("+00:00", "Z")


def hour(value=None):
    return utc(value).replace(minute=0, second=0, microsecond=0)


def year_before(value=None):
    value = utc(value)
    year = value.year - 1
    return value.replace(
        year=year, day=min(value.day, calendar.monthrange(year, value.month)[1])
    )


def hours(start, end):
    current = hour(start)
    if current < utc(start):
        current += timedelta(hours=1)
    while current <= utc(end):
        yield current
        current += timedelta(hours=1)


def encoded(value):
    return json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()


def digest(value):
    return hashlib.sha256(
        value if isinstance(value, bytes) else encoded(value)
    ).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_bytes(encoded(value) + b"\n")
    temporary.replace(path)


def fetch(url, *, attempts=3, timeout=40, limit=16 * 1024 * 1024):
    """Bounded metadata downloads; errors never become successful empty feeds."""
    for attempt in range(attempts):
        try:
            with urlopen(
                Request(url, headers={"User-Agent": "StormSense/1 research"}),
                timeout=timeout,
            ) as response:
                body = response.read(limit + 1)
            if len(body) > limit:
                raise ValueError(f"metadata response exceeds limit: {url}")
            return body
        except HTTPError as error:
            if error.code < 500 and error.code != 429:
                raise
            if attempt == attempts - 1:
                raise
        except (URLError, TimeoutError, OSError):
            if attempt == attempts - 1:
                raise
        time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def finite(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def category(wind_ms):
    if wind_ms is None:
        return None
    knots = wind_ms / KNOT
    for level, threshold in [(5, 137), (4, 113), (3, 96), (2, 83), (1, 64), (0, 34)]:
        if knots >= threshold - 1e-5:
            return level
    return -1
