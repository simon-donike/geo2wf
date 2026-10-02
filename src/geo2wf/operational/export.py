"""Compact, immutable browser releases; unchanged storm objects are reused."""

from __future__ import annotations

from collections import Counter
from datetime import timedelta
import json
from pathlib import Path
import re
import subprocess

from . import SCHEMA_VERSION
from .common import (
    category,
    digest,
    encoded,
    hour,
    hours,
    iso,
    utc,
    write_json,
    year_before,
)
from .models import MANIFEST, version

REMOTE = "r2:tcd/explorer/stormsense"


def coverage(store, start, end):
    result = []
    for storm in store.storms():
        begin, finish = max(utc(start), utc(storm["start"])), min(
            utc(end), utc(end) if storm.get("active") else utc(storm["end"])
        )
        expected = {iso(t) for t in hours(begin, finish)}
        if not expected:
            continue
        samples = {}
        for sample in store.samples(storm["id"], version("nowcast")):
            if sample["time"] not in expected:
                continue
            previous = samples.get(sample["time"])
            if previous is None or (
                sample["status"] == "ready"
                and (previous["status"] != "ready" or sample["kind"] == "live")
            ):
                samples[sample["time"]] = sample
        ready = sum(s["status"] == "ready" for s in samples.values())
        reasons = Counter(
            s.get("reason", "unknown")
            for s in samples.values()
            if s["status"] != "ready"
        )
        result.append(
            {
                "storm_id": storm["id"],
                "expected": len(expected),
                "predictions": ready,
                "gaps": sum(reasons.values()),
                "pending": len(expected) - len(samples),
                "reasons": dict(reasons),
            }
        )
    discovery = store.get_status("history")
    complete = bool(
        discovery
        and not discovery.get("failures")
        and discovery["start"] <= iso(start)
        and hour(discovery["end"]) >= hour(end)
        and all(row["pending"] == 0 for row in result)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "start": iso(start),
        "end": iso(end),
        "generated_at": iso(),
        "storms": result,
        "totals": {
            key: sum(r[key] for r in result)
            for key in ("expected", "predictions", "gaps", "pending")
        },
        "discovery": discovery,
        "complete": complete,
    }


def export_release(store, output, start=None, end=None, release=None):
    with store.read_snapshot():
        return _export_release(store, output, start, end, release)


def _export_release(store, output, start=None, end=None, release=None):
    output = Path(output)
    end = utc(end)
    start = utc(start) if start else year_before(end)
    release = release or utc().strftime("%Y%m%dT%H%M%S%fZ")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", release):
        raise ValueError("invalid release ID")
    release_dir = output / "releases" / release
    if release_dir.exists():
        raise FileExistsError("release IDs must be immutable")
    (output / "objects").mkdir(parents=True, exist_ok=True)
    report = coverage(store, start, end)
    accounted = {row["storm_id"]: row for row in report["storms"]}
    summaries = []
    for storm in store.storms():
        records = [
            s
            for s in store.samples(storm["id"], version("nowcast"))
            if start <= utc(s["time"]) <= end
        ]
        track = [f for f in storm.get("track", []) if start <= utc(f["time"]) <= end]
        if not track and not records and not storm.get("active"):
            continue
        forecasts = [
            f
            for f in store.forecasts(storm["id"])
            if start <= utc(f["anchor_time"]) <= end
        ]
        provenance = {
            key: store.source(storm.get(key + "_snapshot"))
            for key in ("track", "advisory")
        }
        provenance["track_retrieved_at"] = storm.get("track_retrieved_at")
        provenance["advisory_retrieved_at"] = storm.get("advisory_retrieved_at")
        provenance["record_sources"] = {
            key: store.source(key)
            for key in {
                key
                for r in records
                for key in [r.get("source_snapshot"), *r.get("source_snapshots", [])]
            }
            if key
        }
        series = {
            "schema_version": SCHEMA_VERSION,
            "storm_id": storm["id"],
            "track": track,
            "records": records,
            "forecasts": forecasts,
            "provenance": provenance,
        }
        object_path = f"objects/{digest(series)}.json"
        if not (output / object_path).exists():
            write_json(output / object_path, series)
        ready = [r for r in records if r["status"] == "ready"]
        latest = (
            max(ready, key=lambda r: (r["time"], r["kind"] == "live"))
            if ready
            else None
        )
        metrics = latest["metrics"] if latest else None
        last24 = next(
            (
                r
                for r in reversed(ready)
                if latest
                and r["time"] == iso(utc(latest["time"]) - timedelta(hours=24))
            ),
            None,
        )
        unique = {}
        for row in records:
            previous = unique.get(row["time"])
            if previous is None or (
                row["status"] == "ready"
                and (previous["status"] != "ready" or row["kind"] == "live")
            ):
                unique[row["time"]] = row
        counts = accounted.get(storm["id"], {})
        advisory = storm.get("advisory")
        references = ([advisory] if advisory else []) + track[-1:]
        latest_fix = (
            max(references, key=lambda fix: fix["time"]) if references else None
        )
        levels = [
            category(f["wind_ms"])
            for f in storm.get("track", [])
            if f.get("wind_ms") is not None
        ]
        summaries.append(
            {
                "id": storm["id"],
                "name": storm["name"],
                "basin": storm["basin"],
                "active": bool(storm.get("active")),
                "start": max(storm["start"], iso(start)),
                "end": storm["end"],
                "advisory": advisory,
                "latest_fix": latest_fix,
                "peak_category": storm.get(
                    "peak_official_category", max(levels) if levels else None
                ),
                "latest_prediction": latest,
                "metrics": metrics,
                "change_24h_ms": (
                    metrics["vmax_ms"] - last24["metrics"]["vmax_ms"]
                    if last24
                    else None
                ),
                "change_24h_reference_kind": last24["kind"] if last24 else None,
                "prediction_count": counts.get("predictions", 0),
                "gap_count": counts.get("gaps", 0),
                "record_count": len(unique),
                "expected_count": counts.get("expected", 0),
                "pending_count": counts.get("pending", 0),
                "series": object_path,
            }
        )
    evaluation = store.get_status("evaluation")
    catalog = {
        "schema_version": SCHEMA_VERSION,
        "release": release,
        "generated_at": iso(),
        "window": {"start": iso(start), "end": iso(end)},
        "units": {"wind": "m/s", "radius": "km", "time": "UTC"},
        "models": {
            role: {"id": m["id"], "version": version(role), "sha256": m["sha256"]}
            for role, m in MANIFEST["models"].items()
        },
        "source_status": {
            key: store.get_status(key)
            for key in ("discovery", "update", "backfill", "history")
        },
        "coverage": report["totals"],
        "reports": {
            "coverage": f"releases/{release}/coverage.json",
            "evaluation": f"releases/{release}/evaluation.json" if evaluation else None,
        },
        "storms": sorted(
            summaries, key=lambda s: (not s["active"], -utc(s["end"]).timestamp())
        ),
    }
    write_json(release_dir / "catalog.json", catalog)
    write_json(release_dir / "coverage.json", report)
    if evaluation:
        write_json(release_dir / "evaluation.json", evaluation)
    pointer = {
        "schema_version": SCHEMA_VERSION,
        "version": release,
        "manifest": f"releases/{release}/catalog.json",
    }
    write_json(output / "latest.json", pointer)
    return catalog


def publish(output, remote=REMOTE, run=subprocess.run):
    if remote != REMOTE:
        raise ValueError(f"Publication is restricted to {REMOTE}")
    output = Path(output)
    pointer = json.loads((output / "latest.json").read_text())
    release = pointer["version"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", release):
        raise ValueError("invalid release version")
    catalog = json.loads((output / "releases" / release / "catalog.json").read_text())
    for storm in catalog["storms"]:
        if not (output / storm["series"]).is_file():
            raise FileNotFoundError(storm["series"])
    run(
        [
            "rclone",
            "copy",
            str(output / "objects"),
            remote + "/objects",
            "--immutable",
            "--include",
            "*.json",
        ],
        check=True,
    )
    run(
        [
            "rclone",
            "copy",
            str(output / "releases" / release),
            remote + "/releases/" + release,
            "--immutable",
            "--include",
            "*.json",
        ],
        check=True,
    )
    run(
        ["rclone", "copyto", str(output / "latest.json"), remote + "/latest.json"],
        check=True,
    )


def prune_local(output, keep=7, apply=False):
    """Reference-aware local release GC; never apply age rules to shared objects."""
    output = Path(output)
    latest = json.loads((output / "latest.json").read_text())["version"]
    releases = sorted((output / "releases").glob("*/catalog.json"), reverse=True)
    retained = set(p.parent.name for p in releases[: max(keep, 1)]) | {latest}
    references = set()
    for path in releases:
        if path.parent.name in retained:
            references.update(
                s["series"] for s in json.loads(path.read_text())["storms"]
            )
    obsolete = [
        p
        for p in (output / "objects").glob("*.json")
        if str(p.relative_to(output)) not in references
    ]
    obsolete += [
        p
        for path in releases
        if path.parent.name not in retained
        for p in path.parent.glob("*.json")
    ]
    if apply:
        for path in obsolete:
            path.unlink()
        for path in releases:
            if path.parent.name not in retained:
                path.parent.rmdir()
    return [str(p.relative_to(output)) for p in obsolete]


def prune_remote(keep=7, apply=False, run=subprocess.run):
    """Reference-aware GC for the single publishing runner, with pointer guards.

    A 24-hour grace period protects newly uploaded, not-yet-referenced objects.
    Serialize publication and GC on the runner; do not run a second publisher.
    """

    def read(key):
        return json.loads(
            run(
                ["rclone", "cat", REMOTE + "/" + key],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )

    pointer = read("latest.json")
    listing = json.loads(
        run(
            [
                "rclone",
                "lsjson",
                REMOTE,
                "--recursive",
                "--files-only",
                "--include",
                "*.json",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    files = {row["Path"]: row for row in listing}
    catalogs = sorted(
        (
            key
            for key in files
            if re.fullmatch(r"releases/[A-Za-z0-9_-]+/catalog\.json", key)
        ),
        reverse=True,
    )
    latest = f"releases/{pointer['version']}/catalog.json"
    if latest not in catalogs:
        raise ValueError("Remote latest catalog is absent; refusing retention")
    cutoff = utc() - timedelta(days=1)
    retained = (
        set(catalogs[: max(1, keep)])
        | {latest}
        | {key for key in catalogs if utc(files[key]["ModTime"]) >= cutoff}
    )
    references = set()
    for key in retained:
        catalog = read(key)
        for storm in catalog["storms"]:
            path = storm["series"]
            if (
                not re.fullmatch(r"objects/[a-f0-9]{64}\.json", path)
                or path not in files
            ):
                raise ValueError(
                    "Remote catalog has an invalid or missing series; refusing retention"
                )
            references.add(path)
    obsolete_versions = {key.split("/")[1] for key in catalogs if key not in retained}
    obsolete = []
    for key, row in files.items():
        old_release = (
            bool(
                re.fullmatch(
                    r"releases/[A-Za-z0-9_-]+/(catalog|coverage|evaluation)\.json", key
                )
            )
            and key.split("/")[1] in obsolete_versions
        )
        unused_object = (
            bool(re.fullmatch(r"objects/[a-f0-9]{64}\.json", key))
            and key not in references
        )
        if (old_release or unused_object) and utc(row["ModTime"]) < cutoff:
            obsolete.append(key)
    if apply:
        # Recheck catalogs as well as the pointer before deleting any objects.
        after = json.loads(
            run(
                [
                    "rclone",
                    "lsjson",
                    REMOTE + "/releases",
                    "--recursive",
                    "--files-only",
                    "--include",
                    "catalog.json",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        if read("latest.json") != pointer or {
            "releases/" + row["Path"] for row in after
        } != set(catalogs):
            raise RuntimeError(
                "A remote publisher changed the release set; retention aborted"
            )
        for key in sorted(obsolete, reverse=True):
            run(["rclone", "deletefile", REMOTE + "/" + key], check=True)
    return obsolete
