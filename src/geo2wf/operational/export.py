"""Compact, immutable browser releases; unchanged storm objects are reused."""

from __future__ import annotations

from collections import Counter
from datetime import timedelta
import json
from pathlib import Path
import re
import shutil
import subprocess

from . import SCHEMA_VERSION
from .common import (
    digest,
    hour,
    hours,
    iso,
    utc,
    write_json,
    year_before,
)
from .models import MANIFEST, version
from .geocolor import VERSION as IMAGE_VERSION, asset_root, frame_files
from .image_bundles import daily_bundles, display_slot
from .intensification import official_summary
from .schedule import visible_slots, metadata, cadence

REMOTE = "r2:tcd/explorer/stormsense"
IMAGE_PATH = r"imagery/[a-f0-9]{64}\.(?:json|webp(?:\.aux\.xml)?)"
BUNDLE_PATH = r"bundles/[a-f0-9]{64}\.zip"
ASSET_PATH = rf"(?:{IMAGE_PATH}|{BUNDLE_PATH})"


def image_references(catalog, read):
    """A release pins its complete asset set without bloating the browser catalog."""
    manifest = catalog.get("imagery", {}).get("manifest")
    if not manifest:
        return set()
    if not re.fullmatch(r"objects/[a-f0-9]{64}\.json", manifest):
        raise ValueError("invalid imagery manifest")
    paths = read(manifest)["files"]
    if any(not re.fullmatch(ASSET_PATH, path) for path in paths):
        raise ValueError("invalid imagery asset path")
    return {manifest, *paths}


def coverage(store, start, end):
    result = []
    for storm in store.storms():
        begin, finish = max(utc(start), utc(storm["start"])), min(
            utc(end), utc(end) if storm.get("active") else utc(storm["end"])
        )
        expected = {iso(t) for t in visible_slots(storm, begin, finish)}
        if finish < begin:
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
                "prediction_schedule": metadata(storm),
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
    image_files = set()
    image_counts = Counter()
    for storm in store.storms():
        expected_slots = {iso(t) for t in visible_slots(storm, start, end)}
        records = [
            s
            for s in store.samples(storm["id"], version("nowcast"))
            if start <= utc(s["time"]) <= end
            and (cadence() == 1 or s["time"] in expected_slots)
        ]
        track = [f for f in storm.get("track", []) if start <= utc(f["time"]) <= end]
        if not track and not records and not storm.get("active"):
            continue
        forecasts = [
            f
            for f in store.forecasts(storm["id"])
            if start <= utc(f["anchor_time"]) <= end
            and f["model_version"] == version("forecast") + ":" + version("nowcast")
            and (cadence() == 1 or f["anchor_time"] in expected_slots)
        ]
        visuals = [
            v
            for v in store.visuals(storm["id"], IMAGE_VERSION)
            if start <= utc(v["time"]) <= end and display_slot(v["time"])
        ]
        visual_counts = {
            "ready": sum(v["status"] == "ready" for v in visuals),
            "gaps": sum(v["status"] == "gap" for v in visuals),
        }
        for visual in visuals:
            for key in frame_files(visual):
                if not re.fullmatch(IMAGE_PATH, key):
                    raise ValueError("invalid imagery asset path")
                source, target = asset_root(store) / key, output / key
                if not source.is_file():
                    raise FileNotFoundError(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copy2(source, target)
                image_files.add(key)
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
            "prediction_schedule": metadata(storm),
            "track": track,
            "records": records,
            "forecasts": forecasts,
            "provenance": provenance,
        }
        if visuals:
            series["imagery"] = [
                {k: v for k, v in frame.items() if k != "files"} for frame in visuals
            ]
            series["imagery_bundles"] = daily_bundles(storm["id"], visuals, output)
            image_files.update(b["path"] for b in series["imagery_bundles"])
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
        visual_counts["expected"] = sum(
            display_slot(t)
            for t in hours(
                max(start, utc(storm["start"])),
                min(end, end if storm.get("active") else utc(storm["end"])),
            )
        )
        visual_counts["pending"] = max(0, visual_counts["expected"] - len(visuals))
        image_counts.update(visual_counts)
        advisory = storm.get("advisory")
        references = ([advisory] if advisory else []) + track[-1:]
        latest_fix = (
            max(references, key=lambda fix: fix["time"]) if references else None
        )
        official = storm.get("official_summary") or official_summary(
            storm.get("track", []), advisory
        )
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
                "prediction_schedule": metadata(storm),
                "peak_category": max(
                    [
                        v
                        for v in [
                            storm.get("peak_official_category"),
                            official["peak_category"],
                        ]
                        if v is not None
                    ],
                    default=None,
                ),
                "peak_official_wind_ms": official["peak_wind_ms"],
                "has_ri": official["has_ri"],
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
                "imagery_coverage": visual_counts,
            }
        )
    evaluation = store.get_status("evaluation")
    image_manifest = {
        "schema_version": 1,
        "version": IMAGE_VERSION,
        "files": sorted(image_files),
    }
    image_manifest_path = f"objects/{digest(image_manifest)}.json"
    if not (output / image_manifest_path).exists():
        write_json(output / image_manifest_path, image_manifest)
    catalog = {
        "schema_version": SCHEMA_VERSION,
        "release": release,
        "prediction_cadence_hours": cadence(),
        "quality_gate": store.get_status("migration_quality"),
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
        "imagery": {
            "version": IMAGE_VERSION,
            "manifest": image_manifest_path,
            "coverage": dict(image_counts),
            "source": "NASA GIBS GOES GeoColor",
            "status": store.get_status("geocolor"),
        },
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


def publish(output, remote=REMOTE, run=subprocess.run, advance_pointer=True):
    if remote != REMOTE:
        raise ValueError(f"Publication is restricted to {REMOTE}")
    output = Path(output)
    pointer = json.loads((output / "latest.json").read_text())
    release = pointer["version"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", release):
        raise ValueError("invalid release version")
    catalog = json.loads((output / "releases" / release / "catalog.json").read_text())
    if catalog.get("prediction_cadence_hours", 1) == 2:
        gate = catalog.get("quality_gate") or {}
        if not (
            gate.get("passed")
            and gate.get("phase") == "full"
            and gate.get("candidate_version") == catalog["models"]["nowcast"]["version"]
        ):
            raise ValueError(
                "Two-hour publication requires a passing full-archive quality gate for this model"
            )
    for storm in catalog["storms"]:
        if not (output / storm["series"]).is_file():
            raise FileNotFoundError(storm["series"])
    assets = image_references(
        catalog, lambda key: json.loads((output / key).read_text())
    )
    for key in assets:
        if not (output / key).is_file():
            raise FileNotFoundError(key)
    if any(key.startswith("imagery/") for key in assets):
        # Every display image and georeferencing sidecar precedes the release
        # catalog and pointer. Never publish a partially available image archive.
        run(
            [
                "rclone",
                "copy",
                str(output / "imagery"),
                remote + "/imagery",
                "--immutable",
                "--checksum",
                "--s3-no-head",
                "--transfers",
                "16",
                "--checkers",
                "16",
                "--include",
                "*.webp",
                "--include",
                "*.webp.aux.xml",
                "--include",
                "*.json",
            ],
            check=True,
        )
    if any(key.startswith("bundles/") for key in assets):
        run(
            [
                "rclone",
                "copy",
                str(output / "bundles"),
                remote + "/bundles",
                "--immutable",
                "--checksum",
                "--s3-no-head",
                "--include",
                "*.zip",
            ],
            check=True,
        )
    # Restored exports can have different mtimes but identical bytes. Compare
    # hashes so repeated publication neither rejects nor rewrites R2 metadata.
    run(
        [
            "rclone",
            "copy",
            str(output / "objects"),
            remote + "/objects",
            "--immutable",
            "--checksum",
            "--s3-no-head",
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
            "--checksum",
            "--s3-no-head",
            "--include",
            "*.json",
        ],
        check=True,
    )
    # This endpoint returns a version ID but rejects version-specific HEADs.
    # Avoid rclone's post-PUT HEAD, then independently compare normal object
    # checksums before making the release visible. Upload success alone is not
    # sufficient to advance the pointer.
    directories = ["objects", "releases/" + release]
    if any(key.startswith("imagery/") for key in assets):
        directories.insert(0, "imagery")
    if any(key.startswith("bundles/") for key in assets):
        directories.insert(0, "bundles")
    for directory in directories:
        run(
            [
                "rclone",
                "check",
                str(output / directory),
                remote + "/" + directory,
                "--one-way",
                "--checkers",
                "16",
            ],
            check=True,
        )
    if advance_pointer:
        run(
            [
                "rclone",
                "copyto",
                "--checksum",
                "--s3-no-head",
                str(output / "latest.json"),
                remote + "/latest.json",
            ],
            check=True,
        )


def prune_local(output, keep=7, apply=False):
    """Reference-aware local release GC; never apply age rules to shared objects."""
    output = Path(output)
    latest = json.loads((output / "latest.json").read_text())["version"]
    releases = sorted((output / "releases").glob("*/catalog.json"), reverse=True)
    retained = set(p.parent.name for p in releases[: max(keep, 1)]) | {latest}
    retained.update(p.parent.name for p in releases if (p.parent / "pin.json").exists())
    references = set()
    for path in releases:
        if path.parent.name in retained:
            catalog = json.loads(path.read_text())
            references.update(s["series"] for s in catalog["storms"])
            references.update(
                image_references(
                    catalog, lambda key: json.loads((output / key).read_text())
                )
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
    obsolete += [
        p
        for p in (output / "imagery").glob("*")
        if re.fullmatch(IMAGE_PATH, str(p.relative_to(output)))
        and str(p.relative_to(output)) not in references
    ]
    obsolete += [
        p
        for p in (output / "bundles").glob("*.zip")
        if re.fullmatch(BUNDLE_PATH, str(p.relative_to(output)))
        and str(p.relative_to(output)) not in references
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

    def modified(row):
        # Rclone emits nanosecond RFC3339 times; Python 3.10 accepts at most
        # microseconds. Sub-microsecond precision is irrelevant to a day of grace.
        return utc(re.sub(r"(\.\d{6})\d+", r"\1", row["ModTime"]))

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
                # Retention grace starts at upload time. Also avoids a HEAD
                # request per image just to retrieve its original file mtime.
                "--use-server-modtime",
                "--no-mimetype",
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
        | {key for key in catalogs if modified(files[key]) >= cutoff}
        | {key for key in catalogs if key.replace("catalog.json", "pin.json") in files}
    )
    references = set()
    for key in retained:
        catalog = read(key)
        image_refs = image_references(catalog, read)
        if not image_refs.issubset(files):
            raise ValueError("Remote imagery assets are missing; refusing retention")
        references.update(image_refs)
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
            bool(
                re.fullmatch(r"objects/[a-f0-9]{64}\.json", key)
                or re.fullmatch(ASSET_PATH, key)
            )
            and key not in references
        )
        if (old_release or unused_object) and modified(row) < cutoff:
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
                    "--use-server-modtime",
                    "--no-mimetype",
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
