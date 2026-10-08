"""Restartable discovery, hourly inference, and causally isolated forecasts."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import timedelta
from itertools import groupby
import json
import logging
import time
import multiprocessing
from aiohttp import ClientError
from urllib.error import URLError

from .common import fetch, hour, hours, iso, utc, year_before
from .models import version
from .schedule import slots, qualified
from .satellite import DataGap, acquire, shared_reads
from .tracks import (
    CURRENT,
    ATCF,
    parse_current,
    parse_atcf,
    track_urls,
    historical_center,
    live_center,
    latest_live_center,
)

LOG = logging.getLogger(__name__)
_worker_models = None


def initialize_worker(model_root, device, manifest_path=None):
    from .models import Models, select_manifest
    import torch

    if manifest_path:
        select_manifest(manifest_path)
    if device == "cuda" and torch.cuda.device_count() > 1:
        index = multiprocessing.current_process()._identity[0] - 1
        device = f"cuda:{index % torch.cuda.device_count()}"
    global _worker_models
    _worker_models = Models(model_root, device)


def process_group_worker(tasks):
    with shared_reads():
        return [process(storm, at, "hindcast", _worker_models) for storm, at in tasks]


def discover(store, now=None):
    now = utc(now)
    try:
        body = fetch(CURRENT)
        active = parse_current(body)
        snapshot = store.snapshot(CURRENT, body, now)
    except (URLError, OSError, ValueError, KeyError) as error:
        previous = store.get_status("discovery") or {}
        store.put_status(
            "discovery", {**previous, "last_attempt": iso(now), "error": str(error)}
        )
        raise
    ids = {s["id"] for s in active}
    for previous in store.storms():
        if previous.get("active") and previous["id"] not in ids:
            store.put_storm({**previous, "active": False})
    for item in active:
        previous = store.storm(item["id"]) or {}
        storm = {
            **previous,
            **item,
            "active": True,
            "advisory_snapshot": snapshot,
            "advisory_retrieved_at": iso(now),
        }
        if (
            not storm.get("prediction_qualified_at")
            and qualified(item["advisory"].get("classification"))
            and utc(item["advisory"]["time"]) <= now
        ):
            # Only observed nonfuture advice opens the live gate. Keep its
            # issue time and the observation time separately for provenance.
            storm["prediction_qualified_at"] = item["advisory"]["time"]
            storm["prediction_qualification_observed_at"] = iso(now)
        try:
            url = f"{ATCF}/btk/b{item['id'].lower()}.dat"
            tracks_body = fetch(url)
            fixes = parse_atcf(tracks_body, item["id"])
            if fixes:
                storm.update(
                    track=fixes,
                    track_snapshot=store.snapshot(url, tracks_body, now),
                    track_retrieved_at=iso(now),
                )
                storm.pop("track_error", None)
        except (URLError, OSError, ValueError) as error:
            storm["track_error"] = str(error)
        fixes = storm.get("track", [])
        storm["start"] = fixes[0]["time"] if fixes else item["advisory"]["time"]
        storm["end"] = max([item["advisory"]["time"]] + [f["time"] for f in fixes])
        store.put_storm(storm)
    store.put_status(
        "discovery",
        {
            "last_attempt": iso(now),
            "last_success": iso(now),
            "error": None,
            "active_count": len(active),
            "snapshot": snapshot,
        },
    )
    return active


def discover_history(store, start, end):
    end = min(utc(end), utc())
    failures = []
    for year in range(utc(start).year, utc(end).year + 1):
        try:
            urls = track_urls(year)
        except (OSError, URLError, ValueError) as error:
            failures.append(
                {"year": year, "error": str(error), "stage": "track_listing"}
            )
            continue
        for sid, url in urls:
            try:
                body = fetch(url)
                fixes = parse_atcf(body, sid)
                if (
                    not fixes
                    or utc(fixes[-1]["time"]) < utc(start)
                    or utc(fixes[0]["time"]) > utc(end)
                ):
                    continue
                previous = store.storm(sid) or {}
                names = [
                    f["name"]
                    for f in fixes
                    if f["name"] and f["name"].upper() not in {"INVEST", "UNNAMED"}
                ]
                store.put_storm(
                    {
                        **previous,
                        "id": sid,
                        "name": previous.get("name") or (names[-1] if names else sid),
                        "basin": previous.get("basin")
                        or (
                            "AL"
                            if sid.startswith("AL")
                            else ("CP" if fixes[-1]["lon"] < -140 else "EP")
                        ),
                        "active": previous.get("active", False),
                        "track": fixes,
                        "start": fixes[0]["time"],
                        "end": (
                            max(
                                fixes[-1]["time"],
                                previous.get("advisory", {}).get("time", ""),
                            )
                            if previous.get("active")
                            else fixes[-1]["time"]
                        ),
                        "track_snapshot": store.snapshot(url, body),
                        "track_retrieved_at": iso(),
                    }
                )
                LOG.info("track %s %s %s", sid, fixes[0]["time"], fixes[-1]["time"])
            except (OSError, URLError, ValueError) as error:
                failures.append({"storm_id": sid, "url": url, "error": str(error)})
    store.put_status(
        "history",
        {
            "start": iso(start),
            "end": iso(end),
            "retrieved_at": iso(),
            "failures": failures,
        },
    )
    if failures:
        raise RuntimeError(
            f"Historical discovery incomplete for {len(failures)} storms; see history status"
        )


def process(storm, at, kind, models, available_at=None):
    started = time.monotonic()
    at = utc(at)
    center = (
        latest_live_center(storm["advisory"], storm.get("track", []), at)
        if kind == "live"
        else historical_center(storm.get("track", []), at)
    )
    previous = storm.get("_rebuild_source") if kind == "hindcast" else None
    if previous:
        center = previous["center"]
    row = {
        "storm_id": storm["id"],
        "time": iso(at),
        "kind": kind,
        "model_version": version("nowcast"),
        "generated_at": iso(),
        "status": "gap",
        "metrics": None,
        "center": center,
        "source_snapshot": (
            storm.get("advisory_snapshot")
            if kind == "live"
            else storm.get("track_snapshot")
        ),
    }
    if previous:
        row.update(
            source_snapshot=previous.get("source_snapshot"),
            source_snapshots=previous.get("source_snapshots", []),
            reprocessed_from={
                "model_version": previous["model_version"],
                "kind": previous["kind"],
                "generated_at": previous["generated_at"],
            },
        )
    if kind == "live" and center and center.get("position_source") == "track":
        row["source_snapshot"] = storm.get("track_snapshot")
        row["source_snapshots"] = [
            key
            for key in (
                storm.get("track_snapshot"),
                storm.get("advisory_snapshot") if center["age_hours"] else None,
            )
            if key
        ]
    if center is None:
        return {
            **row,
            "reason": "stale_or_missing_center",
            "detail": (
                "Recorded fixes do not bracket this hour within six hours."
                if kind == "hindcast"
                else "No nonfuture position and motion report is fresh enough for this hour."
            ),
            "retryable": True,
        }
    try:
        array, mask, source = acquire(
            at, center, available_at if kind == "live" else None
        )
        values = models.infer(array, mask, center["lat"], center["lon"], source["end"])
        return {
            **row,
            "status": "ready",
            "metrics": values,
            "imagery": source,
            "reason": None,
            "retryable": False,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "generated_at": iso(),
        }
    except DataGap as error:
        return {**row, "reason": error.reason, "detail": str(error), "retryable": True}
    except (URLError, TimeoutError, OSError, ClientError) as error:
        return {
            **row,
            "reason": "source_unavailable",
            "detail": str(error)[:500],
            "retryable": True,
        }


def forecast_rows(samples, models, existing=None, issued_at=None):
    lookup = {
        (r["storm_id"], r["time"], r["kind"], r["model_version"]): r
        for r in samples
        if r["status"] == "ready"
    }
    for row in samples:
        if row["status"] != "ready":
            continue
        if existing and (row["time"], row["kind"], row["model_version"]) in existing:
            continue
        at = utc(row["time"])
        cutoff = utc(issued_at)

        def input_at(offset):
            stamp = iso(at - timedelta(hours=offset))
            kinds = (
                ("live", "hindcast")
                if row["kind"] == "live" and offset
                else (("hindcast", "live") if offset else (row["kind"],))
            )
            for kind in kinds:
                candidate = lookup.get(
                    (row["storm_id"], stamp, kind, row["model_version"])
                )
                if candidate is not None and (
                    row["kind"] != "live"
                    or (at <= cutoff and utc(candidate["generated_at"]) <= cutoff)
                ):
                    return candidate
            return None

        anchors = [input_at(h) for h in (0, 6, 12)]
        if any(a is None for a in anchors):
            continue
        values = [a["metrics"]["vmax_ms"] for a in anchors]
        predictions = models.predict_future(*values)
        yield {
            "storm_id": row["storm_id"],
            "anchor_time": row["time"],
            "kind": row["kind"],
            "generated_at": iso(),
            "model_version": version("forecast") + ":" + version("nowcast"),
            "input_model_version": row["model_version"],
            "input_times": [a["time"] for a in anchors],
            "input_kinds": [a["kind"] for a in anchors],
            "input_generated_at": [a["generated_at"] for a in anchors],
            "input_vmax_ms": values,
            "experimental": True,
            "predictions": [
                {
                    "lead_hours": lead,
                    "valid_time": iso(at + timedelta(hours=lead)),
                    "vmax_ms": prediction,
                }
                for lead, prediction in zip((6, 12), predictions)
            ],
        }


def generate_forecasts(store, models, sid, live_anchor=None):
    existing = {
        (f["anchor_time"], f["kind"], f["input_model_version"])
        for f in store.forecasts(sid)
        if f["model_version"] == version("forecast") + ":" + version("nowcast")
    }
    samples = store.samples(sid, version("nowcast"))
    # A historical rerun must not manufacture a previously unissued live issue.
    existing.update(
        (row["time"], row["kind"], row["model_version"])
        for row in samples
        if row["kind"] == "live" and row["time"] != live_anchor
    )
    for row in forecast_rows(samples, models, existing):
        store.put_forecast(row)


def update(store, models, now=None, workers=4, model_root="downloads/models"):
    now = utc(now)
    active = discover(store, now)
    results = []
    for item in active:
        storm = store.storm(item["id"])
        at = hour(now)
        if not list(slots(storm, at, at, live=True)):
            continue
        previous = store.sample(
            storm["id"], at, "live", version("nowcast")
        ) or store.sample(storm["id"], at, "hindcast", version("nowcast"))
        if previous:
            generate_forecasts(store, models, storm["id"], live_anchor=iso(at))
            continue
        result = process(storm, at, "live", models, now)
        store.put_sample(result, retry=True)
        generate_forecasts(store, models, storm["id"], live_anchor=iso(at))
        results.append(result)
    # Scan the entire retained window, rather than a high-water mark: a newer
    # live result must never conceal interior holes or storms ended while offline.
    start = hour(year_before(now)) - timedelta(hours=12)
    discover_history(store, start, now)
    caught_up = backfill(
        store,
        models,
        start,
        now,
        workers=workers,
        model_root=model_root,
        reconcile=True,
    )
    if not store.get_status("backfill")["complete"]:
        raise RuntimeError(
            "Update catch-up interrupted; committed work will resume next update"
        )
    # Catch-up may have supplied missing forecast context, including after a
    # restart between committing a prediction and its forecast.
    for storm in store.storms():
        generate_forecasts(
            store,
            models,
            storm["id"],
            live_anchor=iso(hour(now)) if storm.get("active") else None,
        )
    store.put_status(
        "update",
        {
            "last_completed": iso(),
            "requested_at": iso(now),
            "results": len(results),
            "ready": sum(r["status"] == "ready" for r in results),
            "catchup": caught_up,
            "catchup_start": iso(start),
            "catchup_end": iso(hour(now)),
        },
    )
    return results


def backfill(
    store,
    models,
    start,
    end,
    workers=4,
    limit=None,
    retry_gaps=False,
    storm_ids=None,
    model_root="downloads/models",
    reconcile=False,
):
    end = min(utc(end), utc())
    tasks = []
    storms = store.storms()
    for storm in storms:
        if storm_ids and storm["id"] not in storm_ids:
            continue
        first, last = max(utc(start), utc(storm["start"])), min(
            utc(end), utc(end) if storm.get("active") else utc(storm["end"])
        )
        from .models import MANIFEST

        baseline = MANIFEST.get("baseline_model_version")
        originals = {}
        if baseline:
            for row in store.samples(storm["id"], baseline):
                if row["status"] == "ready" and row.get("center"):
                    originals[row["time"]] = row
        for at in slots(storm, first, last):
            if reconcile:
                live = store.sample(storm["id"], at, "live", version("nowcast"))
                if live:
                    continue
            previous = store.sample(storm["id"], at, "hindcast", version("nowcast"))
            if previous:
                if previous["status"] == "ready":
                    continue
                if not retry_gaps:
                    continue
            source = originals.get(iso(at))
            tasks.append(
                ({**storm, "_rebuild_source": source} if source else storm, at)
            )
    tasks.sort(key=lambda item: item[1], reverse=True)
    if limit is not None:
        tasks = tasks[:limit]
    LOG.info("backfill queued %d hours (%d workers)", len(tasks), workers)
    counts = {"ready": 0, "gap": 0}
    # Bounded submission avoids retaining thousands of futures or satellite crops.
    from pathlib import Path

    interrupted = False
    groups = [list(group) for _, group in groupby(tasks, key=lambda item: item[1])]
    iterator = iter(groups)
    touched = set()
    last_forecast_count = 0
    from .models import MANIFEST_PATH

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_worker,
        initargs=(str(model_root), models.device, str(MANIFEST_PATH)),
    ) as pool:
        futures = {
            pool.submit(process_group_worker, next(iterator))
            for _ in range(min(workers, len(groups)))
        }
        while futures:
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                futures.remove(future)
                for row in future.result():
                    store.put_sample(row, retry=retry_gaps)
                    touched.add(row["storm_id"])
                    counts[row["status"]] += 1
                    LOG.info(
                        "%s %s %s %s",
                        row["storm_id"],
                        row["time"],
                        row["status"],
                        row.get("reason") or "",
                    )
                interrupted = (
                    interrupted
                    or Path(str(store.path) + ".update-requested").exists()
                    or Path(str(store.path) + ".stop-backfill").exists()
                )
                task = None if interrupted else next(iterator, None)
                if task:
                    futures.add(pool.submit(process_group_worker, task))
            store.put_status(
                "backfill",
                {
                    "start": iso(start),
                    "end": iso(end),
                    "last_progress": iso(),
                    "counts": counts,
                    "queued": len(tasks),
                    "processed": sum(counts.values()),
                    "complete": False,
                },
            )
            if sum(counts.values()) - last_forecast_count >= 32:
                for sid in touched:
                    generate_forecasts(store, models, sid)
                touched.clear()
                last_forecast_count = sum(counts.values())
    for sid in touched:
        generate_forecasts(store, models, sid)
    store.put_status(
        "backfill",
        {
            "start": iso(start),
            "end": iso(end),
            "last_progress": iso(),
            "counts": counts,
            "queued": len(tasks),
            "processed": sum(counts.values()),
            "complete": not interrupted,
        },
    )
    return counts
