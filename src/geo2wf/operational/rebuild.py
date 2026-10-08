"""Stage and assess the two-hour migration without touching the live database.

Run with STORMSENSE_MODEL_MANIFEST pointing to models-finetuned.json.
The pilot is a quality gate: a failed comparison must not be published.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3

from .common import iso, utc, write_json, KNOT
from .models import MANIFEST, version
from .schedule import slots
from .store import Store
from .evaluate import training_storms


def reconcile(source, destination):
    """Merge a stopped publisher's latest state without losing candidate rows.

    Caller must hold the source cycle lock. New tracks may extend the candidate
    cohort, so invalidate the previous full quality gate before another export.
    """
    with sqlite3.connect(destination) as db:
        db.execute("ATTACH DATABASE ? AS production", (str(source.resolve()),))
        with db:
            for table in ("storms", "sources", "visuals", "status"):
                db.execute(
                    f"INSERT OR REPLACE INTO main.{table} SELECT * FROM production.{table}"
                )
            for table in ("samples", "forecasts"):
                db.execute(
                    f"INSERT OR IGNORE INTO main.{table} SELECT * FROM production.{table}"
                )
            db.execute("DELETE FROM main.status WHERE key='migration_quality'")
        db.execute("DETACH DATABASE production")


def prepare(source, destination, export, model_root):
    if destination.exists():
        raise FileExistsError("Staging database already exists; resume it instead")
    pointer = json.loads((export / "latest.json").read_text())
    catalog = json.loads((export / pointer["manifest"]).read_text())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(f"file:{source.resolve()}?mode=ro", uri=True) as src:
        with sqlite3.connect(destination) as dst:
            src.backup(dst)
    # Imagery is immutable and shared; no multi-gigabyte duplicate archive.
    imagery = destination.parent / "geocolor"
    original = source.parent / "geocolor"
    if not imagery.exists() and original.exists():
        imagery.symlink_to(original.resolve(), target_is_directory=True)
    store = Store(destination)
    try:
        seen = training_storms(model_root)
        counts = {}
        candidates = defaultdict(list)
        for storm in store.storms():
            if storm["id"] not in {s["id"] for s in catalog["storms"]}:
                continue
            start = utc(catalog["window"]["start"])
            # Keep pre-window context for six/twelve-hour forecasts.
            from datetime import timedelta

            expected = {
                iso(t)
                for t in slots(
                    storm, start - timedelta(hours=12), catalog["window"]["end"]
                )
            }
            ready = {
                r["time"] for r in store.samples(storm["id"]) if r["status"] == "ready"
            }
            counts[storm["id"]] = len(expected)
            if not storm.get("active") and storm["id"] not in seen:
                candidates[storm["basin"]].append((len(expected & ready), storm["id"]))
        pilot = [max(candidates[b])[1] for b in ("AL", "EP", "CP") if candidates[b]]
        active = sorted(
            s["id"] for s in store.storms() if s.get("active") and s["id"] in counts
        )
        if active:
            pilot.append(active[0])
        if len(pilot) < 4:
            raise ValueError(
                "Pilot requires three basin representatives and one active storm"
            )
        report = {
            "baseline_pointer": pointer,
            "window": catalog["window"],
            "pilot_storms": pilot,
            "eligible_slots": counts,
            "total_slots": sum(counts.values()),
            "candidate_version": version("nowcast"),
        }
        write_json(destination.parent / "migration.json", report)
        # Local pin is copied to R2 only during the approved cutover.
        write_json(
            export / pointer["manifest"].replace("catalog.json", "pin.json"),
            {"reason": "pre-finetuning rollback and image preservation"},
        )
        return report
    finally:
        store.close()


def compare(store, storm_ids, start, end, seen=frozenset()):
    original = json.loads(Path(__file__).with_name("models.json").read_text())
    from . import PIPELINE_VERSION

    def original_version(role):
        m = original["models"][role]
        return f"{PIPELINE_VERSION}:{m['id']}:{m['sha256'][:12]}"

    baseline = original_version("nowcast")
    candidate = version("nowcast")
    forecasts = {
        "baseline": original_version("forecast") + ":" + baseline,
        "candidate": version("forecast") + ":" + candidate,
    }
    errors = {name: defaultdict(list) for name in forecasts}
    missing = []
    membership = defaultdict(int)
    source_bytes = 0
    for storm in store.storms():
        if storm["id"] not in storm_ids:
            continue
        expected = {iso(t) for t in slots(storm, start, end)}
        rows = {}
        for name, model in (("baseline", baseline), ("candidate", candidate)):
            rows[name] = {}
            for r in store.samples(storm["id"], model):
                if r["time"] in expected and r["status"] == "ready":
                    old = rows[name].get(r["time"])
                    if old is None or r["kind"] == "hindcast":
                        rows[name][r["time"]] = r
        absent = set(rows["baseline"]) - set(rows["candidate"])
        missing.extend({"storm": storm["id"], "time": t} for t in sorted(absent))
        for row in rows["candidate"].values():
            source_bytes += row.get("imagery", {}).get("range_bytes", 0)
        for fix in storm.get("track", []):
            at = fix["time"]
            if at not in rows["baseline"] or at not in rows["candidate"]:
                continue
            membership["seen" if storm["id"] in seen else "unseen"] += 1
            targets = {
                "vmax_ms": fix.get("wind_ms"),
                **{key + "_km": val for key, val in fix.get("radii_km", {}).items()},
            }
            # Match the training label contract: below a wind threshold its
            # wind radius is physically absent, even when the B-deck omits it.
            if fix.get("wind_ms") is not None:
                for threshold in (34, 50, 64):
                    if fix["wind_ms"] < threshold * KNOT:
                        targets[f"r{threshold}_km"] = 0.0
            for key in ("vmax_ms", "rmw_km", "r34_km", "r50_km", "r64_km"):
                if targets.get(key) is not None:
                    for name in rows:
                        errors[name][key].append(
                            rows[name][at]["metrics"][key] - targets[key]
                        )
        reference = {f["time"]: f["wind_ms"] for f in storm.get("track", [])}
        frows = {name: {} for name in forecasts}
        for f in store.forecasts(storm["id"]):
            if f["anchor_time"] not in expected or f["kind"] != "hindcast":
                continue
            for name, model in forecasts.items():
                if f["model_version"] == model:
                    for p in f["predictions"]:
                        target = reference.get(p["valid_time"])
                        if target is not None:
                            frows[name][(f["anchor_time"], p["lead_hours"])] = (
                                p["vmax_ms"] - target
                            )
        for key in frows["baseline"].keys() & frows["candidate"].keys():
            for name in frows:
                errors[name][f"forecast_{key[1]}h"].append(frows[name][key])
    metrics = {
        name: {
            key: {
                "n": len(values),
                "mae": sum(map(abs, values)) / len(values),
                "bias": sum(values) / len(values),
            }
            for key, values in groups.items()
        }
        for name, groups in errors.items()
    }
    failures = []
    for key in (
        "vmax_ms",
        "rmw_km",
        "r34_km",
        "r50_km",
        "r64_km",
        "forecast_6h",
        "forecast_12h",
    ):
        a, b = metrics["baseline"].get(key), metrics["candidate"].get(key)
        if not a or not b:
            failures.append(f"No matched labels for {key}")
        elif b["mae"] > a["mae"] or (key == "vmax_ms" and b["mae"] == a["mae"]):
            failures.append(f"MAE regression: {key}")
    if missing:
        failures.append(f"Missing previously ready eligible slots: {len(missing)}")
    return {
        "passed": not failures,
        "failures": failures,
        "metrics": metrics,
        "missing": missing,
        "pretraining_membership": dict(membership),
        "candidate_source_bytes": source_bytes,
        "storms": storm_ids,
        "window": {"start": start, "end": end},
    }


def apply_quality_policy(report, policy):
    """Keep measured regressions visible when wind-priority release is authorized."""
    report["quality_policy"] = policy
    report["warnings"] = []
    if policy == "wind-priority":
        radius_regressions = {
            f"MAE regression: {key}" for key in ("rmw_km", "r34_km", "r50_km", "r64_km")
        }
        report["warnings"] = [f for f in report["failures"] if f in radius_regressions]
        report["failures"] = [f for f in report["failures"] if f not in radius_regressions]
        report["passed"] = not report["failures"]
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "compare", "reconcile"])
    p.add_argument("--source", type=Path, default=Path("var/stormsense/state.sqlite"))
    p.add_argument(
        "--db", type=Path, default=Path("var/stormsense-migration/state.sqlite")
    )
    p.add_argument("--export", type=Path, default=Path("var/stormsense/export"))
    p.add_argument("--model-root", default="downloads/models")
    p.add_argument("--full", action="store_true")
    p.add_argument("--quality-policy", choices=["strict", "wind-priority"], default="strict")
    a = p.parse_args()
    if not MANIFEST.get("prediction_policy", {}).get("classification_start_gate"):
        p.error("Select the finetuned manifest using STORMSENSE_MODEL_MANIFEST")
    if a.command == "reconcile":
        from .cli import lock

        with (
            lock(str(a.source) + ".cycle.lock"),
            lock(str(a.source) + ".lock"),
            lock(str(a.db) + ".lock"),
        ):
            reconcile(a.source, a.db)
        # The post-reconciliation comparison must cover the new live window.
        migration = json.loads((a.db.parent / "migration.json").read_text())
        pointer = json.loads((a.export / "latest.json").read_text())
        catalog = json.loads((a.export / pointer["manifest"]).read_text())
        migration["window"] = catalog["window"]
        migration["eligible_slots"] = {s["id"]: 0 for s in catalog["storms"]}
        write_json(a.db.parent / "migration.json", migration)
        report = {"reconciled": True, "window": migration["window"]}
    elif a.command == "prepare":
        report = prepare(a.source, a.db, a.export, a.model_root)
    else:
        seen = training_storms(a.model_root)
        migration = json.loads((a.db.parent / "migration.json").read_text())
        store = Store(a.db)
        try:
            report = compare(
                store,
                (
                    list(migration["eligible_slots"])
                    if a.full
                    else migration["pilot_storms"]
                ),
                migration["window"]["start"],
                migration["window"]["end"],
                seen,
            )
            report.update(
                candidate_version=version("nowcast"),
                phase="full" if a.full else "pilot",
            )
            apply_quality_policy(report, a.quality_policy)
            store.put_status(
                "migration_quality" if a.full else "migration_pilot", report
            )
        finally:
            store.close()
        write_json(
            a.db.parent
            / ("full-comparison.json" if a.full else "pilot-comparison.json"),
            report,
        )
    print(json.dumps(report, indent=2))
    if report.get("passed") is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
