from datetime import timedelta
import json
from pathlib import Path
from subprocess import CalledProcessError
from urllib.error import URLError

import numpy as np
import pytest

from geo2wf.operational import pipeline, satellite
from geo2wf.operational.common import iso, utc, KNOT
from geo2wf.operational.export import coverage, export_release, publish, prune_local
from geo2wf.operational.models import version
from geo2wf.operational.store import Store
from geo2wf.operational.tracks import (
    parse_atcf,
    parse_current,
    live_center,
    historical_center,
)


@pytest.fixture
def store(tmp_path):
    value = Store(tmp_path / "state.sqlite")
    yield value
    value.close()


def fix(at="2026-09-01T00:00:00Z", lon=-139):
    return dict(
        time=at,
        lat=18,
        lon=lon,
        wind_ms=40,
        classification="HU",
        name="Test",
        radii_km={},
        source="ATCF",
    )


def storm():
    return dict(
        id="EP012026",
        name="Test",
        basin="EP",
        active=True,
        start="2026-09-01T00:00:00Z",
        end="2026-09-01T12:00:00Z",
        track=[
            fix(),
            fix("2026-09-01T06:00:00Z", -141),
            fix("2026-09-01T12:00:00Z", -142),
        ],
        advisory={**fix(), "motion_direction": 270, "motion_speed_kt": 10},
    )


def sample(hour, kind="hindcast", status="ready"):
    return dict(
        storm_id="EP012026",
        time=iso(utc("2026-09-01T00:00:00Z") + timedelta(hours=hour)),
        kind=kind,
        model_version=version("nowcast"),
        generated_at="2020-10-02T00:00:00Z",
        status=status,
        metrics=(
            {
                "vmax_ms": 20 + hour,
                "rmw_km": 20,
                "r34_km": 100,
                "r50_km": 60,
                "r64_km": 30,
            }
            if status == "ready"
            else None
        ),
        center={"lat": 18, "lon": -139},
        reason=None if status == "ready" else "missing_scan",
    )


class FakeModels:
    def __init__(self):
        self.inputs = []

    def predict_future(self, *values):
        self.inputs.append(values)
        return values[0] + 1, values[0] + 2

    def infer(self, *values):
        raise AssertionError("inference should not run without inputs")


def test_duplicate_atcf_thresholds_and_basin_crossing():
    row = [
        "EP",
        "01",
        "2026090100",
        "03",
        "BEST",
        "0",
        "180N",
        "1390W",
        "80",
        "960",
        "HU",
        "34",
        "NEQ",
        "100",
        "90",
        "80",
        "70",
        "",
        "",
        "20",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "TEST",
    ]
    row2 = row.copy()
    row2[11] = "50"
    row2[13:17] = ["40", "30", "20", "10"]
    later = row.copy()
    later[0] = "CP"
    later[2] = "2026090106"
    later[7] = "1410W"
    result = parse_atcf(
        "\n".join(",".join(r) for r in [row, row, row2, later]), "EP012026"
    )
    assert len(result) == 2
    assert result[0]["wind_ms"] == 80 * KNOT
    assert set(result[0]["radii_km"]) == {"r34", "r50", "rmw"}
    assert historical_center(result, "2026-09-01T03:00:00Z")["lon"] == -140


def test_live_center_is_causal_and_expires():
    s = storm()
    assert live_center(s["advisory"], "2026-08-31T23:00:00Z") is None
    assert live_center(s["advisory"], "2026-09-01T07:00:00Z") is None
    center = live_center(s["advisory"], "2026-09-01T06:00:00Z")
    assert center["method"] == "motion_estimate" and center["lon"] < -139
    assert center["age_hours"] == 6
    assert (
        historical_center([fix(), fix("2026-09-01T12:00:00Z")], "2026-09-01T06:00:00Z")
        is None
    )
    assert (
        abs(
            abs(
                historical_center(
                    [fix(lon=179), fix("2026-09-01T06:00:00Z", -179)],
                    "2026-09-01T03:00:00Z",
                )["lon"]
            )
            - 180
        )
        < 1e-5
    )


def test_current_preserves_identity_but_updates_basin_and_excludes_invests():
    payload = {
        "activeStorms": [
            {
                "id": "EP152026",
                "lastUpdate": "2026-10-02T09:00:00Z",
                "name": "Nolo",
                "latitudeNumeric": 23.3,
                "longitudeNumeric": -166.7,
            },
            {"id": "EP902026"},
        ]
    }
    value = parse_current(json.dumps(payload))[0]
    assert value["id"] == "EP152026" and value["basin"] == "CP"
    with pytest.raises(ValueError):
        parse_current("{}")


def test_empty_feed_is_distinct_from_failure(store, monkeypatch):
    store.put_storm(storm())
    monkeypatch.setattr(
        pipeline, "fetch", lambda _: (_ for _ in ()).throw(URLError("offline"))
    )
    with pytest.raises(URLError):
        pipeline.discover(store, "2026-10-02T00:00:00Z")
    assert store.storm("EP012026")["active"]
    assert store.get_status("discovery")["error"]
    monkeypatch.setattr(pipeline, "fetch", lambda _: b'{"activeStorms":[]}')
    assert pipeline.discover(store, "2026-10-02T00:00:00Z") == []
    assert not store.storm("EP012026")["active"]
    assert store.get_status("discovery")["error"] is None


def test_forecasts_use_only_matching_past_model_estimates():
    models = FakeModels()
    rows = [sample(h) for h in (0, 6, 12, 18)]
    rows.append(sample(12, "live"))
    results = list(
        pipeline.forecast_rows(rows, models, issued_at="2026-10-02T01:00:00Z")
    )
    assert models.inputs == [(32, 26, 20), (38, 32, 26), (32, 26, 20)]
    assert [r["kind"] for r in results] == ["hindcast", "hindcast", "live"]
    assert results[-1]["input_kinds"] == ["live", "hindcast", "hindcast"]
    assert results[0]["predictions"][1]["valid_time"] == "2026-09-02T00:00:00Z"
    assert (
        list(
            pipeline.forecast_rows(
                [sample(0), sample(6, status="gap"), sample(12)], FakeModels()
            )
        )
        == []
    )
    other = sample(6)
    other["model_version"] = "different"
    assert (
        list(pipeline.forecast_rows([sample(0), other, sample(12)], FakeModels())) == []
    )


def test_missing_scan_and_stale_fix_never_use_official_winds(monkeypatch):
    s = storm()
    monkeypatch.setattr(
        pipeline,
        "acquire",
        lambda *args: (_ for _ in ()).throw(satellite.DataGap("missing_scan")),
    )
    row = pipeline.process(s, "2026-09-01T00:00:00Z", "live", FakeModels())
    assert row["metrics"] is None and row["reason"] == "missing_scan"
    row = pipeline.process(s, "2026-09-01T08:00:00Z", "live", FakeModels())
    assert row["metrics"] is None and row["reason"] == "stale_or_missing_center"


def test_scan_selection_excludes_future_acquisitions_and_unpublished_data(monkeypatch):
    scans = [
        dict(end="2026-09-01T00:00:01Z", published_at="2026-09-01T00:00:01Z"),
        dict(end="2026-08-31T23:59:00Z", published_at="2026-09-01T00:04:00Z"),
        dict(end="2026-08-31T23:49:00Z", published_at="2026-08-31T23:51:00Z"),
    ]
    monkeypatch.setattr(satellite, "list_hour", lambda *args: scans)
    selected = satellite.select_scan(19, "2026-09-01T00:00:00Z", "2026-09-01T00:00:00Z")
    assert selected["end"] == "2026-08-31T23:49:00Z"
    assert (
        satellite.select_scan(19, "2026-09-01T00:00:00Z")["end"]
        == "2026-08-31T23:59:00Z"
    )
    monkeypatch.setattr(satellite, "list_hour", lambda *args: [])
    with pytest.raises(satellite.DataGap, match="No complete"):
        satellite.select_scan(19, "2026-09-01T00:00:00Z")


def test_satellite_geometry_and_operational_dates():
    assert satellite.satellite_order("2026-01-01", 20, -60) == [19, 18]
    assert satellite.satellite_order("2026-01-01", 20, -160) == [18, 19]
    assert satellite.satellite_order("2024-01-01", 20, -60) == [16, 18]


def test_committed_jobs_survive_reopen_and_ready_results_are_immutable(tmp_path):
    path = tmp_path / "state.sqlite"
    a = Store(path)
    a.put_sample(sample(0, status="gap"))
    a.close()
    b = Store(path)
    b.put_sample(sample(0), retry=True)
    b.put_sample(sample(0, status="gap"), retry=True)
    assert (
        b.sample("EP012026", "2026-09-01T00:00:00Z", "hindcast", version("nowcast"))[
            "status"
        ]
        == "ready"
    )
    b.close()


def test_coverage_accounts_for_every_expected_hour(store, tmp_path):
    store.put_storm({**storm(), "active": False})
    store.put_sample(sample(0))
    store.put_sample(sample(1, status="gap"))
    report = coverage(store, "2026-09-01T00:00:00Z", "2026-09-01T12:00:00Z")
    assert report["totals"] == {
        "expected": 13,
        "predictions": 1,
        "gaps": 1,
        "pending": 11,
    }
    # A live slot newer than the last recorded fix must not conceal an older hole.
    store.put_sample(sample(13, kind="live"))
    catalog = export_release(
        store, tmp_path / "coverage-export", "2026-09-01", "2026-09-02"
    )
    assert catalog["storms"][0]["pending_count"] == 11
    assert catalog["storms"][0]["record_count"] == 3
    assert catalog["reports"]["evaluation"] is None


def test_release_objects_reused_and_failed_upload_never_advances_pointer(
    store, tmp_path
):
    store.put_storm(storm())
    store.put_sample(sample(0))
    output = tmp_path / "export"
    a = export_release(store, output, "2026-09-01", "2026-09-02", release="first")
    b = export_release(store, output, "2026-09-01", "2026-09-02", release="second")
    assert a["storms"][0]["series"] == b["storms"][0]["series"]
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        if len(calls) == 2:
            raise CalledProcessError(1, command)

    with pytest.raises(CalledProcessError):
        publish(output, run=fail)
    assert not any(c[1] == "copyto" for c in calls)
    calls.clear()
    publish(output, run=lambda command, **kwargs: calls.append(command))
    assert (
        calls[-1][1] == "copyto"
        and calls[-1][-1] == "r2:tcd/explorer/stormsense/latest.json"
    )
    assert not any("rcat" in c for c in calls)
    calls.clear()

    def bad_checksum(command, **kwargs):
        calls.append(command)
        if command[1] == "check":
            raise CalledProcessError(1, command)

    with pytest.raises(CalledProcessError):
        publish(output, run=bad_checksum)
    assert not any(c[1] == "copyto" for c in calls)
    obsolete = prune_local(output, keep=1, apply=True)
    assert "releases/first/catalog.json" in obsolete
    assert (output / b["storms"][0]["series"]).exists()
    with pytest.raises(FileExistsError):
        export_release(store, output, release="second")


def test_publication_reuses_equal_bytes_with_different_mtimes_and_rejects_corruption(
    store, tmp_path
):
    import os
    import shutil
    import subprocess
    from geo2wf.operational.export import REMOTE

    if not shutil.which("rclone"):
        pytest.skip("rclone is required for the publication integration check")
    store.put_storm(storm())
    store.put_sample(sample(0))
    output, remote = tmp_path / "export", tmp_path / "remote"
    first = export_release(store, output, "2026-09-01", "2026-09-02", release="first")

    def local_remote(command, **kwargs):
        mapped = [
            str(remote) + item[len(REMOTE) :] if item.startswith(REMOTE) else item
            for item in command
        ]
        return subprocess.run(mapped, capture_output=True, **kwargs)

    publish(output, run=local_remote)
    object_path = remote / first["storms"][0]["series"]
    changed_mtime = object_path.stat().st_mtime - 3600
    os.utime(object_path, (changed_mtime, changed_mtime))
    export_release(store, output, "2026-09-01", "2026-09-02", release="second")
    publish(output, run=local_remote)
    assert object_path.stat().st_mtime == changed_mtime
    assert json.loads((remote / "latest.json").read_text())["version"] == "second"
    # Same-sized corruption must still fail, leaving the previous pointer intact.
    content = object_path.read_bytes()
    object_path.write_bytes(b"!" + content[1:])
    export_release(store, output, "2026-09-01", "2026-09-02", release="third")
    with pytest.raises(CalledProcessError):
        publish(output, run=local_remote)
    assert json.loads((remote / "latest.json").read_text())["version"] == "second"


def test_remote_retention_preserves_shared_and_recent_references():
    from types import SimpleNamespace
    from geo2wf.operational.export import prune_remote

    shared = "objects/" + "a" * 64 + ".json"
    obsolete = "objects/" + "b" * 64 + ".json"
    recent = "objects/" + "c" * 64 + ".json"
    catalogs = {
        f"releases/{name}/catalog.json": {"storms": [{"series": obj}]}
        for name, obj in [
            ("a-old", obsolete),
            ("b-recent", recent),
            ("z-latest", shared),
        ]
    }
    listing = [
        {"Path": key, "ModTime": iso() if "b-recent" in key else "2025-01-01T00:00:00Z"}
        for key in [*catalogs, shared, obsolete, recent]
    ]
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[1] == "cat":
            key = command[2].split("/stormsense/")[1]
            value = {"version": "z-latest"} if key == "latest.json" else catalogs[key]
        elif command[1] == "lsjson":
            value = listing
        else:
            value = {}
        return SimpleNamespace(stdout=json.dumps(value))

    result = prune_remote(keep=1, run=run)
    assert obsolete in result and shared not in result and recent not in result
    assert not any(command[1] == "deletefile" for command in calls)


def test_transition_dates_and_forecast_storm_isolation():
    assert satellite.satellite_order("2025-04-06T16:00:00Z", 20, -60)[0] == 16
    assert satellite.satellite_order("2025-04-07T16:00:00Z", 20, -60)[0] == 19
    assert satellite.satellite_order("2023-01-04T19:00:00Z", 20, -160)[0] == 18
    rows = [sample(0), sample(6), sample(12)]
    rows[1]["storm_id"] = "EP022026"
    assert list(pipeline.forecast_rows(rows, FakeModels())) == []


def test_export_reads_consistent_snapshot_during_backfill(store, tmp_path, monkeypatch):
    store.put_storm(storm())
    writer = Store(store.path)
    original = store.samples
    injected = False

    def concurrent(sid, model=None):
        nonlocal injected
        if not injected:
            writer.put_sample(sample(0))
            injected = True
        return original(sid, model)

    monkeypatch.setattr(store, "samples", concurrent)
    first = export_release(store, tmp_path / "exports", "2026-09-01", "2026-09-02")
    assert first["coverage"]["predictions"] == 0
    second = export_release(store, tmp_path / "exports", "2026-09-01", "2026-09-02")
    assert second["coverage"]["predictions"] == 1
    writer.close()


def test_history_listing_failure_is_reported(store, monkeypatch):
    monkeypatch.setattr(
        pipeline, "track_urls", lambda _: (_ for _ in ()).throw(URLError("offline"))
    )
    with pytest.raises(RuntimeError):
        pipeline.discover_history(store, "2025-10-02", "2026-10-02")
    assert len(store.get_status("history")["failures"]) == 2
    assert not coverage(store, "2025-10-02", "2026-10-02")["complete"]


def test_live_predictions_count_toward_archive_coverage(store):
    store.put_storm(storm())
    store.put_sample(sample(0, "live"))
    store.put_sample(sample(0, status="gap"))
    report = coverage(store, "2026-09-01T00:00:00Z", "2026-09-01T00:00:00Z")
    assert report["totals"] == {
        "expected": 1,
        "predictions": 1,
        "gaps": 0,
        "pending": 0,
    }


def test_shared_scan_handles_are_scoped_and_discarded(monkeypatch):
    created = []

    class Handle:
        closed = False

        def seek(self, offset):
            pass

        def close(self):
            self.closed = True

    class FileSystem:
        def open(self, *args, **kwargs):
            handle = Handle()
            created.append(handle)
            return handle

    monkeypatch.setattr(
        satellite.fsspec, "filesystem", lambda *args, **kwargs: FileSystem()
    )
    with satellite.shared_reads():
        with satellite.window_file("test-scan") as first:
            pass
        with satellite.window_file("test-scan") as second:
            assert first is second
        assert not first.closed
    assert first.closed and len(created) == 1
    with pytest.raises(RuntimeError):
        with satellite.shared_reads():
            with satellite.window_file("test-scan"):
                raise RuntimeError("interrupted")
    assert all(handle.closed for handle in created)
    with satellite.window_file("test-scan") as third:
        assert third is not first
    assert third.closed


def test_backfill_cannot_manufacture_old_live_issues(store):
    for at in (0, 6, 12):
        store.put_sample(sample(at, "live"))
    models = FakeModels()
    pipeline.generate_forecasts(store, models, "EP012026")
    assert store.forecasts("EP012026") == []
    pipeline.generate_forecasts(
        store, models, "EP012026", live_anchor="2026-09-01T12:00:00Z"
    )
    assert len(store.forecasts("EP012026")) == 1
    assert store.forecasts("EP012026")[0]["kind"] == "live"


def test_live_forecast_rejects_future_generated_prehistory():
    rows = [sample(0), sample(6), sample(12, "live")]
    rows[0]["generated_at"] = "2026-09-02T00:00:00Z"
    assert (
        list(
            pipeline.forecast_rows(rows, FakeModels(), issued_at="2026-09-01T12:30:00Z")
        )
        == []
    )


def test_peak_category_is_storm_lifetime_metadata(store, tmp_path):
    item = storm()
    item["track"][0]["wind_ms"] = 80
    store.put_storm(item)
    catalog = export_release(
        store, tmp_path / "peak", "2026-09-01T06:00:00Z", "2026-09-02"
    )
    assert catalog["storms"][0]["peak_category"] == 5


def test_live_center_uses_newest_known_position_and_fresh_motion():
    from geo2wf.operational.tracks import latest_live_center

    advisory = {**fix(), "motion_direction": 270, "motion_speed_kt": 10}
    fixes = [
        fix(),
        {**fix("2026-09-01T06:00:00Z", -140), "lat": 19},
        fix("2026-09-01T12:00:00Z", -145),
    ]
    exact = latest_live_center(advisory, fixes, "2026-09-01T06:00:00Z")
    assert exact["lat"] == 19 and exact["method"] == "live_track_fix"
    assert latest_live_center(advisory, fixes, "2026-09-01T08:00:00Z") is None
    advisory["time"] = "2026-09-01T03:00:00Z"
    estimated = latest_live_center(advisory, fixes, "2026-09-01T08:00:00Z")
    assert estimated["fix_time"] == "2026-09-01T06:00:00Z" and estimated["lon"] < -140
    assert estimated["motion_time"] == advisory["time"]


def test_evaluation_keeps_training_membership_modes_and_baselines_separate(
    store, monkeypatch
):
    from geo2wf.operational import evaluate as evaluation

    monkeypatch.setattr(evaluation, "training_storms", lambda _: {"EP012026"})
    current_version = version("forecast") + ":" + version("nowcast")
    for sid, prediction, kind in (
        ("EP012026", 30, "hindcast"),
        ("AL012026", 35, "hindcast"),
        ("AL012026", 50, "live"),
    ):
        store.put_storm({**storm(), "id": sid})
        store.put_forecast(
            {
                "storm_id": sid,
                "anchor_time": "2026-09-01T00:00:00Z",
                "kind": kind,
                "model_version": current_version,
                "input_vmax_ms": [30, 25, 20],
                "predictions": [
                    {
                        "valid_time": "2026-09-01T06:00:00Z",
                        "lead_hours": 6,
                        "vmax_ms": prediction,
                    }
                ],
            }
        )
    store.put_forecast(
        {
            "storm_id": "AL012026",
            "anchor_time": "2026-09-01T00:00:00Z",
            "kind": "hindcast",
            "model_version": "obsolete-model",
            "input_vmax_ms": [999, 999, 999],
            "predictions": [
                {"valid_time": "2026-09-01T06:00:00Z", "lead_hours": 6, "vmax_ms": 999}
            ],
        }
    )
    result = evaluation.evaluate(store, "unused")
    all_storms = result["groups"]["hindcast/all/+6h"]
    assert all_storms["samples"] == 2 and all_storms["mae_ms"] == 7.5
    assert all_storms["persistence_mae_ms"] == 10 and all_storms["trend_mae_ms"] == 5
    excluded = result["groups"]["hindcast/excluded_from_nowcast_training/+6h"]
    assert excluded["storm_ids"] == ["AL012026"] and excluded["mae_ms"] == 5
    assert result["groups"]["live/all/+6h"]["mae_ms"] == 10
    assert (
        evaluation.evaluate(store, "unused", start="2026-09-01T06:00:00Z")["groups"]
        == {}
    )


def test_runner_exports_failure_status_and_preserves_failed_exit_code(tmp_path):
    import os
    import subprocess

    runner = Path(__file__).resolve().parents[2] / "apps/stormsense/runner/cycle.sh"
    stub = tmp_path / "python-stub"
    calls = tmp_path / "calls"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import os,sys\n"
        "command=next(c for c in sys.argv if c in ('update','imagery','evaluate','export','publish'))\n"
        "with open(os.environ['STORMSENSE_TEST_CALLS'],'a') as f: f.write(command+'\\n')\n"
        "sys.exit(7 if command=='update' else 0)\n"
    )
    stub.chmod(0o755)
    result = subprocess.run(
        ["bash", str(runner)],
        env={
            **os.environ,
            "STORMSENSE_PYTHON": str(stub),
            "STORMSENSE_TEST_CALLS": str(calls),
            "STORMSENSE_PUBLISH": "0",
            "STORMSENSE_DB": str(tmp_path / "state.sqlite"),
        },
        check=False,
    )
    assert result.returncode == 7
    assert calls.read_text().splitlines() == ["update", "evaluate", "export"]


def test_discovery_never_claims_to_cover_future_storms(store, monkeypatch):
    clock = utc("2026-09-01T12:34:00Z")
    monkeypatch.setattr(
        pipeline, "utc", lambda value=None: utc(value) if value is not None else clock
    )
    years = []
    monkeypatch.setattr(pipeline, "track_urls", lambda year: years.append(year) or [])
    pipeline.discover_history(store, "2025-09-01", "2099-01-01")
    assert years == [2025, 2026]
    assert store.get_status("history")["end"] == iso(clock)


def test_hourly_coverage_remains_complete_between_slots(store):
    store.put_storm(storm())
    for at in range(13):
        store.put_sample(sample(at))
    store.put_status(
        "history",
        {
            "start": "2026-09-01T00:00:00Z",
            "end": "2026-09-01T12:00:00Z",
            "failures": [],
        },
    )
    assert coverage(store, "2026-09-01T00:00:00Z", "2026-09-01T12:45:00Z")["complete"]
    assert not coverage(store, "2026-09-01T00:00:00Z", "2026-09-01T13:00:00Z")[
        "complete"
    ]


def test_active_hours_beyond_the_last_fix_are_accounted_for(store):
    store.put_storm(storm())
    for at in range(13):
        store.put_sample(sample(at))
    gap = pipeline.process(storm(), "2026-09-01T13:00:00Z", "hindcast", FakeModels())
    assert gap["reason"] == "stale_or_missing_center" and gap["retryable"]
    store.put_sample(gap)
    store.put_sample(sample(14, kind="live"))
    report = coverage(store, "2026-09-01", "2026-09-01T14:00:00Z")
    assert report["totals"] == {
        "expected": 15,
        "predictions": 14,
        "gaps": 1,
        "pending": 0,
    }
