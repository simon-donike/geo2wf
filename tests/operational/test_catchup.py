"""Catch-up regressions: holes, ended storms, retries and restart safety."""

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import pytest

from geo2wf.operational import cli, geocolor, models, pipeline
from geo2wf.operational.common import hour, iso, utc, year_before
from geo2wf.operational.models import version
from geo2wf.operational.store import Store
from test_pipeline import FakeModels, sample, storm


@pytest.fixture
def setup(tmp_path, monkeypatch):
    store = Store(tmp_path / "state.sqlite")
    model = FakeModels()
    model.device = "cpu"
    calls = []
    monkeypatch.setattr(
        pipeline,
        "ProcessPoolExecutor",
        lambda max_workers, **kw: ThreadPoolExecutor(max_workers=max_workers),
    )

    def process_group(tasks):
        result = []
        for s, at in tasks:
            calls.append((s["id"], iso(at)))
            row = sample(int((at - utc("2026-09-01")).total_seconds() / 3600))
            row["storm_id"] = s["id"]
            result.append(row)
        return result

    monkeypatch.setattr(pipeline, "process_group_worker", process_group)
    yield store, model, calls
    store.close()


def test_update_fills_interior_holes_and_discovers_ended_storms(setup, monkeypatch):
    store, model, calls = setup
    store.put_storm(storm())
    # A newer ready live record is not a watermark for archive completeness.
    store.put_sample(sample(0))
    store.put_sample(sample(6, "live"))
    store.put_sample(sample(10))
    original = store.sample("EP012026", sample(6)["time"], "live", version("nowcast"))
    discovery = []
    monkeypatch.setattr(pipeline, "discover", lambda *args: [storm()])

    def history(db, start, end):
        discovery.append((start, end))
        db.put_storm(
            {**storm(), "id": "EP022026", "active": False, "end": sample(2)["time"]}
        )

    monkeypatch.setattr(pipeline, "discover_history", history)
    monkeypatch.setattr(pipeline, "process", lambda *args: sample(12, "live"))
    now = utc("2026-09-01T12:30:00Z")
    pipeline.update(store, model, now=now, workers=2)
    assert discovery == [(hour(year_before(now)) - timedelta(hours=12), now)]
    assert set(calls) == {
        *(
            ("EP012026", sample(h)["time"])
            for h in range(13)
            if h not in (0, 6, 10, 12)
        ),
        *(("EP022026", sample(h)["time"]) for h in range(3)),
    }
    assert (
        store.sample("EP012026", sample(6)["time"], "live", version("nowcast"))
        == original
    )
    assert any(
        f["kind"] == "live" and f["anchor_time"] == sample(12)["time"]
        for f in store.forecasts("EP012026")
    )
    assert store.get_status("backfill")["complete"]
    calls.clear()
    pipeline.update(store, model, now=now, workers=2)
    assert calls == []


def test_reconcile_leaves_recorded_gaps_and_only_fills_unattempted_slots(setup):
    store, model, calls = setup
    store.put_storm({**storm(), "active": False, "end": sample(4)["time"]})
    for h in (0, 1, 2):
        row = sample(h, kind="live" if h == 2 else "hindcast", status="gap")
        row["generated_at"] = iso(utc() - timedelta(days=10))
        if h == 0:
            row["reason"] = "stale_or_missing_center"
        store.put_sample(row)
    pipeline.backfill(store, model, "2026-09-01", sample(4)["time"], reconcile=True)
    assert {at for _, at in calls} == {sample(3)["time"], sample(4)["time"]}
    assert (
        store.sample("EP012026", sample(0)["time"], "hindcast", version("nowcast"))[
            "status"
        ]
        == "gap"
    )
    calls.clear()
    pipeline.backfill(store, model, "2026-09-01", sample(4)["time"], retry_gaps=True)
    assert {at for _, at in calls} == {sample(h)["time"] for h in (0, 1, 2)}


@pytest.mark.parametrize("kind", ["live", "hindcast"])
def test_update_does_not_retry_a_failed_current_hour(setup, monkeypatch, kind):
    store, model, calls = setup
    store.put_storm(storm())
    store.put_sample(sample(12, kind, status="gap"))
    monkeypatch.setattr(pipeline, "discover", lambda *args: [storm()])
    monkeypatch.setattr(pipeline, "discover_history", lambda *args: None)
    monkeypatch.setattr(
        pipeline, "process", lambda *args: pytest.fail("already attempted live hour")
    )
    pipeline.update(store, model, now="2026-09-01T12:00:00Z", workers=2)
    assert sample(12)["time"] not in {at for _, at in calls}
    assert (
        store.sample("EP012026", sample(12)["time"], kind, version("nowcast"))["status"]
        == "gap"
    )


def test_interrupted_catchup_does_not_mark_update_complete(setup, monkeypatch):
    store, model, calls = setup
    store.put_storm({**storm(), "active": False})
    monkeypatch.setattr(pipeline, "discover", lambda *args: [])
    monkeypatch.setattr(pipeline, "discover_history", lambda *args: None)
    Path(str(store.path) + ".stop-backfill").touch()
    with pytest.raises(RuntimeError, match="interrupted"):
        pipeline.update(store, model, now="2026-09-01T12:00:00Z", workers=1)
    assert store.get_status("update") is None
    assert not store.get_status("backfill")["complete"]
    Path(str(store.path) + ".stop-backfill").unlink()
    pipeline.update(store, model, now="2026-09-01T12:00:00Z", workers=1)
    assert store.get_status("backfill")["complete"]
    assert len(calls) == 13


def test_cli_clears_own_priority_marker_before_catchup(tmp_path, monkeypatch):
    db = tmp_path / "state.sqlite"
    marker = Path(str(db) + ".update-requested")
    monkeypatch.setattr(models, "Models", lambda *args: object())
    calls = []

    def update(store, model, **kwargs):
        assert not marker.exists()
        calls.append(kwargs)

    monkeypatch.setattr(pipeline, "update", update)
    monkeypatch.setattr(
        geocolor, "backfill_images", lambda *args, **kwargs: {"interrupted": False}
    )
    cli.main(["--db", str(db), "update", "--workers", "2"])
    assert calls[0]["workers"] == 2
    assert not marker.exists()


@pytest.mark.parametrize("prediction_fails", [False, True])
def test_python_update_builds_and_exports_frames_even_without_predictions(
    tmp_path, monkeypatch, prediction_fails
):
    from test_geocolor import source
    from geo2wf.operational.export import export_release
    import json

    db = tmp_path / "state.sqlite"
    store = Store(db)
    store.put_storm({**storm(), "active": False, "start": sample(12)["time"]})
    store.close()
    client, requests = source()
    monkeypatch.setattr(geocolor, "Client", lambda: client)
    monkeypatch.setattr(models, "Models", lambda *args: object())
    monkeypatch.setattr(cli, "utc", lambda value=None: utc(value or "2026-09-02"))

    def predict(*args, **kwargs):
        if prediction_fails:
            raise RuntimeError("prediction source offline")
        return []  # No new predictions must still create missing website images.

    monkeypatch.setattr(pipeline, "update", predict)
    command = ["--db", str(db), "update", "--imagery-workers", "1"]
    if prediction_fails:
        with pytest.raises(RuntimeError, match="prediction source offline"):
            cli.main(command)
    else:
        cli.main(command)
        request_count = len(requests)
        cli.main(command)
        assert len(requests) == request_count  # Successful frames are reused.
    store = Store(db)
    frames = store.visuals("EP012026", geocolor.VERSION)
    assert len(frames) == 1 and frames[0]["status"] == "ready"
    output = tmp_path / "export"
    catalog = export_release(store, output, "2026-09-01", "2026-09-02")
    series = json.loads((output / catalog["storms"][0]["series"]).read_text())
    image = series["imagery"][0]["parts"][0]["image"]
    assert (output / image).is_file()
    assert all(
        (output / bundle["path"]).is_file() for bundle in series["imagery_bundles"]
    )
    store.close()


@pytest.mark.parametrize("update_status", [0, 7])
def test_runner_exports_and_publishes_even_after_update_failure(
    tmp_path, update_status
):
    import json
    import os
    import subprocess

    runner = Path(__file__).resolve().parents[2] / "apps/stormsense/runner/cycle.sh"
    stub = tmp_path / "python-stub"
    calls = tmp_path / "calls"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import json,os,sys\n"
        "with open(os.environ['STORMSENSE_TEST_CALLS'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        f"sys.exit({update_status} if 'update' in sys.argv else 0)\n"
    )
    stub.chmod(0o755)
    result = subprocess.run(
        ["bash", str(runner)],
        env={
            **os.environ,
            "STORMSENSE_PYTHON": str(stub),
            "STORMSENSE_TEST_CALLS": str(calls),
            "STORMSENSE_PUBLISH": "1",
            "STORMSENSE_DB": str(tmp_path / "state.sqlite"),
        },
    )
    assert result.returncode == update_status
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    assert [
        next(
            arg
            for arg in args
            if arg in {"update", "imagery", "evaluate", "export", "publish"}
        )
        for args in commands
    ] == ["update", "evaluate", "export", "publish"]
    assert "--imagery-workers" in commands[0]


def test_imagery_reconciles_old_ended_storm_and_missing_files(tmp_path, monkeypatch):
    from test_geocolor import source

    store = Store(tmp_path / "state.sqlite")
    store.put_storm(
        {
            **storm(),
            "active": False,
            "start": sample(12)["time"],
            "end": sample(12)["time"],
        }
    )
    client, calls = source()
    args = (store, "2026-09-01", "2026-09-04")
    result = geocolor.backfill_images(*args, client=client)
    assert result["results"] == {"ready": 1}
    assert geocolor.backfill_images(*args, client=client)["processed"] == 0
    frame = store.visuals("EP012026", geocolor.VERSION)[0]
    (geocolor.asset_root(store) / frame["files"][0]).unlink()
    assert geocolor.backfill_images(*args, client=client)["results"] == {"ready": 1}
    store.put_visual({**frame, "status": "gap", "reason": "no_center", "center": None})
    # A newly available track center repairs a former no-center gap immediately.
    assert geocolor.backfill_images(*args, client=client)["results"] == {"ready": 1}
    store.close()
