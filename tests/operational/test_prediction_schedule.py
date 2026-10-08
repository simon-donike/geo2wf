from datetime import timedelta
import json

import pytest

from geo2wf.operational import schedule, pipeline
from geo2wf.operational.common import iso, utc
from geo2wf.operational.export import coverage, export_release, prune_local, publish
from geo2wf.operational.models import version
from geo2wf.operational.store import Store


@pytest.fixture
def policy(monkeypatch):
    monkeypatch.setitem(
        schedule.MANIFEST,
        "prediction_policy",
        {"cadence_hours": 2, "classification_start_gate": True},
    )


def storm():
    return dict(
        id="AL012026",
        name="Test",
        basin="AL",
        active=False,
        start="2026-09-01T00:00:00Z",
        end="2026-09-02T00:00:00Z",
        track=[
            dict(
                time="2026-09-01T03:00:00Z",
                classification="TD",
                lat=20,
                lon=-70,
                wind_ms=15,
                radii_km={},
            ),
            dict(
                time="2026-09-01T12:00:00Z",
                classification="LO",
                lat=21,
                lon=-70,
                wind_ms=10,
                radii_km={},
            ),
        ],
        advisory=None,
    )


@pytest.mark.parametrize(
    "code", [" td ", "Tropical Depression", "subtropical-storm", "HU", "SD"]
)
def test_classifications(code):
    assert schedule.qualified(code)


@pytest.mark.parametrize(
    "code", [None, "", "DB", "LO", "EX", "Potential Tropical Cyclone"]
)
def test_unknown_and_precursors_do_not_qualify(code):
    assert not schedule.qualified(code)


def test_start_gate_continues_through_weakening(policy):
    s = storm()
    times = [iso(t) for t in schedule.slots(s, s["start"], s["end"])]
    assert times[0] == "2026-09-01T04:00:00Z"
    assert times[-1] == "2026-09-02T00:00:00Z"
    assert len(times) == 11
    assert schedule.metadata(s)["eligible_start"] == times[0]
    s["track"] = []
    assert list(schedule.slots(s, s["start"], s["end"])) == []


def test_live_gate_requires_observed_qualification_and_survives_restart(
    policy, tmp_path
):
    s = storm()
    assert list(schedule.slots(s, s["start"], s["end"], live=True)) == []
    s["prediction_qualified_at"] = "2026-09-01T07:00:00Z"
    store = Store(tmp_path / "state.sqlite")
    store.put_storm(s)
    store.close()
    store = Store(tmp_path / "state.sqlite")
    try:
        times = list(
            schedule.slots(store.storm(s["id"]), s["start"], s["end"], live=True)
        )
        assert iso(times[0]) == "2026-09-01T08:00:00Z"
    finally:
        store.close()


def test_coverage_and_track_only_export(policy, tmp_path):
    store = Store(tmp_path / "state.sqlite")
    try:
        s = storm()
        store.put_storm(s)
        assert coverage(store, s["start"], s["end"])["totals"]["expected"] == 11
        s["track"][0]["classification"] = "DB"
        store.put_storm(s)
        report = coverage(store, s["start"], s["end"])
        assert report["totals"] == dict(expected=0, predictions=0, gaps=0, pending=0)
        catalog = export_release(store, tmp_path / "export", s["start"], s["end"])
        assert len(catalog["storms"]) == 1
        assert catalog["storms"][0]["prediction_schedule"]["eligible_start"] is None
        assert catalog["prediction_cadence_hours"] == 2
    finally:
        store.close()


def test_forecast_uses_exact_six_and_twelve_hour_context(policy):
    class Models:
        def predict_future(self, *values):
            return [20, 21]

    rows = [
        dict(
            storm_id="AL012026",
            time=iso(utc("2026-09-01T04:00:00Z") + timedelta(hours=h)),
            kind="hindcast",
            model_version=version("nowcast"),
            status="ready",
            generated_at="2026-09-03T00:00:00Z",
            metrics={"vmax_ms": 20},
        )
        for h in range(0, 13, 2)
    ]
    assert list(pipeline.forecast_rows(rows[:-1], Models())) == []
    forecasts = list(pipeline.forecast_rows(rows, Models()))
    assert len(forecasts) == 1
    assert forecasts[0]["anchor_time"] == "2026-09-01T16:00:00Z"
    assert (
        list(
            pipeline.forecast_rows(
                [r for r in rows if r["time"] != "2026-09-01T10:00:00Z"], Models()
            )
        )
        == []
    )


def test_retention_keeps_pinned_release(tmp_path):
    root = tmp_path
    (root / "latest.json").write_text(json.dumps({"version": "new"}))
    for name in ("old", "new"):
        p = root / "releases" / name
        p.mkdir(parents=True)
        (p / "catalog.json").write_text(json.dumps({"storms": []}))
    (root / "releases/old/pin.json").write_text("{}")
    assert prune_local(root, keep=1) == []


def test_candidate_publication_requires_full_matching_quality_gate(tmp_path):
    (tmp_path / "latest.json").write_text(json.dumps({"version": "candidate"}))
    folder = tmp_path / "releases/candidate"
    folder.mkdir(parents=True)
    catalog = {"prediction_cadence_hours": 2, "models": {"nowcast": {"version": "new"}}}
    for gate in (
        None,
        {"passed": False},
        {"passed": True, "phase": "pilot"},
        {"passed": True, "phase": "full", "candidate_version": "old"},
    ):
        catalog["quality_gate"] = gate
        (folder / "catalog.json").write_text(json.dumps(catalog))
        with pytest.raises(ValueError, match="quality gate"):
            publish(tmp_path, run=lambda *a, **kw: pytest.fail("Upload must not start"))


def test_reprocessed_live_center_remains_a_hindcast(monkeypatch):
    source = {
        "center": {"lat": 20, "lon": -70},
        "source_snapshot": "original",
        "model_version": "old",
        "kind": "live",
        "generated_at": "2026-09-02T00:05:00Z",
    }
    s = {**storm(), "_rebuild_source": source}
    calls = []

    def acquire(at, center, available):
        calls.append(center)
        return None, None, {"end": iso(at)}

    class Models:
        def infer(self, *args):
            return {"vmax_ms": 20}

    monkeypatch.setattr(pipeline, "acquire", acquire)
    result = pipeline.process(s, "2026-09-02T00:00:00Z", "hindcast", Models())
    assert calls == [source["center"]]
    assert result["status"] == "ready" and result["kind"] == "hindcast"
    assert result["source_snapshot"] == "original"
    assert result["reprocessed_from"]["kind"] == "live"


def test_live_advice_is_visible_before_best_track_arrives(policy, tmp_path):
    s = storm()
    s.update(active=True, track=[], prediction_qualified_at="2026-09-01T07:00:00Z")
    store = Store(tmp_path / "state.sqlite")
    try:
        store.put_storm(s)
        assert list(schedule.slots(s, s["start"], s["end"])) == []
        assert (
            iso(schedule.visible_slots(s, s["start"], s["end"])[0])
            == "2026-09-01T08:00:00Z"
        )
        catalog = export_release(store, tmp_path / "export", s["start"], s["end"])
        assert (
            catalog["storms"][0]["prediction_schedule"]["eligible_start"]
            == "2026-09-01T08:00:00Z"
        )
        assert catalog["coverage"]["expected"] == 9
    finally:
        store.close()


def test_odd_hour_update_discovers_without_inference(policy, tmp_path, monkeypatch):
    s = storm()
    s.update(active=True, prediction_qualified_at=s["track"][0]["time"])
    store = Store(tmp_path / "state.sqlite")
    store.put_storm(s)
    discovered = []

    def discover(store, now):
        discovered.append(now)
        return [{"id": s["id"]}]

    def backfill(store, *args, **kwargs):
        store.put_status("backfill", {"complete": True})
        return {"ready": 0, "gap": 0}

    monkeypatch.setattr(pipeline, "discover", discover)
    monkeypatch.setattr(pipeline, "discover_history", lambda *a: None)
    monkeypatch.setattr(pipeline, "backfill", backfill)
    monkeypatch.setattr(pipeline, "generate_forecasts", lambda *a, **kw: None)
    monkeypatch.setattr(
        pipeline, "process", lambda *a: pytest.fail("Odd-hour inference")
    )
    try:
        assert pipeline.update(store, object(), now="2026-09-01T13:05:00Z") == []
        assert len(discovered) == 1
    finally:
        store.close()
