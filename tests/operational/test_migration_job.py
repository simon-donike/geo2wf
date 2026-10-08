import json
from pathlib import Path

import pytest

from geo2wf.operational import migration_job as job
from geo2wf.operational.models import version
from geo2wf.operational.rebuild import apply_quality_policy


def test_wind_priority_keeps_radius_warning_and_all_other_failures():
    radius = "MAE regression: rmw_km"
    strict = apply_quality_policy({"passed": False, "failures": [radius]}, "strict")
    assert not strict["passed"]
    accepted = apply_quality_policy({"passed": False, "failures": [radius]}, "wind-priority")
    assert accepted["passed"] and accepted["warnings"] == [radius]
    for failure in ("MAE regression: vmax_ms", "MAE regression: forecast_6h",
                    "No matched labels for rmw_km", "Missing previously ready eligible slots: 1"):
        report = apply_quality_policy({"passed": False, "failures": [radius, failure]}, "wind-priority")
        assert not report["passed"] and report["failures"] == [failure]


def test_quality_gate_requires_success_phase_and_exact_model(tmp_path):
    path = tmp_path / "report.json"
    good = {"passed": True, "phase": "full", "candidate_version": version("nowcast")}
    path.write_text(json.dumps(good))
    assert job.require_quality(path, "full") == good
    for bad in (
        {**good, "passed": False},
        {**good, "passed": "true"},
        {**good, "phase": "pilot"},
        {**good, "candidate_version": "different"},
    ):
        path.write_text(json.dumps(bad))
        with pytest.raises(RuntimeError, match="quality gate failed"):
            job.require_quality(path, "full")


def test_image_verification_checks_bytes_not_only_metadata(tmp_path):
    before, after = tmp_path / "before", tmp_path / "after"
    frame = {
        "time": "2026-09-01T02:00:00Z",
        "status": "ready",
        "metadata": "frame.json",
        "parts": [
            {"image": "frame.webp", "preview": "preview.webp", "sidecar": "frame.xml"}
        ],
    }
    catalog = {
        "storms": [{"id": "AL012026", "series": "storm.json"}],
        "window": {"start": "2026-09-01T00:00:00Z", "end": "2026-09-02T00:00:00Z"},
    }
    for directory in (before, after):
        directory.mkdir()
        (directory / "storm.json").write_text(json.dumps({"imagery": [frame]}))
        for name in ("frame.json", "frame.webp", "preview.webp", "frame.xml"):
            (directory / name).write_bytes(b"same")
    assert job.verify_images(before, catalog, after, catalog) == 1
    (after / "preview.webp").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="bytes changed"):
        job.verify_images(before, catalog, after, catalog)


def test_image_verification_allows_recovering_an_old_gap(tmp_path):
    before, after = tmp_path / "before", tmp_path / "after"
    catalog = {
        "storms": [{"id": "AL012026", "series": "storm.json"}],
        "window": {"start": "2026-09-01T00:00:00Z", "end": "2026-09-02T00:00:00Z"},
    }
    for directory, status in ((before, "gap"), (after, "ready")):
        directory.mkdir()
        (directory / "storm.json").write_text(json.dumps({"imagery": [
            {"time": "2026-09-01T02:00:00Z", "status": status, "parts": []}
        ]}))
    assert job.verify_images(before, catalog, after, catalog) == 0


def test_failed_pilot_cannot_run_commands_or_pause_production(tmp_path, monkeypatch):
    folder = tmp_path / "var/stormsense-migration"
    folder.mkdir(parents=True)
    monkeypatch.setattr(job, "ROOT", tmp_path)
    monkeypatch.setattr(job, "fingerprint", lambda: "source")
    monkeypatch.setattr(job.sys, "argv", ["migration_job", "--deploy"])
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (folder / "pilot-comparison.json").write_text(json.dumps({"passed": False}))
    monkeypatch.setattr(
        job.subprocess, "run", lambda *a, **kw: pytest.fail("No command may run")
    )
    with pytest.raises(RuntimeError, match="pilot quality gate failed"):
        job.main()
    assert json.loads((folder / "job-status.json").read_text())["stage"] == "stopped"
