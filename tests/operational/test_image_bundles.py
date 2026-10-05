import json
from zipfile import ZipFile
from subprocess import CalledProcessError

import pytest

from geo2wf.operational.common import KNOT, digest
from geo2wf.operational.image_bundles import daily_bundles, display_slot
from geo2wf.operational.intensification import official_summary
from geo2wf.operational.export import export_release, publish, prune_local
from geo2wf.operational.store import Store


def frame(output, day, hour):
    image = f"imagery/{digest(f'{day}-{hour}'.encode())}.webp"
    files = [image, image + ".aux.xml"]
    for path, data in zip(
        files, [b"RIFFtest-webp", b"<PAMDataset>georeference</PAMDataset>"]
    ):
        target = output / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return {
        "storm_id": "EP012026",
        "time": f"2026-09-{day:02d}T{hour:02d}:00:00Z",
        "status": "ready",
        "files": files,
        "parts": [{"image": image, "bbox": [179, 10, 180, 20]}],
    }


def test_daily_zip_reuse_georeferencing_and_append(tmp_path):
    frames = [
        frame(tmp_path, 1, 0),
        frame(tmp_path, 1, 1),
        frame(tmp_path, 1, 2),
        frame(tmp_path, 2, 0),
    ]
    first = daily_bundles("EP012026", frames, tmp_path)
    assert len(first) == 2
    assert display_slot(frames[0]["time"]) and not display_slot(frames[1]["time"])
    assert len(first[0]["images"]) == 2
    with ZipFile(tmp_path / first[0]["path"]) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["crs"] == "EPSG:3857"
        assert len(manifest["frames"]) == 2
        for record in [frames[0], frames[2]]:
            for path in record["files"]:
                assert archive.read(path) == (tmp_path / path).read_bytes()
    assert daily_bundles("EP012026", list(reversed(frames)), tmp_path) == first
    second = daily_bundles("EP012026", frames + [frame(tmp_path, 2, 2)], tmp_path)
    assert first[0] == second[0]
    assert first[1]["path"] != second[1]["path"]
    (tmp_path / first[0]["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="immutable"):
        daily_bundles("EP012026", frames, tmp_path)


def fixes():
    return [
        {
            "time": f"2026-09-{1 + h // 24:02d}T{h % 24:02d}:00:00Z",
            "wind_ms": (40 + h * 30 / 24) * KNOT,
            "classification": "TS",
            "lat": 20,
            "lon": -110,
        }
        for h in range(0, 25, 6)
    ]


def test_official_ri_and_lifetime_peaks_survive_retention(tmp_path):
    track = fixes()
    summary = official_summary(track)
    assert summary["has_ri"] is True
    assert summary["peak_wind_ms"] == 70 * KNOT
    assert summary["ri_events"][0]["windows"] == 1
    assert official_summary(track[:2])["has_ri"] is None
    assert (
        official_summary([f for i, f in enumerate(track) if i != 2])["has_ri"] is None
    )
    assert official_summary([{**f, "wind_ms": 30} for f in track])["has_ri"] is False
    assert (
        official_summary([{**f, "classification": "EX"} for f in track])["has_ri"]
        is None
    )
    store = Store(tmp_path / "state.sqlite")
    store.put_storm(
        {
            "id": "EP012026",
            "name": "Test",
            "basin": "EP",
            "active": True,
            "start": track[0]["time"],
            "end": track[-1]["time"],
            "track": track,
        }
    )
    store.retain("2026-09-02")
    catalog = export_release(
        store, tmp_path / "export", "2026-09-02", "2026-09-02T06:00:00Z"
    )
    assert catalog["storms"][0]["peak_official_wind_ms"] == 70 * KNOT
    assert catalog["storms"][0]["has_ri"] is True
    store.close()


def test_bundle_failure_never_advances_pointer_and_retention_follows_manifest(tmp_path):
    data = daily_bundles("EP012026", [frame(tmp_path, 1, 0)], tmp_path)[0]
    manifest = {"files": [data["path"]]}
    path = f"objects/{digest(manifest)}.json"
    (tmp_path / "objects").mkdir()
    (tmp_path / path).write_text(json.dumps(manifest))
    release = tmp_path / "releases" / "test"
    release.mkdir(parents=True)
    (release / "catalog.json").write_text(
        json.dumps({"storms": [], "imagery": {"manifest": path}})
    )
    (tmp_path / "latest.json").write_text(json.dumps({"version": "test"}))
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        if command[1] == "check" and command[2].endswith("bundles"):
            raise CalledProcessError(1, command)

    with pytest.raises(CalledProcessError):
        publish(tmp_path, run=fail)
    assert any(
        command[1] == "copy" and command[2].endswith("bundles") for command in calls
    )
    assert not any(command[1] == "copyto" for command in calls)
    calls.clear()
    publish(
        tmp_path,
        run=lambda command, **kwargs: calls.append(command),
        advance_pointer=False,
    )
    assert any(command[1] == "check" for command in calls)
    assert not any(command[1] == "copyto" for command in calls)
    unused = tmp_path / "bundles" / ("f" * 64 + ".zip")
    unused.write_bytes(b"unused")
    removed = prune_local(tmp_path, apply=True)
    assert str(unused.relative_to(tmp_path)) in removed
    assert (tmp_path / data["path"]).exists()
