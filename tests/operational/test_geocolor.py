from io import BytesIO
import json
from pathlib import Path
from subprocess import CalledProcessError
from urllib.error import URLError
from urllib.parse import parse_qs, urlsplit

from PIL import Image
import pytest
import rasterio
from rasterio.transform import xy
from pyproj import Transformer

from geo2wf.operational import geocolor as geo
from geo2wf.operational.common import utc, iso
from geo2wf.operational.export import export_release, publish, prune_local, prune_remote
from geo2wf.operational.store import Store


def job(lon=-110):
    return {
        "storm_id": "EP012026",
        "time": "2026-09-01T12:00:00Z",
        "center": {"lat": 20, "lon": lon},
        "center_kind": "hindcast",
    }


def source(empty=False, fail=False):
    calls = []

    def read(url):
        calls.append(url)
        if fail:
            raise URLError("offline")
        if url.endswith(".xml"):
            return b"<Domains><Domain>2026-09-01T11:40:00Z/2026-09-01T12:10:00Z/PT10M</Domain></Domains>"
        p = parse_qs(urlsplit(url).query)
        out = BytesIO()
        Image.new(
            "RGBA",
            (int(p["WIDTH"][0]), int(p["HEIGHT"][0])),
            (60, 120, 180, 0 if empty else 255),
        ).save(out, format="PNG")
        return out.getvalue()

    return geo.Client(read), calls


def test_georeferencing_roundtrip_and_preview_grid(tmp_path):
    client, calls = source()
    frame = geo.acquire(job(), client, tmp_path, now="2026-09-01T12:05:00Z")
    assert frame["status"] == "ready"
    assert frame["acquired_at"] == frame["time"]  # advertised future 12:10 excluded
    part = frame["parts"][0]
    item = json.loads((tmp_path / frame["metadata"]).read_text())
    assert item["properties"]["datetime"] == frame["acquired_at"]
    assert item["properties"]["stormsense:slot_time"] == frame["time"]
    assert all((tmp_path / f).is_file() for f in frame["files"])
    with rasterio.open(tmp_path / part["image"]) as full, rasterio.open(
        tmp_path / part["preview"]
    ) as preview:
        assert full.crs.to_epsg() == preview.crs.to_epsg() == 3857
        assert full.shape == (768, 768) and preview.shape == (256, 256)
        assert full.bounds == preview.bounds
        assert preview.transform.a == pytest.approx(3 * full.transform.a)
        assert list(full.transform) == item["assets"]["visual-0"]["proj:transform"]
        x, y = xy(full.transform, 0, 0)
        assert x == pytest.approx(full.bounds.left + full.transform.a / 2)
        assert y == pytest.approx(full.bounds.top + full.transform.e / 2)
        lon, lat = Transformer.from_crs(3857, 4326, always_xy=True).transform(
            (full.bounds.left + full.bounds.right) / 2,
            (full.bounds.top + full.bounds.bottom) / 2,
        )
        assert (lon, lat) == pytest.approx((-110, 20), abs=1e-6)
    assert len([u for u in calls if "GetMap" in u]) == 1


@pytest.mark.parametrize("lon", [-178, 178])
def test_dateline_split_has_canonical_and_continuous_bounds(tmp_path, lon):
    client, _ = source()
    frame = geo.acquire(job(lon), client, tmp_path, now="2026-10-02")
    parts = frame["parts"]
    assert len(parts) == 2
    assert parts[0]["display_bbox"][2] == pytest.approx(
        parts[1]["display_bbox"][0], abs=1e-6
    )
    for part in parts:
        assert -180 <= part["bbox"][0] < part["bbox"][2] <= 180
        with rasterio.open(tmp_path / part["image"]) as image:
            assert image.transform.a > 0 and image.transform.e < 0
            assert image.bounds.left >= -geo.WORLD - 0.01
            assert image.bounds.right <= geo.WORLD + 0.01


def test_empty_images_and_source_errors_are_distinct_and_retries_bounded(tmp_path):
    client, calls = source(empty=True)
    frame = geo.acquire(job(), client, tmp_path, now="2026-09-01T12:05:00Z")
    assert frame["reason"] == "empty_provider_image"
    assert len([u for u in calls if "GetMap" in u]) == 3
    assert not list(tmp_path.rglob("*.webp"))
    client, calls = source(empty=True)
    assert geo.acquire(job(), client, tmp_path, now="2026-10-02")["status"] == "gap"
    assert len([u for u in calls if "GetMap" in u]) == 1
    client, _ = source(fail=True)
    assert geo.acquire(job(), client, tmp_path)["reason"] == "source_unavailable"
    client, calls = source()
    assert (
        geo.acquire({**job(), "center": None}, client, tmp_path)["reason"]
        == "no_center"
    )
    assert calls == []


def test_resume_export_publication_and_retention(tmp_path):
    store = Store(tmp_path / "state.sqlite")
    store.put_storm(
        {
            "id": "EP012026",
            "name": "Test",
            "basin": "EP",
            "active": False,
            "start": job()["time"],
            "end": job()["time"],
            "track": [{"time": job()["time"], "lat": 20, "lon": -110, "wind_ms": 30}],
        }
    )
    client, calls = source()
    result = geo.backfill_images(store, "2026-09-01", "2026-09-02", client=client)
    assert result["results"] == {"ready": 1}
    count = len(calls)
    assert (
        geo.backfill_images(store, "2026-09-01", "2026-09-02", client=client)[
            "processed"
        ]
        == 0
    )
    assert len(calls) == count
    output = tmp_path / "export"
    a = export_release(store, output, "2026-09-01", "2026-09-02", release="a")
    b = export_release(store, output, "2026-09-01", "2026-09-02", release="b")
    assert a["imagery"]["coverage"] == {
        "expected": 1,
        "ready": 1,
        "gaps": 0,
        "pending": 0,
    }
    assert a["storms"][0]["series"] == b["storms"][0]["series"]
    calls = []

    def fail(command, **_):
        calls.append(command)
        raise CalledProcessError(1, command)

    with pytest.raises(CalledProcessError):
        publish(output, run=fail)
    assert "/imagery" in calls[0][3]
    assert not any(c[1] == "copyto" for c in calls)
    visuals = store.visuals("EP012026", geo.VERSION)
    for key in visuals[0]["files"]:
        assert (output / key).exists()
    obsolete = prune_local(output, keep=1, apply=True)
    assert "releases/a/catalog.json" in obsolete
    assert not any(k.startswith("imagery/") for k in obsolete)
    # A published release still pins its images after processing-state retention.
    store.retain("2026-09-02")
    assert geo.retain_assets(store, apply=True)
    assert all((output / key).exists() for key in visuals[0]["files"])
    store.close()


def test_identical_pixels_on_different_grids_get_distinct_asset_names(tmp_path):
    image = Image.new("RGBA", (32, 32), (80, 100, 120, 255))
    a = geo.encode_asset(image, geo.geometry(20, -110)[0], tmp_path)
    b = geo.encode_asset(image, geo.geometry(21, -110)[0], tmp_path)
    assert a["sha256"] == b["sha256"] and a["path"] != b["path"]


@pytest.mark.parametrize(
    "now,early,due",
    [
        ("2026-09-01T12:05:00Z", "2026-09-01T13:04:59Z", "2026-09-01T13:05:00Z"),
        ("2026-09-04T12:00:00Z", "2026-09-05T11:59:59Z", "2026-09-05T12:00:00Z"),
    ],
)
@pytest.mark.parametrize("force", [False, True])
def test_source_gaps_retry_after_cooldown_or_explicit_retry(
    tmp_path, now, early, due, force
):
    store = Store(tmp_path / "state.sqlite")
    store.put_storm(
        {
            "id": job()["storm_id"],
            "name": "Test",
            "basin": "EP",
            "active": False,
            "start": job()["time"],
            "end": job()["time"],
            "track": [{"time": job()["time"], **job()["center"], "wind_ms": 30}],
        }
    )
    offline, _ = source(fail=True)
    args = (store, "2026-09-01", "2026-09-02")
    assert geo.backfill_images(*args, client=offline, now=now)["results"] == {"gap": 1}
    client, calls = source()
    assert geo.backfill_images(*args, client=client, now=early)["processed"] == 0
    assert calls == []
    assert geo.backfill_images(
        *args, client=client, now=early if force else due, retry_gaps=force
    )["results"] == {"ready": 1}
    assert geo.backfill_images(*args, client=client, now=due)["processed"] == 0
    frame = store.visuals(job()["storm_id"], geo.VERSION)[0]
    assert all((geo.asset_root(store) / path).is_file() for path in frame["files"])
    store.close()


def test_missing_center_waits_for_track_and_legacy_gaps_retry():
    frame = {
        **job(),
        "status": "gap",
        "reason": "source_unavailable",
        "checked_at": "2026-09-01T12:05:00Z",
    }
    assert not geo.retry_due(frame, job()["center"], utc("2026-09-01T12:10:00Z"))
    frame["reason"] = "no_center"
    assert not geo.retry_due(frame, None, utc("2026-10-01"))
    assert geo.retry_due(frame, job()["center"], utc("2026-09-01T12:10:00Z"))
    frame.pop("checked_at")
    frame["reason"] = "no_reported_frame"
    assert geo.retry_due(frame, job()["center"], utc("2026-10-01"))


def test_storage_failure_is_not_reported_as_a_provider_gap(tmp_path, monkeypatch):
    client, _ = source()

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(geo, "atomic_bytes", fail)
    with pytest.raises(OSError, match="disk full"):
        geo.acquire(job(), client, tmp_path)


def test_unavailable_time_domain_is_not_confused_with_empty_availability():
    assert geo.available_times(b"<Domains><Domain/></Domains>") == []
    with pytest.raises(ValueError, match="no time domain"):
        geo.available_times(b"<ServiceExceptionReport/>")


def test_remote_retention_preserves_image_manifest_and_rejects_missing_assets():
    from types import SimpleNamespace

    series = "objects/" + "a" * 64 + ".json"
    manifest = "objects/" + "b" * 64 + ".json"
    image = "imagery/" + "c" * 64 + ".webp"
    sidecar = image + ".aux.xml"
    unused = "imagery/" + "d" * 64 + ".webp"
    values = {
        "latest.json": {"version": "latest"},
        "releases/latest/catalog.json": {
            "storms": [{"series": series}],
            "imagery": {"manifest": manifest},
        },
        manifest: {"files": [image, sidecar]},
    }
    listing = [
        {"Path": path, "ModTime": "2025-01-01T00:00:00.123456789Z"}
        for path in [*values, series, image, sidecar, unused]
    ]

    def run(command, **kwargs):
        if command[1] == "cat":
            result = values[command[2].split("/stormsense/")[1]]
        else:
            result = listing
        return SimpleNamespace(stdout=json.dumps(result))

    assert prune_remote(keep=1, run=run) == [unused]
    listing[:] = [row for row in listing if row["Path"] != sidecar]
    with pytest.raises(ValueError, match="imagery assets are missing"):
        prune_remote(keep=1, run=run)
