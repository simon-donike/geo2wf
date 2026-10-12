#!/usr/bin/env python3
"""Package the fixed StormSense showcase; never modifies the operational release.

Run from the repository root. --check verifies the committed bundle against its
source archive and checksums without writing any files.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "apps/stormsense/public"
SOURCE = PUBLIC / "data"
DEST = PUBLIC / "showcase/v1"
RELEASE = "20261011T120622220077Z"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def package():
    catalog_path = f"releases/{RELEASE}/catalog.json"
    catalog_bytes = (SOURCE / catalog_path).read_bytes()
    catalog = json.loads(catalog_bytes)
    original = next(s for s in catalog["storms"] if s["id"] == "EP152026")
    series_bytes = (SOURCE / original["series"]).read_bytes()
    full_series = json.loads(series_bytes)
    canonical = json.dumps(full_series, sort_keys=True, separators=(",", ":")).encode()
    assert digest(canonical) == Path(original["series"]).stem
    series = dict(full_series)
    # Keep the entire published history, including the start gate and real gaps.
    # Delivery packs below replace the archive's daily imagery ZIPs only.
    series["imagery_bundles"] = []
    assert len(series["records"]) == 253
    assert all(r["kind"] == "hindcast" for r in series["records"])
    assert sum(r["status"] == "ready" for r in series["records"]) == 252
    assert len(series["imagery"]) == 292
    assert all(i["status"] == "ready" for i in series["imagery"])
    series_data = encoded(series)
    series_path = f"objects/{digest(series_data)}.json"
    assets = {series_path: series_data}
    for frame in series["imagery"]:
        paths = [frame["metadata"]]
        for part in frame["parts"]:
            image = (SOURCE / part["image"]).read_bytes()
            assert digest(image) == part["sha256"]
            paths.extend(part[k] for k in ("image", "preview", "sidecar", "preview_sidecar"))
        for path in paths:
            assets[path] = (SOURCE / path).read_bytes()
    storm = dict(original, series=series_path)
    window = {"start": original["start"], "end": original["end"]}
    catalog = dict(catalog, release="showcase-v1", window=window,
                   storms=[storm], reports={},
                   coverage={"expected": storm["expected_count"],
                             "predictions": storm["prediction_count"],
                             "gaps": storm["gap_count"], "pending": storm["pending_count"]})
    catalog.pop("imagery", None)  # The full-release image manifest is outside this subset.
    assets["releases/showcase-v1/catalog.json"] = encoded(catalog)
    assets["latest.json"] = encoded({"schema_version": 1, "version": "showcase-v1",
                                     "manifest": "releases/showcase-v1/catalog.json"})
    recorded = next(r for r in series["records"] if r["time"] == "2026-09-28T06:00:00Z")
    shared = [PUBLIC / "land.geojson", *sorted((PUBLIC / "method").iterdir()),
              *sorted((PUBLIC / "brand").iterdir())]
    shared = [p for p in shared if p.is_file() and p.suffix in (".json", ".geojson", ".webp", ".svg", ".png")]
    files = [{"url": f"/showcase/v1/{path}", "sha256": digest(data), "bytes": len(data)}
             for path, data in sorted(assets.items())]
    files += [{"url": "/" + str(p.relative_to(PUBLIC)), "sha256": digest(p.read_bytes()),
               "bytes": p.stat().st_size} for p in shared]
    # A full storm has >1,500 small files. Fetch bounded packs to avoid thousands
    # of network round trips; retain and verify every original asset separately.
    packs, group, group_bytes = [], [], 0

    def pack_group():
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for path, data in group:
                info = zipfile.ZipInfo(path, date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, data, compresslevel=6)
        data = buffer.getvalue()
        path = f"packs/{digest(data)}.zip"
        assets[path] = data
        packs.append({"url": f"/showcase/v1/{path}", "sha256": digest(data),
                      "bytes": len(data),
                      "start": sum(p["count"] for p in packs), "count": len(group)})

    for path, data in sorted(assets.items()):
        if group and group_bytes + len(data) > 5 * 1024 * 1024:
            pack_group()
            group, group_bytes = [], 0
        group.append((path, data))
        group_bytes += len(data)
    if group:
        pack_group()
    assets["manifest.json"] = encoded({
        "schema_version": 1, "version": "v1", "label": "Nolo · recorded historical hindcast",
        "window": window, "source_release": RELEASE,
        "source_catalog": {"path": catalog_path, "sha256": digest(catalog_bytes)},
        "source_series": {"path": original["series"], "sha256": digest(series_bytes)},
        "source_storm": original, "recorded_prediction": recorded,
        "note": "The complete published Nolo history: records, track fixes, forecasts, imagery, schedule and provenance are unchanged. Catalog includes Nolo only; local delivery packs replace daily imagery bundles. No model was rerun.",
        "files": files, "packs": packs,
    })
    return assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    assets = package()
    old_manifest = json.loads((DEST / "manifest.json").read_text()) if (DEST / "manifest.json").exists() else {}
    for path, data in assets.items():
        target = DEST / path
        if args.check:
            assert target.read_bytes() == data, f"Showcase differs from source: {path}"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    if not args.check:
        # Remove only obsolete files owned by the previous generated manifest.
        for entry in old_manifest.get("files", []) + old_manifest.get("packs", []):
            prefix = "/showcase/v1/"
            if entry["url"].startswith(prefix):
                path = entry["url"][len(prefix):]
                assert ".." not in Path(path).parts
                if path not in assets:
                    (DEST / path).unlink(missing_ok=True)
    existing = {str(p.relative_to(DEST)) for p in DEST.rglob("*") if p.is_file()}
    assert existing == set(assets), "Unexpected files in the versioned showcase bundle"
    print(f"{'Verified' if args.check else 'Packaged'} {len(assets)} files, {sum(map(len, assets.values())):,} bytes")


if __name__ == "__main__":
    main()
