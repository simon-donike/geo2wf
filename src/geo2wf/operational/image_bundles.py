"""Deterministic daily ZIPs with portable pixels, GIS sidecars and STAC metadata."""

from collections import defaultdict
from io import BytesIO
from zipfile import ZipFile, ZipInfo, ZIP_STORED

from .common import digest, encoded, utc
from .geocolor import atomic_bytes, frame_files

VERSION = "geocolor-daily-zip-v1"


def display_slot(time):
    at = utc(time)
    return at.hour % 2 == 0 and at.minute == 0 and at.second == 0


def daily_bundles(storm_id, frames, output):
    days = defaultdict(list)
    for frame in frames:
        if frame["status"] == "ready" and display_slot(frame["time"]):
            days[frame["time"][:10]].append(frame)
    result = []
    for day, group in sorted(days.items()):
        group.sort(key=lambda frame: frame["time"])
        files = sorted({path for frame in group for path in frame_files(frame)})
        manifest = {
            "schema_version": 1,
            "format": VERSION,
            "storm_id": storm_id,
            "date": day,
            "crs": "EPSG:3857",
            "frames": [
                {k: v for k, v in frame.items() if k != "files"} for frame in group
            ],
        }
        data = BytesIO()
        # Stored WebP members need no CPU decompression. Fixed metadata makes
        # unchanged days byte-identical, regardless of local mtime/export time.
        with ZipFile(data, "w", compression=ZIP_STORED) as archive:
            for name, content in [("manifest.json", encoded(manifest))] + [
                (key, (output / key).read_bytes()) for key in files
            ]:
                info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = ZIP_STORED
                info.external_attr = 0o644 << 16
                archive.writestr(info, content)
        content = data.getvalue()
        sha = digest(content)
        path = f"bundles/{sha}.zip"
        target = output / path
        atomic_bytes(target, content)
        result.append(
            {
                "schema_version": 1,
                "date": day,
                "path": path,
                "sha256": sha,
                "bytes": len(content),
                "images": [key for key in files if key.endswith(".webp")],
            }
        )
    return result
