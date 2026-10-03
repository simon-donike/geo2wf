"""Optional hourly display imagery. GIBS only: unavailable crops remain gaps.

WebP pixels are north-up EPSG:3857 PixelIsArea rasters, accompanied by GDAL
PAM sidecars and STAC Items. No raw ABI downloads or model inference occur.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, Future
from datetime import timedelta
from io import BytesIO
import json
import logging
import math
from pathlib import Path
import threading
from urllib.parse import urlencode
from xml.etree import ElementTree as ET

from PIL import Image
from pyproj import CRS

from .common import digest, encoded, fetch, hours, iso, utc, write_json
from .models import version
from .tracks import historical_center

VERSION = "gibs-geocolor-webp-v1"
GIBS = "https://gibs.earthdata.nasa.gov"
PIXELS, PREVIEW, QUALITY, CROP_KM = 768, 256, 75, 1800
R = 6378137
WORLD = math.pi * R
WKT = CRS.from_epsg(3857).to_wkt()
LOG = logging.getLogger(__name__)


class Gap(Exception):
    pass


def side_for(lon):
    return (
        "West"
        if math.cos(math.radians(lon + 137)) > math.cos(math.radians(lon + 75.2))
        else "East"
    )


def available_times(xml):
    root = ET.fromstring(xml)
    domain = next(
        (
            node.text or ""
            for node in root.iter()
            if node.tag.split("}")[-1] == "Domain"
        ),
        None,
    )
    if domain is None:
        raise ValueError("imagery availability response has no time domain")
    result = set()
    for interval in domain.split(","):
        if not interval.strip():
            continue
        parts = interval.strip().split("/")
        start = utc(parts[0])
        if len(parts) == 1:
            result.add(start)
        elif len(parts) == 3 and parts[2].startswith("PT") and parts[2].endswith("M"):
            step = int(parts[2][2:-1])
            end = utc(parts[1])
            if step <= 0 or end - start > timedelta(days=2):
                raise ValueError("invalid imagery availability interval")
            while start <= end:
                result.add(start)
                start += timedelta(minutes=step)
        else:
            raise ValueError("unsupported imagery availability interval")
    return sorted(result)


class Client:
    def __init__(self, read=None):
        self.read = read or (lambda url: fetch(url, attempts=2, timeout=12))
        self.domains = {}
        self.lock = threading.Lock()

    def availability(self, side, day):
        key = (side, day)
        with self.lock:
            owner = key not in self.domains
            if owner:
                self.domains[key] = Future()
            future = self.domains[key]
        if owner:
            url = f"{GIBS}/wmts/epsg3857/best/1.0.0/GOES-{side}_ABI_GeoColor/default/GoogleMapsCompatible_Level7/all/{day}T00:00:00Z--{day}T23:59:59Z.xml"
            try:
                raw = self.read(url)
                future.set_result(
                    (
                        available_times(raw),
                        {"url": url, "retrieved_at": iso(), "sha256": digest(raw)},
                    )
                )
            except Exception as error:
                future.set_exception(error)
        return future.result()


def geometry(lat, lon):
    if not math.isfinite(lat + lon) or abs(lat) > 75 or abs(lon) > 180:
        raise Gap("unsupported_center")
    x = R * math.radians(lon)
    y = R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    half = CROP_KM * 500 / math.cos(math.radians(lat))
    south, north = max(-WORLD, y - half), min(WORLD, y + half)

    def unproject(px, py):
        return [
            math.degrees(px / R),
            math.degrees(2 * math.atan(math.exp(py / R)) - math.pi / 2),
        ]

    result = []
    for world in range(
        math.floor((x - half + WORLD) / (2 * WORLD)),
        math.floor((x + half + WORLD) / (2 * WORLD)) + 1,
    ):
        left, right = max(x - half, -WORLD + world * 2 * WORLD), min(
            x + half, WORLD + world * 2 * WORLD
        )
        if right <= left:
            continue
        bbox = [
            round(v, 2)
            for v in (left - world * 2 * WORLD, south, right - world * 2 * WORLD, north)
        ]
        west, low = unproject(bbox[0], bbox[1])
        east, high = unproject(bbox[2], bbox[3])
        # Canonical geometry stays within +/-180. Display coordinates unwrap
        # around the storm; clients can select a different world themselves.
        west, east = max(-180, west), min(180, east)
        result.append(
            {
                "proj_bbox": bbox,
                "bbox": [west, low, east, high],
                "display_bbox": [west + world * 360, low, east + world * 360, high],
                "width": max(1, round(PIXELS * (right - left) / (2 * half))),
                "height": PIXELS,
            }
        )
    return result


def source_url(part, side, at):
    return (
        GIBS
        + "/wms/epsg3857/best/wms.cgi?"
        + urlencode(
            {
                "SERVICE": "WMS",
                "VERSION": "1.1.1",
                "REQUEST": "GetMap",
                "LAYERS": f"GOES-{side}_ABI_GeoColor",
                "STYLES": "",
                "SRS": "EPSG:3857",
                "BBOX": ",".join(map(str, part["proj_bbox"])),
                "WIDTH": part["width"],
                "HEIGHT": part["height"],
                "FORMAT": "image/png",
                "TRANSPARENT": "TRUE",
                "TIME": iso(at),
            }
        )
    )


def atomic_bytes(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != body:
            raise ValueError(f"immutable imagery asset changed: {path}")
        return
    temporary = path.with_name(path.name + f".{threading.get_ident()}.tmp")
    temporary.write_bytes(body)
    temporary.replace(path)


def encode_asset(image, part, root):
    width, height = image.size
    left, bottom, right, top = part["proj_bbox"]
    dx, dy = (right - left) / width, (top - bottom) / height
    transform = [dx, 0, left, 0, -dy, top, 0, 0, 1]
    out = BytesIO()
    image.save(out, format="WEBP", quality=QUALITY, method=4, exact=True)
    body = out.getvalue()
    sha = digest(body)
    # Include the grid in the address: identical pixels at different locations
    # must never share a contradictory georeferencing sidecar.
    key = digest(
        {"sha256": sha, "transform": transform, "shape": [height, width], "crs": 3857}
    )
    path = f"imagery/{key}.webp"
    pam = ET.Element("PAMDataset")
    ET.SubElement(pam, "SRS", dataAxisToSRSAxisMapping="1,2").text = WKT
    ET.SubElement(pam, "GeoTransform").text = ",".join(
        map(str, [left, dx, 0, top, 0, -dy])
    )
    ET.SubElement(ET.SubElement(pam, "Metadata"), "MDI", key="AREA_OR_POINT").text = (
        "Area"
    )
    atomic_bytes(root / path, body)
    atomic_bytes(root / (path + ".aux.xml"), ET.tostring(pam, encoding="utf-8"))
    return {
        "path": path,
        "sidecar": path + ".aux.xml",
        "sha256": sha,
        "bytes": len(body),
        "proj:epsg": 3857,
        "proj:shape": [height, width],
        "proj:transform": transform,
        "proj:bbox": part["proj_bbox"],
    }


def save_frame(record, images, root):
    assets, parts, polygons = {}, [], []
    files = []
    for i, (part, image, url, retrieved) in enumerate(images):
        full = encode_asset(image, part, root)
        preview = encode_asset(
            image.resize(
                (max(1, round(image.width * PREVIEW / PIXELS)), PREVIEW),
                Image.Resampling.LANCZOS,
            ),
            part,
            root,
        )
        for role, asset in (("visual", full), ("thumbnail", preview)):
            name = f"{role}-{i}"
            assets[name] = {
                "href": Path(asset["path"]).name,
                "type": "image/webp",
                "roles": [role],
                "proj:wkt2": WKT,
                **{k: v for k, v in asset.items() if k.startswith("proj:")},
                "stormsense:sha256": asset["sha256"],
                "stormsense:bytes": asset["bytes"],
            }
            assets[name + "-georeference"] = {
                "href": Path(asset["sidecar"]).name,
                "type": "application/xml",
                "roles": ["metadata"],
            }
            files.extend([asset["path"], asset["sidecar"]])
        west, south, east, north = part["bbox"]
        polygons.append(
            [
                [
                    [west, south],
                    [east, south],
                    [east, north],
                    [west, north],
                    [west, south],
                ]
            ]
        )
        parts.append(
            {
                "image": full["path"],
                "preview": preview["path"],
                "sidecar": full["sidecar"],
                "preview_sidecar": preview["sidecar"],
                "bbox": part["bbox"],
                "display_bbox": part["display_bbox"],
                "source_url": url,
                "retrieved_at": retrieved,
                "sha256": full["sha256"],
                "bytes": full["bytes"],
                "preview_bytes": preview["bytes"],
                "valid_fraction": sum(image.getchannel("A").histogram()[1:])
                / (image.width * image.height),
            }
        )
    bboxes = [part["bbox"] for part in parts]
    item = {
        "type": "Feature",
        "stac_version": "1.0.0",
        "stac_extensions": [
            "https://stac-extensions.github.io/projection/v1.1.0/schema.json"
        ],
        "id": f"{record['storm_id']}-{record['time']}-{VERSION}",
        "geometry": {"type": "MultiPolygon", "coordinates": polygons},
        "bbox": [
            min(b[0] for b in bboxes),
            min(b[1] for b in bboxes),
            max(b[2] for b in bboxes),
            max(b[3] for b in bboxes),
        ],
        "properties": {
            "datetime": record["acquired_at"],
            "created": iso(),
            "proj:epsg": 3857,
            "stormsense:slot_time": record["time"],
            "stormsense:center": record["center"],
            "stormsense:version": VERSION,
            "stormsense:source": "NASA GIBS / NOAA / CIRA GeoColor",
            "stormsense:pixel_interpretation": "Area",
            "stormsense:availability": record["availability"],
            "stormsense:parts": parts,
        },
        "links": [
            {"rel": "derived_from", "href": part["source_url"], "type": "image/png"}
            for part in parts
        ],
        "assets": assets,
    }
    metadata = f"imagery/{digest(item)}.json"
    atomic_bytes(root / metadata, encoded(item) + b"\n")
    return {
        **record,
        "status": "ready",
        "reason": None,
        "parts": parts,
        "metadata": metadata,
        "files": [*files, metadata],
        "generated_at": iso(),
    }


def acquire(job, client, root, now=None):
    now = utc(now)
    record = {
        **job,
        "version": VERSION,
        "checked_at": iso(now),
        "parts": [],
        "files": [],
    }
    try:
        if not job.get("center"):
            raise Gap("no_center")
        at = utc(job["time"])
        if at > now:
            raise Gap("future_slot")
        side = side_for(job["center"]["lon"])
        record["satellite"] = side
        times, provenance = client.availability(side, at.date().isoformat())
        if at.hour == 0:
            prior, prior_provenance = client.availability(
                side, (at - timedelta(days=1)).date().isoformat()
            )
            times, provenance = times + prior, [provenance, prior_provenance]
        record["availability"] = provenance
        candidates = sorted(
            (t for t in times if at - timedelta(minutes=30) <= t <= at), reverse=True
        )
        if not candidates:
            raise Gap("no_reported_frame")
        # Older crops get one inexpensive attempt. For fresh slots allow the
        # provider's newest regional tiles time to arrive by trying predecessors.
        candidates = candidates[: 3 if now - at < timedelta(hours=6) else 1]
        parts = geometry(job["center"]["lat"], job["center"]["lon"])
        for candidate in candidates:
            images = []
            for part in parts:
                url = source_url(part, side, candidate)
                raw = client.read(url)
                image = Image.open(BytesIO(raw))
                if image.size != (part["width"], part["height"]):
                    raise Gap("invalid_image_dimensions")
                image = image.convert("RGBA")
                # Empty or almost empty imagery is not a successful observation.
                valid = sum(image.getchannel("A").histogram()[1:]) / (
                    image.width * image.height
                )
                if valid < 0.05:
                    break
                images.append((part, image, url, iso()))
            if len(images) == len(parts):
                record["acquired_at"] = iso(candidate)
                break
        else:
            raise Gap("empty_provider_image")
    except Gap as error:
        return {**record, "status": "gap", "reason": str(error)}
    except (OSError, ValueError, ET.ParseError) as error:
        return {
            **record,
            "status": "gap",
            "reason": "source_unavailable",
            "detail": str(error)[:240],
        }
    # Local encoding/storage failures must stop the job for safe resumption;
    # they are never misreported as a provider's missing observation.
    return save_frame(record, images, Path(root))


def asset_root(store):
    return store.path.parent / "geocolor"


def frame_files(frame):
    return frame.get("files", []) if frame.get("status") == "ready" else []


def backfill_images(
    store,
    start,
    end,
    workers=4,
    limit=None,
    retry_gaps=False,
    storm_ids=None,
    active_only=False,
    client=None,
):
    end = min(utc(end), utc())
    jobs = []
    for storm in store.storms():
        if (storm_ids and storm["id"] not in storm_ids) or (
            active_only and not storm.get("active")
        ):
            continue
        samples = {}
        for sample in store.samples(storm["id"], version("nowcast")):
            old = samples.get(sample["time"])
            if old is None or (
                sample.get("center")
                and (not old.get("center") or sample["kind"] == "live")
            ):
                samples[sample["time"]] = sample
        saved = {r["time"]: r for r in store.visuals(storm["id"], VERSION)}
        for at in hours(
            max(utc(start), utc(storm["start"])),
            min(end, end if storm.get("active") else utc(storm["end"])),
        ):
            previous = saved.get(iso(at))
            if previous:
                if previous["status"] == "ready":
                    if all(
                        (asset_root(store) / f).is_file() for f in frame_files(previous)
                    ):
                        continue
                elif not retry_gaps and not (
                    at >= utc() - timedelta(hours=6)
                    and utc(previous["checked_at"]) < utc() - timedelta(minutes=10)
                ):
                    continue
            sample = samples.get(iso(at))
            center = sample.get("center") if sample else None
            center = center or historical_center(storm.get("track", []), at)
            jobs.append(
                {
                    "storm_id": storm["id"],
                    "time": iso(at),
                    "center": center,
                    "center_kind": (
                        sample["kind"]
                        if sample and sample.get("center")
                        else "retrospective_track"
                    ),
                }
            )
    jobs.sort(key=lambda j: (j["time"], j["storm_id"]), reverse=True)
    if limit is not None:
        jobs = jobs[:limit]
    client = client or Client()
    counts = Counter()
    iterator = iter(jobs)
    stopped = False
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {}
        while True:
            stopped = (
                stopped
                or Path(str(store.path) + ".update-requested").exists()
                or Path(str(store.path) + ".stop-imagery").exists()
            )
            while not stopped and len(pending) < workers:
                job = next(iterator, None)
                if job is None:
                    break
                pending[executor.submit(acquire, job, client, asset_root(store))] = job
            if not pending:
                break
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                result = future.result()
                store.put_visual(result)
                counts[result["status"]] += 1
                pending.pop(future)
                if sum(counts.values()) % 50 == 0:
                    LOG.info(
                        "GeoColor %s/%s: %s",
                        sum(counts.values()),
                        len(jobs),
                        dict(counts),
                    )
    result = {
        "version": VERSION,
        "start": iso(start),
        "end": iso(end),
        "finished_at": iso(),
        "scheduled": len(jobs),
        "processed": sum(counts.values()),
        "results": dict(counts),
        "interrupted": stopped,
    }
    store.put_status("geocolor", result)
    return result


def retain_assets(store, apply=False):
    references = {
        path
        for storm in store.storms()
        for frame in store.visuals(storm["id"], VERSION)
        for path in frame_files(frame)
    }
    root = asset_root(store)
    obsolete = [
        p
        for p in (root / "imagery").glob("*")
        if str(p.relative_to(root)) not in references
    ]
    if apply:
        for path in obsolete:
            path.unlink()
    return [str(p.relative_to(root)) for p in obsolete]
