"""Read only storm windows from public GOES ABI multi-channel CMI NetCDFs."""

from __future__ import annotations

from datetime import datetime, timedelta
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
import math
import re
import time
from urllib.parse import urlencode
import xml.etree.ElementTree as ET

import fsspec
import h5py
from aiohttp import ClientError
import numpy as np
from pyproj import CRS, Transformer
import xarray as xr

from .common import UTC, fetch, iso, utc

BANDS = tuple(f"CMI_C{i:02d}" for i in range(7, 17))
PRODUCT = "ABI-L2-MCMIPF"
_read_scope = ContextVar("stormsense_scan_read_scope", default=None)


@contextmanager
def shared_reads():
    """Reuse compressed strips for storms in one hourly batch, then discard them."""
    handles = {}
    token = _read_scope.set(handles)
    try:
        yield
    finally:
        for handle in handles.values():
            handle.close()
        _read_scope.reset(token)


@contextmanager
def window_file(url):
    fs = fsspec.filesystem(
        "http", client_kwargs={"timeout": __import__("aiohttp").ClientTimeout(total=90)}
    )
    handles = _read_scope.get()
    handle = handles.get(url) if handles is not None else None
    if handle is None or handle.closed:
        handle = fs.open(
            url,
            "rb",
            block_size=128 * 1024,
            cache_type="blockcache",
            cache_options={"maxblocks": 512},
        )
        if handles is not None:
            handles[url] = handle
    handle.seek(0)
    try:
        yield handle
    except BaseException:
        handle.close()
        if handles is not None:
            handles.pop(url, None)
        raise
    finally:
        if handles is None:
            handle.close()


class DataGap(RuntimeError):
    def __init__(self, reason, detail=""):
        self.reason = reason
        super().__init__(detail or reason)


def satellite_order(at, lat, lon):
    at = utc(at)
    # NOAA transition notices: MSG_20250407_1510 and MSG_20230104_1805.
    east = 19 if at >= utc("2025-04-07T15:10:00Z") else 16
    west = 18 if at >= utc("2023-01-04T18:00:00Z") else 17
    # Larger cos(viewing angle) gives the smaller geocentric viewing angle.
    choices = [(east, -75.2), (west, -137.0)]
    return [
        sat
        for sat, _ in sorted(
            choices,
            key=lambda pair: math.cos(math.radians(lat))
            * math.cos(math.radians(lon - pair[1])),
            reverse=True,
        )
    ]


def filename_time(text):
    return datetime.strptime(text[:13], "%Y%j%H%M%S").replace(tzinfo=UTC)


@lru_cache(maxsize=4096)
def list_hour(satellite, date_hour, freshness_bucket=0):
    stamp = utc(date_hour)
    base = f"https://noaa-goes{satellite}.s3.amazonaws.com/"
    prefix = f"{PRODUCT}/{stamp:%Y/%j/%H}/"
    result, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        root = ET.fromstring(fetch(base + "?" + urlencode(params)))
        ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
        for item in root.findall("s:Contents", ns):
            key = item.findtext("s:Key", namespaces=ns)
            match = re.search(r"_s(\d+)_e(\d+)_c", key or "")
            if match:
                result.append(
                    {
                        "url": base + key,
                        "satellite": satellite,
                        "start": iso(filename_time(match[1])),
                        "end": iso(filename_time(match[2])),
                        "etag": item.findtext("s:ETag", namespaces=ns),
                        "size_bytes": int(item.findtext("s:Size", namespaces=ns)),
                        "published_at": iso(
                            item.findtext("s:LastModified", namespaces=ns)
                        ),
                    }
                )
        if root.findtext("s:IsTruncated", namespaces=ns) != "true":
            break
        token = root.findtext("s:NextContinuationToken", namespaces=ns)
    return result


def select_scan(satellite, at, available_at=None):
    at = utc(at)
    # Don't retain a negative listing for an hour that is still arriving.
    freshness = int(time.time() // 900) if (utc() - at).total_seconds() < 7200 else 0
    candidates = list_hour(satellite, iso(at), freshness) + list_hour(
        satellite, iso(at - timedelta(hours=1)), freshness
    )
    candidates = [
        c
        for c in candidates
        if 0 <= (at - utc(c["end"])).total_seconds() <= 1800
        and (available_at is None or utc(c["published_at"]) <= utc(available_at))
    ]
    if not candidates:
        raise DataGap(
            "missing_scan", f"No complete GOES-{satellite} scan before {iso(at)}"
        )
    return max(candidates, key=lambda c: c["end"])


def target_grid(lat, lon):
    offset = (np.arange(256, dtype=np.float64) - 127.5) * 0.027
    # Existing training rasters are north-up, with pixel centers inside bounds.
    x, y = np.meshgrid(lon + offset, lat - offset)
    return y, x


def read_window(scan, lat, lon):
    """HTTP byte ranges + lazy variable slicing; no retained satellite files."""
    with window_file(scan["url"]) as handle:
        before = handle.cache.total_requested_bytes
        with xr.open_dataset(
            handle, engine="h5netcdf", mask_and_scale=True, cache=False
        ) as ds:
            missing = [band for band in BANDS if band not in ds]
            if missing:
                raise DataGap("missing_bands", ", ".join(missing))
            projection = ds["goes_imager_projection"].attrs
            height = projection["perspective_point_height"]
            geos = CRS.from_cf(projection)
            forward = Transformer.from_crs("EPSG:4326", geos, always_xy=True)
            inverse = Transformer.from_crs(geos, "EPSG:4326", always_xy=True)
            grid_lat, grid_lon = target_grid(lat, lon)
            tx, ty = forward.transform(grid_lon, grid_lat)
            if not (np.isfinite(tx).all() and np.isfinite(ty).all()):
                raise DataGap("outside_satellite_coverage")
            xs, ys = ds.x.values * height, ds.y.values * height
            ix = np.flatnonzero((xs >= tx.min() - 6000) & (xs <= tx.max() + 6000))
            iy = np.flatnonzero((ys >= ty.min() - 6000) & (ys <= ty.max() + 6000))
            if not len(ix) or not len(iy):
                raise DataGap("outside_satellite_coverage")
            crop = ds.isel(
                x=slice(ix.min(), ix.max() + 1), y=slice(iy.min(), iy.max() + 1)
            )
            # Inspect compressed chunk offsets without reading the full arrays, then
            # overlap independent HTTP range requests. h5py decoding itself remains
            # serial; only immutable byte-cache fills run in the thread pool.
            blocks = set()
            with h5py.File(handle, "r") as raw:
                for name in (*BANDS, *(band.replace("CMI_", "DQF_") for band in BANDS)):
                    if name not in raw:
                        raise DataGap(
                            (
                                "missing_quality_flags"
                                if name.startswith("DQF_")
                                else "missing_bands"
                            ),
                            name,
                        )
                    dataset = raw[name]
                    if dataset.chunks is None:
                        continue
                    cy, cx = dataset.chunks
                    for yy in range(int(iy.min()) // cy * cy, int(iy.max()) + 1, cy):
                        for xx in range(
                            int(ix.min()) // cx * cx, int(ix.max()) + 1, cx
                        ):
                            chunk = dataset.id.get_chunk_info_by_coord((yy, xx))
                            if chunk.byte_offset is not None and chunk.size:
                                blocks.update(
                                    range(
                                        chunk.byte_offset // handle.blocksize,
                                        (chunk.byte_offset + chunk.size - 1)
                                        // handle.blocksize
                                        + 1,
                                    )
                                )
            # Bound prefetch to the cache; larger windows use lazy range reads.
            if len(blocks) <= 450:
                with ThreadPoolExecutor(max_workers=8) as pool:
                    list(pool.map(handle.cache._fetch_block_cached, sorted(blocks)))
            px, py = np.meshgrid(crop.x.values * height, crop.y.values * height)
            source_lon, source_lat = inverse.transform(px, py)
            from .grid import regrid

            fields, masks = [], []
            for band in BANDS:
                if crop[band].attrs.get("units") != "K":
                    raise DataGap("unexpected_channel_units", band)
                values = np.asarray(crop[band].values, dtype=np.float32)
                dqf_name = band.replace("CMI_", "DQF_")
                if dqf_name not in crop:
                    raise DataGap("missing_quality_flags", band)
                quality = crop[dqf_name].values
                values = np.where(quality <= 1, values, np.nan)
                # Unwrap longitudes about the crop center before nearest-neighbor sampling.
                unwrapped = lon + (source_lon - lon + 180) % 360 - 180
                field, mask = regrid(values, source_lat, unwrapped, grid_lat, grid_lon)
                fields.append(field)
                masks.append(mask & np.isfinite(field))
            array = np.stack(fields)
            valid = np.logical_and.reduce(masks)
            fraction = float(valid[32:224, 32:224].mean())
            if fraction < 0.9 or not valid[126:130, 126:130].all():
                raise DataGap("poor_coverage", f"valid fraction {fraction:.3f}")
            transferred = handle.cache.total_requested_bytes - before
    return (
        array,
        valid,
        {
            **scan,
            "valid_fraction": fraction,
            "range_bytes": transferred,
            "retrieved_at": iso(),
            "bands": list(BANDS),
            "units": "K",
            "grid": {
                "height": 256,
                "width": 256,
                "resolution_degrees": 0.027,
                "crop": [32, 224],
            },
        },
    )


def acquire(at, center, available_at=None):
    failures = []
    for satellite in satellite_order(at, center["lat"], center["lon"]):
        try:
            scan = select_scan(satellite, at, available_at)
            for attempt in range(3):
                try:
                    return read_window(scan, center["lat"], center["lon"])
                except (OSError, TimeoutError, ClientError):
                    if attempt == 2:
                        raise
                    time.sleep(2**attempt)
        except DataGap as error:
            failures.append((error.reason, f"GOES-{satellite}: {error}"))
    raise DataGap(
        failures[0][0] if failures else "missing_scan",
        "; ".join(detail for _, detail in failures),
    )
