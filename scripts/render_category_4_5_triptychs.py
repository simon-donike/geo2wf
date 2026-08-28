#!/usr/bin/env python3
"""Render GEO--SAR--prediction triptychs for Category 4--5 validation times."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Colormap  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    ROOT / "logs/latent-matrix/sar/era5/max-wind-radii/20260827-190834_modular"
)
DEFAULT_CATEGORY_MANIFEST = ROOT / "data/unet_intensity_structure_v3/val/manifest.csv"
DEFAULT_OUTPUT_DIR = ROOT / "docs/assets/images/final-results/category-4-5-triptychs"
DEFAULT_VALIDATION_L1_MS = 2.4017400469868218
WIND_DISPLAY_RANGE_MS = (0.0, 80.0)
REQUIRED_CATEGORY_COLUMNS = frozenset(
    {
        "sample_id",
        "storm_id",
        "observation_timestamp",
        "target_wind_ms",
        "target_category",
        "center_lat",
        "center_lon",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_RUN_DIR / "resolved-config.yaml",
        help="Resolved config for the selected wind-field model.",
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=DEFAULT_RUN_DIR / "result.json",
        help="Completed-run metadata containing best_model_path.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional checkpoint override; otherwise use result.json.",
    )
    parser.add_argument(
        "--category-manifest",
        type=Path,
        default=DEFAULT_CATEGORY_MANIFEST,
        help="Validation manifest containing IBTrACS target_category labels.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="val",
        help="Configured dataset split from which to run inference.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--geo-channel",
        default="C13",
        help="Single geostationary channel to render (default: C13).",
    )
    parser.add_argument(
        "--categories",
        type=int,
        nargs="+",
        default=[4, 5],
        help="IBTrACS Saffir--Simpson categories to retain.",
    )
    parser.add_argument(
        "--include-land",
        action="store_true",
        help="Retain land-centered occurrences (excluded by default).",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=25,
        help="Fail unless the category filter yields this many images.",
    )
    parser.add_argument(
        "--selection",
        choices=("all", "strongest-per-storm"),
        default="all",
        help="Optional deterministic selection within the eligible cohort.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum images after category, ocean, and selection filtering.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def select_category_rows(
    frame: pd.DataFrame,
    categories: tuple[int, ...] | list[int],
    *,
    ocean_only: bool = False,
    expected_count: int | None = None,
) -> pd.DataFrame:
    """Return a stable, validated category subset from an intensity manifest."""

    missing = REQUIRED_CATEGORY_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"category manifest is missing columns: {sorted(missing)}")
    selected_categories = tuple(sorted(set(int(value) for value in categories)))
    if not selected_categories:
        raise ValueError("at least one category is required")

    selected = frame.loc[
        frame["target_category"].isin(selected_categories),
        sorted(REQUIRED_CATEGORY_COLUMNS),
    ].copy()
    selected["observation_timestamp"] = pd.to_datetime(
        selected["observation_timestamp"], errors="raise", utc=True
    )
    if ocean_only:
        from global_land_mask import globe

        latitude = pd.to_numeric(selected["center_lat"], errors="coerce").to_numpy()
        longitude = pd.to_numeric(selected["center_lon"], errors="coerce").to_numpy()
        if (
            not np.isfinite(latitude).all()
            or not np.isfinite(longitude).all()
            or (np.abs(latitude) > 90.0).any()
            or (np.abs(longitude) > 180.0).any()
        ):
            raise ValueError(
                "category selection contains invalid storm-center coordinates"
            )
        selected = selected.loc[~globe.is_land(latitude, longitude)].copy()
    selected = selected.sort_values(
        ["storm_id", "observation_timestamp", "sample_id"], kind="stable"
    ).reset_index(drop=True)
    if selected["sample_id"].astype(str).duplicated().any():
        raise ValueError("category selection contains duplicate sample_id values")
    if expected_count is not None and len(selected) != int(expected_count):
        raise ValueError(
            f"category filter {list(selected_categories)} yielded {len(selected)} "
            f"rows, expected {expected_count}"
        )
    return selected


def choose_render_rows(
    frame: pd.DataFrame,
    *,
    strategy: str = "all",
    limit: int | None = None,
    expected_count: int | None = None,
) -> pd.DataFrame:
    """Choose a deterministic subset from already eligible category rows."""

    if strategy == "all":
        selected = frame.copy()
    elif strategy == "strongest-per-storm":
        selected = (
            frame.sort_values(
                [
                    "storm_id",
                    "target_category",
                    "target_wind_ms",
                    "observation_timestamp",
                    "sample_id",
                ],
                ascending=[True, False, False, True, True],
                kind="stable",
            )
            .drop_duplicates("storm_id", keep="first")
            .copy()
        )
    else:
        raise ValueError(f"unsupported selection strategy: {strategy!r}")
    selected = selected.sort_values(
        ["storm_id", "observation_timestamp", "sample_id"], kind="stable"
    )
    if limit is not None:
        if int(limit) < 1:
            raise ValueError("limit must be positive")
        selected = selected.head(int(limit)).copy()
    selected = selected.reset_index(drop=True)
    if expected_count is not None and len(selected) != int(expected_count):
        raise ValueError(
            f"render selection yielded {len(selected)} rows, expected {expected_count}"
        )
    return selected


def geostationary_channel_index(names: list[str], requested: str) -> int:
    """Resolve CMI/AHI/ABI and Bxx/Cxx spellings to one channel index."""

    def normalized(value: str) -> str:
        value = str(value).strip().upper()
        for prefix in ("CMI_", "ABI_", "AHI_"):
            value = value.removeprefix(prefix)
        if re.fullmatch(r"B\d{2}", value):
            value = f"C{value[1:]}"
        return value

    wanted = normalized(requested)
    for index, name in enumerate(names):
        if normalized(name) == wanted:
            return index
    raise ValueError(
        f"geostationary channel {requested!r} is unavailable; channels are {names}"
    )


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def _bounds_extent(bounds: Any) -> tuple[float, float, float, float]:
    values = _as_numpy(bounds).reshape(-1)
    if values.size != 4 or not np.isfinite(values).all():
        raise ValueError(f"expected four finite bounds, got {values}")
    left, right, bottom, top = (float(value) for value in values)
    return left, right, bottom, top


def _masked_colormap(name: str, bad_color: str) -> Colormap:
    colormap = matplotlib.colormaps[name].copy()
    colormap.set_bad(bad_color)
    return colormap


def _display_limits(values: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    finite = values[valid & np.isfinite(values)]
    if not finite.size:
        return 0.0, 1.0
    low, high = np.nanpercentile(finite, [2.0, 98.0])
    if not math.isfinite(float(low)) or not math.isfinite(float(high)):
        return 0.0, 1.0
    if math.isclose(float(low), float(high)):
        high = float(low) + 1.0
    return float(low), float(high)


def _plot_center(axis: Any, center: Any) -> None:
    values = _as_numpy(center).reshape(-1)
    if values.size == 2 and np.isfinite(values).all():
        latitude, longitude = (float(value) for value in values)
        axis.plot(
            longitude,
            latitude,
            marker="+",
            color="#D62728",
            markersize=8.0,
            markeredgewidth=1.3,
            linestyle="none",
            zorder=5,
        )


def _format_storm_id(storm_id: str) -> str:
    match = re.fullmatch(r"([A-Z]{2})(\d{2})(\d{4})", str(storm_id).upper())
    return f"{match.group(1)}{match.group(2)} {match.group(3)}" if match else storm_id


def render_triptych(
    *,
    condition: Any,
    condition_mask: Any,
    condition_channels: list[str],
    target: Any,
    target_mask: Any,
    prediction: Any,
    bounds: Any,
    center: Any,
    category_row: Mapping[str, Any],
    geo_channel: str,
):
    """Build one horizontal GEO, SAR, and selected-model prediction figure."""

    condition_array = _as_numpy(condition)
    target_array = _as_numpy(target).squeeze()
    prediction_array = _as_numpy(prediction).squeeze()
    condition_valid = _as_numpy(condition_mask).astype(bool).squeeze()
    target_valid = _as_numpy(target_mask).astype(bool).squeeze()
    if condition_array.ndim != 3:
        raise ValueError(f"condition must be CHW, got {condition_array.shape}")
    if target_array.ndim != 2 or prediction_array.ndim != 2:
        raise ValueError("target and prediction must each resolve to one 2D field")
    if target_array.shape != prediction_array.shape:
        raise ValueError("target and prediction shapes differ")

    channel_index = geostationary_channel_index(condition_channels, geo_channel)
    geo = condition_array[channel_index]
    geo_low, geo_high = _display_limits(geo, condition_valid)
    extent = _bounds_extent(bounds)
    timestamp = pd.Timestamp(category_row["observation_timestamp"])
    category = int(category_row["target_category"])
    target_wind = float(category_row["target_wind_ms"])
    storm_id = str(category_row["storm_id"])

    with plt.rc_context(
        {
            "font.size": 9.0,
            "axes.titlesize": 10.2,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        figure, axes = plt.subplots(
            1,
            3,
            figsize=(12.4, 4.25),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        geo_image = np.ma.masked_where(~condition_valid, geo)
        sar_image = np.ma.masked_where(
            ~target_valid | ~np.isfinite(target_array), target_array
        )
        prediction_image = np.ma.masked_invalid(prediction_array)

        axes[0].imshow(
            geo_image,
            extent=extent,
            origin="upper",
            cmap=_masked_colormap("gray_r", "#777777"),
            vmin=geo_low,
            vmax=geo_high,
            interpolation="nearest",
        )
        wind_colormap = _masked_colormap("turbo", "#8A8A8A")
        wind_artist = axes[1].imshow(
            sar_image,
            extent=extent,
            origin="upper",
            cmap=wind_colormap,
            vmin=WIND_DISPLAY_RANGE_MS[0],
            vmax=WIND_DISPLAY_RANGE_MS[1],
            interpolation="nearest",
        )
        axes[2].imshow(
            prediction_image,
            extent=extent,
            origin="upper",
            cmap=wind_colormap,
            vmin=WIND_DISPLAY_RANGE_MS[0],
            vmax=WIND_DISPLAY_RANGE_MS[1],
            interpolation="nearest",
        )

        sensor = str(category_row.get("geo_sensor", "")).strip()
        sensor_prefix = f"{sensor} " if sensor else ""
        axes[0].set_title(f"(a) Geostationary {sensor_prefix}{geo_channel.upper()}")
        axes[1].set_title("(b) SAR observed wind field")
        axes[2].set_title("(c) Best-model prediction")
        for index, axis in enumerate(axes):
            _plot_center(axis, center)
            axis.set_xlabel("Longitude (°)")
            if index == 0:
                axis.set_ylabel("Latitude (°)")
            axis.set_aspect("equal", adjustable="box")
            axis.tick_params(length=2.5, pad=1.5)

        figure.colorbar(
            wind_artist,
            ax=axes[1:],
            location="right",
            shrink=0.82,
            pad=0.025,
            label=r"10 m wind speed (m s$^{-1}$)",
        )
        figure.suptitle(
            f"{_format_storm_id(storm_id)} · "
            f"{timestamp.strftime('%Y-%m-%d %H:%M UTC')} · "
            f"IBTrACS Category {category} ({target_wind:.1f} m s$^{{-1}}$)",
            fontsize=11.2,
        )
    return figure


def _sample_mae(target: Any, target_mask: Any, prediction: Any) -> float:
    target_array = _as_numpy(target).squeeze()
    prediction_array = _as_numpy(prediction).squeeze()
    valid = (
        _as_numpy(target_mask).astype(bool).squeeze()
        & np.isfinite(target_array)
        & np.isfinite(prediction_array)
    )
    return float(np.abs(target_array[valid] - prediction_array[valid]).mean())


def _output_filename(sequence: int, row: Mapping[str, Any]) -> str:
    timestamp = pd.Timestamp(row["observation_timestamp"]).strftime("%Y%m%dT%H%M%SZ")
    return (
        f"{sequence:02d}-{str(row['storm_id']).lower()}-{timestamp.lower()}-"
        f"cat{int(row['target_category'])}.png"
    )


def _resolve_checkpoint(checkpoint: Path | None, result_path: Path) -> Path:
    if checkpoint is not None:
        resolved = checkpoint.expanduser().resolve()
    else:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed":
            raise ValueError(f"model run is not completed: {result_path}")
        resolved = Path(str(result.get("best_model_path", ""))).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def main() -> None:
    args = parse_args()
    if args.expected_count < 1 or args.batch_size < 1 or args.dpi < 1:
        raise ValueError("expected-count, batch-size, and dpi must be positive")
    for path in (args.config, args.result, args.category_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)

    # Keep workspace imports independent of editable-install state.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from geo2wf.config import (  # noqa: PLC0415
        instantiate_datamodule,
        load_config_file,
    )
    from geo2wf.data.collation import collate_wind_field_samples  # noqa: PLC0415
    from geo2wf.inference import CheckpointLoader  # noqa: PLC0415

    categories = tuple(sorted(set(args.categories)))
    all_category_rows = select_category_rows(
        pd.read_csv(args.category_manifest),
        categories,
    )
    ocean_category_rows = select_category_rows(
        all_category_rows,
        categories,
        ocean_only=not args.include_land,
    )
    category_rows = choose_render_rows(
        ocean_category_rows,
        strategy=args.selection,
        limit=args.limit,
        expected_count=args.expected_count,
    )
    config = load_config_file(args.config)
    checkpoint = _resolve_checkpoint(args.checkpoint, args.result)
    datamodule = instantiate_datamodule(config)
    split_name = str(getattr(datamodule, f"{args.split}_split"))
    dataset = datamodule._make_dataset(split_name)  # noqa: SLF001
    model = CheckpointLoader.load(config, checkpoint, strict=True)
    model.validate_data_spec(dataset.data_spec)

    dataset_index = {
        str(sample_id): index
        for index, sample_id in enumerate(dataset.samples["sample_id"].astype(str))
    }
    requested_ids = category_rows["sample_id"].astype(str).tolist()
    missing_ids = sorted(set(requested_ids).difference(dataset_index))
    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} category samples are absent from the model cohort: "
            f"{missing_ids[:5]}"
        )
    indices = [dataset_index[sample_id] for sample_id in requested_ids]
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_wind_field_samples,
    )

    device = torch.device(args.device)
    model = model.eval().to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    row_by_id = {str(row["sample_id"]): row for _, row in category_rows.iterrows()}
    image_records: list[dict[str, Any]] = []
    sequence_by_id = {
        sample_id: index for index, sample_id in enumerate(requested_ids, start=1)
    }
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = move_to_device(cpu_batch, device)
            prediction = model.predict_physical(batch)
            for batch_index, sample_id in enumerate(cpu_batch["sample_id"]):
                sample_id = str(sample_id)
                row = row_by_id[sample_id]
                meta = cpu_batch["meta"][batch_index]
                row = row.copy()
                row["geo_sensor"] = str(meta.get("condition_sensor", ""))
                output = args.output_dir / _output_filename(
                    sequence_by_id[sample_id], row
                )
                figure = render_triptych(
                    condition=cpu_batch["condition"][batch_index],
                    condition_mask=cpu_batch["condition_mask"][batch_index],
                    condition_channels=[
                        str(value) for value in meta["condition_channels"]
                    ],
                    target=cpu_batch["target_physical"][batch_index],
                    target_mask=cpu_batch["target_mask"][batch_index],
                    prediction=prediction[batch_index],
                    bounds=cpu_batch["target_bounds"][batch_index],
                    center=cpu_batch["center"][batch_index],
                    category_row=row,
                    geo_channel=args.geo_channel,
                )
                figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
                plt.close(figure)
                image_records.append(
                    {
                        "sequence": sequence_by_id[sample_id],
                        "file": output.name,
                        "sample_id": sample_id,
                        "storm_id": str(row["storm_id"]),
                        "observation_timestamp": pd.Timestamp(
                            row["observation_timestamp"]
                        ).isoformat(),
                        "ibtracs_target_wind_ms": float(row["target_wind_ms"]),
                        "ibtracs_category": int(row["target_category"]),
                        "valid_sar_pixel_mae_ms": _sample_mae(
                            cpu_batch["target_physical"][batch_index],
                            cpu_batch["target_mask"][batch_index],
                            prediction[batch_index],
                        ),
                    }
                )

    image_records.sort(key=lambda record: int(record["sequence"]))
    if len(image_records) != args.expected_count:
        raise RuntimeError(
            f"rendered {len(image_records)} images, expected {args.expected_count}"
        )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "image_count": len(image_records),
        "filter": {
            "source": str(args.category_manifest.resolve()),
            "split": args.split,
            "target": "interpolated IBTrACS USA_WIND",
            "categories": list(categories),
            "category_counts": {
                str(category): int((category_rows["target_category"] == category).sum())
                for category in categories
            },
            "storm_count": int(category_rows["storm_id"].nunique()),
            "ocean_only": not args.include_land,
            "land_criterion": (
                "interpolated IBTrACS storm center classified by global_land_mask"
                if not args.include_land
                else None
            ),
            "land_occurrences_excluded": len(all_category_rows)
            - len(ocean_category_rows),
            "eligible_occurrences": len(ocean_category_rows),
            "selection": args.selection,
            "selection_limit": args.limit,
        },
        "panels": [
            f"geostationary {args.geo_channel.upper()}",
            "SAR observed 10 m wind field",
            "selected-model 10 m wind-field prediction",
        ],
        "wind_display_range_ms": list(WIND_DISPLAY_RANGE_MS),
        "model": {
            "name": "Latent MLP · SAR · ERA5 · wind + radii",
            "selection_basis": "lowest current all-validation wind-field L1",
            "validation_l1_ms": DEFAULT_VALIDATION_L1_MS,
            "config": str(args.config.resolve()),
            "checkpoint": str(checkpoint),
        },
        "images": image_records,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(image_records)} triptychs and {manifest_path}")


if __name__ == "__main__":
    main()
