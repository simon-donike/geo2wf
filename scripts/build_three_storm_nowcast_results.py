#!/usr/bin/env python3
"""Build paper-ready maximum-wind nowcasts for three validation storms."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_intensity_comparison_storm_inference import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    STORM_NAMES,
    STORMS,
    _atomic_csv,
    _atomic_json,
    _cohort_fingerprint,
)


DOCS_START = "<!-- three-storm-nowcast:start -->"
DOCS_END = "<!-- three-storm-nowcast:end -->"
CORE_MODELS = (
    ("unet_raw_max", "Raw field U-Net", "unet_raw_max_ms", "#E69F00"),
    ("unet_correction", "U-Net + correction", "unet_correction_ms", "#0072B2"),
    ("joint_unet_mlp", "Joint U-Net + MLP", "joint_unet_mlp_ms", "#009E73"),
)
OPTIONAL_ABLATIONS = (
    (
        "ablation_max_wind_only",
        "Joint ablation: max wind only",
        "ablation_max_wind_only_ms",
        "#CC79A7",
    ),
    (
        "ablation_max_wind_radii",
        "Joint ablation: max wind + radii",
        "ablation_max_wind_radii_ms",
        "#56B4E9",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-era5-csv",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "with-era5.csv",
    )
    parser.add_argument(
        "--without-era5-csv",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "without-era5.csv",
    )
    parser.add_argument("--storms", nargs="+", default=list(STORMS))
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path(
            "docs/assets/data/final-results/three-storm-nowcast-predictions.csv"
        ),
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("docs/assets/data/final-results/three-storm-nowcast-metrics.csv"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("docs/assets/data/final-results/three-storm-nowcast.json"),
    )
    parser.add_argument(
        "--png-output",
        type=Path,
        default=Path("docs/assets/images/final-results/three-storm-nowcasts.png"),
    )
    parser.add_argument(
        "--pdf-output",
        type=Path,
        default=Path("docs/assets/images/final-results/three-storm-nowcasts.pdf"),
    )
    parser.add_argument("--docs-page", type=Path, default=Path("docs/results.md"))
    parser.add_argument(
        "--no-docs-update",
        action="store_true",
        help="Write data and figures without replacing the results-page section.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_frame(path: Path, storms: Sequence[str]) -> pd.DataFrame:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {
        "observation_id",
        "storm_id",
        "source_split",
        "observation_timestamp",
        "target_ms",
        "inference_valid",
        *(column for _, _, column, _ in CORE_MODELS),
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    selected = frame.loc[frame["storm_id"].astype(str).isin(storms)].copy()
    selected["storm_id"] = selected["storm_id"].astype(str)
    selected["observation_id"] = selected["observation_id"].astype(str)
    selected["observation_timestamp"] = pd.to_datetime(
        selected["observation_timestamp"], utc=True
    )
    if selected["observation_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate observation IDs")
    found_storms = set(selected["storm_id"])
    if found_storms != set(storms):
        raise ValueError(
            f"{path} storms differ: expected {list(storms)}, got {sorted(found_storms)}"
        )
    if set(selected["source_split"].astype(str)) != {"val"}:
        raise ValueError(f"{path} contains observations outside the validation split")
    return selected.sort_values(
        ["storm_id", "observation_timestamp", "observation_id"]
    ).reset_index(drop=True)


def load_frames(
    with_era5_path: Path,
    without_era5_path: Path,
    storms: Sequence[str] = STORMS,
) -> dict[str, pd.DataFrame]:
    normalized_storms = tuple(
        dict.fromkeys(str(storm).strip().upper() for storm in storms)
    )
    if not normalized_storms or not all(normalized_storms):
        raise ValueError("at least one non-empty storm ID is required")
    frames = {
        "with_era5": _read_frame(with_era5_path, normalized_storms),
        "without_era5": _read_frame(without_era5_path, normalized_storms),
    }
    reference = frames["with_era5"]
    candidate = frames["without_era5"]
    for column in (
        "observation_id",
        "storm_id",
        "source_split",
        "observation_timestamp",
    ):
        if not reference[column].equals(candidate[column]):
            raise ValueError(f"ERA5/no-ERA5 cohorts differ in {column}")
    if not np.allclose(
        reference["target_ms"].to_numpy(float),
        candidate["target_ms"].to_numpy(float),
        rtol=1.0e-7,
        atol=1.0e-6,
    ):
        raise ValueError("ERA5/no-ERA5 cohorts use different maximum-wind targets")
    ri_columns = ("ri_24h_change_ms", "is_rapid_intensification")
    reference_has_ri = all(column in reference for column in ri_columns)
    candidate_has_ri = all(column in candidate for column in ri_columns)
    if not reference_has_ri:
        raise ValueError("with-ERA5 inference is missing RI diagnostics")
    if not candidate_has_ri:
        # RI is defined entirely by the shared IBTrACS storm trajectory and
        # observation timestamp.  Schema-1 no-ERA5 outputs can therefore inherit
        # it from a verified, row-identical with-ERA5 inference result.
        for column in ri_columns:
            candidate[column] = reference[column].to_numpy(copy=True)
    else:
        if not np.allclose(
            reference["ri_24h_change_ms"].to_numpy(float),
            candidate["ri_24h_change_ms"].to_numpy(float),
            rtol=1.0e-7,
            atol=1.0e-6,
            equal_nan=True,
        ):
            raise ValueError("ERA5/no-ERA5 cohorts use different RI changes")
        if (
            not reference["is_rapid_intensification"]
            .astype(bool)
            .equals(candidate["is_rapid_intensification"].astype(bool))
        ):
            raise ValueError("ERA5/no-ERA5 cohorts use different RI labels")
    return frames


def _series(frames: Mapping[str, pd.DataFrame]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for model_key, model_label, column, color in CORE_MODELS:
        for regime, regime_label, linestyle in (
            ("with_era5", "with ERA5", "-"),
            ("without_era5", "without ERA5", "--"),
        ):
            result.append(
                {
                    "key": f"{model_key}_{regime}",
                    "model_key": model_key,
                    "model": model_label,
                    "regime": regime,
                    "conditioning": regime_label,
                    "column": column,
                    "color": color,
                    "linestyle": linestyle,
                    "label": f"{model_label} · {regime_label}",
                }
            )
    for model_key, model_label, column, color in OPTIONAL_ABLATIONS:
        if column in frames["with_era5"].columns:
            result.append(
                {
                    "key": model_key,
                    "model_key": model_key,
                    "model": model_label,
                    "regime": "with_era5",
                    "conditioning": "with ERA5",
                    "column": column,
                    "color": color,
                    "linestyle": ":",
                    "label": model_label,
                }
            )
    return result


def prediction_rows(
    frames: Mapping[str, pd.DataFrame], series: Sequence[Mapping[str, str]]
) -> pd.DataFrame:
    rows = []
    for item in series:
        frame = frames[item["regime"]]
        for row in frame.itertuples(index=False):
            prediction = float(getattr(row, item["column"]))
            target = float(row.target_ms)
            if not math_isfinite(prediction) or not math_isfinite(target):
                continue
            rows.append(
                {
                    "observation_id": str(row.observation_id),
                    "storm_id": str(row.storm_id),
                    "observation_timestamp": row.observation_timestamp.isoformat(),
                    "model_key": item["model_key"],
                    "model": item["model"],
                    "conditioning": item["conditioning"],
                    "prediction_ms": prediction,
                    "target_ms": target,
                    "error_ms": prediction - target,
                    "is_rapid_intensification": bool(row.is_rapid_intensification),
                    "ri_24h_change_ms": float(row.ri_24h_change_ms),
                }
            )
    return pd.DataFrame(rows)


def math_isfinite(value: float) -> bool:
    return bool(np.isfinite(value))


def metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["model_key", "model", "conditioning"]
    for keys, model_frame in predictions.groupby(group_columns, sort=False):
        scopes = [("all_three_storms", model_frame)]
        scopes.extend(
            (str(storm_id), storm)
            for storm_id, storm in model_frame.groupby("storm_id", sort=False)
        )
        ri_frame = model_frame.loc[model_frame["is_rapid_intensification"]]
        if not ri_frame.empty:
            scopes.append(("rapid_intensification_all_storms", ri_frame))
            scopes.extend(
                (f"{storm_id}_rapid_intensification", storm)
                for storm_id, storm in ri_frame.groupby("storm_id", sort=False)
            )
        for scope, frame in scopes:
            error = frame["error_ms"].to_numpy(float)
            rows.append(
                {
                    "model_key": keys[0],
                    "model": keys[1],
                    "conditioning": keys[2],
                    "scope": scope,
                    "samples": len(frame),
                    "mae_ms": float(np.mean(np.abs(error))),
                    "rmse_ms": float(np.sqrt(np.mean(np.square(error)))),
                    "bias_ms": float(np.mean(error)),
                }
            )
    return pd.DataFrame(rows)


def _hourly(frame: pd.DataFrame, storm_id: str, columns: Sequence[str]) -> pd.DataFrame:
    storm = frame.loc[frame["storm_id"] == storm_id]
    return (
        storm.set_index("observation_timestamp")[list(columns)]
        .resample("1h")
        .mean()
        .dropna(how="all")
    )


def write_figure(
    frames: Mapping[str, pd.DataFrame],
    series: Sequence[Mapping[str, str]],
    storms: Sequence[str],
    png_path: Path,
    pdf_path: Path,
) -> None:
    figure, axes = plt.subplots(
        len(storms),
        1,
        figsize=(13.0, max(3.15 * len(storms), 5.0)),
        squeeze=False,
        sharey=True,
    )
    reference = frames["with_era5"]
    for axis, storm_id in zip(axes[:, 0], storms):
        truth = _hourly(reference, storm_id, ["target_ms"])
        axis.plot(
            truth.index,
            truth["target_ms"],
            color="#111111",
            linewidth=2.6,
            label="IBTrACS USA_WIND",
            zorder=10,
        )
        for item in series:
            frame = frames[item["regime"]]
            hourly = _hourly(frame, storm_id, [item["column"]])
            axis.plot(
                hourly.index,
                hourly[item["column"]],
                color=item["color"],
                linestyle=item["linestyle"],
                linewidth=1.45,
                alpha=0.92,
                label=item["label"],
            )
        native_samples = int((reference["storm_id"] == storm_id).sum())
        storm_name = STORM_NAMES.get(storm_id, storm_id)
        axis.set_title(f"{storm_name} ({storm_id}) · {native_samples} GEO nowcasts")
        axis.set_ylabel("Maximum wind (m/s)")
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.22)
        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axes[-1, 0].set_xlabel("Observation time (UTC)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    figure.suptitle(
        "Maximum-wind nowcasts for three validation storms\n"
        "Hourly means shown; metrics use every native GEO observation",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.105, 1.0, 0.94))
    for path, options in (
        (png_path, {"dpi": 300}),
        (pdf_path, {}),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, bbox_inches="tight", facecolor="white", **options)
    plt.close(figure)


def _relative_link(target: Path, page: Path) -> str:
    return Path(os.path.relpath(target.resolve(), page.parent.resolve())).as_posix()


def _docs_section(
    metrics: pd.DataFrame,
    docs_page: Path,
    png_path: Path,
    predictions_path: Path,
    metrics_path: Path,
    metadata_path: Path,
) -> str:
    overall = metrics.loc[metrics["scope"] == "all_three_storms"]
    ri = metrics.loc[metrics["scope"] == "rapid_intensification_all_storms"].set_index(
        ["model_key", "conditioning"]
    )
    table = [
        "| Model | Conditioning | All n | All MAE | All RMSE | All bias | RI n | RI MAE | RI RMSE | RI bias |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall.itertuples(index=False):
        ri_row = ri.loc[(row.model_key, row.conditioning)]
        table.append(
            f"| {row.model} | {row.conditioning} | {row.samples} | "
            f"{row.mae_ms:.3f} | {row.rmse_ms:.3f} | {row.bias_ms:.3f} | "
            f"{ri_row.samples} | {ri_row.mae_ms:.3f} | {ri_row.rmse_ms:.3f} | "
            f"{ri_row.bias_ms:.3f} |"
        )
    return "\n".join(
        [
            DOCS_START,
            "## Three-storm maximum-wind nowcasts",
            "",
            "Humberto 2025, Kiko 2025, and Otis 2023 are validation-storm case",
            "studies, not held-out test estimates. Each prediction is an independent",
            "single-observation nowcast. The figure uses hourly means for readability;",
            "the metrics use every native GEO observation. RI columns apply the same",
            "metrics only where IBTrACS increased by at least 30 kt in the preceding",
            "24 hours.",
            "",
            *table,
            "",
            f"![All-model maximum-wind nowcasts]({_relative_link(png_path, docs_page)})",
            "",
            f"[Download predictions]({_relative_link(predictions_path, docs_page)})"
            "{ .md-button }",
            f"[Download metrics]({_relative_link(metrics_path, docs_page)})"
            "{ .md-button }",
            f"[Download provenance]({_relative_link(metadata_path, docs_page)})"
            "{ .md-button }",
            DOCS_END,
        ]
    )


def update_docs_page(path: Path, section: str) -> None:
    content = path.read_text(encoding="utf-8")
    if content.count(DOCS_START) != 1 or content.count(DOCS_END) != 1:
        raise ValueError(f"{path} must contain exactly one nowcast result marker pair")
    start = content.index(DOCS_START)
    end = content.index(DOCS_END, start) + len(DOCS_END)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content[:start] + section + content[end:], encoding="utf-8")
    os.replace(temporary, path)


def build(args: argparse.Namespace) -> dict[str, Any]:
    storms = tuple(dict.fromkeys(str(storm).strip().upper() for storm in args.storms))
    frames = load_frames(args.with_era5_csv, args.without_era5_csv, storms)
    series = _series(frames)
    predictions = prediction_rows(frames, series)
    metrics = metric_rows(predictions)
    _atomic_csv(predictions, args.predictions_output)
    _atomic_csv(metrics, args.metrics_output)
    write_figure(frames, series, storms, args.png_output, args.pdf_output)
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": "dense_validation_storm_case_study",
        "storms": list(storms),
        "cohort": {
            "samples": len(frames["with_era5"]),
            "sha256": _cohort_fingerprint(frames["with_era5"]),
        },
        "inputs": {
            "with_era5": {
                "path": str(args.with_era5_csv.resolve()),
                "sha256": _sha256(args.with_era5_csv),
            },
            "without_era5": {
                "path": str(args.without_era5_csv.resolve()),
                "sha256": _sha256(args.without_era5_csv),
            },
        },
        "series": list(series),
        "metrics": metrics.to_dict(orient="records"),
    }
    _atomic_json(payload, args.metadata_output)
    if not args.no_docs_update:
        section = _docs_section(
            metrics,
            args.docs_page,
            args.png_output,
            args.predictions_output,
            args.metrics_output,
            args.metadata_output,
        )
        update_docs_page(args.docs_page, section)
    return payload


def main() -> None:
    payload = build(parse_args())
    print(
        "Wrote three-storm nowcast outputs for "
        f"{len(payload['series'])} model/conditioning series"
    )


if __name__ == "__main__":
    main()
