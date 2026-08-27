#!/usr/bin/env python3
"""Evaluate the paired joint U-Net/latent-MLP structure runs for the docs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geo2wf.config import (  # noqa: E402
    instantiate_datamodule,
    instantiate_model,
    load_config_file,
)
from geo2wf.data.joint_intensity import (  # noqa: E402
    IBTRACS_STRUCTURE_TARGET_NAMES,
)


COMMON_RADIUS_NAMES = ("rmw", "r34", "r50", "r64")
STRUCTURE_INDEX = {
    name.replace("_equivalent", ""): index
    for index, name in enumerate(IBTRACS_STRUCTURE_TARGET_NAMES)
    if name != "eye_size"
}
STRUCTURE_METRIC_NAME = {
    name.replace("_equivalent", ""): name
    for name in IBTRACS_STRUCTURE_TARGET_NAMES
    if name != "eye_size"
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-wind-run", type=Path, required=True)
    parser.add_argument("--radii-run", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/assets/data/latent-structure"),
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("docs/assets/images/latent-structure"),
    )
    parser.add_argument(
        "--docs-page",
        type=Path,
        default=Path("docs/experiments/latent-structure-results.md"),
    )
    parser.add_argument(
        "--accelerator",
        choices=("auto", "cpu", "gpu"),
        default="gpu" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--devices", type=int, default=1)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _completed_run(run_dir: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    run_dir = run_dir.resolve()
    result_path = run_dir / "result.json"
    config_path = run_dir / "resolved-config.yaml"
    if not result_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"run is missing result/config artifacts: {run_dir}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "completed":
        raise ValueError(f"run has not completed successfully: {run_dir}")
    checkpoint_value = result.get("best_model_path")
    if not checkpoint_value:
        raise ValueError(f"run has no best validation checkpoint: {run_dir}")
    checkpoint = Path(checkpoint_value).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"best checkpoint does not exist: {checkpoint}")
    return load_config_file(config_path), checkpoint, result


def _float_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for name, value in metrics.items():
        if torch.is_tensor(value):
            value = value.detach().cpu().item()
        values[str(name)] = float(value)
    return values


def _evaluate_run(
    config: dict[str, Any],
    checkpoint: Path,
    *,
    accelerator: str,
    devices: int,
) -> tuple[dict[str, float], int, dict[str, int]]:
    datamodule = instantiate_datamodule(config)
    datamodule.setup("test")
    model = instantiate_model(config)
    model.validate_data_spec(datamodule.data_spec)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=False,
        enable_checkpointing=False,
    )
    result = trainer.test(model=model, datamodule=datamodule, verbose=False)
    if len(result) != 1:
        raise RuntimeError("expected exactly one test dataloader result")
    structure_samples: dict[str, int] = {}
    statistics = getattr(model, "_test_structure_statistics", None)
    if torch.is_tensor(statistics):
        for name, index in STRUCTURE_INDEX.items():
            structure_samples[name] = int(statistics[index, 0].detach().cpu())
    return _float_metrics(result[0]), len(datamodule.test_dataset), structure_samples


def _metric(metrics: dict[str, float], name: str) -> float:
    if name not in metrics:
        raise KeyError(f"evaluation did not emit required metric {name!r}")
    return metrics[name]


def _summary_rows(
    max_metrics: dict[str, float],
    radii_metrics: dict[str, float],
    max_samples: int,
    radii_samples: int,
    structure_samples: dict[str, int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant, metrics, samples in (
        ("max_wind_only", max_metrics, max_samples),
        ("max_wind_plus_radii", radii_metrics, radii_samples),
    ):
        for metric in ("mae", "rmse", "bias"):
            rows.append(
                {
                    "variant": variant,
                    "source": "latent_mlp",
                    "target": "maximum_wind",
                    "metric": f"{metric}_ms",
                    "value": _metric(metrics, f"test/intensity_{metric}_ms"),
                    "samples": samples,
                }
            )
        for metric in ("mae", "rmse", "bias"):
            rows.append(
                {
                    "variant": variant,
                    "source": "2d_unet_field",
                    "target": "wind_field",
                    "metric": f"{metric}_ms",
                    "value": _metric(metrics, f"test/image_{metric}_ms"),
                    "samples": samples,
                }
            )

    for radius in COMMON_RADIUS_NAMES:
        for metric in ("mae", "rmse", "bias"):
            rows.append(
                {
                    "variant": "max_wind_plus_radii",
                    "source": "latent_mlp",
                    "target": radius,
                    "metric": f"{metric}_km",
                    "value": _metric(
                        radii_metrics,
                        f"test/structure_{STRUCTURE_METRIC_NAME[radius]}_{metric}_km",
                    ),
                    "samples": structure_samples[radius],
                }
            )
        for metric in ("mae", "bias"):
            rows.append(
                {
                    "variant": "max_wind_plus_radii",
                    "source": "2d_unet_field",
                    "target": radius,
                    "metric": f"{metric}_km",
                    "value": _metric(
                        radii_metrics, f"test/ibtracs_{radius}_{metric}_km"
                    ),
                    "samples": int(
                        _metric(radii_metrics, f"test/ibtracs_{radius}_samples")
                    ),
                }
            )
    return rows


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("variant", "source", "target", "metric", "value", "samples"),
        )
        writer.writeheader()
        writer.writerows(rows)


def _row_value(
    rows: list[dict[str, object]], variant: str, source: str, target: str, metric: str
) -> tuple[float, int]:
    matches = [
        row
        for row in rows
        if row["variant"] == variant
        and row["source"] == source
        and row["target"] == target
        and row["metric"] == metric
    ]
    if len(matches) != 1:
        raise KeyError((variant, source, target, metric))
    return float(matches[0]["value"]), int(matches[0]["samples"])


def _write_figures(image_dir: Path, rows: list[dict[str, object]]) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    variants = ("max_wind_only", "max_wind_plus_radii")
    labels = ("Max wind only", "Max wind + radii")
    intensity_mae = [
        _row_value(rows, variant, "latent_mlp", "maximum_wind", "mae_ms")[0]
        for variant in variants
    ]
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    bars = axis.bar(labels, intensity_mae, color=("#4878CF", "#D65F5F"))
    axis.bar_label(bars, fmt="%.2f")
    axis.set_ylabel("Held-out MAE (m/s)")
    axis.set_title("Latent MLP maximum-wind error")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(image_dir / "maximum-wind-mae.png", dpi=180)
    plt.close(figure)

    positions = torch.arange(len(COMMON_RADIUS_NAMES), dtype=torch.float32).numpy()
    mlp_values = [
        _row_value(rows, "max_wind_plus_radii", "latent_mlp", radius, "mae_km")[0]
        for radius in COMMON_RADIUS_NAMES
    ]
    field_values = [
        _row_value(rows, "max_wind_plus_radii", "2d_unet_field", radius, "mae_km")[0]
        for radius in COMMON_RADIUS_NAMES
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    width = 0.38
    axis.bar(positions - width / 2, mlp_values, width, label="Latent MLP")
    axis.bar(positions + width / 2, field_values, width, label="2D U-Net field")
    axis.set_xticks(positions, [name.upper() for name in COMMON_RADIUS_NAMES])
    axis.set_ylabel("Held-out MAE (km)")
    axis.set_title("Radius error by extraction source")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(image_dir / "radius-mae-by-source.png", dpi=180)
    plt.close(figure)


def _write_docs_page(
    path: Path,
    rows: list[dict[str, object]],
    *,
    created_utc: str,
    max_run: Path,
    radii_run: Path,
    max_checkpoint: dict[str, str],
    radii_checkpoint: dict[str, str],
) -> None:
    intensity_lines = []
    for variant, label in (
        ("max_wind_only", "Maximum wind only"),
        ("max_wind_plus_radii", "Maximum wind + radii"),
    ):
        mae, samples = _row_value(rows, variant, "latent_mlp", "maximum_wind", "mae_ms")
        rmse, _ = _row_value(rows, variant, "latent_mlp", "maximum_wind", "rmse_ms")
        bias, _ = _row_value(rows, variant, "latent_mlp", "maximum_wind", "bias_ms")
        intensity_lines.append(
            f"| {label} | {mae:.3f} | {rmse:.3f} | {bias:.3f} | {samples} |"
        )

    radius_lines = []
    for radius in COMMON_RADIUS_NAMES:
        mlp_mae, mlp_samples = _row_value(
            rows, "max_wind_plus_radii", "latent_mlp", radius, "mae_km"
        )
        field_mae, field_samples = _row_value(
            rows, "max_wind_plus_radii", "2d_unet_field", radius, "mae_km"
        )
        radius_lines.append(
            f"| {radius.upper()} | {mlp_mae:.2f} | {mlp_samples} | "
            f"{field_mae:.2f} | {field_samples} |"
        )

    content = f"""# Joint U-Net/latent-MLP structure results

Generated `{created_utc}` from the best validation checkpoint of each seed-42
run. The table below is the first evaluation of the held-out test split; no
test metric was used for checkpoint selection.

## Maximum wind

| Training objective | MLP MAE (m/s) | MLP RMSE (m/s) | MLP bias (m/s) | Samples |
|---|---:|---:|---:|---:|
{chr(10).join(intensity_lines)}

![Maximum-wind MAE](../assets/images/latent-structure/maximum-wind-mae.png)

## Radii from two sources

Both columns evaluate the radii-supervised run. “Latent MLP” is the direct
multi-task head; “2D field” diagnoses the same quantity from the reconstructed
wind image. Missing targets are masked. Field-derived counts can be smaller
because a target outside the image's complete circular domain is excluded.

| Radius | Latent MLP MAE (km) | MLP n | 2D field MAE (km) | Field n |
|---|---:|---:|---:|---:|
{chr(10).join(radius_lines)}

![Radius MAE by source](../assets/images/latent-structure/radius-mae-by-source.png)

## Provenance

- Maximum-wind run: `{max_run.resolve()}`
- Maximum-wind checkpoint SHA-256: `{max_checkpoint['sha256']}`
- Radii-supervised run: `{radii_run.resolve()}`
- Radii-supervised checkpoint SHA-256: `{radii_checkpoint['sha256']}`
- Machine-readable metrics: [summary.csv](../assets/data/latent-structure/summary.csv)
- Full result metadata: [results.json](../assets/data/latent-structure/results.json)

The two arms use the same ERA5-conditioned cohort (568 train, 159 validation,
139 test), seed, U-Net/MLP architecture, and optimization settings. They differ
only in the enabled structure head and its `0.25` masked-loss weight. Strict
CUDA determinism is disabled for both because reflection-padding backward has
no deterministic CUDA implementation.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    max_config, max_checkpoint_path, max_result = _completed_run(args.max_wind_run)
    radii_config, radii_checkpoint_path, radii_result = _completed_run(args.radii_run)
    if max_config["data"] != radii_config["data"]:
        raise ValueError("the paired runs do not use identical data configurations")
    if max_config.get("seed") != radii_config.get("seed"):
        raise ValueError("the paired runs do not use the same random seed")

    max_metrics, max_samples, _ = _evaluate_run(
        max_config,
        max_checkpoint_path,
        accelerator=args.accelerator,
        devices=args.devices,
    )
    radii_metrics, radii_samples, structure_samples = _evaluate_run(
        radii_config,
        radii_checkpoint_path,
        accelerator=args.accelerator,
        devices=args.devices,
    )
    if max_samples != radii_samples:
        raise ValueError("paired test splits contain different sample counts")
    rows = _summary_rows(
        max_metrics,
        radii_metrics,
        max_samples,
        radii_samples,
        structure_samples,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.csv"
    _write_summary_csv(summary_path, rows)
    _write_figures(args.image_dir, rows)

    created_utc = datetime.now(timezone.utc).isoformat()
    max_checkpoint = {
        "path": str(max_checkpoint_path),
        "sha256": _sha256(max_checkpoint_path),
    }
    radii_checkpoint = {
        "path": str(radii_checkpoint_path),
        "sha256": _sha256(radii_checkpoint_path),
    }
    payload = {
        "schema_version": 1,
        "created_utc": created_utc,
        "split": "test",
        "samples": max_samples,
        "runs": {
            "max_wind_only": {
                "path": str(args.max_wind_run.resolve()),
                "checkpoint": max_checkpoint,
                "fit_result": max_result,
                "test_metrics": max_metrics,
            },
            "max_wind_plus_radii": {
                "path": str(args.radii_run.resolve()),
                "checkpoint": radii_checkpoint,
                "fit_result": radii_result,
                "test_metrics": radii_metrics,
            },
        },
        "summary": rows,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_docs_page(
        args.docs_page,
        rows,
        created_utc=created_utc,
        max_run=args.max_wind_run,
        radii_run=args.radii_run,
        max_checkpoint=max_checkpoint,
        radii_checkpoint=radii_checkpoint,
    )
    print(
        f"Wrote {summary_path}, {args.output_dir / 'results.json'}, and {args.docs_page}"
    )


if __name__ == "__main__":
    main()
