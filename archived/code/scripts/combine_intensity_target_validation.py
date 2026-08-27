#!/usr/bin/env python3
"""Combine matched target experiments and publish one validation report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd


DOCUMENTATION_START = "<!-- matched-validation-results:start -->"
DOCUMENTATION_END = "<!-- matched-validation-results:end -->"
KNOTS_PER_MPS = 1.0 / 0.514444


def _speed(value: float) -> str:
    """Format an m/s result with its knots equivalent in parentheses."""
    return f"{value:.3f} ({value * KNOTS_PER_MPS:.3f} kt)"


def _speed_interval(low: float, high: float) -> str:
    return (
        f"{low:.3f}–{high:.3f} m/s "
        f"({low * KNOTS_PER_MPS:.3f}–{high * KNOTS_PER_MPS:.3f} kt)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--divergence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb-project", default="geo2wf")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument(
        "--documentation",
        type=Path,
        default=None,
        help="Experiment page whose marked validation-results section is updated.",
    )
    parser.add_argument("--disable-wandb", action="store_true")
    args = parser.parse_args()
    if len(args.result) != 4:
        parser.error("exactly four --result files are required")
    return args


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) < 2:
        raise ValueError(f"{path} is not a dual-reference evaluation")
    if payload.get("split") != "val":
        raise ValueError(f"{path} is not a validation result")
    return payload


def _metric_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    seen_raw = set()
    for result in results:
        era5 = str(result["conditioning"]["label"])
        target_source = str(result["target_fingerprint"]["source"])
        for reference, reference_payload in result["reference_evaluation"].items():
            for subset, summary_key, bootstrap_key in (
                (
                    "overall",
                    "overall",
                    "overall_storm_bootstrap",
                ),
                (
                    "rapid_intensification",
                    "rapid_intensification",
                    "rapid_intensification_storm_bootstrap",
                ),
            ):
                summaries = reference_payload.get(summary_key)
                bootstrap = reference_payload.get(bootstrap_key)
                if not summaries:
                    continue
                for model_key, summary in summaries.items():
                    if model_key == "unet_raw_max":
                        raw_identity = (
                            era5,
                            reference,
                            subset,
                            result["checkpoints"]["unet"]["sha256"],
                        )
                        if raw_identity in seen_raw:
                            continue
                        seen_raw.add(raw_identity)
                        trained_target = "sar_field_only"
                    else:
                        trained_target = target_source
                    interval = [None, None]
                    if bootstrap:
                        interval = (
                            bootstrap.get("models", {})
                            .get(model_key, {})
                            .get("mae_ms_95ci", [None, None])
                        )
                    rows.append(
                        {
                            "era5": era5,
                            "trained_target": trained_target,
                            "model_key": model_key,
                            "model": (
                                (
                                    "U-Net raw field maximum"
                                    if reference == "ibtracs"
                                    else "U-Net raw field robust peak"
                                )
                                if model_key == "unet_raw_max"
                                else result["models"][model_key]["label"]
                            ),
                            "evaluation_reference": reference,
                            "subset": subset,
                            "samples": summary["samples"],
                            "storms": summary["storms"],
                            "mae_ms": summary["regression"]["mae_ms"],
                            "mae_95ci_low_ms": interval[0],
                            "mae_95ci_high_ms": interval[1],
                            "rmse_ms": summary["regression"]["rmse_ms"],
                            "bias_ms": summary["regression"]["bias_ms"],
                            "storm_macro_mae_ms": summary["storm_macro_mae_ms"],
                            "category_accuracy": summary["category"]["accuracy"],
                            "category_macro_f1": summary["category"]["macro_f1"],
                            "within_one_category_accuracy": summary["category"][
                                "within_one_accuracy"
                            ],
                        }
                    )
    return rows


def _prediction_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        era5 = str(result["conditioning"]["label"])
        target_source = str(result["target_fingerprint"]["source"])
        for model_key, predictions in result["prediction_rows"].items():
            if model_key == "unet_raw_max" and target_source != "ibtracs":
                continue
            for row in predictions:
                rows.append(
                    {
                        "era5": era5,
                        "trained_target": (
                            "sar_field_only"
                            if model_key == "unet_raw_max"
                            else target_source
                        ),
                        "model_key": model_key,
                        **row,
                    }
                )
    return rows


def _markdown(rows: list[dict[str, Any]]) -> str:
    best: dict[tuple[str, str, str, str], float] = {}
    for row in rows:
        group = (row["era5"], row["evaluation_reference"], row["subset"])
        for metric in ("mae_ms", "rmse_ms", "bias_ms", "storm_macro_mae_ms"):
            value = abs(row[metric]) if metric == "bias_ms" else row[metric]
            key = (*group, metric)
            best[key] = min(best.get(key, value), value)

    def formatted(row: dict[str, Any], metric: str, value: str) -> str:
        group = (row["era5"], row["evaluation_reference"], row["subset"], metric)
        candidate = abs(row[metric]) if metric == "bias_ms" else row[metric]
        return f"**{value}**" if candidate == best[group] else value

    lines = [
        "# Matched IBTrACS versus SAR intensity validation",
        "",
        "All models use the identical SAR-center-valid cohort. RI denotes an IBTrACS gain of at least 30 kt in the preceding 24 hours.",
        "",
        "| ERA5 | Trained target | Model | Evaluated against | Subset | Samples | Storms | MAE, m/s (kt); 95% CI | RMSE, m/s (kt) | Bias, m/s (kt) | Storm-macro MAE, m/s (kt) |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        interval = ""
        if row["mae_95ci_low_ms"] is not None:
            interval = "; 95% CI " + _speed_interval(
                row["mae_95ci_low_ms"], row["mae_95ci_high_ms"]
            )
        lines.append(
            f"| {row['era5']} | {row['trained_target']} | {row['model']} | "
            f"{row['evaluation_reference']} | {row['subset']} | {row['samples']} | "
            f"{row['storms']} | {formatted(row, 'mae_ms', _speed(row['mae_ms']) + interval)} | "
            f"{formatted(row, 'rmse_ms', _speed(row['rmse_ms']))} | "
            f"{formatted(row, 'bias_ms', _speed(row['bias_ms']))} | "
            f"{formatted(row, 'storm_macro_mae_ms', _speed(row['storm_macro_mae_ms']))} |"
        )
    return "\n".join(lines) + "\n"


def _update_documentation(
    path: Path,
    *,
    payload: dict[str, Any],
    metric_rows: list[dict[str, Any]],
) -> None:
    path = path.expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    if text.count(DOCUMENTATION_START) != 1 or text.count(DOCUMENTATION_END) != 1:
        raise ValueError(
            f"documentation must contain one validation marker pair: {path}"
        )
    before, marked = text.split(DOCUMENTATION_START, 1)
    _, after = marked.split(DOCUMENTATION_END, 1)
    report_lines = _markdown(metric_rows).splitlines()[2:]
    generated = str(payload["created_utc"])
    cohort = payload["cohort"]
    section = "\n".join(
        [
            DOCUMENTATION_START,
            "",
            f"Generated on `{generated}` from the completed seed-42 validation matrix. "
            f"All rows use the same cohort fingerprint `{cohort['sha256']}` "
            f"({cohort['samples']} samples from {cohort['storms']} storms).",
            "",
            *report_lines,
            DOCUMENTATION_END,
        ]
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(before + section + after, encoding="utf-8")
    temporary.replace(path)


def _divergence_figure(divergence: dict[str, Any], output: Path) -> Path | None:
    rows = pd.DataFrame(divergence.get("rows", []))
    if rows.empty:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(
        rows["ibtracs_target_ms"],
        rows["sar_robust_peak_target_ms"],
        s=10,
        alpha=0.45,
    )
    low = float(
        min(rows["ibtracs_target_ms"].min(), rows["sar_robust_peak_target_ms"].min())
    )
    high = float(
        max(rows["ibtracs_target_ms"].max(), rows["sar_robust_peak_target_ms"].max())
    )
    axes[0].plot([low, high], [low, high], color="black", linestyle="--")
    axes[0].set(xlabel="IBTrACS (m/s)", ylabel="SAR robust peak (m/s)")
    difference = rows["sar_robust_peak_target_ms"] - rows["ibtracs_target_ms"]
    axes[1].hist(difference, bins=30, color="#0072B2", alpha=0.8)
    axes[1].axvline(0.0, color="black", linestyle="--")
    axes[1].set(xlabel="SAR robust peak − IBTrACS (m/s)", ylabel="Samples")
    figure.tight_layout()
    path = output.with_name(output.stem + "-divergence.png")
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _log_wandb(
    args: argparse.Namespace,
    payload: dict[str, Any],
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    files: list[Path],
    divergence_figure: Path | None,
) -> str | None:
    if args.disable_wandb:
        return None
    try:
        import wandb
    except ImportError:
        return None
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name
        or f"intensity-target-validation-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}",
        group=args.wandb_group,
        job_type="evaluation",
        config={
            "split": "val",
            "cohort": payload["cohort"],
            "targets": ["ibtracs", "sar_robust_peak"],
            "era5_regimes": ["with_era5", "without_era5"],
        },
    )
    overall = metrics.loc[metrics["subset"] == "overall"].reset_index(drop=True)
    ri = metrics.loc[metrics["subset"] == "rapid_intensification"].reset_index(
        drop=True
    )
    divergence_rows = []
    for row in payload["divergence"].get("table", []):
        divergence_rows.append(
            {
                key: value
                for key, value in row.items()
                if not isinstance(value, (dict, list))
            }
        )
    checkpoint_rows = []
    config_rows = []
    for source in payload["source_results"]:
        config_rows.append(
            {
                "era5": source["conditioning"]["label"],
                "target": source["target_fingerprint"]["source"],
                "path": source["data_config"]["path"],
                "sha256": source["data_config"]["sha256"],
            }
        )
        for model, checkpoint in source["checkpoints"].items():
            checkpoint_rows.append(
                {
                    "era5": source["conditioning"]["label"],
                    "target": source["target_fingerprint"]["source"],
                    "model": model,
                    "path": checkpoint["path"],
                    "sha256": checkpoint["sha256"],
                }
            )
    run.log(
        {
            "validation/metrics": wandb.Table(dataframe=metrics),
            "validation/overall_metrics": wandb.Table(dataframe=overall),
            "validation/ri_metrics": wandb.Table(dataframe=ri),
            "validation/predictions": wandb.Table(dataframe=predictions),
            "validation/divergence": wandb.Table(
                dataframe=pd.DataFrame(divergence_rows)
            ),
            "validation/checkpoints": wandb.Table(
                dataframe=pd.DataFrame(checkpoint_rows)
            ),
            "validation/configs": wandb.Table(dataframe=pd.DataFrame(config_rows)),
            **(
                {"validation/sar_ibtracs_divergence": wandb.Image(divergence_figure)}
                if divergence_figure is not None
                else {}
            ),
        }
    )
    artifact = wandb.Artifact("intensity-target-validation", type="evaluation")
    for path in files:
        artifact.add_file(str(path))
    run.log_artifact(artifact)
    run_id = str(run.id)
    run.finish()
    return run_id


def combine(args: argparse.Namespace) -> dict[str, Any]:
    result_paths = [path.expanduser().resolve() for path in args.result]
    results = [_load(path) for path in result_paths]
    cohort_hashes = {result["cohort"]["sha256"] for result in results}
    if len(cohort_hashes) != 1:
        raise ValueError("target/ERA5 results do not share one cohort fingerprint")
    combinations = {
        (
            str(result["conditioning"]["label"]),
            str(result["target_fingerprint"]["source"]),
        )
        for result in results
    }
    expected = {
        (era5, target)
        for era5 in ("with_era5", "without_era5")
        for target in ("ibtracs", "sar_robust_peak")
    }
    if combinations != expected:
        raise ValueError(f"incomplete validation matrix: {sorted(combinations)}")
    for era5 in ("with_era5", "without_era5"):
        unet_hashes = {
            result["checkpoints"]["unet"]["sha256"]
            for result in results
            if result["conditioning"]["label"] == era5
        }
        if len(unet_hashes) != 1:
            raise ValueError(
                f"{era5} target variants did not reuse one field-only U-Net"
            )
    divergence_path = args.divergence.expanduser().resolve()
    divergence = json.loads(divergence_path.read_text(encoding="utf-8"))
    metric_rows = _metric_rows(results)
    prediction_rows = _prediction_rows(results)
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": results[0]["cohort"],
        "source_results": [
            {
                "path": str(path),
                "conditioning": result["conditioning"],
                "target_fingerprint": result["target_fingerprint"],
                "checkpoints": result["checkpoints"],
                "data_config": result["data_config"],
            }
            for path, result in zip(result_paths, results)
        ],
        "divergence": divergence,
        "metrics": metric_rows,
        "prediction_rows": prediction_rows,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics_path = output.with_suffix(".csv")
    predictions_path = output.with_name(output.stem + "-predictions.csv")
    markdown_path = output.with_suffix(".md")
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    markdown_path.write_text(_markdown(metric_rows), encoding="utf-8")
    figure = _divergence_figure(divergence, output)
    files = [
        output,
        metrics_path,
        predictions_path,
        markdown_path,
        divergence_path,
    ]
    for suffix in (".csv", ".md"):
        sibling = divergence_path.with_suffix(suffix)
        if sibling.is_file():
            files.append(sibling)
    if figure is not None:
        files.append(figure)
    documentation = getattr(args, "documentation", None)
    if documentation is not None:
        _update_documentation(
            documentation,
            payload=payload,
            metric_rows=metric_rows,
        )
        files.append(Path(documentation).expanduser().resolve())
    wandb_id = _log_wandb(args, payload, metrics, predictions, files, figure)
    if wandb_id:
        payload["wandb_run_id"] = wandb_id
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(_markdown(metric_rows))
    return payload


def main() -> None:
    combine(parse_args())


if __name__ == "__main__":
    main()
