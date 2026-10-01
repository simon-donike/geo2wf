"""Verify and reproduce the fixed conference release (never selects latest runs)."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def registry():
    return json.loads((ROOT / "release/registry.json").read_text())


def paper_rows(reg):
    architecture = []
    for regime in ("with-era5", "without-era5"):
        report = json.loads((ROOT / reg["architecture"][regime]["report"]).read_text())
        for all_row, ri_row in zip(
            report["table"], report["rapid_intensification_table"]
        ):
            if all_row["model_key"] != ri_row["model_key"]:
                raise ValueError("Architecture report row alignment changed")
            architecture.append(
                [
                    f"{all_row['intensity_mae_ms']:.3f}",
                    f"{ri_row['intensity_mae_ms']:.3f}",
                ]
            )
    report = json.loads((ROOT / reg["validation_report"]).read_text())
    lookup = {
        (r["experiment"], r["subset"], r["output"], r["target"], r["metric"]): r[
            "value"
        ]
        for r in report["metrics"]
    }
    latent = []
    for era in ("era5", "no_era5"):
        for sar in ("sar", "no_sar"):
            for radii in (False, True):
                key = f"latent_{sar}_{era}_max_wind" + ("_radii" if radii else "")
                wind = [
                    f"{lookup[key, subset, 'scalar_head', 'maximum_wind', 'mae']:.3f}"
                    for subset in ("all_validation", "ri_validation")
                ]
                radius = (
                    [
                        f"{lookup[key, subset, 'scalar_radius_head', 'rmw', 'mae']:.2f}"
                        for subset in ("all_validation", "ri_validation")
                    ]
                    if radii
                    else ["--", "--"]
                )
                latent.append(wind + radius)
    return {
        "vmax_architecture_ablation": architecture,
        "latent_supervision_ablation": latent,
    }


def tables(reg, output):
    output.mkdir(parents=True, exist_ok=True)
    for name, expected in paper_rows(reg).items():
        source = ROOT / "release/paper" / (name + ".tex")
        lines = source.read_text().splitlines(keepends=True)
        found = []
        for line in lines:
            if not re.match(
                r"\s*(Yes|No|Field diagnostic|Post-hoc MLP|Joint latent MLP)\s*&", line
            ):
                continue
            plain = re.sub(r"\\textbf\{([^}]+)\}", r"\1", line)
            found.append(
                [c.strip().removesuffix(r"\\").strip() for c in plain.split("&")][
                    -len(expected[0]) :
                ]
            )
        if found != expected:
            raise ValueError(
                f"{name}: saved metrics do not reproduce publication: {found} != {expected}"
            )
        # Exact publication layout and emphasis, with all numeric cells verified.
        (output / source.name).write_text("".join(lines))
    (output / "table-values.json").write_text(
        json.dumps(paper_rows(reg), indent=2) + "\n"
    )
    print("Both publication tables verified at their published precision.")


def verify(reg, artifacts):
    for path, info in reg["files"].items():
        if digest(ROOT / path) != info["sha256"]:
            raise ValueError(f"Provenance checksum mismatch: {path}")
    for key, model in reg["models"].items():
        if digest(artifacts / model["checkpoint"]) != model["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {key}")
    print(
        f"Verified {len(reg['models'])} checkpoints and {len(reg['files'])} provenance files."
    )


def portable(value, reg, artifacts, data, work):
    if isinstance(value, dict):
        return {k: portable(v, reg, artifacts, data, work) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v, reg, artifacts, data, work) for v in value]
    if not isinstance(value, str):
        return value
    for model in reg["models"].values():
        if value == model["original_checkpoint"]:
            return str(artifacts / model["checkpoint"])
    old = reg["source_root"]
    for prefix in (old + "/data/", "data/"):
        if value.startswith(prefix):
            return str(data / value[len(prefix) :])
    if value.startswith(old + "/"):
        return str(work / value[len(old) + 1 :])
    if value.startswith("logs/"):
        return str(work / value)
    return value


def prepare(reg, artifacts, data, work):
    import yaml

    work.mkdir(parents=True, exist_ok=True)
    for path in sorted((ROOT / "release/provenance").rglob("*")):
        if not path.is_file():
            continue
        dest = work / path.relative_to(ROOT / "release/provenance")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if (
            dest.name == "cache-metadata.json"
            and dest.exists()
            and any(dest.parent.glob("*/fields/*.npz"))
        ):
            # A regenerated cache records its own relocated config and cohort.
            continue
        if path.suffix == ".gz":
            dest.with_suffix("").write_bytes(gzip.decompress(path.read_bytes()))
        elif path.suffix in (".yaml", ".json"):
            value = (
                yaml.safe_load(path.read_text())
                if path.suffix == ".yaml"
                else json.loads(path.read_text())
            )
            value = portable(value, reg, artifacts, data, work)
            dest.write_text(
                yaml.safe_dump(value, sort_keys=False)
                if path.suffix == ".yaml"
                else json.dumps(value, indent=2) + "\n"
            )
        else:
            dest.write_bytes(path.read_bytes())
    # Correction datasets are generated caches, separate from supplied raw data.
    for key, model in reg["models"].items():
        if not model["config"]:
            continue
        config = work / Path(model["config"]).relative_to("release/provenance")
        value = yaml.safe_load(config.read_text())
        if "loader" in value["data"]:
            value["data"]["loader"].update(num_workers=0, persistent_workers=False)
        else:
            value["data"]["num_workers"] = 0
            if "persistent_workers" in value["data"]:
                value["data"]["persistent_workers"] = False
        if "correction" in key:
            if key.startswith("architecture"):
                regime = "without-era5" if "without-era5" in key else "with-era5"
                cache_meta = reg["architecture"][regime]["cache_metadata"]
                value["data"]["root"] = str(
                    (work / Path(cache_meta).relative_to("release/provenance")).parent
                )
            elif key == "dashboard-correction":
                value["data"]["root"] = str(
                    work / "data/unet_intensity_geostat_nopmw_v2"
                )
            else:
                value["data"]["root"] = str(work / "data/unet_intensity_structure_v3")
        config.write_text(yaml.safe_dump(value, sort_keys=False))
    # Relocation changes serialized config hashes, not the archived originals.
    for metadata in work.rglob("cache-metadata.json"):
        value = json.loads(metadata.read_text())
        ref = value.get("unet_config", {})
        if ref.get("path") and Path(ref["path"]).is_file():
            ref.setdefault("original_sha256", ref.get("sha256"))
            ref["sha256"] = digest(ref["path"])
            metadata.write_text(json.dumps(value, indent=2) + "\n")
    print(f"Portable runtime metadata: {work}")


def run(script, *args):
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
        cwd=ROOT,
        check=True,
    )


def figures(output):
    import pandas as pd
    from scripts.publish_current_experiment_results import (
        load_storm_frames,
        _regime_series,
        storm_prediction_rows,
        storm_metric_rows,
        write_family_figure,
    )

    data = ROOT / "docs/assets/data/final-results"
    frames = load_storm_frames(
        data / "current-three-storm-with-era5.csv.gz",
        data / "current-three-storm-without-era5.csv.gz",
    )
    series = {regime: _regime_series(frame, regime) for regime, frame in frames.items()}
    predictions = storm_prediction_rows(frames, series)
    metrics = storm_metric_rows(predictions)
    expected = pd.read_csv(data / "current-three-storm-metrics.csv")
    pd.testing.assert_frame_equal(
        metrics.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
        atol=1e-9,
        rtol=1e-9,
    )
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "current-three-storm-predictions.csv", index=False)
    metrics.to_csv(output / "current-three-storm-metrics.csv", index=False)
    for family in ("core", "latent", "radii"):
        write_family_figure(
            frames,
            series,
            family,
            3,
            output / f"{family}.png",
            output / f"{family}.pdf",
        )
    print(
        "Storm metrics reproduced from native observations; three figure families rendered."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "verify",
            "tables",
            "prepare",
            "evaluate-latent",
            "evaluate-architecture",
            "cache",
            "storm-inference",
            "figures",
            "train",
            "dashboard-mlp",
            "smoke",
        ),
    )
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Directory containing geotiff/, IBTrACs/, and intensity_forecast/.",
    )
    parser.add_argument(
        "--inference-root",
        type=Path,
        help="Original storm observations (inf_data directory).",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("build/conference"))
    parser.add_argument("--output", type=Path, default=Path("build/conference/results"))
    parser.add_argument("--model", help="Exact model ID from release/registry.json")
    parser.add_argument(
        "--era5", choices=("with-era5", "without-era5"), default="with-era5"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--limit", type=int, help="Optional smoke-test observations per storm."
    )
    args = parser.parse_args()
    reg, work, output = registry(), args.work_dir.resolve(), args.output.resolve()
    if args.action == "tables":
        tables(reg, output)
        return
    if args.action == "figures":
        figures(output)
        return
    if args.artifact_root is None:
        parser.error("--artifact-root is required for this action")
    artifacts = args.artifact_root.resolve()
    verify(reg, artifacts)
    if args.action == "verify":
        return
    if args.action == "smoke":
        smoke(reg, artifacts)
        return
    if args.data_root is None:
        parser.error("--data-root is required for this action")
    data = args.data_root.resolve()
    prepare(reg, artifacts, data, work)
    output.mkdir(parents=True, exist_ok=True)

    def config(key):
        return work / Path(reg["models"][key]["config"]).relative_to(
            "release/provenance"
        )

    def checkpoint(key):
        return artifacts / reg["models"][key]["checkpoint"]

    arch = reg["architecture"][args.era5]
    cache_root = (
        work / Path(arch["cache_metadata"]).relative_to("release/provenance")
    ).parent
    if args.action == "prepare":
        return
    if args.action == "evaluate-latent":
        runs = [
            part
            for key, model in reg["models"].items()
            if key.startswith("latent_")
            for part in ("--run", f"{key}={config(key).parent}")
        ]
        run(
            "evaluate_current_experiments.py",
            *runs,
            "--output-dir",
            output,
            "--accelerator",
            "gpu" if args.device == "cuda" else "cpu",
        )
    elif args.action == "cache":
        run(
            "export_joint_intensity_cache.py",
            "--config",
            config(arch["models"]["unet"]),
            "--checkpoint",
            checkpoint(arch["models"]["unet"]),
            "--output-root",
            cache_root,
            "--device",
            args.device,
        )
        structure = reg["structure_cache"]
        run(
            "export_joint_intensity_cache.py",
            "--config",
            work / Path(structure["export_config"]).relative_to("release/provenance"),
            "--checkpoint",
            checkpoint(structure["field_model"]),
            "--output-root",
            work / "data/unet_intensity_structure_v3",
            "--device",
            args.device,
        )
    elif args.action == "evaluate-architecture":
        run(
            "evaluate_intensity_models.py",
            "--data-config",
            config(arch["models"]["unet"]),
            "--cache-root",
            cache_root,
            "--joint-checkpoint",
            checkpoint(arch["models"]["joint"]),
            "--correction-checkpoint",
            checkpoint(arch["models"]["correction"]),
            "--split",
            "test",
            "--output",
            output / f"{args.era5}.json",
            "--device",
            args.device,
        )
    elif args.action == "train":
        if args.model not in reg["models"] or not reg["models"][args.model]["config"]:
            parser.error(
                "--model must identify a retained model with a training config"
            )
        import yaml

        path = config(args.model)
        value = yaml.safe_load(path.read_text())
        value["trainer"].update(
            accelerator="gpu" if args.device == "cuda" else "cpu",
            devices=1,
            default_root_dir=str(output / args.model),
        )
        value["trainer"].pop("strategy", None)
        value.setdefault("logging", {}).setdefault("wandb", {})["enabled"] = False
        path = output / f"{args.model}-train.yaml"
        path.write_text(yaml.safe_dump(value))
        command = [sys.executable, "-m", "geo2wf.training", "--config", str(path)]
        manifest_path = reg["models"][args.model].get("run_manifest")
        if manifest_path:
            inputs = json.loads((ROOT / manifest_path).read_text()).get(
                "input_checkpoints", {}
            )
            for kind, flag in (
                ("weights_only", "--weights-only-path"),
                ("resume", "--ckpt-path"),
            ):
                if inputs.get(kind):
                    command += [
                        flag,
                        portable(inputs[kind]["path"], reg, artifacts, data, work),
                    ]
        subprocess.run(command, cwd=ROOT, check=True)
    elif args.action == "dashboard-mlp":
        run(
            "run_dashboard_mlp_forecast_inference.py",
            "--checkpoint",
            checkpoint("dashboard-mlp"),
            "--ibtracs-file",
            data / "IBTrACs/ibtracs.ALL.list.v04r01.csv",
            "--output-root",
            output,
            "--device",
            args.device,
        )
    elif args.action == "storm-inference":
        if args.inference_root is None:
            parser.error("--inference-root is required")
        extra_report = json.loads((ROOT / reg["case_studies"][args.era5]).read_text())
        extra = [
            part
            for key in extra_report["extra_checkpoints"]
            for part in ("--extra-run", f"{key}={config(key).parent}")
        ]
        run(
            "run_intensity_comparison_storm_inference.py",
            "--era5",
            args.era5.split("-")[0],
            "--data-root",
            args.inference_root,
            "--manifest",
            args.inference_root / "index-files/observation_manifest_v6.csv",
            "--stats",
            data / "geotiff/geo_sar_10bands_era5_v2_pmw/stats.json",
            "--ibtracs-file",
            data / "IBTrACs/ibtracs.ALL.list.v04r01.csv",
            "--unet-checkpoint",
            checkpoint(arch["models"]["unet"]),
            "--joint-checkpoint",
            checkpoint(arch["models"]["joint"]),
            "--correction-checkpoint",
            checkpoint(arch["models"]["correction"]),
            "--intensity-cache-metadata",
            cache_root / "cache-metadata.json",
            "--output-root",
            output,
            "--device",
            args.device,
            *extra,
            *(["--limit", args.limit] if args.limit else []),
        )


def smoke(reg, artifacts):
    import torch
    from geo2wf.config import instantiate_model, load_config_file

    torch.set_num_threads(2)
    for key, ref in reg["models"].items():
        state = torch.load(
            artifacts / ref["checkpoint"], map_location="cpu", weights_only=False
        )
        if ref["config"] and ref.get("model_target"):
            model = instantiate_model(load_config_file(ROOT / ref["config"]))
            model.load_state_dict(state["state_dict"], strict=True)
        else:
            from geo2wf.models.deterministic_residual import ERA5ResidualRegressor

            model = ERA5ResidualRegressor.load_from_checkpoint(
                artifacts / ref["checkpoint"], map_location="cpu"
            )
        model.eval()
        with torch.inference_mode():
            if "intensity_forecast" in type(model).__module__:
                result = model.predict_two_steps(
                    torch.tensor([30.0]), torch.tensor([25.0]), torch.tensor([20.0])
                )
            elif "intensity_correction" in type(model).__module__:
                from geo2wf.data.intensity import INTENSITY_METADATA_NAMES

                result = model(
                    torch.ones(1, 32, 32) * 30,
                    torch.ones(1, 32, 32),
                    torch.zeros(1, 32, 32),
                    torch.zeros(1, model.metadata_features),
                )
            else:
                condition = torch.zeros(1, model.condition_channels, 32, 32)
                mask = torch.ones(1, 1, 32, 32)
                extra = (
                    (mask * 20, mask)
                    if "deterministic_residual" in type(model).__module__
                    else ()
                )
                result = model(condition, mask, *extra)
            values = (
                result
                if isinstance(result, tuple)
                else (
                    tuple(vars(result).values())
                    if hasattr(result, "__dataclass_fields__")
                    else (result,)
                )
            )
            for value in values:
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite smoke output: {key}")
        print(f"Loaded and inferred: {key}")


if __name__ == "__main__":
    main()
