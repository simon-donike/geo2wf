"""Capture the publication's fixed provenance and external checkpoint bundle.

Run against the original research checkout, never against arbitrary latest runs.
Original metadata is copied byte-for-byte; portable runtime files are separate.
"""

from __future__ import annotations

import argparse
import hashlib
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE = {
    "with-era5": "20260820T144011Z-with-era5",
    "without-era5": "20260820T155344Z-without-era5",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    args = parser.parse_args()
    source, artifacts = args.source_root.resolve(), args.artifact_root.resolve()
    release = ROOT / "release"
    release.mkdir(exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    registry = {
        "schema_version": 1,
        "source_root": str(source),
        "source_commit": subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "models": {},
        "files": {},
        "external_dependencies": [],
        "architecture": {},
        "case_studies": {},
    }

    def copy(path, destination, tracked=True):
        path, destination = Path(path), Path(destination)
        if not path.is_file():
            raise FileNotFoundError(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        info = {"sha256": sha256(destination), "size_bytes": destination.stat().st_size}
        if tracked:
            registry["files"][str(destination.relative_to(ROOT))] = info
        return info

    def metadata(path):
        path = Path(path)
        try:
            rel = path.relative_to(source)
        except ValueError:
            return None
        dest = release / "provenance" / rel
        if path.suffix == ".csv":
            dest = dest.with_suffix(".csv.gz")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
            registry["files"][str(dest.relative_to(ROOT))] = {
                "sha256": sha256(dest),
                "size_bytes": dest.stat().st_size,
                "uncompressed_sha256": sha256(path),
            }
            return str(dest.relative_to(ROOT))
        copy(path, dest)
        return str(dest.relative_to(ROOT))

    def add_model(identifier, checkpoint, expected=None, role=None):
        checkpoint = Path(checkpoint)
        for key, value in registry["models"].items():
            if value["original_checkpoint"] == str(checkpoint):
                return key
        digest = sha256(checkpoint)
        if expected and digest != expected:
            raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")
        run = checkpoint.parent.parent
        config = run / "resolved-config.yaml"
        if not config.exists():
            config = run / "config.yaml"
        info = {
            "original_checkpoint": str(checkpoint),
            "checkpoint": f"checkpoints/{identifier}.ckpt",
            "sha256": digest,
            "size_bytes": checkpoint.stat().st_size,
            "role": role,
            "dependencies": [],
            "config": metadata(config) if config.exists() else None,
        }
        registry["models"][identifier] = info
        copy(checkpoint, artifacts / info["checkpoint"], tracked=False)
        for name in ("run-manifest.json", "result.json"):
            if (run / name).exists():
                info[name.removesuffix(".json").replace("-", "_")] = metadata(
                    run / name
                )
        # Keep source patches and captured untracked source in the external bundle.
        for name in (
            "source-diff.patch",
            "source-snapshot",
            "metrics",
            "metric-history.jsonl",
        ):
            path = run / name
            dest = artifacts / "run-provenance" / identifier / name
            if path.is_dir():
                shutil.copytree(path, dest, dirs_exist_ok=True)
            elif path.is_file():
                copy(path, dest, tracked=False)
        conf = yaml.safe_load(config.read_text()) if config.exists() else {}
        if not config.exists():
            info["reproduction_gap"] = (
                "Initialization weights retained; original training config unavailable locally."
            )
        info["model_target"] = conf.get("model", {}).get("_target_")
        info["include_test_in_train"] = conf.get("data", {}).get(
            "include_test_in_train", False
        )
        if "run_manifest" in info:
            manifest = json.loads((ROOT / info["run_manifest"]).read_text())
            info["source_commit"] = manifest.get("git_commit")
            info["source_dirty"] = manifest.get("source_provenance", {}).get(
                "git_dirty"
            )
            for dep in manifest.get("input_checkpoints", {}).values():
                if isinstance(dep, dict) and dep.get("path"):
                    dep_id = add_model(
                        "training-input-" + dep["sha256"][:12],
                        dep["path"],
                        dep["sha256"],
                        "training initialization",
                    )
                    info["dependencies"].append(dep_id)
        data = conf.get("data", {})
        data_root = Path(data.get("root", "."))
        if not data_root.is_absolute():
            data_root = source / data_root
        info["data_metadata"] = []
        for path in sorted(data_root.glob("*/manifest.csv")) + [
            data_root / "stats.json",
            data_root / "cache-metadata.json",
        ]:
            if path.is_file():
                info["data_metadata"].append(metadata(path))
        for key in ("stats_file", "ibtracs_file"):
            if data.get(key):
                p = Path(data[key])
                p = p if p.is_absolute() else source / p
                if p.is_file():
                    # The full IBTrACS corpus stays external; retain its fingerprint.
                    info[key] = {"original_path": str(p), "sha256": sha256(p)}
        return identifier

    for regime, directory in ARCHITECTURE.items():
        base = source / "logs/intensity-comparisons" / directory
        report = json.loads((base / "test-comparison.json").read_text())
        registry["architecture"][regime] = {
            "report": metadata(base / "test-comparison.json"),
            "models": {},
        }
        for name, ref in report["checkpoints"].items():
            registry["architecture"][regime]["models"][name] = add_model(
                f"architecture-{regime}-{name}",
                ref["path"],
                ref["sha256"],
                "paper architecture table and storm case studies",
            )
        for path in base.glob("test-comparison*"):
            if path.is_file():
                metadata(path)
        for path in (base / "unet-intensity-cache").glob("*/manifest.csv"):
            metadata(path)
        metadata(base / "unet-intensity-cache/cache-metadata.json")
        registry["architecture"][regime]["cache_metadata"] = metadata(
            base / "unet-intensity-cache/cache-metadata.json"
        )
        registry["architecture"][regime]["workflow"] = metadata(base / "workflow.json")

    validation = source / "logs/current-experiment-evaluation/validation-results.json"
    registry["validation_report"] = metadata(validation)
    metadata(validation.with_name("validation-metrics.csv"))
    for key, value in json.loads(validation.read_text())["runs"].items():
        add_model(
            key,
            value["checkpoint"],
            value["checkpoint_sha256"],
            (
                "paper latent ablation"
                if key.startswith("latent")
                else "storm case studies"
            ),
        )
    cache = source / "data/unet_intensity_structure_v3/cache-metadata.json"
    meta = json.loads(cache.read_text())
    dep = add_model(
        "structure-cache-field",
        meta["unet_checkpoint"]["path"],
        meta["unet_checkpoint"]["sha256"],
        "frozen field used to train radii correction heads",
    )
    for key in ("correction_image_radii", "correction_mlp_radii"):
        registry["models"][key]["dependencies"].append(dep)
    registry["structure_cache"] = {
        "metadata": metadata(cache),
        "export_config": metadata(meta["unet_config"]["path"]),
        "field_model": dep,
    }
    for regime in ARCHITECTURE:
        p = source / "logs/current-experiment-evaluation/three-storm" / f"{regime}.json"
        registry["case_studies"][regime] = metadata(p)
    for p in (source / "tables").glob("*.tex"):
        copy(p, release / "paper" / p.name)
    for p in (source / "inference/forecasts").glob("**/summary.json"):
        ref = json.loads(p.read_text())
        metadata(p)
        if "/mlp/" in str(p):
            add_model(
                "dashboard-mlp",
                ref["checkpoint"],
                ref["checkpoint_sha256"],
                "stormtracker retrospective 12-hour forecast",
            )
        elif not registry["external_dependencies"]:
            registry["external_dependencies"].append(
                {
                    "id": "dashboard-convlstm",
                    "checkpoint": ref["checkpoint"],
                    "sha256": ref["checkpoint_sha256"],
                    "config": ref.get("experiment_config"),
                    "status": "external HPC checkpoint unavailable locally; exported forecasts retained",
                }
            )
    registry["dashboard"] = {"field": dep, "forecast": "dashboard-mlp"}
    for p in (source / "inference/inf_unet_mlp").glob("*/correction-provenance.json"):
        ref = json.loads(p.read_text())
        metadata(p)
        correction = ref["correction_checkpoint"]
        key = add_model(
            "dashboard-correction",
            correction["path"],
            correction["sha256"],
            "stormtracker UNet+MLP nowcasts",
        )
        registry["models"][key]["dependencies"] = [dep]
        registry["dashboard"]["correction"] = key
        registry["dashboard"]["cache_metadata"] = metadata(
            ref["intensity_cache_metadata"]
        )
    for name in ("inf_unet", "inf_unet_mlp", "inf_vit"):
        for p in (source / "inference" / name).glob("*/inference-summary.csv"):
            metadata(p)
    registry["external_dependencies"].append(
        {
            "id": "dashboard-vit",
            "status": "Precomputed ViT fields and summaries supplied externally; no checkpoint identity recorded in local bundles. Browser exports retained.",
        }
    )
    # Historical local metadata may have changed since evaluation. Keep that visible.
    registry["provenance_notes"] = [
        "Architecture table uses its original test cohort; latent table uses configured validation cohorts.",
        "Source reports preserve historical hashes; files records hash the available captured metadata.",
        "Structure-cache field was trained with include_test_in_train=true; its export config says false. Preserve both; it supports case-study correction heads, not the six paper architecture checkpoints.",
    ]
    (release / "registry.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n"
    )
    if (artifacts / "release").exists():
        shutil.rmtree(artifacts / "release")
    shutil.copytree(release, artifacts / "release")
    files = sorted(
        p for p in artifacts.rglob("*") if p.is_file() and p.name != "SHA256SUMS"
    )
    (artifacts / "SHA256SUMS").write_text(
        "".join(f"{sha256(p)}  {p.relative_to(artifacts)}\n" for p in files)
    )
    archive = artifacts.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as output:
        output.add(artifacts, arcname="conference-artifacts")
    archive.with_suffix(archive.suffix + ".sha256").write_text(
        f"{sha256(archive)}  {archive.name}\n"
    )
    print(f"Captured {len(registry['models'])} checkpoints; bundle: {archive}")


if __name__ == "__main__":
    main()
