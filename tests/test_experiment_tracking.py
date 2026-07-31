from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml

from src.experiment_tracking import initialize_run_manifest, record_run_failure
from train import build_model, resolve_runtime_config


def _run_git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )


def test_runtime_config_materializes_environment_baseline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "baseline.ckpt"
    checkpoint.write_bytes(b"baseline")
    monkeypatch.setenv("GEO2WF_BASELINE_CKPT", str(checkpoint))
    config = {
        "model": {
            "type": "diffusion_residual",
            "residual": {
                "baseline": {
                    "source": "deterministic",
                    "checkpoint_path": None,
                }
            },
        }
    }

    resolved = resolve_runtime_config(config)

    assert resolved["model"]["residual"]["baseline"]["checkpoint_path"] == str(
        checkpoint.resolve()
    )


def test_manifest_snapshots_dirty_source_and_input_checkpoints(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _run_git(repository, "init", "-q")
    tracked = repository / "tracked.py"
    tracked.write_text("VALUE = 1\n", encoding="utf-8")
    _run_git(repository, "add", "tracked.py")
    _run_git(
        repository,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "initial",
    )
    tracked.write_text("VALUE = 2\n", encoding="utf-8")
    untracked_config = repository / "configs" / "new.yaml"
    untracked_config.parent.mkdir()
    untracked_config.write_text("enabled: true\n", encoding="utf-8")
    untracked_script = repository / "scripts" / "new.sh"
    untracked_script.parent.mkdir()
    untracked_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (repository / "notes.txt").write_text("not source\n", encoding="utf-8")

    checkpoint = tmp_path / "baseline.ckpt"
    checkpoint.write_bytes(b"checkpoint-content")
    config_path = tmp_path / "config.yaml"
    config = {"data": {"include_test_in_train": True}}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_dir = tmp_path / "run"
    monkeypatch.chdir(repository)

    initialize_run_manifest(
        run_dir,
        config_path=config_path,
        config=config,
        checkpoint_paths={"deterministic_baseline": checkpoint},
    )

    manifest = json.loads((run_dir / "run-manifest.json").read_text(encoding="utf-8"))
    source = manifest["source_provenance"]
    assert manifest["schema_version"] == 2
    assert manifest["include_test_in_train"] is True
    assert source["git_dirty"] is True
    assert source["git_commit"]
    assert (
        source["tracked_diff_sha256"]
        == hashlib.sha256((run_dir / "source-diff.patch").read_bytes()).hexdigest()
    )
    assert "VALUE = 2" in (run_dir / "source-diff.patch").read_text(encoding="utf-8")
    snapshot_paths = {
        item["repository_path"]: Path(item["snapshot_path"])
        for item in source["untracked_snapshots"]
    }
    assert snapshot_paths["configs/new.yaml"].read_text(encoding="utf-8") == (
        "enabled: true\n"
    )
    assert snapshot_paths["scripts/new.sh"].is_file()
    assert "notes.txt" not in snapshot_paths
    checkpoint_record = manifest["input_checkpoints"]["deterministic_baseline"]
    assert checkpoint_record["path"] == str(checkpoint.resolve())
    assert (
        checkpoint_record["sha256"]
        == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
    resolved_config = yaml.safe_load(
        (run_dir / "resolved-config.yaml").read_text(encoding="utf-8")
    )
    assert resolved_config == config


def test_record_run_failure_is_durable_and_idempotent(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run-manifest.json").write_text(
        json.dumps({"schema_version": 2, "status": "running"}),
        encoding="utf-8",
    )

    record_run_failure(run_dir, RuntimeError("setup broke"), phase="model_build")
    record_run_failure(run_dir, RuntimeError("later"), phase="trainer_fit")

    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run-manifest.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["failure_phase"] == "model_build"
    assert result["exception"] == "setup broke"
    assert manifest["status"] == "failed"
    assert manifest["finished_at"] == result["finished_at"]


def test_builder_wires_image_logging_switch() -> None:
    diffusion = build_model(
        {
            "model": {
                "type": "diffusion",
                "in_channels": 2,
                "out_channels": 1,
                "num_timesteps": 4,
                "unet": {
                    "dim": 4,
                    "dim_mults": [1, 2],
                    "channels": 4,
                    "out_dim": 1,
                },
            },
            "validation": {"log_reconstruction_images": False},
        }
    )
    deterministic = build_model(
        {
            "model": {
                "type": "deterministic_residual",
                "condition_channels": 2,
                "residual": {"base_channels": 4, "channel_mults": [1, 2]},
            },
            "validation": {"log_reconstruction_images": False},
        }
    )

    assert diffusion.log_reconstruction_images is False
    assert deterministic.log_reconstruction_images is False
