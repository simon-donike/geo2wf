from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return value if torch.isfinite(torch.tensor(value)) else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _file_provenance(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    payload: dict[str, Any] = {
        "path": str(resolved),
        "exists": resolved.is_file(),
        "size_bytes": None,
        "sha256": None,
    }
    if payload["exists"]:
        payload["size_bytes"] = resolved.stat().st_size
        payload["sha256"] = _sha256_file(resolved)
    return payload


def _git_result(
    repository: Path,
    *arguments: str,
    text: bool = False,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        capture_output=True,
        text=text,
        check=False,
    )


def _git_repository_root() -> Path | None:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return Path(result.stdout.strip()).resolve()


def _is_relevant_untracked_source(relative_path: Path) -> bool:
    relevant_roots = {"configs", "data", "run_scripts", "scripts", "src", "tests"}
    relevant_suffixes = {
        ".cfg",
        ".ini",
        ".json",
        ".py",
        ".sh",
        ".toml",
        ".yaml",
        ".yml",
    }
    return relative_path.suffix.lower() in relevant_suffixes and (
        len(relative_path.parts) == 1 or relative_path.parts[0] in relevant_roots
    )


def _snapshot_source_state(run_dir: Path) -> dict[str, Any]:
    """Save enough dirty-worktree state to reconstruct the code used by a run."""
    repository = _git_repository_root()
    if repository is None:
        return {
            "available": False,
            "repository_root": None,
            "git_commit": None,
            "git_dirty": None,
            "git_status": [],
            "tracked_diff_path": None,
            "tracked_diff_sha256": None,
            "untracked_snapshots": [],
            "source_state_sha256": None,
        }

    commit_result = _git_result(repository, "rev-parse", "HEAD", text=True)
    commit = (
        commit_result.stdout.strip()
        if commit_result.returncode == 0 and commit_result.stdout.strip()
        else None
    )
    status_result = _git_result(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "-z",
    )
    status_entries = (
        [
            entry.decode("utf-8", errors="replace")
            for entry in status_result.stdout.split(b"\0")
            if entry
        ]
        if status_result.returncode == 0
        else []
    )

    diff_result = _git_result(repository, "diff", "HEAD", "--binary", "--no-ext-diff")
    tracked_diff = diff_result.stdout if diff_result.returncode == 0 else b""
    tracked_diff_path = run_dir / "source-diff.patch"
    _atomic_bytes(tracked_diff_path, tracked_diff)

    untracked_result = _git_result(
        repository,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    untracked_paths = (
        [
            Path(item.decode("utf-8", errors="strict"))
            for item in untracked_result.stdout.split(b"\0")
            if item
        ]
        if untracked_result.returncode == 0
        else []
    )
    snapshot_root = run_dir / "source-snapshot" / "untracked"
    snapshots = []
    for relative_path in sorted(untracked_paths, key=lambda item: item.as_posix()):
        if not _is_relevant_untracked_source(relative_path):
            continue
        source = repository / relative_path
        if (
            not source.is_file()
            or source.is_symlink()
            or not source.resolve().is_relative_to(repository)
        ):
            continue
        destination = snapshot_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        snapshots.append(
            {
                "repository_path": relative_path.as_posix(),
                "snapshot_path": str(destination.resolve()),
                "size_bytes": destination.stat().st_size,
                "sha256": _sha256_file(destination),
            }
        )

    tracked_diff_sha256 = _sha256_bytes(tracked_diff)
    source_identity = {
        "git_commit": commit,
        "tracked_diff_sha256": tracked_diff_sha256,
        "untracked": [
            {
                "repository_path": item["repository_path"],
                "sha256": item["sha256"],
            }
            for item in snapshots
        ],
    }
    source_state_sha256 = _sha256_bytes(
        json.dumps(source_identity, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    return {
        "available": True,
        "repository_root": str(repository),
        "git_commit": commit,
        "git_dirty": bool(status_entries),
        "git_status": status_entries,
        "tracked_diff_path": str(tracked_diff_path.resolve()),
        "tracked_diff_size_bytes": len(tracked_diff),
        "tracked_diff_sha256": tracked_diff_sha256,
        "untracked_snapshots": snapshots,
        "source_state_sha256": source_state_sha256,
    }


def initialize_run_manifest(
    run_dir: str | Path,
    *,
    config_path: str | Path,
    config: dict[str, Any],
    checkpoint_paths: dict[str, str | Path | None] | None = None,
) -> None:
    """Write resolved configuration and a durable initial run record."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    import yaml

    (run_dir / "resolved-config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    source_provenance = _snapshot_source_state(run_dir)
    checkpoint_provenance = {
        name: _file_provenance(path) for name, path in (checkpoint_paths or {}).items()
    }
    payload = {
        "schema_version": 2,
        "status": "running",
        "started_at": _utc_now(),
        "finished_at": None,
        "config_path": str(Path(config_path).resolve()),
        "resolved_config_path": str((run_dir / "resolved-config.yaml").resolve()),
        "run_dir": str(run_dir.resolve()),
        "command": [str(item) for item in sys.argv],
        "pid": os.getpid(),
        "git_commit": source_provenance["git_commit"],
        "source_provenance": source_provenance,
        "input_checkpoints": checkpoint_provenance,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "include_test_in_train": bool(
            config.get("data", {}).get("include_test_in_train", False)
        ),
        "result_path": str((run_dir / "result.json").resolve()),
        "metric_history_path": str((run_dir / "metric-history.jsonl").resolve()),
        "csv_metrics_dir": str((run_dir / "metrics").resolve()),
    }
    _atomic_json(run_dir / "run-manifest.json", payload)


def record_run_failure(
    run_dir: str | Path,
    exception: BaseException,
    *,
    phase: str | None = None,
) -> None:
    """Persist failures that occur before Lightning can invoke callback hooks."""
    run_dir = Path(run_dir)
    result_path = run_dir / "result.json"
    manifest_path = run_dir / "run-manifest.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") in {"completed", "failed"}:
            if existing.get("status") == "failed" and manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["status"] = "failed"
                manifest["finished_at"] = existing.get("finished_at", _utc_now())
                _atomic_json(manifest_path, manifest)
            return

    finished_at = _utc_now()
    result = {
        "schema_version": 1,
        "status": "failed",
        "finished_at": finished_at,
        "epoch": None,
        "global_step": None,
        "metrics": {},
        "failure_phase": phase,
        "exception_type": type(exception).__name__,
        "exception": str(exception),
    }
    _atomic_json(result_path, result)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"schema_version": 2, "run_dir": str(run_dir.resolve())}
    manifest["status"] = "failed"
    manifest["finished_at"] = finished_at
    _atomic_json(manifest_path, manifest)


class MachineReadableRunCallback(pl.Callback):
    """Persist validation history and a final scalar result without W&B."""

    def __init__(self, run_dir: str | Path) -> None:
        super().__init__()
        self.run_dir = Path(run_dir)
        self.manifest_path = self.run_dir / "run-manifest.json"
        self.history_path = self.run_dir / "metric-history.jsonl"

    @staticmethod
    def _metrics(trainer: pl.Trainer) -> dict[str, Any]:
        metrics = {}
        for name, value in trainer.callback_metrics.items():
            converted = _json_value(value)
            if converted is not None:
                metrics[str(name)] = converted
        return metrics

    def _is_global_zero(self, trainer: pl.Trainer) -> bool:
        return bool(getattr(trainer, "is_global_zero", True))

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        del pl_module
        if not self._is_global_zero(trainer) or trainer.sanity_checking:
            return
        record = {
            "timestamp": _utc_now(),
            "epoch": int(trainer.current_epoch),
            "global_step": int(trainer.global_step),
            "metrics": self._metrics(trainer),
        }
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        if not self._is_global_zero(trainer):
            return
        checkpoint = next(
            (
                callback
                for callback in trainer.callbacks
                if isinstance(callback, ModelCheckpoint)
            ),
            None,
        )
        result = {
            "schema_version": 1,
            "status": "completed",
            "finished_at": _utc_now(),
            "epoch": int(trainer.current_epoch),
            "global_step": int(trainer.global_step),
            "metrics": self._metrics(trainer),
            "best_model_path": (
                str(checkpoint.best_model_path) if checkpoint is not None else None
            ),
            "best_model_score": (
                _json_value(checkpoint.best_model_score)
                if checkpoint is not None
                else None
            ),
            "last_model_path": (
                str(checkpoint.last_model_path) if checkpoint is not None else None
            ),
        }
        _atomic_json(self.run_dir / "result.json", result)
        self._finish_manifest("completed")

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ) -> None:
        del pl_module
        if not self._is_global_zero(trainer):
            return
        result = {
            "schema_version": 1,
            "status": "failed",
            "finished_at": _utc_now(),
            "epoch": int(trainer.current_epoch),
            "global_step": int(trainer.global_step),
            "metrics": self._metrics(trainer),
            "exception_type": type(exception).__name__,
            "exception": str(exception),
        }
        _atomic_json(self.run_dir / "result.json", result)
        self._finish_manifest("failed")

    def _finish_manifest(self, status: str) -> None:
        if self.manifest_path.exists():
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        else:
            payload = {"schema_version": 1, "run_dir": str(self.run_dir.resolve())}
        payload["status"] = status
        payload["finished_at"] = _utc_now()
        _atomic_json(self.manifest_path, payload)
