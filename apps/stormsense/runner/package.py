#!/usr/bin/env python3
"""Package the CPU runner, pinned models and a consistent SQLite snapshot.

Does not include credentials, imagery, a virtualenv, or install/activate services.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import tarfile
import tempfile


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "var/stormsense/runner.tar.gz")
    args = parser.parse_args()
    files = {Path("pyproject.toml"), Path("uv.lock")}
    for directory in ("src", "apps/stormsense/runner"):
        files.update(p.relative_to(root) for p in (root / directory).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    pins = json.loads((root / "src/geo2wf/operational/models.json").read_text())
    for spec in pins["models"].values():
        for field in ("checkpoint", "config", "stats", "training_manifest"):
            if field not in spec:
                continue
            relative = Path("downloads/models") / spec[field]
            expected = spec["sha256" if field == "checkpoint" else field + "_sha256"]
            if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Pinned model asset checksum mismatch: {relative}")
            files.add(relative)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stormsense-runner-") as directory:
        snapshot = Path(directory) / "state.sqlite"
        source = sqlite3.connect((root / "var/stormsense/state.sqlite").as_uri() + "?mode=ro", uri=True)
        destination = sqlite3.connect(snapshot)
        try:
            source.backup(destination)
            if destination.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("SQLite snapshot failed its integrity check")
        finally:
            destination.close()
            source.close()
        staging = output.with_suffix(output.suffix + ".tmp")
        with tarfile.open(staging, "w:gz", dereference=True) as archive:
            for relative in sorted(files):
                archive.add(root / relative, arcname=str(relative), recursive=False)
            archive.add(snapshot, arcname="var/stormsense/state.sqlite")
        staging.replace(output)
    report = {
        "path": str(output.relative_to(root)) if output.is_relative_to(root) else str(output),
        "bytes": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "files": len(files) + 1,
        "sqlite_integrity": "ok",
        "pinned_models_verified": True,
        "credentials_included": False,
        "imagery_included": False,
        "scheduler_activated": False,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
