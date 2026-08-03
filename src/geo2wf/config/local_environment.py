"""Load optional repository-local environment overrides."""

from __future__ import annotations

import os
from pathlib import Path


def load_local_env(repo_root: Path | None = None) -> None:
    root = repo_root or Path(__file__).resolve().parents[3]
    env_file = root / ".local.env"
    if not env_file.exists():
        env_file = root / ".local.example.env"
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
