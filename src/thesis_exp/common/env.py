"""Helpers for loading simple environment files."""

from __future__ import annotations

import os
from pathlib import Path


def load_env_file(project_root: Path, env_filename: str = ".env") -> None:
    """Load key-value pairs from a local .env file into os.environ."""

    env_path = project_root / env_filename
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or key in os.environ:
            continue
        os.environ[key] = value
