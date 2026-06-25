from __future__ import annotations

import os
from pathlib import Path

ENV_CACHE_DIR = "MATCHER_HDW_CACHE_DIR"
DEFAULT_CACHE_DIR = ".cache/models"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_cache_dir(cache_dir: str | Path | None = None, create: bool = False) -> Path:
    selected = os.environ.get(ENV_CACHE_DIR) or cache_dir or DEFAULT_CACHE_DIR
    path = Path(selected).expanduser()
    if not path.is_absolute():
        path = repo_root() / path
    path = path.resolve()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def builtin_matcher_config_path(name: str) -> Path:
    filename = name if name.endswith(".yaml") else f"{name}.yaml"
    return repo_root() / "config" / "matchers" / filename
