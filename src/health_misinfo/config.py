from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = PROJECT_ROOT / path if not Path(path).is_absolute() else Path(path)
    with resolved.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_paths() -> dict[str, Path]:
    return {key: PROJECT_ROOT / value for key, value in load_yaml("configs/paths.yaml").items()}


def load_experiments() -> dict[str, Any]:
    return load_yaml("configs/experiments.yaml")

