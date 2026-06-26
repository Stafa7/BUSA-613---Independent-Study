from __future__ import annotations

from pathlib import Path

from health_misinfo.config import PROJECT_ROOT, load_paths


def ensure_project_dirs() -> None:
    paths = load_paths()
    for key in [
        "data_interim",
        "data_processed",
        "inventory",
        "provenance",
        "profiles",
        "splits",
        "experiments",
    ]:
        paths[key].mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "reports" / "tables").mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)

