from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from health_misinfo.paths import rel


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_files(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(raw_dir.rglob("*")):
        if ".git" in path.parts:
            continue
        if path.is_file():
            stat = path.stat()
            rows.append(
                {
                    "path": rel(path),
                    "dataset_root": rel(raw_dir),
                    "suffix": path.suffix.lower(),
                    "size_bytes": stat.st_size,
                }
            )
    return pd.DataFrame(rows)


def hash_inventory(files: pd.DataFrame, project_root: Path, max_regular_size: int = 25_000_000) -> pd.DataFrame:
    rows = []
    for row in files.itertuples(index=False):
        path = project_root / row.path
        if "fakehealth_zenodo/extracted" in row.path:
            rows.append({"path": row.path, "size_bytes": row.size_bytes, "sha256": "skipped_extracted_zenodo_file"})
            continue
        should_hash = row.size_bytes <= max_regular_size or path.name == "FakeHealth.zip"
        if should_hash:
            rows.append({"path": row.path, "size_bytes": row.size_bytes, "sha256": sha256_file(path)})
        else:
            rows.append({"path": row.path, "size_bytes": row.size_bytes, "sha256": "skipped_large_file"})
    return pd.DataFrame(rows)


def _json_shape(obj: Any) -> tuple[int, list[str]]:
    if isinstance(obj, list):
        keys = sorted({key for item in obj[:50] if isinstance(item, dict) for key in item.keys()})
        return len(obj), keys
    if isinstance(obj, dict):
        first_values = list(obj.values())[:50]
        keys = sorted({key for item in first_values if isinstance(item, dict) for key in item.keys()})
        return len(obj), keys
    return 1, []


def table_and_field_inventory(files: pd.DataFrame, project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    table_rows = []
    field_rows = []
    for row in files.itertuples(index=False):
        if "fakehealth_zenodo/extracted" in row.path:
            continue
        path = project_root / row.path
        if path.suffix.lower() == ".csv":
            try:
                sample = pd.read_csv(path, nrows=5)
                count = sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1
                fields = list(sample.columns)
            except Exception as exc:
                count = None
                fields = []
                table_rows.append({"path": row.path, "format": "csv", "records": count, "status": f"error: {exc}"})
                continue
            table_rows.append({"path": row.path, "format": "csv", "records": count, "status": "ok"})
            for field in fields:
                field_rows.append({"path": row.path, "field": field})
        elif path.suffix.lower() == ".json" and path.stat().st_size <= 10_000_000:
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
                count, fields = _json_shape(obj)
                table_rows.append({"path": row.path, "format": "json", "records": count, "status": "ok"})
                for field in fields:
                    field_rows.append({"path": row.path, "field": field})
            except Exception as exc:
                table_rows.append({"path": row.path, "format": "json", "records": None, "status": f"error: {exc}"})
    return pd.DataFrame(table_rows), pd.DataFrame(field_rows)
