#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from health_misinfo.config import PROJECT_ROOT, load_paths
from health_misinfo.datasets.inventory import hash_inventory, scan_files, table_and_field_inventory
from health_misinfo.paths import ensure_project_dirs


def main() -> int:
    ensure_project_dirs()
    paths = load_paths()
    raw = paths["data_raw"]
    out = paths["inventory"]
    roots = [raw / "coaid", raw / "fakehealth", raw / "fakehealth_zenodo"]
    files = []
    for root in roots:
        if root.exists():
            files.append(scan_files(root))
    file_inventory = files[0] if len(files) == 1 else __import__("pandas").concat(files, ignore_index=True)
    table_inventory, field_inventory = table_and_field_inventory(file_inventory, PROJECT_ROOT)
    hashes = hash_inventory(file_inventory, PROJECT_ROOT)
    file_inventory.to_csv(out / "file_inventory.csv", index=False)
    table_inventory.to_csv(out / "table_inventory.csv", index=False)
    field_inventory.to_csv(out / "field_inventory.csv", index=False)
    hashes.to_csv(out / "hash_inventory.csv", index=False)
    summary = [
        "# Dataset inventory summary",
        "",
        f"- Files inventoried: {len(file_inventory):,}",
        f"- Table-like files inventoried: {len(table_inventory):,}",
        f"- Fields inventoried: {len(field_inventory):,}",
        f"- Total bytes: {int(file_inventory['size_bytes'].sum()):,}",
        "",
        "Large files are listed in the hash inventory with `skipped_large_file` unless they are primary archives.",
    ]
    (out / "inventory_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Wrote inventory outputs to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

