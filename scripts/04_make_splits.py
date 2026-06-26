#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from health_misinfo.config import load_experiments, load_paths
from health_misinfo.features.leakage import count_outcome_terms
from health_misinfo.paths import ensure_project_dirs
from health_misinfo.splits import grouped_stratified_split, source_disjoint_or_fallback, split_distribution


def _write_manifest(df: pd.DataFrame, path: Path) -> None:
    columns = ["dataset", "record_id", "harmonized_label", "source_name", "duplicate_group", "split", "split_type", "group_field"]
    df[columns].to_csv(path, index=False)


def main() -> int:
    ensure_project_dirs()
    paths = load_paths()
    config = load_experiments()
    seed = int(config["random_seed"])
    combined = pd.read_parquet(paths["data_processed"] / "combined_items.parquet")
    included = combined[combined["exclusion_reason"].fillna("") == ""].copy()
    manifests = []
    notes = []
    for dataset, df in included.groupby("dataset"):
        standard = grouped_stratified_split(df, seed=seed)
        standard["dataset"] = dataset
        manifests.append(standard)
        controlled, status = source_disjoint_or_fallback(df, seed=seed)
        controlled["dataset"] = dataset
        manifests.append(controlled)
        notes.append(f"- {dataset}: {status}; split type `{controlled['split_type'].iloc[0]}`.")
    standard_all = pd.concat([m for m in manifests if m["split_type"].str.startswith("standard").all()], ignore_index=True)
    controlled_all = pd.concat([m for m in manifests if not m["split_type"].str.startswith("standard").all()], ignore_index=True)
    fallback_all = standard_all.copy()
    fallback_all["split_type"] = "artifact_masked_sensitivity_using_standard_split"
    fallback_all["group_field"] = "duplicate_group"
    _write_manifest(standard_all, paths["splits"] / "standard_split_manifest.csv")
    _write_manifest(controlled_all, paths["splits"] / "source_disjoint_split_manifest.csv")
    _write_manifest(fallback_all, paths["splits"] / "topic_or_fallback_split_manifest.csv")
    distribution = pd.concat(
        [split_distribution(standard_all), split_distribution(controlled_all), split_distribution(fallback_all)],
        ignore_index=True,
    )
    distribution.to_csv(paths["splits"] / "split_distribution_table.csv", index=False)
    leakage_rows = []
    for dataset, df in included.groupby("dataset"):
        leakage_rows.append(
            {
                "dataset": dataset,
                "records": len(df),
                "records_with_url": int(df["url"].fillna("").astype(str).str.len().gt(0).sum()),
                "records_with_source_name": int(df["source_name"].fillna("").astype(str).str.len().gt(0).sum()),
                "outcome_term_hits_in_model_text": int(df["model_text"].map(count_outcome_terms).sum()),
                "outcome_term_hits_in_masked_text": int(df["model_text_masked"].map(count_outcome_terms).sum()),
            }
        )
    pd.DataFrame(leakage_rows).to_csv(paths["splits"] / "leakage_audit_table.csv", index=False)
    masked_spec = """# Masked text specification

`model_text_masked` replaces raw URLs with `[URL]` and source-name mentions with `[SOURCE]`.

This is a sensitivity input, not the primary text representation. It is used to check whether model performance depends heavily on obvious source or URL artifacts.
"""
    (paths["splits"] / "masked_text_specification.md").write_text(masked_spec, encoding="utf-8")
    audit = "# Leakage audit\n\n" + "\n".join(notes) + "\n\nOutcome-term counts are available in `leakage_audit_table.csv`.\n"
    (paths["splits"] / "leakage_audit.md").write_text(audit, encoding="utf-8")
    print(f"Wrote split manifests to {paths['splits']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

