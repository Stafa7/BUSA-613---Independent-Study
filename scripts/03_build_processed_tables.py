#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from health_misinfo.config import load_paths
from health_misinfo.datasets.coaid import load_coaid_items
from health_misinfo.datasets.fakehealth import load_fakehealth_items
from health_misinfo.datasets.labels import REQUIRED_PROCESSED_COLUMNS
from health_misinfo.paths import ensure_project_dirs


def _write_profile(combined: pd.DataFrame, profiles: Path) -> None:
    data_dictionary = pd.DataFrame(
        [
            {"field": col, "dtype": str(combined[col].dtype), "non_null": int(combined[col].notna().sum())}
            for col in combined.columns
        ]
    )
    data_dictionary.to_csv(profiles / "data_dictionary.csv", index=False)
    missingness = (
        combined.groupby("dataset")
        .apply(lambda df: df.isna().mean(), include_groups=False)
        .reset_index()
        .melt(id_vars="dataset", var_name="field", value_name="missing_fraction")
    )
    missingness.to_csv(profiles / "missingness_report.csv", index=False)
    duplicates = (
        combined.groupby(["dataset", "duplicate_group"])
        .size()
        .reset_index(name="records")
        .query("records > 1")
        .sort_values(["dataset", "records"], ascending=[True, False])
    )
    duplicates.to_csv(profiles / "duplicate_report.csv", index=False)
    feasibility_rows = []
    for dataset, df in combined.groupby("dataset"):
        included = df[df["exclusion_reason"].fillna("") == ""]
        complete = included[included["engagement_available"]]
        label_counts = complete["harmonized_label"].value_counts()
        coverage = len(complete) / len(included) if len(included) else 0
        gate = (
            coverage >= 0.60
            and label_counts.get("reliable", 0) >= 100
            and label_counts.get("unreliable", 0) >= 100
            and (included[["tweet_count", "reply_count", "retweet_count"]].sum(axis=1).nunique() > 1)
        )
        feasibility_rows.append(
            {
                "dataset": dataset,
                "included_records": len(included),
                "engagement_complete_records": len(complete),
                "engagement_coverage": coverage,
                "reliable_with_engagement": int(label_counts.get("reliable", 0)),
                "unreliable_with_engagement": int(label_counts.get("unreliable", 0)),
                "predictive_engagement_gate": "pass" if gate else "fail",
            }
        )
    pd.DataFrame(feasibility_rows).to_csv(profiles / "engagement_feasibility.csv", index=False)
    summary = [
        "# Processed data profile",
        "",
        f"- Combined records: {len(combined):,}",
        f"- Included records: {(combined['exclusion_reason'].fillna('') == '').sum():,}",
        f"- Exact duplicate groups with >1 record: {len(duplicates):,}",
        "",
        "Labels are dataset-defined reliability mappings, not verified claim-level truth.",
    ]
    (profiles / "profile_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def main() -> int:
    ensure_project_dirs()
    paths = load_paths()
    processed = paths["data_processed"]
    coaid = load_coaid_items(paths["data_raw"] / "coaid")
    fakehealth = load_fakehealth_items(paths["data_raw"] / "fakehealth", paths["data_raw"])
    combined = pd.concat([coaid, fakehealth], ignore_index=True)
    for col in REQUIRED_PROCESSED_COLUMNS:
        if col not in combined.columns:
            combined[col] = ""
    coaid.to_parquet(processed / "coaid_items.parquet", index=False)
    fakehealth.to_parquet(processed / "fakehealth_items.parquet", index=False)
    combined.to_parquet(processed / "combined_items.parquet", index=False)
    _write_profile(combined, paths["profiles"])
    print(f"Wrote processed tables to {processed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

