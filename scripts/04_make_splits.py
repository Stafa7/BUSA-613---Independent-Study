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
from health_misinfo.splits import (
    controlled_split_or_unavailable,
    grouped_stratified_split,
    repeated_publisher_disjoint_splits,
    split_distribution,
    temporal_split_or_unavailable,
)


def _write_manifest(df: pd.DataFrame, path: Path) -> None:
    columns = [
        "dataset",
        "record_id",
        "harmonized_label",
        "text_unit",
        "analysis_cohort",
        "source_name",
        "publisher_id",
        "source_domain",
        "publication_date",
        "duplicate_group",
        "family_group",
        "split",
        "split_type",
        "group_field",
    ]
    df[columns].to_csv(path, index=False)


def main() -> int:
    ensure_project_dirs()
    paths = load_paths()
    config = load_experiments()
    seed = int(config["random_seed"])
    source_disjoint_repeats = int(config["split"]["source_disjoint_repeats"])
    combined = pd.read_parquet(paths["data_processed"] / "combined_items.parquet")
    included = combined[combined["primary_eligible"]].copy()
    standard_manifests = []
    controlled_manifests = []
    repeated_controlled_manifests = []
    temporal_manifests = []
    temporal_decision_rows = []
    decision_rows = []
    for dataset, df in included.groupby("dataset"):
        standard = grouped_stratified_split(df, seed=seed)
        standard["dataset"] = dataset
        standard_manifests.append(standard)
        controlled, decisions = controlled_split_or_unavailable(df, seed=seed)
        if controlled is not None:
            controlled["dataset"] = dataset
            controlled_manifests.append(controlled)
        repeated = repeated_publisher_disjoint_splits(
            df,
            seed=seed,
            repeats=source_disjoint_repeats,
        )
        if len(repeated):
            repeated_controlled_manifests.append(repeated)
        temporal, temporal_decision = temporal_split_or_unavailable(df)
        if temporal is not None:
            temporal["dataset"] = dataset
            temporal_manifests.append(temporal)
        temporal_decision_rows.append(
            {
                "dataset": dataset,
                "status": temporal_decision.status,
                "split_type": temporal_decision.split_type,
                "group_field": temporal_decision.group_field,
                "excluded_missing_date_records": temporal_decision.excluded_missing_group_records,
                "selected": temporal is not None,
            }
        )
        for order, decision in enumerate(decisions, start=1):
            decision_rows.append(
                {
                    "dataset": dataset,
                    "precedence_order": order,
                    "status": decision.status,
                    "split_type": decision.split_type,
                    "group_field": decision.group_field,
                    "excluded_missing_group_records": decision.excluded_missing_group_records,
                    "selected": bool(
                        controlled is not None
                        and decision.split_type == controlled["split_type"].iloc[0]
                    ),
                }
            )
    standard_all = pd.concat(standard_manifests, ignore_index=True)
    controlled_all = (
        pd.concat(controlled_manifests, ignore_index=True)
        if controlled_manifests
        else standard_all.iloc[0:0].copy()
    )
    repeated_controlled_all = (
        pd.concat(repeated_controlled_manifests, ignore_index=True)
        if repeated_controlled_manifests
        else pd.DataFrame()
    )
    temporal_all = (
        pd.concat(temporal_manifests, ignore_index=True)
        if temporal_manifests
        else standard_all.iloc[0:0].copy()
    )
    masked_all = standard_all.copy()
    masked_all["split_type"] = "artifact_masked_standard_split"
    masked_all["group_field"] = "family_group"
    _write_manifest(standard_all, paths["splits"] / "standard_split_manifest.csv")
    _write_manifest(controlled_all, paths["splits"] / "controlled_split_manifest.csv")
    _write_manifest(masked_all, paths["splits"] / "artifact_masked_split_manifest.csv")
    _write_manifest(temporal_all, paths["splits"] / "temporal_split_manifest.csv")
    repeated_controlled_all.to_csv(
        paths["splits"] / "controlled_split_repeats.csv",
        index=False,
    )
    manifests_for_distribution = [standard_all, masked_all]
    if len(controlled_all):
        manifests_for_distribution.append(controlled_all)
    if len(temporal_all):
        manifests_for_distribution.append(temporal_all)
    distribution = pd.concat([split_distribution(m) for m in manifests_for_distribution], ignore_index=True)
    distribution.to_csv(paths["splits"] / "split_distribution_table.csv", index=False)
    decisions_frame = pd.DataFrame(decision_rows)
    decisions_frame.to_csv(paths["splits"] / "controlled_split_decisions.csv", index=False)
    pd.DataFrame(temporal_decision_rows).to_csv(
        paths["splits"] / "temporal_split_decisions.csv",
        index=False,
    )

    quality_rows = []
    for manifest_name, manifest in [
        ("standard", standard_all),
        ("controlled", controlled_all),
        ("masked", masked_all),
        ("temporal", temporal_all),
    ]:
        for dataset, df in manifest.groupby("dataset"):
            total = len(df)
            for split_name, split in df.groupby("split"):
                group_field = str(split["group_field"].iloc[0])
                group_sizes = split.groupby(group_field).size()
                quality_rows.append(
                    {
                        "manifest": manifest_name,
                        "dataset": dataset,
                        "split_type": split["split_type"].iloc[0],
                        "split": split_name,
                        "records": len(split),
                        "record_fraction": len(split) / total,
                        "unreliable_fraction": split["harmonized_label"].eq("unreliable").mean(),
                        "unique_publishers": split["publisher_id"].replace("", pd.NA).nunique(),
                        "unique_families": split["family_group"].nunique(),
                        "bootstrap_groups": int(len(group_sizes)),
                        "effective_bootstrap_groups": float(
                            group_sizes.sum() ** 2 / (group_sizes.pow(2).sum())
                        ),
                    }
                )
    pd.DataFrame(quality_rows).to_csv(paths["splits"] / "split_quality_table.csv", index=False)

    leakage_rows = []
    for dataset, df in included.groupby("dataset"):
        leakage_rows.append(
            {
                "dataset": dataset,
                "records": len(df),
                "records_with_url": int(df["url"].fillna("").astype(str).str.len().gt(0).sum()),
                "records_with_source_name": int(df["source_name"].fillna("").astype(str).str.len().gt(0).sum()),
                "records_with_publisher_id": int(df["publisher_id"].fillna("").astype(str).str.len().gt(0).sum()),
                "outcome_term_hits_in_model_text": int(df["model_text"].map(count_outcome_terms).sum()),
                "outcome_term_hits_in_masked_text": int(df["model_text_masked"].map(count_outcome_terms).sum()),
                "mask_marker_hits": int(
                    df["model_text_masked"]
                    .fillna("")
                    .str.contains(r"\[(?:SOURCE|ARTIFACT|URL)\]", case=False, regex=True)
                    .sum()
                ),
            }
        )
    pd.DataFrame(leakage_rows).to_csv(paths["splits"] / "leakage_audit_table.csv", index=False)
    masked_spec = """# Masked text specification

`model_text_masked` removes raw URLs, source-name mentions, common platform names,
review-process terms, release-type terms, and explicit label-bearing terms. It
does not insert placeholder tokens because placeholder frequency can itself
encode the removed artifact.

This is a sensitivity input, not the primary text representation. Scraper and interface pages are excluded before masking. Masking cannot prove that all remaining text is artifact-free.
"""
    (paths["splits"] / "masked_text_specification.md").write_text(masked_spec, encoding="utf-8")
    integrity_rows = []
    for manifest_name, manifest in [
        ("standard", standard_all),
        ("controlled", controlled_all),
        ("temporal", temporal_all),
    ]:
        for dataset, dataset_manifest in manifest.groupby("dataset"):
            group_field = str(dataset_manifest["group_field"].iloc[0])
            integrity_rows.append(
                {
                    "manifest": manifest_name,
                    "dataset": dataset,
                    "group_field": group_field,
                    "max_group_split_count": int(
                        dataset_manifest.groupby(group_field)["split"].nunique().max()
                    ),
                    "max_family_split_count": int(
                        dataset_manifest.groupby("family_group")["split"].nunique().max()
                    ),
                    "max_publisher_split_count": (
                        int(dataset_manifest.groupby("publisher_id")["split"].nunique().max())
                        if manifest_name == "controlled"
                        else pd.NA
                    ),
                    "crossing_url_domains": int(
                        (
                            dataset_manifest[
                                dataset_manifest["source_domain"].fillna("").ne("")
                            ]
                            .groupby("source_domain")["split"]
                            .nunique()
                            .gt(1)
                        ).sum()
                    ),
                }
            )
    pd.DataFrame(integrity_rows).to_csv(
        paths["splits"] / "split_integrity_audit.csv",
        index=False,
    )
    selected_notes = []
    for dataset in sorted(included["dataset"].unique()):
        rows = decisions_frame[decisions_frame["dataset"] == dataset]
        selected = rows[rows["selected"]]
        if len(selected):
            selected_notes.append(
                f"- {dataset}: selected `{selected.iloc[0]['split_type']}` after applying the controlled-split precedence."
            )
        else:
            selected_notes.append(f"- {dataset}: no controlled split passed; controlled evaluation is unavailable.")
    audit = (
        "# Leakage and controlled-split audit\n\n"
        + "\n".join(selected_notes)
        + "\n\nFull gate decisions are available in `controlled_split_decisions.csv`. "
        "Outcome-term counts are available in `leakage_audit_table.csv`.\n"
    )
    (paths["splits"] / "leakage_audit.md").write_text(audit, encoding="utf-8")
    print(f"Wrote split manifests to {paths['splits']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
