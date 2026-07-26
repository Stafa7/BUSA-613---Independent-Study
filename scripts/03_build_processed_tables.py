#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from health_misinfo.config import load_experiments, load_paths
from health_misinfo.datasets.coaid import load_coaid_items
from health_misinfo.datasets.fakehealth import load_fakehealth_items
from health_misinfo.datasets.labels import REQUIRED_PROCESSED_COLUMNS
from health_misinfo.features.entities import assign_family_groups, assign_publisher_ids
from health_misinfo.paths import ensure_project_dirs


def _blank_fraction(series: pd.Series) -> float:
    missing = series.isna()
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        missing = missing | series.fillna("").astype(str).str.strip().eq("")
    return float(missing.mean())


def _apply_quality_rules(combined: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic exclusions while retaining every parsed row for audit."""
    result = combined.copy()
    result["exclusion_reason"] = result["exclusion_reason"].fillna("").astype(str)

    eligible = result["exclusion_reason"].eq("")
    exact_conflicts = (
        result[eligible]
        .groupby(["dataset", "duplicate_group"])["harmonized_label"]
        .nunique()
    )
    exact_conflict_keys = set(exact_conflicts[exact_conflicts > 1].index)
    exact_conflict_mask = pd.Series(
        [
            eligible.iloc[i] and (row.dataset, row.duplicate_group) in exact_conflict_keys
            for i, row in enumerate(result.itertuples(index=False))
        ],
        index=result.index,
    )
    result.loc[exact_conflict_mask, "exclusion_reason"] = "conflicting_exact_duplicate_labels"

    eligible = result["exclusion_reason"].eq("")
    duplicate_rank = (
        result[eligible]
        .sort_values(["dataset", "duplicate_group", "record_id"])
        .groupby(["dataset", "duplicate_group"])
        .cumcount()
    )
    duplicate_indices = duplicate_rank[duplicate_rank > 0].index
    result.loc[duplicate_indices, "exclusion_reason"] = "exact_duplicate"
    return result


def _normalize_publication_dates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    raw = result["publish_date"].fillna("").astype(str).str.strip()
    numeric = pd.to_numeric(raw, errors="coerce")
    normalized = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns, UTC]")

    def nanosecond_safe(values: pd.Series) -> pd.Series:
        """Discard parsed dates outside pandas' nanosecond timestamp range."""

        years = values.dt.year
        safe = values.where(years.between(1678, 2261))
        return safe.astype("datetime64[ns, UTC]")

    seconds = numeric.between(100_000_000, 20_000_000_000, inclusive="both")
    milliseconds = numeric.between(100_000_000_000, 20_000_000_000_000, inclusive="both")
    normalized.loc[seconds] = nanosecond_safe(
        pd.to_datetime(numeric.loc[seconds], unit="s", errors="coerce", utc=True)
    )
    normalized.loc[milliseconds] = nanosecond_safe(
        pd.to_datetime(
            numeric.loc[milliseconds],
            unit="ms",
            errors="coerce",
            utc=True,
        )
    )
    textual = ~(seconds | milliseconds) & raw.ne("")
    textual_values = raw.loc[textual].str.replace(r"\bat\b", "", regex=True)
    timezone_offsets = {
        "PDT": "-0700",
        "PST": "-0800",
        "EDT": "-0400",
        "EST": "-0500",
        "CDT": "-0500",
        "CST": "-0600",
    }
    for abbreviation, offset in timezone_offsets.items():
        textual_values = textual_values.str.replace(
            rf"\b{abbreviation}\b",
            offset,
            regex=True,
        )
    normalized.loc[textual] = nanosecond_safe(
        pd.to_datetime(
            textual_values,
            format="mixed",
            errors="coerce",
            utc=True,
        )
    )
    result["publication_date"] = normalized.dt.strftime("%Y-%m-%dT%H:%M:%SZ").fillna("")
    return result


def _assign_primary_cohorts(
    frame: pd.DataFrame,
    minimum_body_words: int,
    coaid_units: set[str],
    fakehealth_units: set[str],
) -> pd.DataFrame:
    result = frame.copy()
    result["body_word_count"] = (
        result["body_text"].fillna("").astype(str).str.findall(r"\b[\w'-]+\b").str.len()
    )
    text = result["model_text"].fillna("").astype(str)
    alphabetic = text.map(lambda value: sum(character.isalpha() for character in value))
    latin = text.map(
        lambda value: sum(
            ("a" <= character.lower() <= "z") for character in value if character.isalpha()
        )
    )
    result["latin_letter_fraction"] = (
        latin / alphabetic.where(alphabetic.gt(0), 1)
    ).astype(float)
    result["language_audit_flag"] = (
        alphabetic.ge(50) & result["latin_letter_fraction"].lt(0.80)
    )
    result["unicode_replacement_count"] = text.str.count("\ufffd")
    result["analysis_cohort"] = result["dataset"] + "_" + result["text_unit"]
    result["primary_exclusion_reason"] = ""

    quality_excluded = result["exclusion_reason"].fillna("").ne("")
    result.loc[quality_excluded, "primary_exclusion_reason"] = result.loc[
        quality_excluded,
        "exclusion_reason",
    ]
    allowed_unit = (
        (result["dataset"].eq("coaid") & result["text_unit"].isin(coaid_units))
        | (result["dataset"].eq("fakehealth") & result["text_unit"].isin(fakehealth_units))
    )
    result.loc[
        result["primary_exclusion_reason"].eq("") & ~allowed_unit,
        "primary_exclusion_reason",
    ] = "non_primary_text_unit"
    result.loc[
        result["primary_exclusion_reason"].eq("")
        & result["body_word_count"].lt(minimum_body_words),
        "primary_exclusion_reason",
    ] = "insufficient_body_text"
    result["primary_eligible"] = result["primary_exclusion_reason"].eq("")
    return result


def _write_profile(
    combined: pd.DataFrame,
    profiles: Path,
    family_audit: pd.DataFrame,
    minimum_body_words: int,
) -> None:
    data_dictionary = pd.DataFrame(
        [
            {
                "field": col,
                "dtype": str(combined[col].dtype),
                "non_missing": int(round(len(combined) * (1 - _blank_fraction(combined[col])))),
            }
            for col in combined.columns
        ]
    )
    data_dictionary.to_csv(profiles / "data_dictionary.csv", index=False)
    missingness_rows = []
    for dataset, df in combined.groupby("dataset"):
        for field in combined.columns:
            missingness_rows.append(
                {
                    "dataset": dataset,
                    "field": field,
                    "missing_fraction": _blank_fraction(df[field]),
                    "missing_records": int(round(len(df) * _blank_fraction(df[field]))),
                }
            )
    missingness = pd.DataFrame(missingness_rows)
    missingness.to_csv(profiles / "missingness_report.csv", index=False)
    (
        combined[combined["parse_status"].ne("parsed")][
            [
                "dataset",
                "record_id",
                "source_record_id",
                "source_file",
                "parse_status",
                "content_file_available",
            ]
        ]
        .sort_values(["dataset", "parse_status", "source_record_id"])
        .to_csv(profiles / "parse_issue_ledger.csv", index=False)
    )
    exact_duplicates = (
        combined.groupby(["dataset", "duplicate_group"])
        .size()
        .reset_index(name="records")
        .query("records > 1")
        .sort_values(["dataset", "records"], ascending=[True, False])
    )
    exact_duplicates["group_type"] = "exact_text"
    family_duplicates = (
        combined.groupby(["dataset", "family_group"])
        .size()
        .reset_index(name="records")
        .query("records > 1")
        .sort_values(["dataset", "records"], ascending=[True, False])
        .rename(columns={"family_group": "duplicate_group"})
    )
    family_duplicates["group_type"] = "parent_or_near_duplicate_family"
    duplicates = pd.concat([exact_duplicates, family_duplicates], ignore_index=True)
    duplicates.to_csv(profiles / "duplicate_report.csv", index=False)
    primary_family_rows = []
    for dataset, dataset_frame in combined[combined["primary_eligible"]].groupby("dataset"):
        sizes = dataset_frame["family_group"].value_counts()
        primary_family_rows.append(
            {
                "dataset": dataset,
                "primary_family_groups": int(len(sizes)),
                "primary_multi_record_families": int((sizes > 1).sum()),
                "largest_primary_family": int(sizes.max()) if len(sizes) else 0,
            }
        )
    family_audit.merge(
        pd.DataFrame(primary_family_rows),
        on="dataset",
        how="left",
    ).to_csv(profiles / "family_construction_audit.csv", index=False)

    flow = (
        combined.assign(
            analysis_status=combined["exclusion_reason"].replace("", "included")
        )
        .groupby(["dataset", "analysis_status"])
        .size()
        .reset_index(name="records")
        .sort_values(["dataset", "analysis_status"])
    )
    flow.to_csv(profiles / "inclusion_exclusion_flow.csv", index=False)
    (
        combined.assign(
            primary_analysis_status=combined["primary_exclusion_reason"].replace("", "included")
        )
        .groupby(["dataset", "primary_analysis_status"])
        .size()
        .reset_index(name="records")
        .sort_values(["dataset", "primary_analysis_status"])
        .to_csv(profiles / "primary_inclusion_exclusion_flow.csv", index=False)
    )

    (
        combined[combined["primary_eligible"]]
        .groupby(["dataset", "text_unit", "harmonized_label"])
        .size()
        .reset_index(name="records")
        .to_csv(profiles / "unit_label_distribution.csv", index=False)
    )
    (
        combined.groupby(["dataset", "source_name_raw", "source_name_normalized", "source_domain", "publisher_id"])
        .size()
        .reset_index(name="records")
        .sort_values(["dataset", "records"], ascending=[True, False])
        .to_csv(profiles / "publisher_canonicalization_audit.csv", index=False)
    )
    (
        combined.assign(has_publication_date=combined["publication_date"].fillna("").ne(""))
        .groupby("dataset")
        .agg(
            parsed_records=("record_id", "size"),
            records_with_publication_date=("has_publication_date", "sum"),
            unique_publication_dates=("publication_date", lambda values: values[values.ne("")].nunique()),
        )
        .reset_index()
        .to_csv(profiles / "publication_date_audit.csv", index=False)
    )
    (
        combined.groupby(["dataset", "text_unit"])
        .agg(
            records=("record_id", "size"),
            content_files_available=("content_file_available", "sum"),
            median_body_words=("body_word_count", "median"),
            primary_eligible=("primary_eligible", "sum"),
        )
        .reset_index()
        .to_csv(profiles / "content_quality_audit.csv", index=False)
    )
    (
        combined.groupby("dataset")
        .agg(
            records=("record_id", "size"),
            language_audit_flags=("language_audit_flag", "sum"),
            minimum_latin_letter_fraction=("latin_letter_fraction", "min"),
            unicode_replacement_characters=("unicode_replacement_count", "sum"),
        )
        .reset_index()
        .to_csv(profiles / "language_unicode_audit.csv", index=False)
    )
    (
        combined[combined["exclusion_reason"].eq("")]
        .groupby(["dataset", "native_label", "harmonized_label"])
        .size()
        .reset_index(name="records")
        .to_csv(profiles / "native_label_distribution.csv", index=False)
    )

    feasibility_rows = []
    for dataset, df in combined.groupby("dataset"):
        included = df[df["primary_eligible"]]
        available = included[included["engagement_available"]]
        label_counts = available["harmonized_label"].value_counts()
        coverage = len(available) / len(included) if len(included) else 0
        coverage_by_label = (
            included.groupby("harmonized_label")["engagement_available"].mean()
            if len(included)
            else pd.Series(dtype=float)
        )
        coverage_gap = (
            float(coverage_by_label.max() - coverage_by_label.min())
            if len(coverage_by_label) > 1
            else 1.0
        )
        proxy_risk = coverage_gap > 0.20
        gate = (
            coverage >= 0.60
            and label_counts.get("reliable", 0) >= 100
            and label_counts.get("unreliable", 0) >= 100
            and (included[["tweet_count", "reply_count", "retweet_count"]].sum(axis=1).nunique() > 1)
            and not proxy_risk
        )
        feasibility_rows.append(
            {
                "dataset": dataset,
                "included_records": len(included),
                "records_with_any_engagement_ids": len(available),
                "engagement_coverage": coverage,
                "reliable_with_engagement": int(label_counts.get("reliable", 0)),
                "unreliable_with_engagement": int(label_counts.get("unreliable", 0)),
                "reliable_engagement_coverage": float(coverage_by_label.get("reliable", 0)),
                "unreliable_engagement_coverage": float(coverage_by_label.get("unreliable", 0)),
                "coverage_gap": coverage_gap,
                "missingness_proxy_risk": proxy_risk,
                "predictive_engagement_gate": "pass" if gate else "fail",
            }
        )
    pd.DataFrame(feasibility_rows).to_csv(profiles / "engagement_feasibility.csv", index=False)
    included_count = int(combined["primary_eligible"].sum())
    excluded_count = len(combined) - included_count
    summary = [
        "# Processed data profile",
        "",
        f"- Combined records: {len(combined):,}",
        f"- Included primary analytical records: {included_count:,}",
        f"- Excluded from the primary analysis: {excluded_count:,}",
        f"- Minimum primary body length: {minimum_body_words:,} words",
        f"- Exact duplicate groups with >1 parsed record: {len(exact_duplicates):,}",
        f"- Parent or near-duplicate families with >1 parsed record: {len(family_duplicates):,}",
        "",
        "Empty strings are treated as missing. Inclusion and exclusion counts are available in `inclusion_exclusion_flow.csv`.",
        "",
        "Labels are dataset-defined reliability mappings, not verified claim-level truth.",
    ]
    (profiles / "profile_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def main() -> int:
    ensure_project_dirs()
    paths = load_paths()
    experiments = load_experiments()
    quality = experiments["data_quality"]
    minimum_body_words = int(quality["minimum_primary_body_words"])
    processed = paths["data_processed"]
    coaid = load_coaid_items(paths["data_raw"] / "coaid")
    fakehealth = load_fakehealth_items(paths["data_raw"] / "fakehealth", paths["data_raw"])
    combined = pd.concat([coaid, fakehealth], ignore_index=True)
    combined = assign_publisher_ids(combined)
    combined, family_audit = assign_family_groups(
        combined,
        fuzzy_title_similarity=float(quality["fuzzy_title_similarity"]),
    )
    combined = _normalize_publication_dates(combined)
    for col in REQUIRED_PROCESSED_COLUMNS:
        if col not in combined.columns:
            combined[col] = ""
    combined = _apply_quality_rules(combined)
    combined = _assign_primary_cohorts(
        combined,
        minimum_body_words=minimum_body_words,
        coaid_units=set(quality["coaid_primary_units"]),
        fakehealth_units=set(quality["fakehealth_primary_units"]),
    )
    coaid = combined[combined["dataset"] == "coaid"].copy()
    fakehealth = combined[combined["dataset"] == "fakehealth"].copy()
    coaid.to_parquet(processed / "coaid_items.parquet", index=False)
    fakehealth.to_parquet(processed / "fakehealth_items.parquet", index=False)
    combined.to_parquet(processed / "combined_items.parquet", index=False)
    _write_profile(
        combined,
        paths["profiles"],
        family_audit,
        minimum_body_words,
    )
    print(f"Wrote processed tables to {processed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
