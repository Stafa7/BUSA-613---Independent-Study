from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


SPLIT_NAMES = ("train", "validation", "test")
TARGET_FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}


@dataclass(frozen=True)
class SplitDecision:
    status: str
    split_type: str
    group_field: str
    excluded_missing_group_records: int = 0


def _manifest(df: pd.DataFrame, split_map: dict[str, str], split_type: str, group_field: str) -> pd.DataFrame:
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
    ]
    manifest = df[columns].copy()
    manifest["split"] = manifest[group_field].map(split_map)
    manifest["split_type"] = split_type
    manifest["group_field"] = group_field
    return manifest


def _group_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    working = df.assign(
        _stratum=df["text_unit"].astype(str) + "__" + df["harmonized_label"].astype(str)
    )
    counts = (
        working.groupby([group_col, "_stratum"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    stratum_columns = [column for column in counts.columns if column != group_col]
    counts["records"] = counts[stratum_columns].sum(axis=1)
    return counts


def _balanced_group_map(groups: pd.DataFrame, group_col: str, seed: int) -> dict[str, str]:
    """Assign intact groups while balancing record and class counts."""
    rng = np.random.default_rng(seed)
    assignments: dict[str, str] = {}
    groups = groups.copy()
    groups["_tie"] = rng.random(len(groups))
    groups = groups.sort_values(["records", "_tie"], ascending=[False, True])
    strata = tuple(column for column in groups.columns if column not in {group_col, "records", "_tie"})
    totals = {stratum: int(groups[stratum].sum()) for stratum in strata}
    targets = {
        split: {stratum: TARGET_FRACTIONS[split] * totals[stratum] for stratum in strata}
        for split in SPLIT_NAMES
    }
    current = {split: {stratum: 0 for stratum in strata} for split in SPLIT_NAMES}
    for row in groups.itertuples(index=False):
        def assignment_cost(split_name: str) -> tuple[float, int]:
            change = 0.0
            for stratum in strata:
                before = abs(current[split_name][stratum] - targets[split_name][stratum])
                after = abs(
                    current[split_name][stratum]
                    + getattr(row, stratum)
                    - targets[split_name][stratum]
                )
                change += after - before
            return change, SPLIT_NAMES.index(split_name)

        best_split = min(SPLIT_NAMES, key=assignment_cost)
        assignments[getattr(row, group_col)] = best_split
        for stratum in strata:
            current[best_split][stratum] += int(getattr(row, stratum))
    return assignments


def grouped_stratified_split(
    df: pd.DataFrame,
    group_col: str = "family_group",
    seed: int = 613,
) -> pd.DataFrame:
    groups = _group_table(df, group_col)
    split_map = _balanced_group_map(groups, group_col, seed)
    manifest = _manifest(df, split_map, "standard_family_group_stratified", group_col)
    if not _valid_manifest(manifest, group_col=group_col):
        raise ValueError("Unable to create a valid grouped stratified split")
    return manifest


def _candidate_score(manifest: pd.DataFrame) -> float:
    total = len(manifest)
    overall_strata = (
        manifest.assign(
            _stratum=manifest["text_unit"].astype(str)
            + "__"
            + manifest["harmonized_label"].astype(str)
        )["_stratum"]
        .value_counts(normalize=True)
    )
    score = 0.0
    for split_name, target in TARGET_FRACTIONS.items():
        split = manifest[manifest["split"] == split_name]
        score += abs(len(split) / total - target)
        split_strata = (
            split.assign(
                _stratum=split["text_unit"].astype(str)
                + "__"
                + split["harmonized_label"].astype(str)
            )["_stratum"]
            .value_counts(normalize=True)
        )
        score += float(
            sum(
                abs(split_strata.get(stratum, 0.0) - proportion)
                for stratum, proportion in overall_strata.items()
            )
        )
    return float(score)


def _valid_manifest(
    manifest: pd.DataFrame,
    group_col: str,
    controlled: bool = False,
    min_test_records: int = 100,
    min_test_per_class: int = 30,
    min_validation_records: int = 100,
    min_validation_per_class: int = 20,
    maximum_controlled_group_fraction: float = 0.80,
    minimum_effective_controlled_groups: float = 2.0,
    maximum_controlled_prevalence_shift: float = 0.20,
) -> bool:
    if manifest.empty or manifest["split"].isna().any():
        return False
    for split_name in SPLIT_NAMES:
        subset = manifest[manifest["split"] == split_name]
        if subset.empty or subset["harmonized_label"].nunique() < 2:
            return False
    all_strata = set(zip(manifest["text_unit"], manifest["harmonized_label"]))
    for split_name in SPLIT_NAMES:
        subset = manifest[manifest["split"] == split_name]
        if set(zip(subset["text_unit"], subset["harmonized_label"])) != all_strata:
            return False
    if manifest.groupby(["dataset", group_col])["split"].nunique().max() > 1:
        return False
    if controlled:
        overall_positive = manifest["harmonized_label"].eq("unreliable").mean()
        validation = manifest[manifest["split"] == "validation"]
        if len(validation) < min_validation_records:
            return False
        if validation["harmonized_label"].value_counts().min() < min_validation_per_class:
            return False
        test = manifest[manifest["split"] == "test"]
        if len(test) < min_test_records:
            return False
        if test["harmonized_label"].value_counts().min() < min_test_per_class:
            return False
        for subset in (validation, test):
            if (
                abs(
                    subset["harmonized_label"].eq("unreliable").mean()
                    - overall_positive
                )
                > maximum_controlled_prevalence_shift
            ):
                return False
            group_sizes = subset.groupby(group_col).size()
            if group_sizes.max() / len(subset) > maximum_controlled_group_fraction:
                return False
            effective_groups = group_sizes.sum() ** 2 / group_sizes.pow(2).sum()
            if effective_groups < minimum_effective_controlled_groups:
                return False
    return True


def _source_disjoint_split(
    df: pd.DataFrame,
    seed: int,
    attempts: int = 500,
) -> tuple[pd.DataFrame | None, SplitDecision]:
    source = df["publisher_id"].fillna("").astype(str).str.strip()
    usable = df[source.ne("")].copy()
    missing_records = len(df) - len(usable)
    sources_per_class = usable.groupby("harmonized_label")["publisher_id"].nunique()
    if any(sources_per_class.get(label, 0) < 5 for label in ("reliable", "unreliable")):
        return None, SplitDecision(
            "source_disjoint_failed_fewer_than_five_sources_per_class",
            "unavailable",
            "publisher_id",
            missing_records,
        )

    best: pd.DataFrame | None = None
    best_score = float("inf")
    for attempt in range(attempts):
        candidate_seed = seed + attempt * 9_973
        first = GroupShuffleSplit(
            n_splits=1,
            train_size=0.70,
            random_state=candidate_seed,
        )
        train_idx, temp_idx = next(first.split(usable, groups=usable["publisher_id"]))
        train = usable.iloc[train_idx]
        temp = usable.iloc[temp_idx]
        if temp["publisher_id"].nunique() < 2:
            continue
        second = GroupShuffleSplit(
            n_splits=1,
            train_size=0.50,
            random_state=candidate_seed + 10_000_000,
        )
        try:
            validation_idx, test_idx = next(second.split(temp, groups=temp["publisher_id"]))
        except ValueError:
            continue
        split_by_record = {rid: "train" for rid in train["record_id"]}
        split_by_record.update({rid: "validation" for rid in temp.iloc[validation_idx]["record_id"]})
        split_by_record.update({rid: "test" for rid in temp.iloc[test_idx]["record_id"]})
        candidate = usable[
            [
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
            ]
        ].copy()
        candidate["split"] = candidate["record_id"].map(split_by_record)
        candidate["split_type"] = "canonical_publisher_disjoint"
        candidate["group_field"] = "publisher_id"
        if not _valid_manifest(candidate, group_col="publisher_id", controlled=True):
            continue
        if candidate.groupby(["dataset", "family_group"])["split"].nunique().max() > 1:
            continue
        score = _candidate_score(candidate)
        if score < best_score:
            best = candidate
            best_score = score

    if best is None:
        return None, SplitDecision(
            "source_disjoint_failed_no_valid_assignment",
            "unavailable",
            "publisher_id",
            missing_records,
        )
    return best, SplitDecision(
        "canonical_publisher_disjoint_passed",
        "canonical_publisher_disjoint",
        "publisher_id",
        missing_records,
    )


def repeated_publisher_disjoint_splits(
    df: pd.DataFrame,
    seed: int = 613,
    repeats: int = 10,
) -> pd.DataFrame:
    manifests: list[pd.DataFrame] = []
    for repeat in range(repeats):
        split_seed = seed + repeat * 1_000_000
        manifest, _ = _source_disjoint_split(df, split_seed, attempts=100)
        if manifest is None:
            continue
        repeated = manifest.copy()
        repeated["repeat_id"] = repeat
        repeated["split_seed"] = split_seed
        manifests.append(repeated)
    minimum_successful_repeats = min(repeats, max(5, repeats // 2))
    if len(manifests) < minimum_successful_repeats:
        return pd.DataFrame()
    return pd.concat(manifests, ignore_index=True)


def _parse_study_dates(df: pd.DataFrame) -> pd.Series:
    published = pd.to_datetime(df["publication_date"], errors="coerce", utc=True)
    collection = pd.to_datetime(
        df["collection_date"],
        format="%m-%d-%Y",
        errors="coerce",
        utc=True,
    )
    return published.fillna(collection)


def _temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame | None, SplitDecision]:
    dated = df.copy()
    dated["_study_date"] = _parse_study_dates(dated)
    missing = int(dated["_study_date"].isna().sum())
    dated = dated[dated["_study_date"].notna()].copy()
    if dated["_study_date"].nunique() < 3:
        return None, SplitDecision("temporal_failed_too_few_dates", "unavailable", "family_group", missing)

    family_dates = (
        dated.groupby("family_group")
        .agg(
            minimum_date=("_study_date", "min"),
            maximum_date=("_study_date", "max"),
            records=("record_id", "size"),
        )
        .sort_values("maximum_date")
    )
    unique_dates = np.array(sorted(dated["_study_date"].unique()))
    if len(unique_dates) <= 100:
        train_cut_candidates = range(1, len(unique_dates) - 1)
        validation_cut_candidates = range(2, len(unique_dates))
    else:
        train_cut_candidates = sorted(set((np.linspace(0.60, 0.80, 11) * len(unique_dates)).astype(int)))
        validation_cut_candidates = sorted(set((np.linspace(0.75, 0.95, 11) * len(unique_dates)).astype(int)))

    best: pd.DataFrame | None = None
    best_score = float("inf")
    best_boundary_exclusions = 0
    for train_cut in train_cut_candidates:
        for validation_cut in validation_cut_candidates:
            if train_cut >= validation_cut or validation_cut >= len(unique_dates):
                continue
            train_end = unique_dates[train_cut - 1]
            validation_end = unique_dates[validation_cut - 1]
            split_map = {}
            for family, row in family_dates.iterrows():
                if row.maximum_date <= train_end:
                    split_map[family] = "train"
                elif row.minimum_date > train_end and row.maximum_date <= validation_end:
                    split_map[family] = "validation"
                elif row.minimum_date > validation_end:
                    split_map[family] = "test"
            eligible = dated[dated["family_group"].isin(split_map)].copy()
            boundary_exclusions = len(dated) - len(eligible)
            candidate = _manifest(
                eligible,
                split_map,
                "temporal_family_disjoint",
                "family_group",
            )
            if not _valid_manifest(candidate, group_col="family_group", controlled=True):
                continue
            score = _candidate_score(candidate) + boundary_exclusions / len(dated)
            if score < best_score:
                best = candidate
                best_score = score
                best_boundary_exclusions = boundary_exclusions

    if best is None:
        return None, SplitDecision("temporal_failed_no_valid_assignment", "unavailable", "family_group", missing)
    return best, SplitDecision(
        "temporal_passed",
        "temporal_family_disjoint",
        "family_group",
        missing + best_boundary_exclusions,
    )


def temporal_split_or_unavailable(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame | None, SplitDecision]:
    return _temporal_split(df)


def controlled_split_or_unavailable(df: pd.DataFrame, seed: int = 613) -> tuple[pd.DataFrame | None, list[SplitDecision]]:
    """Apply source then temporal controls; topic-held-out remains a separately gated analysis."""
    decisions: list[SplitDecision] = []
    source_manifest, source_decision = _source_disjoint_split(df, seed)
    decisions.append(source_decision)
    if source_manifest is not None:
        return source_manifest, decisions

    decisions.append(
        SplitDecision(
            "topic_held_out_not_attempted_no_frozen_topic_groups",
            "unavailable",
            "topic_group",
        )
    )
    temporal_manifest, temporal_decision = _temporal_split(df)
    decisions.append(temporal_decision)
    return temporal_manifest, decisions


def split_distribution(manifest: pd.DataFrame) -> pd.DataFrame:
    return (
        manifest.groupby(["dataset", "split_type", "split", "harmonized_label"])
        .size()
        .reset_index(name="records")
    )
