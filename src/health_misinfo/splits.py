from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitSettings:
    """Frozen split allocation targets and validity gates."""

    train_size: float = 0.70
    validation_size: float = 0.15
    test_size: float = 0.15
    minimum_controlled_test_records: int = 100
    minimum_controlled_test_records_per_class: int = 30
    minimum_controlled_validation_records: int = 100
    minimum_controlled_validation_records_per_class: int = 20
    minimum_controlled_groups_per_class: int = 5
    maximum_controlled_group_fraction: float = 0.35
    minimum_effective_controlled_groups: float = 5.0
    maximum_controlled_prevalence_shift: float = 0.20
    source_disjoint_attempts: int = 500
    source_disjoint_repeat_attempts: int = 100
    source_disjoint_repeats: int = 10
    minimum_successful_source_disjoint_repeats: int = 5
    hosting_domain_disjoint_attempts: int = 5_000

    def __post_init__(self) -> None:
        fractions = (self.train_size, self.validation_size, self.test_size)
        if any(value <= 0 or value >= 1 for value in fractions):
            raise ValueError("Split sizes must each be between zero and one")
        if not np.isclose(sum(fractions), 1.0):
            raise ValueError("Train, validation, and test sizes must sum to one")
        integer_gates = (
            self.minimum_controlled_test_records,
            self.minimum_controlled_test_records_per_class,
            self.minimum_controlled_validation_records,
            self.minimum_controlled_validation_records_per_class,
            self.minimum_controlled_groups_per_class,
            self.source_disjoint_attempts,
            self.source_disjoint_repeat_attempts,
            self.source_disjoint_repeats,
            self.minimum_successful_source_disjoint_repeats,
            self.hosting_domain_disjoint_attempts,
        )
        if any(value < 1 for value in integer_gates):
            raise ValueError("Split validity counts and attempt counts must be positive")
        if (
            self.minimum_successful_source_disjoint_repeats
            > self.source_disjoint_repeats
        ):
            raise ValueError(
                "minimum_successful_source_disjoint_repeats cannot exceed "
                "source_disjoint_repeats"
            )
        if not 0 < self.maximum_controlled_group_fraction <= 1:
            raise ValueError("maximum_controlled_group_fraction must be in (0, 1]")
        if self.minimum_effective_controlled_groups < 1:
            raise ValueError("minimum_effective_controlled_groups must be at least one")
        if not 0 <= self.maximum_controlled_prevalence_shift <= 1:
            raise ValueError("maximum_controlled_prevalence_shift must be in [0, 1]")

    @property
    def target_fractions(self) -> dict[str, float]:
        return {
            "train": self.train_size,
            "validation": self.validation_size,
            "test": self.test_size,
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "SplitSettings":
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown split configuration keys: {names}")
        float_fields = {
            "train_size",
            "validation_size",
            "test_size",
            "maximum_controlled_group_fraction",
            "minimum_effective_controlled_groups",
            "maximum_controlled_prevalence_shift",
        }
        converted = {
            key: float(value) if key in float_fields else int(value)
            for key, value in values.items()
        }
        return cls(**converted)


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


def _balanced_group_map(
    groups: pd.DataFrame,
    group_col: str,
    seed: int,
    settings: SplitSettings,
) -> dict[str, str]:
    """Assign intact groups while balancing record and class counts."""
    rng = np.random.default_rng(seed)
    assignments: dict[str, str] = {}
    groups = groups.copy()
    groups["_tie"] = rng.random(len(groups))
    groups = groups.sort_values(["records", "_tie"], ascending=[False, True])
    strata = tuple(column for column in groups.columns if column not in {group_col, "records", "_tie"})
    totals = {stratum: int(groups[stratum].sum()) for stratum in strata}
    targets = {
        split: {
            stratum: settings.target_fractions[split] * totals[stratum]
            for stratum in strata
        }
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
    settings: SplitSettings | None = None,
) -> pd.DataFrame:
    settings = settings or SplitSettings()
    groups = _group_table(df, group_col)
    split_map = _balanced_group_map(groups, group_col, seed, settings)
    manifest = _manifest(df, split_map, "standard_family_group_stratified", group_col)
    if not _valid_manifest(manifest, group_col=group_col, settings=settings):
        raise ValueError("Unable to create a valid grouped stratified split")
    return manifest


def _candidate_score(manifest: pd.DataFrame, settings: SplitSettings) -> float:
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
    for split_name, target in settings.target_fractions.items():
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
    settings: SplitSettings | None = None,
) -> bool:
    settings = settings or SplitSettings()
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
        if len(validation) < settings.minimum_controlled_validation_records:
            return False
        if (
            validation["harmonized_label"].value_counts().min()
            < settings.minimum_controlled_validation_records_per_class
        ):
            return False
        test = manifest[manifest["split"] == "test"]
        if len(test) < settings.minimum_controlled_test_records:
            return False
        if (
            test["harmonized_label"].value_counts().min()
            < settings.minimum_controlled_test_records_per_class
        ):
            return False
        for subset in (validation, test):
            if (
                abs(
                    subset["harmonized_label"].eq("unreliable").mean()
                    - overall_positive
                )
                > settings.maximum_controlled_prevalence_shift
            ):
                return False
            group_sizes = subset.groupby(group_col).size()
            if (
                group_sizes.max() / len(subset)
                > settings.maximum_controlled_group_fraction
            ):
                return False
            effective_groups = group_sizes.sum() ** 2 / group_sizes.pow(2).sum()
            if effective_groups < settings.minimum_effective_controlled_groups:
                return False
    return True


def _controlled_group_split(
    df: pd.DataFrame,
    group_col: str,
    split_type: str,
    status_prefix: str,
    seed: int,
    attempts: int,
    settings: SplitSettings,
) -> tuple[pd.DataFrame | None, SplitDecision]:
    group = df[group_col].fillna("").astype(str).str.strip()
    usable = df[group.ne("")].copy()
    missing_records = len(df) - len(usable)
    groups_per_class = usable.groupby("harmonized_label")[group_col].nunique()
    if any(
        groups_per_class.get(label, 0) < settings.minimum_controlled_groups_per_class
        for label in ("reliable", "unreliable")
    ):
        return None, SplitDecision(
            f"{status_prefix}_failed_insufficient_groups_per_class",
            "unavailable",
            group_col,
            missing_records,
        )

    best: pd.DataFrame | None = None
    best_score = float("inf")
    for attempt in range(attempts):
        candidate_seed = seed + attempt * 9_973
        first = GroupShuffleSplit(
            n_splits=1,
            train_size=settings.train_size,
            random_state=candidate_seed,
        )
        train_idx, temp_idx = next(first.split(usable, groups=usable[group_col]))
        train = usable.iloc[train_idx]
        temp = usable.iloc[temp_idx]
        if temp[group_col].nunique() < 2:
            continue
        second = GroupShuffleSplit(
            n_splits=1,
            train_size=(
                settings.validation_size
                / (settings.validation_size + settings.test_size)
            ),
            random_state=candidate_seed + 10_000_000,
        )
        try:
            validation_idx, test_idx = next(second.split(temp, groups=temp[group_col]))
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
        candidate["split_type"] = split_type
        candidate["group_field"] = group_col
        if not _valid_manifest(
            candidate,
            group_col=group_col,
            controlled=True,
            settings=settings,
        ):
            continue
        if candidate.groupby(["dataset", "family_group"])["split"].nunique().max() > 1:
            continue
        score = _candidate_score(candidate, settings)
        if score < best_score:
            best = candidate
            best_score = score

    if best is None:
        return None, SplitDecision(
            f"{status_prefix}_failed_no_valid_assignment",
            "unavailable",
            group_col,
            missing_records,
        )
    return best, SplitDecision(
        f"{status_prefix}_passed",
        split_type,
        group_col,
        missing_records,
    )


def _source_disjoint_split(
    df: pd.DataFrame,
    seed: int,
    attempts: int | None = None,
    settings: SplitSettings | None = None,
) -> tuple[pd.DataFrame | None, SplitDecision]:
    settings = settings or SplitSettings()
    return _controlled_group_split(
        df,
        group_col="publisher_id",
        split_type="canonical_publisher_disjoint",
        status_prefix="canonical_publisher_disjoint",
        seed=seed,
        attempts=attempts or settings.source_disjoint_attempts,
        settings=settings,
    )


def hosting_domain_disjoint_split_or_unavailable(
    df: pd.DataFrame,
    seed: int = 613,
    settings: SplitSettings | None = None,
) -> tuple[pd.DataFrame | None, SplitDecision]:
    """Create a separate hosting-domain-disjoint sensitivity manifest.

    This does not replace or relabel the credited-publisher split. In syndicated
    and press-release data, a hosting domain and a credited publisher represent
    different sources of dependence.
    """
    settings = settings or SplitSettings()
    return _controlled_group_split(
        df,
        group_col="source_domain",
        split_type="hosting_domain_disjoint_sensitivity",
        status_prefix="hosting_domain_disjoint_sensitivity",
        seed=seed,
        attempts=settings.hosting_domain_disjoint_attempts,
        settings=settings,
    )


def repeated_publisher_disjoint_splits(
    df: pd.DataFrame,
    seed: int = 613,
    repeats: int | None = None,
    settings: SplitSettings | None = None,
) -> pd.DataFrame:
    settings = settings or SplitSettings()
    repeats = settings.source_disjoint_repeats if repeats is None else repeats
    manifests: list[pd.DataFrame] = []
    for repeat in range(repeats):
        split_seed = seed + repeat * 1_000_000
        manifest, _ = _source_disjoint_split(
            df,
            split_seed,
            attempts=settings.source_disjoint_repeat_attempts,
            settings=settings,
        )
        if manifest is None:
            continue
        repeated = manifest.copy()
        repeated["repeat_id"] = repeat
        repeated["split_seed"] = split_seed
        manifests.append(repeated)
    minimum_successful_repeats = min(
        repeats,
        settings.minimum_successful_source_disjoint_repeats,
    )
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


def _temporal_split(
    df: pd.DataFrame,
    settings: SplitSettings,
) -> tuple[pd.DataFrame | None, SplitDecision]:
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
        train_low = max(0.05, settings.train_size - 0.10)
        train_high = min(0.90, settings.train_size + 0.10)
        validation_end = settings.train_size + settings.validation_size
        validation_low = max(train_low + 0.05, validation_end - 0.10)
        validation_high = min(0.98, validation_end + 0.10)
        train_cut_candidates = sorted(
            set(
                (
                    np.linspace(train_low, train_high, 11) * len(unique_dates)
                ).astype(int)
            )
        )
        validation_cut_candidates = sorted(
            set(
                (
                    np.linspace(validation_low, validation_high, 11)
                    * len(unique_dates)
                ).astype(int)
            )
        )

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
            if not _valid_manifest(
                candidate,
                group_col="family_group",
                controlled=True,
                settings=settings,
            ):
                continue
            score = (
                _candidate_score(candidate, settings)
                + boundary_exclusions / len(dated)
            )
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
    settings: SplitSettings | None = None,
) -> tuple[pd.DataFrame | None, SplitDecision]:
    return _temporal_split(df, settings or SplitSettings())


def controlled_split_or_unavailable(
    df: pd.DataFrame,
    seed: int = 613,
    settings: SplitSettings | None = None,
) -> tuple[pd.DataFrame | None, list[SplitDecision]]:
    """Apply credited-publisher then temporal controls in the frozen precedence."""
    settings = settings or SplitSettings()
    decisions: list[SplitDecision] = []
    source_manifest, source_decision = _source_disjoint_split(
        df,
        seed,
        settings=settings,
    )
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
    temporal_manifest, temporal_decision = _temporal_split(df, settings)
    decisions.append(temporal_decision)
    return temporal_manifest, decisions


def split_distribution(manifest: pd.DataFrame) -> pd.DataFrame:
    return (
        manifest.groupby(["dataset", "split_type", "split", "harmonized_label"])
        .size()
        .reset_index(name="records")
    )
