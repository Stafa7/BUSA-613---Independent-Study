from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def _group_labels(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grouped = (
        df.groupby(group_col)
        .agg(harmonized_label=("harmonized_label", lambda x: x.value_counts().index[0]), dataset=("dataset", "first"))
        .reset_index()
    )
    return grouped


def grouped_stratified_split(
    df: pd.DataFrame,
    group_col: str = "duplicate_group",
    seed: int = 613,
    train_size: float = 0.70,
) -> pd.DataFrame:
    groups = _group_labels(df, group_col)
    train_groups, temp_groups = train_test_split(
        groups,
        train_size=train_size,
        random_state=seed,
        stratify=groups["harmonized_label"],
    )
    val_groups, test_groups = train_test_split(
        temp_groups,
        train_size=0.5,
        random_state=seed,
        stratify=temp_groups["harmonized_label"],
    )
    split_map = {
        **{group: "train" for group in train_groups[group_col]},
        **{group: "validation" for group in val_groups[group_col]},
        **{group: "test" for group in test_groups[group_col]},
    }
    manifest = df[["dataset", "record_id", "harmonized_label", "source_name", "duplicate_group"]].copy()
    manifest["split"] = manifest[group_col].map(split_map)
    manifest["split_type"] = "standard_duplicate_group_stratified"
    manifest["group_field"] = group_col
    return manifest


def source_disjoint_or_fallback(df: pd.DataFrame, seed: int = 613) -> tuple[pd.DataFrame, str]:
    usable = df[df["source_name"].fillna("").astype(str).str.len() > 0].copy()
    if usable["source_name"].nunique() < 10:
        fallback = grouped_stratified_split(df, seed=seed)
        fallback["split_type"] = "fallback_duplicate_group_stratified_source_unavailable"
        return fallback, "source_disjoint_failed_too_few_sources"
    splitter = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
    groups = usable["source_name"]
    train_idx, temp_idx = next(splitter.split(usable, groups=groups))
    train = usable.iloc[train_idx]
    temp = usable.iloc[temp_idx]
    if temp["source_name"].nunique() < 2:
        fallback = grouped_stratified_split(df, seed=seed)
        fallback["split_type"] = "fallback_duplicate_group_stratified_source_temp_too_small"
        return fallback, "source_disjoint_failed_temp_too_small"
    splitter2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=seed)
    val_idx, test_idx = next(splitter2.split(temp, groups=temp["source_name"]))
    split_map = {rid: "train" for rid in train["record_id"]}
    split_map.update({rid: "validation" for rid in temp.iloc[val_idx]["record_id"]})
    split_map.update({rid: "test" for rid in temp.iloc[test_idx]["record_id"]})
    manifest = df[["dataset", "record_id", "harmonized_label", "source_name", "duplicate_group"]].copy()
    manifest["split"] = manifest["record_id"].map(split_map)
    manifest["split"] = manifest["split"].fillna("train")
    manifest["split_type"] = "source_disjoint_group_shuffle"
    manifest["group_field"] = "source_name"
    if not _valid_manifest(manifest):
        fallback = grouped_stratified_split(df, seed=seed)
        fallback["split_type"] = "fallback_duplicate_group_stratified_invalid_source_split"
        return fallback, "source_disjoint_failed_class_coverage"
    return manifest, "source_disjoint_passed"


def _valid_manifest(manifest: pd.DataFrame) -> bool:
    for split in ["train", "validation", "test"]:
        subset = manifest[manifest["split"] == split]
        if subset.empty or subset["harmonized_label"].nunique() < 2:
            return False
    if manifest.groupby(["dataset", "duplicate_group"])["split"].nunique().max() > 1:
        return False
    return True


def split_distribution(manifest: pd.DataFrame) -> pd.DataFrame:
    return (
        manifest.groupby(["dataset", "split_type", "split", "harmonized_label"])
        .size()
        .reset_index(name="records")
    )
