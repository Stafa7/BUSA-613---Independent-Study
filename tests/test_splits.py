from pathlib import Path

import pandas as pd
import pytest

from health_misinfo.config import PROJECT_ROOT


SPLITS = PROJECT_ROOT / "outputs" / "splits"


@pytest.mark.skipif(not (SPLITS / "standard_split_manifest.csv").exists(), reason="split manifests not generated yet")
def test_standard_split_has_no_record_or_family_overlap():
    manifest = pd.read_csv(SPLITS / "standard_split_manifest.csv")
    assert manifest.groupby(["dataset", "record_id"])["split"].nunique().max() == 1
    assert manifest.groupby(["dataset", "family_group"])["split"].nunique().max() == 1
    assert set(manifest["split"]) == {"train", "validation", "test"}


@pytest.mark.skipif(not (SPLITS / "standard_split_manifest.csv").exists(), reason="split manifests not generated yet")
def test_standard_split_balances_record_counts_and_labels():
    manifest = pd.read_csv(SPLITS / "standard_split_manifest.csv")
    targets = {"train": 0.70, "validation": 0.15, "test": 0.15}
    for _, dataset in manifest.groupby("dataset"):
        overall_positive = dataset["harmonized_label"].eq("unreliable").mean()
        for split_name, target in targets.items():
            split = dataset[dataset["split"] == split_name]
            assert abs(len(split) / len(dataset) - target) <= 0.02
            assert abs(split["harmonized_label"].eq("unreliable").mean() - overall_positive) <= 0.02
            assert set(zip(split["text_unit"], split["harmonized_label"])) == set(
                zip(dataset["text_unit"], dataset["harmonized_label"])
            )


@pytest.mark.skipif(not (SPLITS / "controlled_split_manifest.csv").exists(), reason="controlled manifest not generated yet")
def test_controlled_split_enforces_declared_group_and_minimum_test_size():
    manifest = pd.read_csv(SPLITS / "controlled_split_manifest.csv")
    for _, dataset in manifest.groupby("dataset"):
        group_field = dataset["group_field"].iloc[0]
        assert group_field in {"publisher_id", "family_group"}
        if group_field == "publisher_id":
            assert dataset["split_type"].eq("canonical_publisher_disjoint").all()
        else:
            assert dataset["split_type"].eq("temporal_family_disjoint").all()
        assert dataset.groupby(group_field)["split"].nunique().max() == 1
        assert dataset.groupby("family_group")["split"].nunique().max() == 1
        test = dataset[dataset["split"] == "test"]
        assert len(test) >= 100
        assert test["harmonized_label"].value_counts().min() >= 30


@pytest.mark.skipif(not (SPLITS / "controlled_split_repeats.csv").exists(), reason="repeat manifests not generated yet")
def test_controlled_repeats_have_distinct_assignments():
    manifest = pd.read_csv(SPLITS / "controlled_split_repeats.csv")
    for _, dataset in manifest.groupby("dataset"):
        assignments = dataset.pivot(
            index="record_id",
            columns="repeat_id",
            values="split",
        )
        signatures = {tuple(assignments[column].fillna("")) for column in assignments}
        assert len(signatures) == dataset["repeat_id"].nunique()


@pytest.mark.skipif(not (SPLITS / "temporal_split_manifest.csv").exists(), reason="temporal manifest not generated yet")
def test_temporal_split_is_chronological_and_family_safe():
    manifest = pd.read_csv(SPLITS / "temporal_split_manifest.csv")
    for _, dataset in manifest.groupby("dataset"):
        dates = pd.to_datetime(dataset["publication_date"], utc=True)
        assert dates[dataset["split"].eq("train")].max() <= dates[dataset["split"].eq("validation")].min()
        assert dates[dataset["split"].eq("validation")].max() <= dates[dataset["split"].eq("test")].min()
        assert dataset.groupby("family_group")["split"].nunique().max() == 1
