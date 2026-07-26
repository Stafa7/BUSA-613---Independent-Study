from dataclasses import fields, replace

import pandas as pd
import pytest

from health_misinfo.config import PROJECT_ROOT, load_experiments
from health_misinfo.splits import (
    SplitSettings,
    grouped_stratified_split,
    hosting_domain_disjoint_split_or_unavailable,
)


SPLITS = PROJECT_ROOT / "outputs" / "splits"


def _synthetic_split_frame(domain_count: int = 24) -> pd.DataFrame:
    rows = []
    for domain_index in range(domain_count):
        for label in ("reliable", "unreliable"):
            record_id = f"record_{domain_index}_{label}"
            rows.append(
                {
                    "dataset": "fakehealth",
                    "record_id": record_id,
                    "harmonized_label": label,
                    "text_unit": "story",
                    "analysis_cohort": "fakehealth_story",
                    "source_name": "Shared credited publisher",
                    "publisher_id": "publisher_shared",
                    "source_domain": f"host{domain_index}.example",
                    "publication_date": f"2020-01-{domain_index + 1:02d}",
                    "duplicate_group": f"duplicate_{record_id}",
                    "family_group": f"family_{record_id}",
                }
            )
    return pd.DataFrame(rows)


def _small_split_settings(**overrides) -> SplitSettings:
    settings = SplitSettings(
        minimum_controlled_test_records=2,
        minimum_controlled_test_records_per_class=1,
        minimum_controlled_validation_records=2,
        minimum_controlled_validation_records_per_class=1,
        minimum_controlled_groups_per_class=5,
        maximum_controlled_group_fraction=0.50,
        minimum_effective_controlled_groups=1.0,
        maximum_controlled_prevalence_shift=0.01,
        source_disjoint_attempts=50,
        source_disjoint_repeat_attempts=20,
        hosting_domain_disjoint_attempts=50,
    )
    return replace(settings, **overrides)


def test_all_declared_split_settings_are_loaded_from_experiment_config():
    split_config = load_experiments()["split"]
    configured_fields = {field.name for field in fields(SplitSettings)}
    assert configured_fields == set(split_config)
    settings = SplitSettings.from_mapping(split_config)
    assert settings.maximum_controlled_group_fraction == 0.35
    assert settings.minimum_effective_controlled_groups == 5.0


def test_unknown_split_configuration_is_rejected_instead_of_ignored():
    with pytest.raises(ValueError, match="Unknown split configuration keys"):
        SplitSettings.from_mapping({"unwired_gate": 123})


def test_standard_split_uses_configured_target_sizes():
    frame = _synthetic_split_frame()
    settings = _small_split_settings(
        train_size=0.50,
        validation_size=0.25,
        test_size=0.25,
    )
    manifest = grouped_stratified_split(frame, seed=613, settings=settings)
    fractions = manifest["split"].value_counts(normalize=True)
    assert fractions.to_dict() == {
        "train": 0.50,
        "validation": 0.25,
        "test": 0.25,
    }


def test_hosting_domain_sensitivity_is_domain_disjoint_and_separately_named():
    frame = _synthetic_split_frame()
    settings = _small_split_settings(
        train_size=0.50,
        validation_size=0.30,
        test_size=0.20,
    )
    manifest, decision = hosting_domain_disjoint_split_or_unavailable(
        frame,
        seed=613,
        settings=settings,
    )
    assert manifest is not None
    assert decision.status == "hosting_domain_disjoint_sensitivity_passed"
    assert manifest["split_type"].eq("hosting_domain_disjoint_sensitivity").all()
    assert manifest["group_field"].eq("source_domain").all()
    assert manifest.groupby("source_domain")["split"].nunique().max() == 1
    assert manifest.groupby("family_group")["split"].nunique().max() == 1
    assert manifest.groupby("publisher_id")["split"].nunique().max() > 1
    fractions = manifest["split"].value_counts(normalize=True)
    assert fractions["train"] == 0.50
    assert fractions["validation"] == pytest.approx(7 / 24)
    assert fractions["test"] == pytest.approx(5 / 24)


def test_configured_concentration_gate_can_reject_domain_assignment():
    frame = _synthetic_split_frame()
    manifest, decision = hosting_domain_disjoint_split_or_unavailable(
        frame,
        seed=613,
        settings=_small_split_settings(maximum_controlled_group_fraction=0.20),
    )
    assert manifest is None
    assert decision.status.endswith("failed_no_valid_assignment")


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
