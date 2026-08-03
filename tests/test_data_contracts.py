from pathlib import Path

import pandas as pd
import pytest

from health_misinfo.config import PROJECT_ROOT
from health_misinfo.datasets.labels import REQUIRED_PROCESSED_COLUMNS


PROCESSED = PROJECT_ROOT / "data" / "processed" / "combined_items.parquet"


@pytest.mark.skipif(not PROCESSED.exists(), reason="processed table not generated yet")
def test_processed_table_contract():
    df = pd.read_parquet(PROCESSED)
    for col in REQUIRED_PROCESSED_COLUMNS:
        assert col in df.columns
    assert df.groupby("dataset")["record_id"].apply(lambda values: values.is_unique).all()
    assert df["harmonized_label"].dropna().isin(["reliable", "unreliable"]).all()


@pytest.mark.skipif(not PROCESSED.exists(), reason="processed table not generated yet")
def test_processed_table_has_model_text_for_included_records():
    df = pd.read_parquet(PROCESSED)
    included = df[df["exclusion_reason"].fillna("") == ""]
    assert included["model_text"].fillna("").str.len().gt(0).all()
    assert not included.duplicated(["dataset", "duplicate_group"]).any()
    assert not included["model_text"].str.contains(
        r"access denied|^\W*403 forbidden\b|log in or sign up to view|javascript is disabled in your browser",
        case=False,
        regex=True,
    ).any()


@pytest.mark.skipif(not PROCESSED.exists(), reason="processed table not generated yet")
def test_exact_duplicate_label_conflicts_are_excluded():
    df = pd.read_parquet(PROCESSED)
    included = df[df["exclusion_reason"].fillna("") == ""]
    conflicts = included.groupby(["dataset", "duplicate_group"])["harmonized_label"].nunique()
    assert conflicts.max() == 1


@pytest.mark.skipif(not PROCESSED.exists(), reason="processed table not generated yet")
def test_primary_cohort_has_substantive_body_and_canonical_publishers():
    df = pd.read_parquet(PROCESSED)
    primary = df[df["primary_eligible"]]
    assert primary["body_word_count"].ge(30).all()
    assert primary["publisher_id"].fillna("").str.len().gt(0).all()
    assert set(primary.loc[primary["dataset"].eq("coaid"), "text_unit"]) == {"article"}
