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
