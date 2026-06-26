from pathlib import Path

import pandas as pd
import pytest

from health_misinfo.config import PROJECT_ROOT


SPLITS = PROJECT_ROOT / "outputs" / "splits"
PROCESSED = PROJECT_ROOT / "data" / "processed" / "combined_items.parquet"


@pytest.mark.skipif(not (SPLITS / "standard_split_manifest.csv").exists(), reason="split manifests not generated yet")
def test_standard_split_has_no_record_overlap():
    manifest = pd.read_csv(SPLITS / "standard_split_manifest.csv")
    counts = manifest.groupby("record_id")["split"].nunique()
    assert counts.max() == 1
    assert set(manifest["split"]) == {"train", "validation", "test"}


@pytest.mark.skipif(
    not (SPLITS / "standard_split_manifest.csv").exists() or not PROCESSED.exists(),
    reason="split manifests or processed table not generated yet",
)
def test_duplicate_groups_do_not_cross_standard_splits():
    manifest = pd.read_csv(SPLITS / "standard_split_manifest.csv")
    processed = pd.read_parquet(PROCESSED)[["record_id", "duplicate_group"]]
    joined = manifest.merge(processed, on="record_id", how="left", suffixes=("", "_processed"))
    duplicate_col = "duplicate_group_processed" if "duplicate_group_processed" in joined.columns else "duplicate_group"
    cross = joined.groupby(["dataset", duplicate_col])["split"].nunique()
    assert cross.max() == 1

