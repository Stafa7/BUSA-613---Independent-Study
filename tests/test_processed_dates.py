from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "03_build_processed_tables.py"
)
SPEC = importlib.util.spec_from_file_location("build_processed_tables", SCRIPT_PATH)
assert SPEC and SPEC.loader
BUILD_PROCESSED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD_PROCESSED)


def test_publication_date_normalization_rejects_out_of_bounds_partial_date():
    frame = pd.DataFrame(
        {
            "publish_date": [
                "15-May",
                "2020-05-15",
                "1589500800",
                "1589500800000",
            ]
        }
    )

    normalized = BUILD_PROCESSED._normalize_publication_dates(frame)

    assert normalized.loc[0, "publication_date"] == ""
    assert normalized.loc[1:, "publication_date"].eq("2020-05-15T00:00:00Z").all()
