from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "09_summarize_study.py"
SPEC = importlib.util.spec_from_file_location("summarize_study", SCRIPT_PATH)
assert SPEC and SPEC.loader
SUMMARIZE_STUDY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARIZE_STUDY)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": ["a", "b", "c", "d"],
            "harmonized_label": ["reliable", "unreliable", "reliable", "unreliable"],
            "prediction": ["reliable", "unreliable", "reliable", "unreliable"],
            "score_unreliable": [0.1, 0.9, 0.2, 0.8],
            "bootstrap_group": ["a", "b", "c", "d"],
        }
    )


def test_paired_difference_requires_identical_record_rosters():
    left = _predictions()
    right = left.iloc[:-1].copy()

    with pytest.raises(ValueError, match="identical record IDs"):
        SUMMARIZE_STUDY._paired_difference(left, right, iterations=20, seed=613)


def test_paired_difference_records_observed_metric_change():
    left = _predictions()
    right = left.copy()
    right.loc[right["record_id"].eq("b"), "prediction"] = "reliable"

    result = SUMMARIZE_STUDY._paired_difference(
        left,
        right,
        iterations=20,
        seed=613,
    )
    macro = result[result["metric"].eq("macro_f1")].iloc[0]

    assert macro["observed_difference_left_minus_right"] > 0
    assert macro["resampling_unit"] == "group"


def test_equivalence_summary_requires_every_available_run():
    results = pd.DataFrame(
        {
            "dataset": ["coaid", "coaid"],
            "model": ["distilbert", "distilbert"],
            "seed": [613, 614],
            "metric": ["macro_f1", "macro_f1"],
            "observed_difference_left_minus_right": [0.0, 0.01],
            "equivalence_margin": [0.03, 0.03],
            "supports_practical_equivalence": [True, False],
        }
    )

    summary = SUMMARIZE_STUDY._equivalence_summary(results).iloc[0]

    assert summary["runs"] == 2
    assert summary["runs_supporting_practical_equivalence"] == 1
    assert not summary["all_runs_support_practical_equivalence"]
    assert "seed-specific for transformers" in summary["decision_rule"]
