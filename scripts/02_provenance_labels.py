#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from health_misinfo.config import load_paths
from health_misinfo.datasets.labels import native_label_dictionary
from health_misinfo.paths import ensure_project_dirs


def main() -> int:
    ensure_project_dirs()
    out = load_paths()["provenance"]
    labels = native_label_dictionary()
    labels.to_csv(out / "native_labels.csv", index=False)
    labels[["dataset", "native_label", "harmonized_label", "interpretation_note"]].to_csv(
        out / "harmonized_labels.csv", index=False
    )
    provenance = pd.DataFrame(
        [
            {
                "dataset": "coaid",
                "creator": "Cui and Lee",
                "scope": "COVID-19 healthcare news and claims",
                "text_unit": "article or claim",
                "label_construction": "Native Real/Fake groupings from CoAID files",
                "study_interpretation": "dataset-defined reliability label; not new claim-level verification",
            },
            {
                "dataset": "fakehealth",
                "creator": "Dai, Sun, and Wang",
                "scope": "HealthStory and HealthRelease reviewed health reporting",
                "text_unit": "article or news release",
                "label_construction": "HealthNewsReview expert rating mapped to binary reliability",
                "study_interpretation": "dataset-defined quality/reliability proxy; original rating retained",
            },
        ]
    )
    provenance.to_csv(out / "dataset_provenance.csv", index=False)
    flow = pd.DataFrame(
        [
            {"dataset": "coaid", "stage": "raw_items", "rule": "All article/claim rows loaded from native files"},
            {"dataset": "coaid", "stage": "included", "rule": "Keep rows with non-empty model_text and native Real/Fake label"},
            {"dataset": "fakehealth", "stage": "raw_items", "rule": "All HealthStory/HealthRelease review records with news_id"},
            {"dataset": "fakehealth", "stage": "included", "rule": "Keep rows with article text/title and numeric expert rating"},
        ]
    )
    flow.to_csv(out / "inclusion_exclusion_flow.csv", index=False)
    memo = """# Cross-dataset compatibility memo

Status: provisionally compatible for Phase 0–6 text-model comparison, with limitations.

- Comparable construct: binary dataset-defined reliable/unreliable mapping is available in both datasets.
- Comparable text unit: article/claim text in CoAID and article/release text in FakeHealth can be represented as `model_text`.
- Label provenance differs: CoAID Real/Fake groupings and FakeHealth expert ratings are not identical constructs.
- Outcome-bearing review text is excluded from FakeHealth model input.
- Numerical transfer remains conditional on later review; this first-half milestone focuses on within-dataset evaluation.

Interpretation rule: do not state that either harmonized label is verified claim-level misinformation truth.
"""
    (out / "transfer_compatibility_memo.md").write_text(memo, encoding="utf-8")
    print(f"Wrote provenance outputs to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

