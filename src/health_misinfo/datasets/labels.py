from __future__ import annotations

import pandas as pd


REQUIRED_PROCESSED_COLUMNS = [
    "dataset",
    "record_id",
    "source_record_id",
    "source_file",
    "parse_status",
    "text_unit",
    "title",
    "body_text",
    "model_text",
    "native_label",
    "harmonized_label",
    "label_source",
    "source_name",
    "source_name_raw",
    "source_name_normalized",
    "source_domain",
    "publisher_id",
    "url",
    "alternate_urls",
    "publish_date",
    "publication_date",
    "collection_date",
    "content_file_available",
    "body_word_count",
    "latin_letter_fraction",
    "language_audit_flag",
    "unicode_replacement_count",
    "primary_eligible",
    "primary_exclusion_reason",
    "analysis_cohort",
    "engagement_available",
    "tweet_count",
    "reply_count",
    "retweet_count",
    "network_followers_archive_available",
    "network_following_archive_available",
    "record_network_linkage_available",
    "duplicate_group",
    "family_group",
    "family_size",
    "exclusion_reason",
]


def coaid_harmonized_label(file_name: str) -> tuple[str, str]:
    if "Fake" in file_name:
        return "fake", "unreliable"
    if "Real" in file_name:
        return "real", "reliable"
    raise ValueError(f"Cannot infer CoAID label from {file_name}")


def fakehealth_harmonized_label(rating: object) -> tuple[str, str]:
    score = float(rating)
    native = f"rating_{score:g}"
    harmonized = "reliable" if score >= 3 else "unreliable"
    return native, harmonized


def native_label_dictionary() -> pd.DataFrame:
    rows = [
        {
            "dataset": "coaid",
            "native_label": "real",
            "harmonized_label": "reliable",
            "native_source": "CoAID files containing Real in the filename",
            "interpretation_note": "Dataset-defined reliable health information; not independent claim-level verification by this study.",
        },
        {
            "dataset": "coaid",
            "native_label": "fake",
            "harmonized_label": "unreliable",
            "native_source": "CoAID files containing Fake in the filename",
            "interpretation_note": "Dataset-defined unreliable health information; not independent claim-level verification by this study.",
        },
    ]
    for score in range(0, 6):
        rows.append(
            {
                "dataset": "fakehealth",
                "native_label": f"rating_{score}",
                "harmonized_label": "reliable" if score >= 3 else "unreliable",
                "native_source": "FakeHealth / HealthNewsReview expert rating",
                "interpretation_note": "Binary mapping for robustness audit; the original rating is retained.",
            }
        )
    return pd.DataFrame(rows)
