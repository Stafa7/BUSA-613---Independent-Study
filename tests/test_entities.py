import pandas as pd

from health_misinfo.features.entities import assign_family_groups, assign_publisher_ids


def _row(record_id, source_name, url, title, body):
    return {
        "dataset": "fakehealth",
        "record_id": record_id,
        "source_name": source_name,
        "url": url,
        "alternate_urls": "[]",
        "title": title,
        "body_text": body,
        "model_text": f"{title} {body}",
    }


def test_publisher_aliases_merge_without_transitive_domain_collapse():
    frame = pd.DataFrame(
        [
            _row("a", "Health Day", "https://consumer.healthday.com/a", "First title here", "body"),
            _row("b", "HealthDay", "https://healthday.com/b", "Second title here", "body"),
            _row("c", "Reuters Health", "https://reuters.com/c", "Third title here", "body"),
        ]
    )
    result = assign_publisher_ids(frame)
    assert result.loc[0, "publisher_id"] == result.loc[1, "publisher_id"]
    assert result.loc[0, "publisher_id"] != result.loc[2, "publisher_id"]


def test_family_components_connect_across_different_identifiers():
    repeated_body = " ".join(["substantive"] * 35)
    frame = pd.DataFrame(
        [
            _row(
                "a",
                "Publisher A",
                "https://example.com/a",
                "A sufficiently specific shared article title",
                repeated_body,
            ),
            _row(
                "b",
                "Publisher A",
                "https://example.com/b",
                "A sufficiently specific shared article title",
                " ".join(["different"] * 35),
            ),
            _row(
                "c",
                "Publisher B",
                "https://example.net/c",
                "A completely separate and specific article title",
                repeated_body,
            ),
        ]
    )
    result, _ = assign_family_groups(frame, fuzzy_title_similarity=1.0)
    assert result["family_group"].nunique() == 1
    assert set(result["family_size"]) == {3}


def test_coaid_subdomains_share_registrable_publisher():
    frame = pd.DataFrame(
        [
            {
                **_row("a", "wwwnc.cdc.gov", "https://wwwnc.cdc.gov/a", "Specific title one", "body"),
                "dataset": "coaid",
            },
            {
                **_row("b", "emergency.cdc.gov", "https://emergency.cdc.gov/b", "Specific title two", "body"),
                "dataset": "coaid",
            },
        ]
    )
    result = assign_publisher_ids(frame)
    assert result["publisher_id"].nunique() == 1
