from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from health_misinfo.datasets.labels import fakehealth_harmonized_label
from health_misinfo.features.engagement import optional_user_network_available
from health_misinfo.features.text import clean_text, domain_from_url, join_text, mask_artifacts, normalized_text_hash, stable_id


DATASET_PARTS = {
    "HealthStory": "story",
    "HealthRelease": "news_release",
}


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _content_path(raw_dir: Path, part: str, news_id: str) -> Path:
    return raw_dir / "dataset" / "content" / part / f"{news_id}.json"


def _load_content(raw_dir: Path, part: str, news_id: str) -> dict[str, object]:
    path = _content_path(raw_dir, part, news_id)
    if not path.exists():
        return {}
    return _load_json(path)


def load_fakehealth_items(raw_dir: Path, raw_root: Path) -> pd.DataFrame:
    followers_available, following_available = optional_user_network_available(raw_root)
    rows: list[dict[str, object]] = []
    for part, text_unit in DATASET_PARTS.items():
        reviews = _load_json(raw_dir / "dataset" / "reviews" / f"{part}.json")
        engagements = _load_json(raw_dir / "dataset" / "engagements" / f"{part}.json")
        for review in reviews:
            news_id = clean_text(review.get("news_id"))
            if not news_id:
                continue
            content = _load_content(raw_dir, part, news_id)
            native_label, harmonized_label = fakehealth_harmonized_label(review.get("rating"))
            title = clean_text(review.get("original_title")) or clean_text(content.get("title")) or clean_text(review.get("title"))
            body = clean_text(content.get("text"))
            model_text = join_text(title, body)
            url = clean_text(content.get("url")) or clean_text(review.get("source_link"))
            source_name = clean_text(review.get("news_source")) or clean_text(content.get("source")) or domain_from_url(url)
            engagement = engagements.get(news_id, {})
            tweet_count = len(engagement.get("tweets", []) or [])
            reply_count = len(engagement.get("replies", []) or [])
            retweet_count = len(engagement.get("retweets", []) or [])
            rows.append(
                {
                    "dataset": "fakehealth",
                    "record_id": stable_id("fakehealth", part, news_id, prefix="fh"),
                    "source_record_id": f"{part}:{news_id}",
                    "text_unit": text_unit,
                    "title": title,
                    "body_text": body,
                    "model_text": model_text,
                    "model_text_masked": mask_artifacts(model_text, source_name),
                    "native_label": native_label,
                    "harmonized_label": harmonized_label,
                    "label_source": "FakeHealth HealthNewsReview expert rating mapped to binary reliability",
                    "source_name": source_name,
                    "url": url,
                    "publish_date": clean_text(content.get("publish_date")),
                    "collection_date": "",
                    "engagement_available": tweet_count > 0 or reply_count > 0 or retweet_count > 0,
                    "tweet_count": tweet_count,
                    "reply_count": reply_count,
                    "retweet_count": retweet_count,
                    "follower_file_available": bool(followers_available),
                    "following_file_available": bool(following_available),
                    "duplicate_group": normalized_text_hash(model_text),
                    "exclusion_reason": "" if model_text else "missing_model_text",
                }
            )
    return pd.DataFrame(rows)

