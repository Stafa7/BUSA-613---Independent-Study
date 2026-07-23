from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from health_misinfo.datasets.labels import coaid_harmonized_label
from health_misinfo.features.text import (
    clean_text,
    content_exclusion_reason,
    domain_from_url,
    join_text,
    mask_artifacts,
    normalized_text_hash,
    stable_id,
)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).rename(columns={"Unnamed: 0": "source_index"})


def _count_map(path: Path, key: str, value_key: str) -> dict[str, int]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if key not in df.columns or value_key not in df.columns:
        return {}
    usable = df[[key, value_key]].dropna().drop_duplicates()
    return usable.assign(_key=usable[key].astype(str)).groupby("_key")[value_key].nunique().to_dict()


def _base_files(raw_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in raw_dir.glob("*/*.csv")
        if "_tweets" not in path.name and "_replies" not in path.name
    )


def load_coaid_items(raw_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in _base_files(raw_dir):
        collection_date = path.parent.name
        stem = path.stem
        text_unit = "claim" if stem.startswith("Claim") else "article"
        native_label, harmonized_label = coaid_harmonized_label(path.name)
        tweets = _count_map(path.with_name(f"{stem}_tweets.csv"), "index", "tweet_id")
        replies = _count_map(
            path.with_name(f"{stem}_tweets_replies.csv"),
            "news_id",
            "reply_id",
        )
        df = _read_csv(path)
        for row_number, row in df.iterrows():
            source_index = clean_text(row.get("source_index", row_number))
            source_record_id = f"{collection_date}:{stem}:{source_index}"
            title = clean_text(row.get("newstitle")) or clean_text(row.get("title"))
            body = join_text(row.get("content"), row.get("abstract"))
            model_text = join_text(title, body)
            url = clean_text(row.get("news_url"))
            alternate_urls = [
                clean_text(row.get(f"news_url{index}"))
                for index in range(2, 6)
                if clean_text(row.get(f"news_url{index}"))
            ]
            source_name = domain_from_url(url)
            tweet_count = int(tweets.get(source_index, 0))
            reply_count = int(replies.get(source_index, 0))
            exclusion_reason = content_exclusion_reason(model_text)
            rows.append(
                {
                    "dataset": "coaid",
                    "record_id": stable_id("coaid", source_record_id, prefix="coaid"),
                    "source_record_id": source_record_id,
                    "source_file": str(path.relative_to(raw_dir)),
                    "parse_status": "parsed",
                    "text_unit": text_unit,
                    "title": title,
                    "body_text": body,
                    "model_text": model_text,
                    "model_text_masked": mask_artifacts(model_text, source_name),
                    "native_label": native_label,
                    "harmonized_label": harmonized_label,
                    "label_source": "CoAID Real/Fake source file grouping",
                    "source_name": source_name,
                    "url": url,
                    "alternate_urls": json.dumps(alternate_urls),
                    "publish_date": clean_text(row.get("publish_date")) or collection_date,
                    "collection_date": collection_date,
                    "content_file_available": bool(body),
                    "engagement_available": tweet_count > 0 or reply_count > 0,
                    "tweet_count": tweet_count,
                    "reply_count": reply_count,
                    "retweet_count": 0,
                    "network_followers_archive_available": False,
                    "network_following_archive_available": False,
                    "record_network_linkage_available": False,
                    "duplicate_group": normalized_text_hash(model_text),
                    "family_group": "",
                    "exclusion_reason": exclusion_reason or ("missing_model_text" if not model_text else ""),
                }
            )
    return pd.DataFrame(rows)
