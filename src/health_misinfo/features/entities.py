from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from urllib.parse import urlsplit

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from health_misinfo.features.text import (
    canonical_article_url,
    clean_text,
    normalized_text,
    stable_id,
)


COMMON_SECOND_LEVEL_SUFFIXES = {
    "ac.uk",
    "co.uk",
    "com.au",
    "com.br",
    "com.cn",
    "com.sg",
    "co.nz",
    "co.za",
    "org.uk",
}
GENERIC_ARCHIVE_DOMAINS = {"archive.org", "web.archive.org"}
PUBLISHER_ALIASES = {
    "abc news": "abcnews",
    "abcnews go": "abcnews",
    "ap associated press": "ap",
    "health day": "healthday",
    "healthday news": "healthday",
    "healthdaynews": "healthday",
    "reuters health": "reuters",
    "reutershealth": "reuters",
    "associated press": "ap",
    "associated press ap": "ap",
    "the associated press": "ap",
    "cnn health": "cnn",
    "cbs news": "cbsnews",
    "cbsnews": "cbsnews",
    "centers for disease control and prevention": "cdc",
    "national institutes of health": "nih",
    "nih national institutes of health": "nih",
    "u s food and drug administration": "fda",
    "us food and drug administration": "fda",
}
GENERIC_SOURCE_NAMES = {
    "https web archive",
    "https web archive org",
    "web archive",
    "web archive org",
}


@dataclass
class _DisjointSet:
    parent: list[int]
    rank: list[int]

    @classmethod
    def create(cls, size: int) -> "_DisjointSet":
        return cls(list(range(size)), [0] * size)

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def normalize_publisher_name(value: object) -> str:
    raw = clean_text(value)
    normalized = normalized_text(raw)
    # A small number of source fields contain a URL instead of a credited name.
    # Remove only unambiguous URL boilerplate; the explicit alias table handles
    # known host labels that differ from the corresponding publisher name.
    if re.match(r"^(?:(?:https?|ftp)://|www\.)", raw, flags=re.IGNORECASE):
        normalized = re.sub(r"^(?:https?|ftp)\s+(?:www\s+)?", "", normalized)
        normalized = re.sub(r"^www\s+", "", normalized)
    normalized = re.sub(r"^the\s+", "", normalized)
    normalized = re.sub(r"\s+(?:com|org|net)$", "", normalized)
    normalized = re.sub(
        r"\b(?:incorporated|corporation|company|limited|llc|inc|corp|ltd)\b",
        " ",
        normalized,
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = PUBLISHER_ALIASES.get(normalized, normalized)
    return "" if normalized in GENERIC_SOURCE_NAMES else normalized


def registrable_domain(value: object) -> str:
    url = clean_text(value)
    if not url:
        return ""
    archive_match = re.search(
        r"https?://web\.archive\.org/web/(?:\d+|[a-z_]+)/(?P<target>https?://.+)",
        url,
        flags=re.IGNORECASE,
    )
    if archive_match:
        url = archive_match.group("target")
    parsed = urlsplit(url if "://" in url else f"https://{url}")
    host = parsed.netloc.lower().split(":", 1)[0]
    if host.startswith("www."):
        host = host[4:]
    if not host or host in GENERIC_ARCHIVE_DOMAINS:
        return ""
    labels = [part for part in host.split(".") if part]
    if len(labels) <= 2:
        return host
    last_two = ".".join(labels[-2:])
    if last_two in COMMON_SECOND_LEVEL_SUFFIXES and len(labels) >= 3:
        return ".".join(labels[-3:])
    return last_two


def assign_publisher_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign conservative dataset-local credited-publisher identities.

    CoAID source names are derived from URLs, so its publisher identity is the
    registrable hosting domain. FakeHealth has an explicit credited-source field:
    use that normalized name when present and fall back to the hosting domain.
    Names and domains are deliberately not joined transitively because wire copy
    and release platforms can host material credited to many distinct publishers.
    """
    result = frame.copy()
    result["source_name_raw"] = result["source_name"].fillna("").astype(str)
    result["source_name_normalized"] = result["source_name_raw"].map(normalize_publisher_name)
    result["source_domain"] = result["url"].map(registrable_domain)
    result["publisher_id"] = ""

    for row_index, row in result.iterrows():
        source = row["source_name_normalized"]
        domain = row["source_domain"]
        if row["dataset"] == "coaid" and domain:
            identity = f"domain:{domain}"
        else:
            identity = f"name:{source}" if source else (f"domain:{domain}" if domain else "")
        if identity:
            result.at[row_index, "publisher_id"] = stable_id(
                row["dataset"],
                identity,
                prefix="publisher",
            )
    return result


def _parse_alternate_urls(value: object) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = [part for part in text.split("|") if part]
    if isinstance(parsed, str):
        parsed = [parsed]
    return [clean_text(item) for item in parsed if clean_text(item)]


def _exact_family_keys(row: pd.Series) -> set[str]:
    keys: set[str] = set()
    urls = [row.get("url", ""), *_parse_alternate_urls(row.get("alternate_urls", ""))]
    for url in urls:
        canonical = canonical_article_url(url)
        if canonical:
            keys.add(f"url:{canonical}")
    title = normalized_text(row.get("title", ""))
    if len(title) >= 20 and len(title.split()) >= 6:
        keys.add(f"title:{title}")
    body = normalized_text(row.get("body_text", ""))
    if len(body.split()) >= 30:
        keys.add(f"body:{body}")
    model_text = normalized_text(row.get("model_text", ""))
    if model_text:
        keys.add(f"text:{model_text}")
    return keys


def _union_fuzzy_titles(
    dataset: pd.DataFrame,
    local_positions: dict[int, int],
    dsu: _DisjointSet,
    similarity_threshold: float,
) -> int:
    eligible = dataset[
        dataset["title"]
        .fillna("")
        .map(
            lambda value: len(normalized_text(value)) >= 30
            and len(normalized_text(value).split()) >= 6
        )
    ]
    if len(eligible) < 2:
        return 0
    titles = eligible["title"].map(normalized_text).tolist()
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
    matrix = vectorizer.fit_transform(titles)
    if matrix.shape[1] == 0:
        return 0
    neighbors = min(8, len(eligible))
    model = NearestNeighbors(metric="cosine", n_neighbors=neighbors)
    model.fit(matrix)
    distances, indices = model.kneighbors(matrix)
    eligible_indices = eligible.index.to_list()
    unions = 0
    for left_position, (row_distances, row_neighbors) in enumerate(zip(distances, indices)):
        left_title = titles[left_position]
        for distance, right_position in zip(row_distances[1:], row_neighbors[1:]):
            similarity = 1.0 - float(distance)
            if similarity < similarity_threshold:
                continue
            right_title = titles[int(right_position)]
            length_ratio = min(len(left_title), len(right_title)) / max(len(left_title), len(right_title))
            if length_ratio < 0.85:
                continue
            left = local_positions[eligible_indices[left_position]]
            right = local_positions[eligible_indices[int(right_position)]]
            if dsu.find(left) != dsu.find(right):
                dsu.union(left, right)
                unions += 1
    return unions


def assign_family_groups(
    frame: pd.DataFrame,
    fuzzy_title_similarity: float = 0.97,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = frame.copy()
    result["family_group"] = ""
    result["family_size"] = 1
    audit_rows: list[dict[str, object]] = []

    for dataset_name, dataset in result.groupby("dataset", sort=True):
        indices = dataset.index.to_list()
        positions = {row_index: position for position, row_index in enumerate(indices)}
        dsu = _DisjointSet.create(len(indices))
        key_owner: dict[str, int] = {}
        exact_unions = 0
        for row_index, row in dataset.iterrows():
            position = positions[row_index]
            for key in sorted(_exact_family_keys(row)):
                if key in key_owner:
                    if dsu.find(position) != dsu.find(key_owner[key]):
                        dsu.union(position, key_owner[key])
                        exact_unions += 1
                else:
                    key_owner[key] = position

        fuzzy_unions = _union_fuzzy_titles(
            dataset,
            positions,
            dsu,
            similarity_threshold=fuzzy_title_similarity,
        )
        components: dict[int, list[int]] = defaultdict(list)
        for position, row_index in enumerate(indices):
            components[dsu.find(position)].append(row_index)
        for members in components.values():
            record_ids = sorted(result.loc[members, "record_id"].astype(str))
            family_id = stable_id(dataset_name, record_ids[0], prefix="family")
            result.loc[members, "family_group"] = family_id
            result.loc[members, "family_size"] = len(members)
        audit_rows.append(
            {
                "dataset": dataset_name,
                "records": len(dataset),
                "family_groups": len(components),
                "multi_record_families": sum(len(members) > 1 for members in components.values()),
                "largest_family": max((len(members) for members in components.values()), default=0),
                "exact_identifier_unions": exact_unions,
                "fuzzy_title_unions": fuzzy_unions,
                "fuzzy_title_similarity_threshold": fuzzy_title_similarity,
            }
        )

    return result, pd.DataFrame(audit_rows)
