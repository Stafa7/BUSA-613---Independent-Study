from __future__ import annotations

import hashlib
import re
from urllib.parse import urlparse


WHITESPACE = re.compile(r"\s+")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return WHITESPACE.sub(" ", text).strip()


def join_text(*parts: object) -> str:
    return clean_text(" ".join(clean_text(part) for part in parts if clean_text(part)))


def stable_id(*parts: object, prefix: str = "rec") -> str:
    raw = "||".join(clean_text(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def normalized_text_hash(text: object) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", clean_text(text).lower()).strip()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def domain_from_url(url: object) -> str:
    text = clean_text(url)
    if not text:
        return ""
    parsed = urlparse(text if "://" in text else f"https://{text}")
    host = parsed.netloc.lower()
    return host[4:] if host.startswith("www.") else host


def mask_artifacts(text: object, source_name: object = "") -> str:
    masked = URL_PATTERN.sub(" [URL] ", clean_text(text))
    source = clean_text(source_name)
    if source:
        source_tokens = [source, source.replace(".", " ")]
        for token in source_tokens:
            if token:
                masked = re.sub(re.escape(token), " [SOURCE] ", masked, flags=re.IGNORECASE)
    return clean_text(masked)

