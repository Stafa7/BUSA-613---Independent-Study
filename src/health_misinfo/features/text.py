from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


WHITESPACE = re.compile(r"\s+")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")
MISSING_LITERALS = {"", "nan", "none", "null", "n/a", "na"}
SCRAPER_OR_INTERFACE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("access_denied_page", re.compile(r"\baccess denied\b|\bcloudflare\b.*\brestrict access\b", re.IGNORECASE)),
    ("forbidden_page", re.compile(r"^\W*403 forbidden\b", re.IGNORECASE)),
    (
        "login_or_signup_page",
        re.compile(r"\blog in or sign up to view\b|\byou must log in to continue\b", re.IGNORECASE),
    ),
    (
        "javascript_disabled_page",
        re.compile(
            r"\bjavascript is disabled in your browser\b"
            r"|\bproceed to legacy twitter\b"
            r"|\bmake sure your browser supports javascript and cookies\b",
            re.IGNORECASE,
        ),
    ),
    ("bare_platform_page", re.compile(r"^\W*(facebook|twitter|instagram|youtube)\W*$", re.IGNORECASE)),
)
SENSITIVITY_ARTIFACT_PATTERNS = (
    re.compile(r"\b(?:facebook|twitter|instagram|youtube|healthnewsreview)\b", re.IGNORECASE),
    re.compile(r"\b(?:fake|real|false|true|misleading)\b", re.IGNORECASE),
    re.compile(r"\b(?:news source|news release|press release)\b", re.IGNORECASE),
)


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", html.unescape(str(value)))
    text = text.replace("\u200b", "").replace("\ufeff", "").replace("\x00", " ").strip()
    if text.lower() in MISSING_LITERALS:
        return ""
    return WHITESPACE.sub(" ", text).strip()


def join_text(*parts: object) -> str:
    return clean_text(" ".join(clean_text(part) for part in parts if clean_text(part)))


def stable_id(*parts: object, prefix: str = "rec") -> str:
    raw = "||".join(clean_text(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def normalized_text_hash(text: object) -> str:
    normalized = normalized_text(text)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def normalized_text(text: object) -> str:
    return NON_ALPHANUMERIC.sub(" ", clean_text(text).lower()).strip()


def domain_from_url(url: object) -> str:
    text = clean_text(url)
    if not text:
        return ""
    parsed = urlsplit(text if "://" in text else f"https://{text}")
    host = parsed.netloc.lower()
    return host[4:] if host.startswith("www.") else host


def canonical_article_url(url: object) -> str:
    """Return a conservative URL key for identifying the same parent item."""
    text = clean_text(url)
    if not text:
        return ""
    archive_match = re.search(
        r"https?://web\.archive\.org/web/(?:\d+|[a-z_]+)/(?P<target>https?://.+)",
        text,
        flags=re.IGNORECASE,
    )
    if archive_match:
        text = archive_match.group("target")
    parsed = urlsplit(text if "://" in text else f"https://{text}")
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = re.sub(r"/+", "/", parsed.path).rstrip("/")
    if not host or not path:
        return ""
    tracking_parameters = {
        "fbclid",
        "gclid",
        "mc_cid",
        "mc_eid",
        "ref",
        "ref_src",
        "source",
    }
    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in tracking_parameters
    ]
    query = urlencode(sorted(query_items))
    return urlunsplit(("https", host, path, query, ""))


def family_group_id(title: object, model_text: object, url: object) -> str:
    """Group exact/near duplicates and repeated parent articles for splitting."""
    canonical_url = canonical_article_url(url)
    if canonical_url:
        return stable_id("url", canonical_url, prefix="family")
    normalized_title = normalized_text(title)
    if len(normalized_title) >= 20:
        return stable_id("title", normalized_title, prefix="family")
    return stable_id("text", normalized_text_hash(model_text), prefix="family")


def content_exclusion_reason(text: object) -> str:
    """Return a coded reason when text is not substantive analytical content."""
    value = clean_text(text)
    normalized = normalized_text(value)
    if len(normalized.replace(" ", "")) < 10:
        return "near_empty_text"
    page_prefix = value[:800]
    word_count = len(normalized.split())
    for reason, pattern in SCRAPER_OR_INTERFACE_PATTERNS:
        if pattern.search(page_prefix) and word_count <= 250:
            return reason
    return ""


def mask_artifacts(text: object, source_name: object = "") -> str:
    """Remove known metadata artifacts without leaving countable marker tokens."""
    masked = URL_PATTERN.sub(" ", clean_text(text))
    source = clean_text(source_name)
    if source:
        source_tokens = {source, source.replace(".", " ")}
        source_tokens.update(
            token
            for token in re.split(r"[^a-zA-Z0-9]+", source)
            if len(token) >= 4 and token.lower() not in {"news", "health", "medical"}
        )
        for token in sorted(source_tokens, key=len, reverse=True):
            if token:
                masked = re.sub(
                    rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])",
                    " ",
                    masked,
                    flags=re.IGNORECASE,
                )
    for pattern in SENSITIVITY_ARTIFACT_PATTERNS:
        masked = pattern.sub(" ", masked)
    return clean_text(masked)
