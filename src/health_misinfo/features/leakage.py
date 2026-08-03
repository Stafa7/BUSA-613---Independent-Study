from __future__ import annotations

import re

OUTCOME_PATTERNS = [
    r"\bfake\b",
    r"\breal\b",
    r"\bfalse\b",
    r"\btrue\b",
    r"\bmisleading\b",
    r"\brating\b",
    r"\bhealthnewsreview\b",
    r"\bnews source\b",
    r"\bpress release\b",
    r"\baccess denied\b",
    r"\b403 forbidden\b",
    r"\blog in or sign up\b",
]

ARTIFACT_FEATURE_PATTERNS = [
    *OUTCOME_PATTERNS,
    r"\bfacebook\b",
    r"\btwitter\b",
    r"\binstagram\b",
    r"\byoutube\b",
    r"\bcloudflare\b",
    r"\bjavascript\b",
    r"\bsource\b",
    r"\bartifact\b",
    r"\burl\b",
]


def count_outcome_terms(text: str) -> int:
    value = text or ""
    return sum(len(re.findall(pattern, value, flags=re.IGNORECASE)) for pattern in OUTCOME_PATTERNS)


def is_artifact_feature(feature: str) -> bool:
    return any(re.search(pattern, feature or "", flags=re.IGNORECASE) for pattern in ARTIFACT_FEATURE_PATTERNS)
