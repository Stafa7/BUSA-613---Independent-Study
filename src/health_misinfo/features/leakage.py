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
]


def count_outcome_terms(text: str) -> int:
    value = text or ""
    return sum(len(re.findall(pattern, value, flags=re.IGNORECASE)) for pattern in OUTCOME_PATTERNS)

