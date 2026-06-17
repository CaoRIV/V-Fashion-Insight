"""Conservative preprocessing helpers for review text."""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_review_text(text: str) -> str:
    """Normalize review text without removing sentiment-bearing content."""
    if not isinstance(text, str):
        raise TypeError("review text must be a string")

    normalized = unicodedata.normalize("NFKC", text)
    return _WHITESPACE_PATTERN.sub(" ", normalized).strip()
