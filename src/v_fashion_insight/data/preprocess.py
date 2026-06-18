"""Conservative preprocessing helpers for review text."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable
from numbers import Integral
from typing import Final

_WHITESPACE_PATTERN = re.compile(r"\s+")

REVIEW_ID_CANONICAL_FIELDS: Final[tuple[str, ...]] = (
    "dataset_name",
    "dataset_revision",
    "source_split",
    "source_id",
)
REVIEW_ID_SCHEMA_VERSION: Final[str] = "v1"
REVIEW_ID_PREFIX: Final[str] = "review_"


class ReviewIdCollisionError(RuntimeError):
    """Raised when distinct canonical identities produce the same review ID."""


def normalize_review_text(text: str) -> str:
    """Normalize review text without removing sentiment-bearing content."""
    if not isinstance(text, str):
        raise TypeError("review text must be a string")

    normalized = unicodedata.normalize("NFKC", text)
    return _WHITESPACE_PATTERN.sub(" ", normalized).strip()


def _required_identity_text(field: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    return normalized


def _canonical_source_id(source_id: object) -> str:
    if isinstance(source_id, bool):
        raise TypeError("source_id must be an integer or non-empty string")
    if isinstance(source_id, Integral):
        return str(int(source_id))
    if isinstance(source_id, str):
        normalized = source_id.strip()
        if normalized:
            return normalized
        raise ValueError("source_id must not be empty")
    raise TypeError("source_id must be an integer or non-empty string")


def _canonical_review_identity(
    *,
    dataset_name: str,
    dataset_revision: str,
    source_split: str,
    source_id: object,
) -> str:
    payload = {
        "schema_version": REVIEW_ID_SCHEMA_VERSION,
        "dataset_name": _required_identity_text(
            "dataset_name",
            dataset_name,
        ),
        "dataset_revision": _required_identity_text(
            "dataset_revision",
            dataset_revision,
        ),
        "source_split": _required_identity_text(
            "source_split",
            source_split,
        ),
        "source_id": _canonical_source_id(source_id),
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _hash_canonical_identity(identity: str) -> str:
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _review_id_from_identity(identity: str) -> str:
    return f"{REVIEW_ID_PREFIX}{_hash_canonical_identity(identity)}"


def generate_review_id(
    *,
    dataset_name: str,
    dataset_revision: str,
    source_split: str,
    source_id: object,
) -> str:
    """Generate a stable ID from immutable source-row identity fields."""
    identity = _canonical_review_identity(
        dataset_name=dataset_name,
        dataset_revision=dataset_revision,
        source_split=source_split,
        source_id=source_id,
    )
    return _review_id_from_identity(identity)


def generate_review_ids(
    source_ids: Iterable[object],
    *,
    dataset_name: str,
    dataset_revision: str,
    source_split: str,
) -> list[str]:
    """Generate ordered review IDs and reject duplicates or hash collisions."""
    review_ids: list[str] = []
    identities_by_review_id: dict[str, str] = {}

    for source_id in source_ids:
        identity = _canonical_review_identity(
            dataset_name=dataset_name,
            dataset_revision=dataset_revision,
            source_split=source_split,
            source_id=source_id,
        )
        review_id = _review_id_from_identity(identity)
        previous_identity = identities_by_review_id.get(review_id)
        if previous_identity is not None:
            if previous_identity == identity:
                raise ValueError(
                    "Duplicate canonical review identity for "
                    f"source_id={source_id!r}."
                )
            raise ReviewIdCollisionError(
                "SHA-256 collision between distinct canonical review "
                f"identities for review_id={review_id!r}."
            )

        identities_by_review_id[review_id] = identity
        review_ids.append(review_id)

    return review_ids
