import pytest

from v_fashion_insight.data import preprocess
from v_fashion_insight.data.preprocess import (
    REVIEW_ID_CANONICAL_FIELDS,
    ReviewIdCollisionError,
    generate_review_id,
    generate_review_ids,
    normalize_review_text,
)


def test_normalize_review_text_applies_nfkc_and_collapses_whitespace() -> None:
    text = "A\u0301o\u00a0đẹp\r\n\tkhông  bị xù"

    assert normalize_review_text(text) == "Áo đẹp không bị xù"


def test_normalize_review_text_preserves_sentiment_bearing_content() -> None:
    text = "Không tệ!!! Size M-2024 😊 áo đầm ổn, giá 199k."

    assert normalize_review_text(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "áo đẹp không bị phai màu",
        "Shop giao\nhàng\t nhanh   và tư vấn tốt.",
        "Mã SP AB12 còn nguyên, size L vừa.",
    ],
)
def test_normalize_review_text_is_idempotent(text: str) -> None:
    normalized = normalize_review_text(text)

    assert normalize_review_text(normalized) == normalized


def test_normalize_review_text_keeps_meaningful_punctuation_digits_and_emoji() -> None:
    text = "Vải mỏng quá?! Nhưng màu #12 đẹp 😊, mua 2 cái."

    assert normalize_review_text(text) == text


def test_normalize_review_text_rejects_non_string_values() -> None:
    with pytest.raises(TypeError, match="review text must be a string"):
        normalize_review_text(None)  # type: ignore[arg-type]


def test_review_id_canonical_fields_are_explicit() -> None:
    assert REVIEW_ID_CANONICAL_FIELDS == (
        "dataset_name",
        "dataset_revision",
        "source_split",
        "source_id",
    )


def test_generate_review_id_is_stable_and_uses_source_identity() -> None:
    identity = {
        "dataset_name": "vinhplaykennen/FashionReviews",
        "dataset_revision": "60abb1cef934cb248b88a2ce4c99bb1ea3129c92",
        "source_split": "train",
        "source_id": 42,
    }

    first = generate_review_id(**identity)
    second = generate_review_id(**identity)

    assert first == second
    assert first == (
        "review_f41892c17ab9e333f6cdd75efd549dc4"
        "c0fe5b310eabd13654da67a0f7b169fb"
    )
    assert generate_review_id(**{**identity, "source_id": 43}) != first
    assert (
        generate_review_id(**{**identity, "dataset_revision": "different"})
        != first
    )


def test_generate_review_ids_returns_unique_ids_in_input_order() -> None:
    review_ids = generate_review_ids(
        [10, 11, 12],
        dataset_name="owner/dataset",
        dataset_revision="abc123",
        source_split="train",
    )

    assert len(review_ids) == 3
    assert len(set(review_ids)) == 3
    assert review_ids[0] == generate_review_id(
        dataset_name="owner/dataset",
        dataset_revision="abc123",
        source_split="train",
        source_id=10,
    )


def test_generate_review_ids_rejects_duplicate_source_identity() -> None:
    with pytest.raises(ValueError, match="Duplicate canonical review identity"):
        generate_review_ids(
            [10, 10],
            dataset_name="owner/dataset",
            dataset_revision="abc123",
            source_split="train",
        )


def test_generate_review_ids_detects_hash_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preprocess,
        "_hash_canonical_identity",
        lambda _identity: "0" * 64,
    )

    with pytest.raises(ReviewIdCollisionError, match="SHA-256 collision"):
        generate_review_ids(
            [10, 11],
            dataset_name="owner/dataset",
            dataset_revision="abc123",
            source_split="train",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_name", ""),
        ("dataset_revision", " "),
        ("source_split", None),
        ("source_id", ""),
        ("source_id", 1.5),
    ],
)
def test_generate_review_id_rejects_invalid_identity_fields(
    field: str,
    value: object,
) -> None:
    identity: dict[str, object] = {
        "dataset_name": "owner/dataset",
        "dataset_revision": "abc123",
        "source_split": "train",
        "source_id": 10,
    }
    identity[field] = value

    with pytest.raises((TypeError, ValueError)):
        generate_review_id(**identity)  # type: ignore[arg-type]
