import pytest

from v_fashion_insight.data.preprocess import normalize_review_text


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
