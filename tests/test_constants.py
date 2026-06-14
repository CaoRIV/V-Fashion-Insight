from v_fashion_insight.common.constants import (
    ASPECTS,
    DEFAULT_RANDOM_SEED,
    LABEL_NAMES,
    VALID_LABELS,
)


def test_aspect_order_is_stable() -> None:
    assert ASPECTS == ("material", "design", "size", "price", "service")


def test_label_mapping_is_complete() -> None:
    assert dict(LABEL_NAMES) == {
        0: "not_mentioned",
        1: "negative",
        2: "neutral",
        3: "positive",
    }
    assert VALID_LABELS == frozenset({0, 1, 2, 3})


def test_default_seed_is_fixed() -> None:
    assert DEFAULT_RANDOM_SEED == 42
