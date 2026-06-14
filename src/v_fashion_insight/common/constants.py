"""Project-wide constants with stable ordering."""

from types import MappingProxyType
from typing import Final, Mapping

ASPECTS: Final[tuple[str, ...]] = (
    "material",
    "design",
    "size",
    "price",
    "service",
)

LABEL_NAMES: Final[Mapping[int, str]] = MappingProxyType(
    {
        0: "not_mentioned",
        1: "negative",
        2: "neutral",
        3: "positive",
    }
)

VALID_LABELS: Final[frozenset[int]] = frozenset(LABEL_NAMES)
DEFAULT_RANDOM_SEED: Final[int] = 42
