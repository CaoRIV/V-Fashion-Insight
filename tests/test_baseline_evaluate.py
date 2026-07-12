from pathlib import Path

import pytest

from v_fashion_insight.models.evaluate import TEST_UNLOCK_TOKEN, evaluate_selected_baseline


def test_test_split_requires_explicit_unlock_before_loading_any_artifact() -> None:
    with pytest.raises(PermissionError, match="Test evaluation is locked"):
        evaluate_selected_baseline(split="test", registry_path=Path("missing.json"))


def test_test_unlock_token_is_intentional_and_stable() -> None:
    assert TEST_UNLOCK_TOKEN == "I_UNDERSTAND_TEST_IS_FINAL"
