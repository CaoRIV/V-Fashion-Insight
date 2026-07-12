import pandas as pd
import pytest

from v_fashion_insight.models.metrics import compute_multioutput_metrics


def test_multioutput_metrics_preserve_order_mask_missing_and_compute_exact_match() -> None:
    true = pd.DataFrame(
        {
            "material": [0, 1, 2, 3, None],
            "design": [0, 1, 2, 3, 0],
            "size": [0, 1, 2, 3, 0],
            "price": [0, 1, 2, 3, 0],
            "service": [0, 1, 2, 3, 0],
        }
    )
    predicted = pd.DataFrame(
        {
            "material": [0, 1, 2, 0, 3],
            "design": [0, 1, 2, 3, 0],
            "size": [0, 1, 2, 3, 0],
            "price": [0, 1, 2, 3, 0],
            "service": [0, 1, 2, 3, 0],
        }
    )
    metrics = compute_multioutput_metrics(true, predicted)
    assert metrics["aspect_order"] == ["material", "design", "size", "price", "service"]
    assert metrics["aspects"]["material"]["evaluated_count"] == 4
    assert metrics["aspects"]["material"]["missing_target_count"] == 1
    assert metrics["aspects"]["material"]["macro_f1"] == pytest.approx(0.6666666667)
    assert metrics["summary"]["complete_target_row_count"] == 4
    assert metrics["summary"]["exact_match_ratio"] == pytest.approx(0.75)
    assert metrics["aspects"]["design"]["confusion_matrix"] == [
        [2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
    ]


def test_multioutput_metrics_reject_invalid_predictions() -> None:
    frame = pd.DataFrame({aspect: [0, 1] for aspect in ("material", "design", "size", "price", "service")})
    predicted = frame.copy()
    predicted.loc[0, "price"] = 8
    with pytest.raises(ValueError, match="invalid labels"):
        compute_multioutput_metrics(frame, predicted)
