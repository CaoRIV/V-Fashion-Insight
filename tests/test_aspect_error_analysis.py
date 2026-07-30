import pandas as pd
import pytest

from v_fashion_insight.analysis.aspect_errors import (
    build_aspect_label_analysis,
    build_detailed_aspect_errors,
    classify_error_family,
)
from v_fashion_insight.common.constants import ASPECTS
from v_fashion_insight.models.metrics import compute_multioutput_metrics


def _enriched_frame() -> pd.DataFrame:
    true_rows = [
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3],
        [None, 0, 0, 0, 0],
        [0, 1, 2, 3, 0],
    ]
    predicted_rows = [
        [0, 0, 0, 0, 0],
        [1, 1, 2, 1, 1],
        [0, 2, 3, 2, 2],
        [3, 3, 3, 3, 3],
        [0, 0, 0, 0, 0],
        [1, 1, 2, 3, 0],
    ]
    rows: list[dict] = []
    for index, (true_values, predicted_values) in enumerate(
        zip(true_rows, predicted_rows, strict=True)
    ):
        row = {
            "review_id": f"r{index}",
            "group_id": f"g{index}",
            "text": f"review {index}",
            "text_character_count": 8,
            "text_token_count": 2,
        }
        error_count = 0
        for aspect, true_label, predicted_label in zip(
            ASPECTS,
            true_values,
            predicted_values,
            strict=True,
        ):
            row[f"true_{aspect}"] = true_label
            row[f"predicted_{aspect}"] = predicted_label
            is_error = (
                true_label is not None
                and true_label != predicted_label
            )
            row[f"error_{aspect}"] = is_error
            error_count += int(is_error)
        row["error_aspect_count"] = error_count
        row["complete_target"] = all(
            label is not None for label in true_values
        )
        row["exact_match"] = (
            row["complete_target"] and error_count == 0
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _recorded_report(enriched: pd.DataFrame) -> dict:
    true = pd.DataFrame(
        {
            aspect: enriched[f"true_{aspect}"]
            for aspect in ASPECTS
        }
    )
    predicted = pd.DataFrame(
        {
            aspect: enriched[f"predicted_{aspect}"]
            for aspect in ASPECTS
        }
    )
    return {
        "run_name": "combined_svc",
        "selection_split": "validation",
        "test_set_accessed": False,
        "metrics": compute_multioutput_metrics(true, predicted),
    }


@pytest.mark.parametrize(
    ("true_label", "predicted_label", "family"),
    [
        (2, 0, "missed_mention"),
        (0, 3, "false_mention"),
        (1, 3, "sentiment_confusion"),
    ],
)
def test_classify_error_family(
    true_label: int,
    predicted_label: int,
    family: str,
) -> None:
    assert classify_error_family(true_label, predicted_label) == family


def test_classify_error_family_rejects_correct_prediction() -> None:
    with pytest.raises(ValueError, match="unequal"):
        classify_error_family(2, 2)


def test_detailed_errors_are_traceable_and_classified() -> None:
    detailed = build_detailed_aspect_errors(_enriched_frame())
    assert len(detailed) == 4
    assert list(detailed.columns) == [
        "review_id",
        "group_id",
        "aspect",
        "true_label",
        "true_label_name",
        "predicted_label",
        "predicted_label_name",
        "error_family",
        "review_error_aspect_count",
        "text_character_count",
        "text_token_count",
        "text",
    ]
    assert detailed["error_family"].value_counts().to_dict() == {
        "sentiment_confusion": 2,
        "missed_mention": 1,
        "false_mention": 1,
    }


def test_aspect_label_analysis_finds_weak_labels_and_directions() -> None:
    enriched = _enriched_frame()
    summary, labels, detailed = build_aspect_label_analysis(
        enriched,
        _recorded_report(enriched),
        inputs={},
    )
    assert summary["test_set_accessed"] is False
    assert summary["summary"]["total_aspect_error_count"] == 4
    assert len(labels) == 20
    assert len(detailed) == 4

    material = summary["aspects"]["material"]
    assert material["error_count"] == 2
    assert material["error_family_counts"] == {
        "missed_mention": 1,
        "false_mention": 1,
        "sentiment_confusion": 0,
    }
    size_neutral = labels.loc[
        labels["aspect"].eq("size")
        & labels["label"].eq(2)
    ].iloc[0]
    assert size_neutral["support"] == 2
    assert size_neutral["false_negative_count"] == 1
    assert size_neutral["dominant_wrong_prediction"] == 3
