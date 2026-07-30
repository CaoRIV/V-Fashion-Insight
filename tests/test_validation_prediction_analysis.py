import json
from pathlib import Path

import pandas as pd
import pytest

from v_fashion_insight.analysis.validation_predictions import (
    analyze_validation_predictions,
    build_prediction_analysis,
    load_enriched_validation_predictions,
)
from v_fashion_insight.common.constants import ASPECTS
from v_fashion_insight.models.metrics import compute_multioutput_metrics


def _analysis_inputs(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path]:
    reviews: list[dict] = []
    split_rows: list[dict] = []
    prediction_rows: list[dict] = []
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
    for index, (true_values, predicted_values) in enumerate(
        zip(true_rows, predicted_rows, strict=True)
    ):
        review_id = f"r{index}"
        group_id = f"g{index}"
        reviews.append(
            {
                "review_id": review_id,
                "group_id": group_id,
                "text": f"review validation {index}",
            }
        )
        split_rows.append(
            {
                "review_id": review_id,
                "group_id": group_id,
                "split": "validation",
            }
        )
        prediction = {
            "review_id": review_id,
            "group_id": group_id,
        }
        for aspect, true_label, predicted_label in zip(
            ASPECTS,
            true_values,
            predicted_values,
            strict=True,
        ):
            prediction[f"true_{aspect}"] = true_label
            prediction[f"predicted_{aspect}"] = predicted_label
        prediction_rows.append(prediction)

    reviews.append(
        {
            "review_id": "train-row",
            "group_id": "train-group",
            "text": "must not enter validation analysis",
        }
    )
    split_rows.append(
        {
            "review_id": "train-row",
            "group_id": "train-group",
            "split": "train",
        }
    )
    interim_path = tmp_path / "reviews.csv"
    split_path = tmp_path / "split_ids.csv"
    predictions_path = tmp_path / "predictions.csv"
    recorded_path = tmp_path / "metrics.json"
    pd.DataFrame(reviews).to_csv(interim_path, index=False)
    pd.DataFrame(split_rows).to_csv(split_path, index=False)
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(predictions_path, index=False)
    true = pd.DataFrame(
        {
            aspect: predictions[f"true_{aspect}"]
            for aspect in ASPECTS
        }
    )
    predicted = pd.DataFrame(
        {
            aspect: predictions[f"predicted_{aspect}"]
            for aspect in ASPECTS
        }
    )
    recorded_path.write_text(
        json.dumps(
            {
                "run_name": "combined_svc",
                "selection_split": "validation",
                "test_set_accessed": False,
                "metrics": compute_multioutput_metrics(true, predicted),
            }
        ),
        encoding="utf-8",
    )
    return predictions_path, recorded_path, interim_path, split_path


def test_load_enriched_predictions_proves_validation_coverage(
    tmp_path: Path,
) -> None:
    predictions_path, _, interim_path, split_path = _analysis_inputs(
        tmp_path
    )
    enriched = load_enriched_validation_predictions(
        predictions_path,
        interim_path,
        split_path,
    )
    assert len(enriched) == 6
    assert "train-row" not in set(enriched["review_id"])
    assert enriched["error_aspect_count"].tolist() == [0, 1, 2, 0, 0, 1]
    assert enriched["complete_target"].sum() == 5
    assert enriched["exact_match"].sum() == 2


def test_build_analysis_classifies_error_types_and_confusions(
    tmp_path: Path,
) -> None:
    predictions_path, recorded_path, interim_path, split_path = (
        _analysis_inputs(tmp_path)
    )
    enriched = load_enriched_validation_predictions(
        predictions_path,
        interim_path,
        split_path,
    )
    recorded = json.loads(recorded_path.read_text(encoding="utf-8"))
    summary, aspect_errors, confusions, examples = (
        build_prediction_analysis(
            enriched,
            run_name="combined_svc",
            inputs={},
            recorded_report=recorded,
            examples_per_confusion=2,
        )
    )
    assert summary["test_set_accessed"] is False
    assert summary["metrics_match_recorded_report"] is True
    assert summary["summary"]["row_with_any_error_count"] == 3
    material = aspect_errors.set_index("aspect").loc["material"]
    assert material["error_count"] == 2
    assert material["missed_mention_count"] == 1
    assert material["false_mention_count"] == 1
    size_confusion = confusions.loc[
        confusions["aspect"].eq("size")
        & confusions["true_label"].eq(2)
        & confusions["predicted_label"].eq(3)
    ].iloc[0]
    assert size_confusion["count"] == 1
    assert set(examples["aspect"]).issubset(set(ASPECTS))


def test_analysis_writes_reproducible_outputs_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    predictions_path, recorded_path, interim_path, split_path = (
        _analysis_inputs(tmp_path)
    )
    outputs = {
        "summary_path": tmp_path / "summary.json",
        "aspect_errors_path": tmp_path / "aspects.csv",
        "confusions_path": tmp_path / "confusions.csv",
        "examples_path": tmp_path / "examples.csv",
        "report_path": tmp_path / "report.md",
    }
    summary = analyze_validation_predictions(
        predictions_path=predictions_path,
        recorded_metrics_path=recorded_path,
        interim_path=interim_path,
        split_ids_path=split_path,
        **outputs,
    )
    assert summary["summary"]["row_count"] == 6
    assert all(path.exists() for path in outputs.values())
    assert "Weakest aspect" in outputs["report_path"].read_text(
        encoding="utf-8"
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        analyze_validation_predictions(
            predictions_path=predictions_path,
            recorded_metrics_path=recorded_path,
            interim_path=interim_path,
            split_ids_path=split_path,
            **outputs,
        )


def test_validation_coverage_mismatch_fails(
    tmp_path: Path,
) -> None:
    predictions_path, _, interim_path, split_path = _analysis_inputs(
        tmp_path
    )
    predictions = pd.read_csv(predictions_path).iloc[:-1]
    predictions.to_csv(predictions_path, index=False)
    with pytest.raises(ValueError, match="exactly cover"):
        load_enriched_validation_predictions(
            predictions_path,
            interim_path,
            split_path,
        )
