from pathlib import Path

import pandas as pd
import pytest

from v_fashion_insight.data.check_splits import (
    DEFAULT_MAX_LABEL_PROPORTION_DRIFT,
    SplitCheckError,
    build_split_report,
    check_split_artifacts,
)
from v_fashion_insight.data.processed_contract import (
    GROUP_ID_COLUMN,
    REVIEW_ID_COLUMN,
    SPLIT_COLUMN,
)


def _balanced_interim_frame() -> pd.DataFrame:
    rows = []
    split_names = ("train", "validation", "test")
    label_pattern = [0, 1, 2, 3]
    for split_name in split_names:
        for label in label_pattern:
            index = len(rows)
            rows.append(
                {
                    REVIEW_ID_COLUMN: f"r{index:02d}",
                    GROUP_ID_COLUMN: f"g{index:02d}",
                    "text": f"review {index}",
                    "material": label,
                    "design": label,
                    "size": label,
                    "price": label,
                    "service": label,
                    "_expected_split": split_name,
                }
            )
    return pd.DataFrame(rows)


def _split_ids_for(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.loc[
            :,
            [REVIEW_ID_COLUMN, GROUP_ID_COLUMN, "_expected_split"],
        ]
        .rename(columns={"_expected_split": SPLIT_COLUMN})
        .copy()
    )


def _issue_codes(report: dict) -> set[str]:
    return {issue["code"] for issue in report["issues"]}


def test_build_split_report_passes_valid_grouped_balanced_split() -> None:
    frame = _balanced_interim_frame()
    split_ids = _split_ids_for(frame)

    report = build_split_report(frame, split_ids)

    assert report["valid"] is True
    assert report["status"] == "passed"
    assert report["summary"]["split_counts"] == {
        "train": 4,
        "validation": 4,
        "test": 4,
    }
    assert (
        report["summary"]["max_label_proportion_drift"]
        <= DEFAULT_MAX_LABEL_PROPORTION_DRIFT
    )


def test_build_split_report_fails_when_group_crosses_splits() -> None:
    frame = _balanced_interim_frame()
    frame.loc[1, GROUP_ID_COLUMN] = frame.loc[0, GROUP_ID_COLUMN]
    split_ids = _split_ids_for(frame)
    split_ids.loc[1, SPLIT_COLUMN] = "test"

    report = build_split_report(frame, split_ids)

    assert report["valid"] is False
    assert "group_id_crosses_splits" in _issue_codes(report)


def test_build_split_report_fails_when_exact_duplicate_crosses_splits() -> None:
    frame = _balanced_interim_frame()
    frame.loc[4, "text"] = "  same\ntext "
    frame.loc[8, "text"] = "same text"
    split_ids = _split_ids_for(frame)

    report = build_split_report(frame, split_ids)

    assert report["valid"] is False
    assert "normalized_exact_duplicate_crosses_splits" in _issue_codes(report)


def test_build_split_report_fails_when_label_drift_exceeds_tolerance() -> None:
    frame = _balanced_interim_frame()
    frame.loc[frame["_expected_split"] == "train", "material"] = 3
    frame.loc[frame["_expected_split"] != "train", "material"] = 0
    split_ids = _split_ids_for(frame)

    report = build_split_report(frame, split_ids)

    assert report["valid"] is False
    assert "label_distribution_drift_exceeds_tolerance" in _issue_codes(report)


def test_check_split_artifacts_writes_report_and_raises_on_failure(
    tmp_path: Path,
) -> None:
    frame = _balanced_interim_frame()
    split_ids = _split_ids_for(frame)
    split_ids.loc[0, SPLIT_COLUMN] = "test"
    interim_path = tmp_path / "interim.csv"
    split_ids_path = tmp_path / "split_ids.csv"
    report_path = tmp_path / "split_report.json"
    frame.to_csv(interim_path, index=False)
    split_ids.to_csv(split_ids_path, index=False)

    with pytest.raises(SplitCheckError, match="Split checks failed"):
        check_split_artifacts(
            interim_path=interim_path,
            split_ids_path=split_ids_path,
            report_path=report_path,
        )

    assert report_path.exists()


def test_check_split_artifacts_refuses_to_overwrite_report(
    tmp_path: Path,
) -> None:
    frame = _balanced_interim_frame()
    split_ids = _split_ids_for(frame)
    interim_path = tmp_path / "interim.csv"
    split_ids_path = tmp_path / "split_ids.csv"
    report_path = tmp_path / "split_report.json"
    frame.to_csv(interim_path, index=False)
    split_ids.to_csv(split_ids_path, index=False)
    report_path.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        check_split_artifacts(
            interim_path=interim_path,
            split_ids_path=split_ids_path,
            report_path=report_path,
        )
