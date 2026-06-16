import pandas as pd

from v_fashion_insight.data.processed_contract import (
    GROUP_ID_COLUMN,
    LABEL_COLUMNS,
    REQUIRED_COLUMNS,
    REVIEW_ID_COLUMN,
    SPLIT_COLUMN,
    TEXT_COLUMN,
    validate_processed_frame,
)


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            REVIEW_ID_COLUMN: ["r-001", "r-002", "r-003"],
            GROUP_ID_COLUMN: ["g-001", "g-001", "g-003"],
            TEXT_COLUMN: [
                "áo đẹp chất vải tốt",
                "áo đẹp chất vải tốt",
                "shop giao hàng nhanh",
            ],
            "material": [3, 3, 0],
            "design": [3, 3, 0],
            "size": [0, 0, 0],
            "price": [2, 2, 0],
            "service": [0, 0, 3],
            SPLIT_COLUMN: ["train", "train", "validation"],
        }
    )


def test_processed_contract_columns_match_plan() -> None:
    assert REQUIRED_COLUMNS == (
        "review_id",
        "group_id",
        "text",
        "material",
        "design",
        "size",
        "price",
        "service",
        "split",
    )
    assert LABEL_COLUMNS == (
        "material",
        "design",
        "size",
        "price",
        "service",
    )


def test_validate_processed_frame_accepts_valid_rows() -> None:
    report = validate_processed_frame(
        _valid_frame(),
        source={"task": "P2-T01"},
    )

    assert report["valid"] is True
    assert report["status"] == "passed"
    assert report["source"] == {"task": "P2-T01"}
    assert report["summary"]["error_count"] == 0
    assert report["summary"]["warning_count"] == 0
    assert report["contract"]["split"] == ["train", "validation", "test"]
    assert report["contract"]["valid_labels"] == [0, 1, 2, 3]


def test_missing_labels_are_preserved_as_warnings() -> None:
    frame = _valid_frame()
    frame.loc[1, "material"] = None

    report = validate_processed_frame(frame)

    assert report["valid"] is True
    assert report["summary"]["warning_count"] == 1
    assert report["issues"] == [
        {
            "code": "missing_label",
            "severity": "warning",
            "message": (
                "Missing labels are preserved as missing values and must not "
                "be converted to label 0."
            ),
            "column": "material",
            "count": 1,
            "sample_rows": [1],
        }
    ]


def test_invalid_processed_rows_report_deterministic_errors() -> None:
    frame = _valid_frame().drop(columns=["service"])
    frame.loc[1, REVIEW_ID_COLUMN] = "r-001"
    frame.loc[2, GROUP_ID_COLUMN] = " "
    frame.loc[0, TEXT_COLUMN] = ""
    frame.loc[1, "material"] = 4
    frame["design"] = frame["design"].astype(float)
    frame.loc[2, "design"] = 1.5
    frame.loc[0, SPLIT_COLUMN] = "dev"

    report = validate_processed_frame(frame)
    issue_codes = [issue["code"] for issue in report["issues"]]

    assert report["valid"] is False
    assert report["status"] == "failed"
    assert issue_codes == [
        "missing_required_columns",
        "duplicate_review_id",
        "empty_group_id",
        "empty_text",
        "invalid_split",
        "label_out_of_range",
        "non_integer_label",
    ]


def test_non_numeric_label_type_is_an_error() -> None:
    frame = _valid_frame()
    frame["price"] = ["0", "1", "2"]

    report = validate_processed_frame(frame)

    assert report["valid"] is False
    assert report["issues"] == [
        {
            "code": "invalid_label_type",
            "severity": "error",
            "message": "Expected numeric labels or missing values, got str.",
            "column": "price",
        }
    ]


def test_empty_processed_frame_fails_even_with_columns() -> None:
    report = validate_processed_frame(pd.DataFrame(columns=REQUIRED_COLUMNS))

    assert report["valid"] is False
    assert report["issues"] == [
        {
            "code": "empty_dataset",
            "severity": "error",
            "message": "Processed datasets must contain at least one row.",
            "count": 0,
        }
    ]
