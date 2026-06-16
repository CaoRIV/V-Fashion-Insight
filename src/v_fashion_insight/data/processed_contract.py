"""Processed dataset schema contract and validation helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import pandas as pd

from v_fashion_insight.common.constants import ASPECTS, VALID_LABELS

REVIEW_ID_COLUMN = "review_id"
GROUP_ID_COLUMN = "group_id"
TEXT_COLUMN = "text"
SPLIT_COLUMN = "split"
SPLIT_NAMES = ("train", "validation", "test")
LABEL_COLUMNS = ASPECTS
REQUIRED_COLUMNS = (
    REVIEW_ID_COLUMN,
    GROUP_ID_COLUMN,
    TEXT_COLUMN,
    *LABEL_COLUMNS,
    SPLIT_COLUMN,
)

Severity = Literal["error", "warning"]


def _issue(
    *,
    code: str,
    severity: Severity,
    message: str,
    column: str | None = None,
    count: int | None = None,
    sample_rows: Sequence[int] | None = None,
) -> dict[str, Any]:
    issue: dict[str, Any] = {
        "code": code,
        "severity": severity,
        "message": message,
    }
    if column is not None:
        issue["column"] = column
    if count is not None:
        issue["count"] = int(count)
    if sample_rows:
        issue["sample_rows"] = [int(index) for index in sample_rows]
    return issue


def _sample_indices(mask: pd.Series, limit: int = 10) -> list[int]:
    return [int(index) for index in mask[mask].index[:limit]]


def _string_values(values: pd.Series) -> pd.Series:
    return values.dropna().astype("string")


def _validate_required_columns(frame: pd.DataFrame) -> list[dict[str, Any]]:
    actual_columns = list(frame.columns)
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in actual_columns
    ]
    if not missing_columns:
        return []
    return [
        _issue(
            code="missing_required_columns",
            severity="error",
            count=len(missing_columns),
            message=f"Missing required columns: {missing_columns!r}.",
        )
    ]


def _validate_required_string_column(
    frame: pd.DataFrame,
    column: str,
) -> list[dict[str, Any]]:
    if column not in frame:
        return []

    issues: list[dict[str, Any]] = []
    values = frame[column]
    null_mask = values.isna()
    if null_mask.any():
        issues.append(
            _issue(
                code=f"null_{column}",
                severity="error",
                column=column,
                count=int(null_mask.sum()),
                sample_rows=_sample_indices(null_mask),
                message=f"{column!r} values must not be null.",
            )
        )

    text_values = _string_values(values)
    empty_mask = pd.Series(False, index=frame.index)
    empty_mask.loc[text_values.index] = text_values.str.strip().eq("")
    if empty_mask.any():
        issues.append(
            _issue(
                code=f"empty_{column}",
                severity="error",
                column=column,
                count=int(empty_mask.sum()),
                sample_rows=_sample_indices(empty_mask),
                message=f"{column!r} values must not be empty.",
            )
        )

    return issues


def _validate_review_ids(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if REVIEW_ID_COLUMN not in frame:
        return []

    issues = _validate_required_string_column(frame, REVIEW_ID_COLUMN)
    valid_id_mask = frame[REVIEW_ID_COLUMN].notna() & frame[
        REVIEW_ID_COLUMN
    ].astype("string").str.strip().ne("")
    duplicate_mask = valid_id_mask & frame.duplicated(
        subset=[REVIEW_ID_COLUMN],
        keep=False,
    )
    if duplicate_mask.any():
        issues.append(
            _issue(
                code="duplicate_review_id",
                severity="error",
                column=REVIEW_ID_COLUMN,
                count=int(duplicate_mask.sum()),
                sample_rows=_sample_indices(duplicate_mask),
                message="review_id values must be unique within the dataset.",
            )
        )

    return issues


def _validate_split_column(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if SPLIT_COLUMN not in frame:
        return []

    issues = _validate_required_string_column(frame, SPLIT_COLUMN)
    values = _string_values(frame[SPLIT_COLUMN]).str.strip()
    invalid_mask = pd.Series(False, index=frame.index)
    invalid_mask.loc[values.index] = ~values.isin(SPLIT_NAMES)
    if invalid_mask.any():
        invalid_values = sorted(
            {str(value) for value in values[~values.isin(SPLIT_NAMES)]}
        )
        issues.append(
            _issue(
                code="invalid_split",
                severity="error",
                column=SPLIT_COLUMN,
                count=int(invalid_mask.sum()),
                sample_rows=_sample_indices(invalid_mask),
                message=(
                    f"split must be one of {list(SPLIT_NAMES)!r}; "
                    f"found {invalid_values!r}."
                ),
            )
        )
    return issues


def _validate_label_column(
    frame: pd.DataFrame,
    column: str,
) -> list[dict[str, Any]]:
    if column not in frame:
        return []

    issues: list[dict[str, Any]] = []
    values = frame[column]
    null_mask = values.isna()
    if null_mask.any():
        issues.append(
            _issue(
                code="missing_label",
                severity="warning",
                column=column,
                count=int(null_mask.sum()),
                sample_rows=_sample_indices(null_mask),
                message=(
                    "Missing labels are preserved as missing values and must "
                    "not be converted to label 0."
                ),
            )
        )

    non_null_values = values[~null_mask]
    if non_null_values.empty:
        return issues
    if not pd.api.types.is_numeric_dtype(non_null_values.dtype):
        issues.append(
            _issue(
                code="invalid_label_type",
                severity="error",
                column=column,
                message=(
                    "Expected numeric labels or missing values, got "
                    f"{values.dtype}."
                ),
            )
        )
        return issues

    integer_mask = non_null_values.mod(1).eq(0)
    if not integer_mask.all():
        invalid_mask = pd.Series(False, index=frame.index)
        invalid_mask.loc[integer_mask.index] = ~integer_mask
        issues.append(
            _issue(
                code="non_integer_label",
                severity="error",
                column=column,
                count=int((~integer_mask).sum()),
                sample_rows=_sample_indices(invalid_mask),
                message="Aspect labels must be integer-valued.",
            )
        )

    integer_values = non_null_values.loc[integer_mask]
    allowed_mask = integer_values.isin(VALID_LABELS)
    if not allowed_mask.all():
        invalid_mask = pd.Series(False, index=frame.index)
        invalid_mask.loc[allowed_mask.index] = ~allowed_mask
        invalid_values = sorted(
            {
                float(value)
                for value in integer_values.loc[~allowed_mask].tolist()
            }
        )
        issues.append(
            _issue(
                code="label_out_of_range",
                severity="error",
                column=column,
                count=int((~allowed_mask).sum()),
                sample_rows=_sample_indices(invalid_mask),
                message=(
                    f"Labels must be in {sorted(VALID_LABELS)} or missing; "
                    f"found {invalid_values!r}."
                ),
            )
        )

    return issues


def validate_processed_frame(
    frame: pd.DataFrame,
    *,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate processed rows against the project data contract."""
    issues: list[dict[str, Any]] = []
    issues.extend(_validate_required_columns(frame))

    if frame.empty:
        issues.append(
            _issue(
                code="empty_dataset",
                severity="error",
                count=0,
                message="Processed datasets must contain at least one row.",
            )
        )

    issues.extend(_validate_review_ids(frame))
    issues.extend(_validate_required_string_column(frame, GROUP_ID_COLUMN))
    issues.extend(_validate_required_string_column(frame, TEXT_COLUMN))
    issues.extend(_validate_split_column(frame))
    for column in LABEL_COLUMNS:
        issues.extend(_validate_label_column(frame, column))

    error_count = sum(issue["severity"] == "error" for issue in issues)
    warning_count = sum(issue["severity"] == "warning" for issue in issues)
    return {
        "status": "failed" if error_count else "passed",
        "valid": error_count == 0,
        "source": dict(source or {}),
        "contract": {
            "required_columns": list(REQUIRED_COLUMNS),
            "review_id": (
                "Unique stable sample identifier for each retained review row."
            ),
            "group_id": (
                "Stable identifier shared by original, exact-duplicate, and "
                "high-confidence augmented review variants."
            ),
            "text": "Conservatively normalized review text.",
            "split": list(SPLIT_NAMES),
            "label_columns": list(LABEL_COLUMNS),
            "valid_labels": sorted(VALID_LABELS),
            "missing_label_policy": (
                "Missing labels are allowed, preserved as missing values, and "
                "reported as warnings."
            ),
        },
        "summary": {
            "total_rows": int(len(frame)),
            "column_names": list(frame.columns),
            "dtypes": {
                column: str(dtype) for column, dtype in frame.dtypes.items()
            },
            "error_count": error_count,
            "warning_count": warning_count,
        },
        "issues": issues,
    }
