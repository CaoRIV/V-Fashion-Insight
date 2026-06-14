"""Validate the raw FashionReviews dataset schema and data quality."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from datasets import Dataset, DatasetDict, DownloadConfig, load_dataset

from v_fashion_insight.common.constants import VALID_LABELS
from v_fashion_insight.common.logging import configure_logging
from v_fashion_insight.data.download import (
    DEFAULT_METADATA_PATH,
    write_metadata,
)

DEFAULT_REPORT_PATH = Path("reports/metrics/data_validation.json")

ID_COLUMN = "STT"
TEXT_COLUMN = "Nội dung review"
LABEL_COLUMNS = (
    "Chất liệu",
    "Kiểu dáng",
    "Kích cỡ",
    "Giá cả",
    "Dịch vụ",
)
REQUIRED_COLUMNS = (ID_COLUMN, TEXT_COLUMN, *LABEL_COLUMNS)

Severity = Literal["error", "warning"]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _issue(
    *,
    code: str,
    severity: Severity,
    split: str,
    message: str,
    column: str | None = None,
    count: int | None = None,
    sample_rows: Sequence[int] | None = None,
) -> dict[str, Any]:
    issue: dict[str, Any] = {
        "code": code,
        "severity": severity,
        "split": split,
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


def _validate_columns(
    split_name: str,
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    actual_columns = list(frame.columns)
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in actual_columns
    ]
    extra_columns = [
        column for column in actual_columns if column not in REQUIRED_COLUMNS
    ]

    if missing_columns:
        issues.append(
            _issue(
                code="missing_required_columns",
                severity="error",
                split=split_name,
                count=len(missing_columns),
                message=f"Missing required columns: {missing_columns!r}.",
            )
        )
    if extra_columns:
        issues.append(
            _issue(
                code="unexpected_columns",
                severity="warning",
                split=split_name,
                count=len(extra_columns),
                message=f"Unexpected columns: {extra_columns!r}.",
            )
        )

    return issues


def _validate_id_column(
    split_name: str,
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    if ID_COLUMN not in frame:
        return []

    issues: list[dict[str, Any]] = []
    values = frame[ID_COLUMN]
    null_mask = values.isna()
    if null_mask.any():
        issues.append(
            _issue(
                code="null_identifier",
                severity="error",
                split=split_name,
                column=ID_COLUMN,
                count=int(null_mask.sum()),
                sample_rows=_sample_indices(null_mask),
                message="Identifier values must not be null.",
            )
        )
    if not pd.api.types.is_integer_dtype(values.dtype):
        issues.append(
            _issue(
                code="invalid_identifier_type",
                severity="error",
                split=split_name,
                column=ID_COLUMN,
                message=f"Expected an integer identifier, got {values.dtype}.",
            )
        )

    return issues


def _validate_text_column(
    split_name: str,
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    if TEXT_COLUMN not in frame:
        return []

    issues: list[dict[str, Any]] = []
    values = frame[TEXT_COLUMN]
    null_mask = values.isna()
    if null_mask.any():
        issues.append(
            _issue(
                code="null_review",
                severity="error",
                split=split_name,
                column=TEXT_COLUMN,
                count=int(null_mask.sum()),
                sample_rows=_sample_indices(null_mask),
                message="Review text must not be null.",
            )
        )
    if not pd.api.types.is_string_dtype(values.dtype):
        issues.append(
            _issue(
                code="invalid_review_type",
                severity="error",
                split=split_name,
                column=TEXT_COLUMN,
                message=f"Expected review text, got {values.dtype}.",
            )
        )

    non_null_values = values[~null_mask].astype(str)
    empty_mask = pd.Series(False, index=frame.index)
    empty_mask.loc[non_null_values.index] = non_null_values.str.strip().eq("")
    if empty_mask.any():
        issues.append(
            _issue(
                code="empty_review",
                severity="error",
                split=split_name,
                column=TEXT_COLUMN,
                count=int(empty_mask.sum()),
                sample_rows=_sample_indices(empty_mask),
                message="Review text must not be empty or whitespace-only.",
            )
        )

    return issues


def _validate_label_column(
    split_name: str,
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
                code="null_label",
                severity="error",
                split=split_name,
                column=column,
                count=int(null_mask.sum()),
                sample_rows=_sample_indices(null_mask),
                message="Aspect labels must not be null.",
            )
        )

    non_null_values = values[~null_mask]
    if not pd.api.types.is_numeric_dtype(non_null_values.dtype):
        issues.append(
            _issue(
                code="invalid_label_type",
                severity="error",
                split=split_name,
                column=column,
                message=f"Expected numeric labels, got {values.dtype}.",
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
                split=split_name,
                column=column,
                count=int((~integer_mask).sum()),
                sample_rows=_sample_indices(invalid_mask),
                message="Aspect labels must be integer-valued.",
            )
        )

    allowed_mask = non_null_values.isin(VALID_LABELS)
    if not allowed_mask.all():
        invalid_mask = pd.Series(False, index=frame.index)
        invalid_mask.loc[allowed_mask.index] = ~allowed_mask
        invalid_values = sorted(
            {
                float(value)
                for value in non_null_values.loc[~allowed_mask].tolist()
            }
        )
        issues.append(
            _issue(
                code="label_out_of_range",
                severity="error",
                split=split_name,
                column=column,
                count=int((~allowed_mask).sum()),
                sample_rows=_sample_indices(invalid_mask),
                message=(
                    f"Labels must be in {sorted(VALID_LABELS)}; "
                    f"found {invalid_values!r}."
                ),
            )
        )

    return issues


def _validate_duplicates(
    split_name: str,
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    duplicate_row_mask = frame.duplicated(keep=False)
    if duplicate_row_mask.any():
        issues.append(
            _issue(
                code="duplicate_rows",
                severity="warning",
                split=split_name,
                count=int(duplicate_row_mask.sum()),
                sample_rows=_sample_indices(duplicate_row_mask),
                message="Fully duplicated rows were detected.",
            )
        )

    if TEXT_COLUMN in frame:
        duplicate_text_mask = frame[TEXT_COLUMN].notna() & frame.duplicated(
            subset=[TEXT_COLUMN],
            keep=False,
        )
        if duplicate_text_mask.any():
            issues.append(
                _issue(
                    code="duplicate_review_text",
                    severity="warning",
                    split=split_name,
                    column=TEXT_COLUMN,
                    count=int(duplicate_text_mask.sum()),
                    sample_rows=_sample_indices(duplicate_text_mask),
                    message=(
                        "Repeated review text was detected and must be grouped "
                        "before splitting."
                    ),
                )
            )

    return issues


def validate_split(
    split_name: str,
    split: Dataset,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate one dataset split and return its summary and issues."""
    frame = split.to_pandas()
    issues = _validate_columns(split_name, frame)

    if frame.empty:
        issues.append(
            _issue(
                code="empty_split",
                severity="error",
                split=split_name,
                count=0,
                message="Dataset splits must contain at least one row.",
            )
        )

    issues.extend(_validate_id_column(split_name, frame))
    issues.extend(_validate_text_column(split_name, frame))
    for column in LABEL_COLUMNS:
        issues.extend(_validate_label_column(split_name, frame, column))
    issues.extend(_validate_duplicates(split_name, frame))

    split_summary = {
        "num_rows": len(frame),
        "column_names": list(frame.columns),
        "dtypes": {
            column: str(dtype) for column, dtype in frame.dtypes.items()
        },
        "fingerprint": split._fingerprint,
        "error_count": sum(issue["severity"] == "error" for issue in issues),
        "warning_count": sum(
            issue["severity"] == "warning" for issue in issues
        ),
    }
    return split_summary, issues


def validate_dataset(
    dataset: DatasetDict,
    *,
    source: Mapping[str, Any] | None = None,
    validated_at: datetime | None = None,
) -> dict[str, Any]:
    """Validate all splits and return a machine-readable report."""
    split_summaries: dict[str, Any] = {}
    issues: list[dict[str, Any]] = []

    if not dataset:
        issues.append(
            _issue(
                code="no_splits",
                severity="error",
                split="<dataset>",
                count=0,
                message="The dataset must contain at least one split.",
            )
        )

    for split_name, split in sorted(dataset.items()):
        split_summary, split_issues = validate_split(split_name, split)
        split_summaries[split_name] = split_summary
        issues.extend(split_issues)

    error_count = sum(issue["severity"] == "error" for issue in issues)
    warning_count = sum(issue["severity"] == "warning" for issue in issues)
    timestamp = validated_at or _utc_now()
    return {
        "status": "failed" if error_count else "passed",
        "valid": error_count == 0,
        "validated_at_utc": timestamp.astimezone(UTC).isoformat(),
        "source": dict(source or {}),
        "contract": {
            "required_columns": list(REQUIRED_COLUMNS),
            "id_column": ID_COLUMN,
            "text_column": TEXT_COLUMN,
            "label_columns": list(LABEL_COLUMNS),
            "valid_labels": sorted(VALID_LABELS),
        },
        "summary": {
            "split_count": len(dataset),
            "total_rows": sum(len(split) for split in dataset.values()),
            "error_count": error_count,
            "warning_count": warning_count,
        },
        "splits": split_summaries,
        "issues": issues,
    }


def load_dataset_from_metadata(
    metadata_path: Path = DEFAULT_METADATA_PATH,
    *,
    local_files_only: bool = True,
) -> tuple[DatasetDict, dict[str, Any]]:
    """Load the exact dataset revision recorded by the download task."""
    metadata_path = Path(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required_metadata = (
        "dataset_name",
        "resolved_revision",
        "cache_dir",
    )
    missing_metadata = [
        key for key in required_metadata if not metadata.get(key)
    ]
    if missing_metadata:
        raise ValueError(
            f"Download metadata is missing required keys: {missing_metadata!r}."
        )

    cache_dir = Path(metadata["cache_dir"])
    loaded_dataset = load_dataset(
        path=metadata["dataset_name"],
        name=metadata.get("config_name"),
        revision=metadata["resolved_revision"],
        cache_dir=str(cache_dir),
        download_config=DownloadConfig(
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        ),
    )
    if not isinstance(loaded_dataset, DatasetDict):
        raise TypeError(
            "Expected load_dataset() to return a DatasetDict when no split is "
            f"requested, got {type(loaded_dataset).__name__}."
        )

    source = {
        "metadata_path": metadata_path.as_posix(),
        "dataset_name": metadata["dataset_name"],
        "config_name": metadata.get("config_name"),
        "resolved_revision": metadata["resolved_revision"],
        "cache_dir": cache_dir.as_posix(),
        "downloaded_at_utc": metadata.get("downloaded_at_utc"),
    }
    return loaded_dataset, source


def _console_safe(value: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return value.encode(encoding, errors="backslashreplace").decode(encoding)


def format_summary(report: Mapping[str, Any]) -> str:
    """Format a concise human-readable validation summary."""
    summary = report["summary"]
    lines = [
        (
            f"Validation {str(report['status']).upper()}: "
            f"{summary['total_rows']} rows, "
            f"{summary['error_count']} error(s), "
            f"{summary['warning_count']} warning(s)."
        )
    ]
    for issue in report["issues"]:
        location = issue["split"]
        if issue.get("column"):
            location = f"{location}/{issue['column']}"
        count = (
            f" count={issue['count']}" if "count" in issue else ""
        )
        lines.append(
            f"- {issue['severity'].upper()} {issue['code']} "
            f"[{location}]{count}: {issue['message']}"
        )
    return _console_safe("\n".join(lines))


def validate_downloaded_dataset(
    *,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Load, validate, and write a report for the downloaded dataset."""
    dataset, source = load_dataset_from_metadata(
        metadata_path,
        local_files_only=local_files_only,
    )
    report = validate_dataset(dataset, source=source)
    write_metadata(report, Path(report_path))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the downloaded FashionReviews schema, labels, missing "
            "values, and duplicates."
        )
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help=f"Download metadata path (default: {DEFAULT_METADATA_PATH}).",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help=f"Validation report path (default: {DEFAULT_REPORT_PATH}).",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow Hugging Face network access if the local cache is missing.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = configure_logging()
    try:
        report = validate_downloaded_dataset(
            metadata_path=args.metadata_path,
            report_path=args.report_path,
            local_files_only=not args.allow_network,
        )
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError) as error:
        logger.error("Dataset validation could not run: %s", error)
        return 2

    print(format_summary(report))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
