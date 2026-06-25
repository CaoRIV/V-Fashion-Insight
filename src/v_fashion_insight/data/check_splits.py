"""Leakage and distribution checks for frozen split membership."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import pandas as pd

from v_fashion_insight.common.constants import ASPECTS, VALID_LABELS
from v_fashion_insight.common.logging import configure_logging
from v_fashion_insight.data.download import write_metadata
from v_fashion_insight.data.preprocess import (
    DEFAULT_INTERIM_PATH,
    normalize_review_text,
)
from v_fashion_insight.data.processed_contract import (
    GROUP_ID_COLUMN,
    REVIEW_ID_COLUMN,
    SPLIT_COLUMN,
    SPLIT_NAMES,
    TEXT_COLUMN,
)
from v_fashion_insight.data.split import (
    DEFAULT_SPLIT_IDS_PATH,
    MISSING_LABEL_TOKEN,
)

DEFAULT_SPLIT_REPORT_PATH = Path("reports/metrics/split_report.json")
DEFAULT_MAX_LABEL_PROPORTION_DRIFT: Final[float] = 0.05
SPLIT_CHECK_SCHEMA_VERSION: Final[str] = "v1"
_NORMALIZED_TEXT_COLUMN: Final[str] = "_normalized_text"


class SplitCheckError(ValueError):
    """Raised when split leakage or distribution checks fail."""


def _console_safe(value: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return value.encode(encoding, errors="backslashreplace").decode(encoding)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ensure_output_path_available(path: Path, *, force: bool) -> None:
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(
            "Refusing to overwrite existing split report: "
            f"{path.as_posix()}. Use --force to replace it."
        )


def _issue(
    *,
    code: str,
    severity: str = "error",
    message: str,
    count: int | None = None,
    samples: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    issue: dict[str, Any] = {
        "code": code,
        "severity": severity,
        "message": message,
    }
    if count is not None:
        issue["count"] = int(count)
    if samples:
        issue["samples"] = list(samples)
    return issue


def _check_result(issues: Sequence[dict[str, Any]]) -> dict[str, Any]:
    error_count = sum(issue["severity"] == "error" for issue in issues)
    return {
        "status": "failed" if error_count else "passed",
        "error_count": error_count,
        "issues": list(issues),
    }


def _string_column(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].astype("string").str.strip().astype(str)


def _prepare_interim_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = (
        REVIEW_ID_COLUMN,
        GROUP_ID_COLUMN,
        TEXT_COLUMN,
        *ASPECTS,
    )
    missing_columns = [
        column for column in required_columns if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Interim data is missing required columns: {missing_columns!r}."
        )

    prepared = frame.loc[:, required_columns].copy()
    prepared[REVIEW_ID_COLUMN] = _string_column(prepared, REVIEW_ID_COLUMN)
    prepared[GROUP_ID_COLUMN] = _string_column(prepared, GROUP_ID_COLUMN)
    prepared[TEXT_COLUMN] = _string_column(prepared, TEXT_COLUMN)
    return prepared


def _prepare_split_ids(frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = (REVIEW_ID_COLUMN, GROUP_ID_COLUMN, SPLIT_COLUMN)
    missing_columns = [
        column for column in required_columns if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Split IDs are missing required columns: {missing_columns!r}."
        )

    prepared = frame.loc[:, required_columns].copy()
    prepared[REVIEW_ID_COLUMN] = _string_column(prepared, REVIEW_ID_COLUMN)
    prepared[GROUP_ID_COLUMN] = _string_column(prepared, GROUP_ID_COLUMN)
    prepared[SPLIT_COLUMN] = _string_column(prepared, SPLIT_COLUMN)
    return prepared


def _contract_check(
    interim: pd.DataFrame,
    split_ids: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if interim.empty:
        issues.append(
            _issue(
                code="empty_interim_dataset",
                message="Interim dataset must contain at least one row.",
            )
        )
    if split_ids.empty:
        issues.append(
            _issue(
                code="empty_split_ids",
                message="Split ID artifact must contain at least one row.",
            )
        )

    duplicate_split_ids = split_ids[REVIEW_ID_COLUMN].duplicated(keep=False)
    if duplicate_split_ids.any():
        issues.append(
            _issue(
                code="duplicate_split_review_id",
                count=int(duplicate_split_ids.sum()),
                message="Split ID artifact must contain one row per review_id.",
                samples=[
                    {REVIEW_ID_COLUMN: review_id}
                    for review_id in sorted(
                        split_ids.loc[
                            duplicate_split_ids,
                            REVIEW_ID_COLUMN,
                        ].unique()
                    )[:10]
                ],
            )
        )

    invalid_splits = ~split_ids[SPLIT_COLUMN].isin(SPLIT_NAMES)
    if invalid_splits.any():
        issues.append(
            _issue(
                code="invalid_split_name",
                count=int(invalid_splits.sum()),
                message=f"Split names must be one of {list(SPLIT_NAMES)!r}.",
                samples=[
                    {
                        REVIEW_ID_COLUMN: row[REVIEW_ID_COLUMN],
                        SPLIT_COLUMN: row[SPLIT_COLUMN],
                    }
                    for row in split_ids.loc[invalid_splits]
                    .head(10)
                    .to_dict("records")
                ],
            )
        )

    interim_ids = set(interim[REVIEW_ID_COLUMN])
    split_id_values = set(split_ids[REVIEW_ID_COLUMN])
    missing_ids = sorted(interim_ids.difference(split_id_values))
    unknown_ids = sorted(split_id_values.difference(interim_ids))
    if missing_ids:
        issues.append(
            _issue(
                code="missing_split_review_id",
                count=len(missing_ids),
                message="Every interim review_id must have a split assignment.",
                samples=[
                    {REVIEW_ID_COLUMN: review_id}
                    for review_id in missing_ids[:10]
                ],
            )
        )
    if unknown_ids:
        issues.append(
            _issue(
                code="unknown_split_review_id",
                count=len(unknown_ids),
                message="Split IDs must refer only to interim review_id values.",
                samples=[
                    {REVIEW_ID_COLUMN: review_id}
                    for review_id in unknown_ids[:10]
                ],
            )
        )

    merged = interim.merge(
        split_ids,
        on=REVIEW_ID_COLUMN,
        how="inner",
        suffixes=("_interim", "_split"),
        validate="one_to_one"
        if not duplicate_split_ids.any()
        else "one_to_many",
    )
    if not merged.empty:
        mismatch = (
            merged[f"{GROUP_ID_COLUMN}_interim"]
            != merged[f"{GROUP_ID_COLUMN}_split"]
        )
        if mismatch.any():
            issues.append(
                _issue(
                    code="split_group_id_mismatch",
                    count=int(mismatch.sum()),
                    message=(
                        "group_id in split IDs must match the interim "
                        "group_id for the same review_id."
                    ),
                    samples=[
                        {
                            REVIEW_ID_COLUMN: row[REVIEW_ID_COLUMN],
                            "interim_group_id": row[
                                f"{GROUP_ID_COLUMN}_interim"
                            ],
                            "split_group_id": row[
                                f"{GROUP_ID_COLUMN}_split"
                            ],
                        }
                        for row in merged.loc[mismatch]
                        .head(10)
                        .to_dict("records")
                    ],
                )
            )

    if not split_ids.empty:
        split_counts = split_ids[SPLIT_COLUMN].value_counts().to_dict()
        empty_splits = [
            split_name
            for split_name in SPLIT_NAMES
            if split_counts.get(split_name, 0) == 0
        ]
        if empty_splits:
            issues.append(
                _issue(
                    code="empty_split",
                    count=len(empty_splits),
                    message="All configured splits must contain rows.",
                    samples=[
                        {SPLIT_COLUMN: split_name}
                        for split_name in empty_splits
                    ],
                )
            )

    checked = merged.rename(
        columns={f"{GROUP_ID_COLUMN}_interim": GROUP_ID_COLUMN}
    ).drop(columns=[f"{GROUP_ID_COLUMN}_split"], errors="ignore")
    return checked, _check_result(issues)


def _group_leakage_check(merged: pd.DataFrame) -> dict[str, Any]:
    if merged.empty:
        return _check_result([])

    split_counts = merged.groupby(GROUP_ID_COLUMN)[SPLIT_COLUMN].nunique()
    leaked_group_ids = sorted(split_counts[split_counts > 1].index)
    issues: list[dict[str, Any]] = []
    if leaked_group_ids:
        samples = []
        for group_id in leaked_group_ids[:10]:
            group = merged.loc[merged[GROUP_ID_COLUMN] == group_id]
            samples.append(
                {
                    GROUP_ID_COLUMN: group_id,
                    "splits": sorted(group[SPLIT_COLUMN].unique()),
                    "review_ids": sorted(
                        group[REVIEW_ID_COLUMN].astype(str)
                    )[:10],
                }
            )
        issues.append(
            _issue(
                code="group_id_crosses_splits",
                count=len(leaked_group_ids),
                message="No group_id may appear in more than one split.",
                samples=samples,
            )
        )
    return _check_result(issues)


def _duplicate_text_leakage_check(merged: pd.DataFrame) -> dict[str, Any]:
    if merged.empty:
        return _check_result([])

    checked = merged.copy()
    checked[_NORMALIZED_TEXT_COLUMN] = checked[TEXT_COLUMN].map(
        normalize_review_text
    )
    grouped = checked.groupby(_NORMALIZED_TEXT_COLUMN)[SPLIT_COLUMN].agg(
        ["count", "nunique"]
    )
    leaked_texts = sorted(
        grouped.loc[
            (grouped["count"] > 1) & (grouped["nunique"] > 1)
        ].index
    )

    issues: list[dict[str, Any]] = []
    if leaked_texts:
        samples = []
        for text in leaked_texts[:10]:
            duplicate_rows = checked.loc[
                checked[_NORMALIZED_TEXT_COLUMN] == text
            ]
            samples.append(
                {
                    "normalized_text_sha256": _text_sha256(text),
                    "row_count": int(len(duplicate_rows)),
                    "splits": sorted(duplicate_rows[SPLIT_COLUMN].unique()),
                    "review_ids": sorted(
                        duplicate_rows[REVIEW_ID_COLUMN].astype(str)
                    )[:10],
                }
            )
        issues.append(
            _issue(
                code="normalized_exact_duplicate_crosses_splits",
                count=len(leaked_texts),
                message=(
                    "Rows with the same normalized text must not appear in "
                    "different splits."
                ),
                samples=samples,
            )
        )
    return _check_result(issues)


def _label_counts(frame: pd.DataFrame, aspect: str) -> dict[str, int]:
    values = frame[aspect]
    counts = {
        str(label): int(values.eq(label).sum())
        for label in sorted(VALID_LABELS)
    }
    counts[MISSING_LABEL_TOKEN] = int(values.isna().sum())
    return counts


def _label_distribution_check(
    merged: pd.DataFrame,
    *,
    max_label_proportion_drift: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if max_label_proportion_drift < 0:
        raise ValueError("max_label_proportion_drift must be non-negative.")
    if merged.empty:
        return _check_result([]), {}

    overall_rows = len(merged)
    distributions: dict[str, Any] = {}
    violations: list[dict[str, Any]] = []
    max_observed_drift = 0.0

    for aspect in ASPECTS:
        overall_counts = _label_counts(merged, aspect)
        aspect_distribution: dict[str, Any] = {
            "overall": {
                label: {
                    "count": count,
                    "proportion": round(count / overall_rows, 8),
                }
                for label, count in overall_counts.items()
            },
            "splits": {},
        }
        for split_name in SPLIT_NAMES:
            split_frame = merged.loc[merged[SPLIT_COLUMN] == split_name]
            split_rows = len(split_frame)
            split_counts = _label_counts(split_frame, aspect)
            split_distribution: dict[str, Any] = {
                "row_count": int(split_rows),
                "labels": {},
            }
            for label, split_count in split_counts.items():
                overall_proportion = overall_counts[label] / overall_rows
                split_proportion = (
                    split_count / split_rows if split_rows else 0.0
                )
                drift = abs(split_proportion - overall_proportion)
                max_observed_drift = max(max_observed_drift, drift)
                split_distribution["labels"][label] = {
                    "count": split_count,
                    "proportion": round(split_proportion, 8),
                    "overall_proportion": round(overall_proportion, 8),
                    "absolute_drift": round(drift, 8),
                }
                if drift > max_label_proportion_drift:
                    violations.append(
                        {
                            "aspect": aspect,
                            "label": label,
                            SPLIT_COLUMN: split_name,
                            "split_proportion": round(split_proportion, 8),
                            "overall_proportion": round(
                                overall_proportion,
                                8,
                            ),
                            "absolute_drift": round(drift, 8),
                        }
                    )
            aspect_distribution["splits"][split_name] = split_distribution
        distributions[aspect] = aspect_distribution

    issues = []
    if violations:
        issues.append(
            _issue(
                code="label_distribution_drift_exceeds_tolerance",
                count=len(violations),
                message=(
                    "Aspect-label split proportions must stay within the "
                    "configured absolute drift tolerance."
                ),
                samples=violations[:20],
            )
        )

    return _check_result(issues), {
        "max_label_proportion_drift": round(max_observed_drift, 8),
        "tolerance": max_label_proportion_drift,
        "distributions": distributions,
    }


def build_split_report(
    interim_frame: pd.DataFrame,
    split_ids_frame: pd.DataFrame,
    *,
    max_label_proportion_drift: float = DEFAULT_MAX_LABEL_PROPORTION_DRIFT,
    interim_path: Path | None = None,
    split_ids_path: Path | None = None,
    interim_sha256: str | None = None,
    split_ids_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a machine-readable report for split leakage checks."""
    interim = _prepare_interim_frame(interim_frame)
    split_ids = _prepare_split_ids(split_ids_frame)
    merged, contract = _contract_check(interim, split_ids)
    checks = {
        "split_id_contract": contract,
        "group_leakage": _group_leakage_check(merged),
        "normalized_exact_duplicate_leakage": (
            _duplicate_text_leakage_check(merged)
        ),
    }
    distribution_check, distribution_summary = _label_distribution_check(
        merged,
        max_label_proportion_drift=max_label_proportion_drift,
    )
    checks["label_distribution_drift"] = distribution_check

    issues = [
        issue
        for check in checks.values()
        for issue in check["issues"]
    ]
    error_count = sum(issue["severity"] == "error" for issue in issues)
    split_counts = (
        split_ids[SPLIT_COLUMN]
        .value_counts()
        .reindex(SPLIT_NAMES, fill_value=0)
        .astype(int)
        .to_dict()
    )
    return {
        "schema_version": SPLIT_CHECK_SCHEMA_VERSION,
        "status": "failed" if error_count else "passed",
        "valid": error_count == 0,
        "policy": {
            "max_label_proportion_drift": max_label_proportion_drift,
            "drift_metric": (
                "absolute difference between split and overall label "
                "proportions for every aspect, label, and missing-label bucket"
            ),
        },
        "inputs": {
            "interim_path": Path(interim_path).as_posix()
            if interim_path is not None
            else None,
            "interim_sha256": interim_sha256,
            "split_ids_path": Path(split_ids_path).as_posix()
            if split_ids_path is not None
            else None,
            "split_ids_sha256": split_ids_sha256,
        },
        "summary": {
            "row_count": int(len(interim)),
            "split_id_row_count": int(len(split_ids)),
            "group_count": int(interim[GROUP_ID_COLUMN].nunique()),
            "split_counts": {
                split_name: int(split_counts[split_name])
                for split_name in SPLIT_NAMES
            },
            **distribution_summary,
            "error_count": error_count,
        },
        "checks": checks,
        "issues": issues,
    }


def check_split_artifacts(
    *,
    interim_path: Path = DEFAULT_INTERIM_PATH,
    split_ids_path: Path = DEFAULT_SPLIT_IDS_PATH,
    report_path: Path = DEFAULT_SPLIT_REPORT_PATH,
    max_label_proportion_drift: float = DEFAULT_MAX_LABEL_PROPORTION_DRIFT,
    force: bool = False,
) -> dict[str, Any]:
    """Read split artifacts, write a report, and fail on leakage."""
    interim_path = Path(interim_path)
    split_ids_path = Path(split_ids_path)
    report_path = Path(report_path)
    if not interim_path.exists():
        raise FileNotFoundError(
            f"Interim dataset does not exist: {interim_path.as_posix()}."
        )
    if not split_ids_path.exists():
        raise FileNotFoundError(
            f"Split ID artifact does not exist: {split_ids_path.as_posix()}."
        )
    _ensure_output_path_available(report_path, force=force)

    interim_frame = pd.read_csv(interim_path)
    split_ids_frame = pd.read_csv(split_ids_path)
    report = build_split_report(
        interim_frame,
        split_ids_frame,
        max_label_proportion_drift=max_label_proportion_drift,
        interim_path=interim_path,
        split_ids_path=split_ids_path,
        interim_sha256=_file_sha256(interim_path),
        split_ids_sha256=_file_sha256(split_ids_path),
    )
    write_metadata(report, report_path)
    if not report["valid"]:
        raise SplitCheckError(
            "Split checks failed; see "
            f"{report_path.as_posix()} for details."
        )

    logger = configure_logging()
    logger.info(
        "Split checks passed: %d rows, %d groups, max drift %.6f.",
        report["summary"]["row_count"],
        report["summary"]["group_count"],
        report["summary"]["max_label_proportion_drift"],
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check split membership for group leakage, exact duplicate "
            "leakage, and aspect-label distribution drift."
        )
    )
    parser.add_argument(
        "--interim-path",
        type=Path,
        default=DEFAULT_INTERIM_PATH,
        help=f"Interim review CSV (default: {DEFAULT_INTERIM_PATH}).",
    )
    parser.add_argument(
        "--split-ids-path",
        type=Path,
        default=DEFAULT_SPLIT_IDS_PATH,
        help=f"Split membership CSV (default: {DEFAULT_SPLIT_IDS_PATH}).",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_SPLIT_REPORT_PATH,
        help=f"Split report JSON (default: {DEFAULT_SPLIT_REPORT_PATH}).",
    )
    parser.add_argument(
        "--max-label-proportion-drift",
        type=float,
        default=DEFAULT_MAX_LABEL_PROPORTION_DRIFT,
        help=(
            "Maximum allowed absolute split-vs-overall label proportion "
            f"drift (default: {DEFAULT_MAX_LABEL_PROPORTION_DRIFT})."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing split report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = check_split_artifacts(
            interim_path=args.interim_path,
            split_ids_path=args.split_ids_path,
            report_path=args.report_path,
            max_label_proportion_drift=args.max_label_proportion_drift,
            force=args.force,
        )
    except (
        FileNotFoundError,
        FileExistsError,
        SplitCheckError,
        TypeError,
        ValueError,
    ) as error:
        print(_console_safe(f"Split checks failed: {error}"), file=sys.stderr)
        return 2

    summary = report["summary"]
    print(
        _console_safe(
            "Split checks passed: "
            f"rows={summary['row_count']}, "
            f"groups={summary['group_count']}, "
            "max_label_proportion_drift="
            f"{summary['max_label_proportion_drift']:.6f}."
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
