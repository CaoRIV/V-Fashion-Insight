"""Profile aspect label distributions for the downloaded dataset."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict

from v_fashion_insight.common.constants import (
    ASPECTS,
    LABEL_NAMES,
    VALID_LABELS,
)
from v_fashion_insight.common.logging import configure_logging
from v_fashion_insight.data.download import (
    DEFAULT_METADATA_PATH,
    write_metadata,
)
from v_fashion_insight.data.validate import (
    LABEL_COLUMNS,
    load_dataset_from_metadata,
)

DEFAULT_JSON_PATH = Path("reports/metrics/label_distribution.json")
DEFAULT_LABEL_CSV_PATH = Path("reports/metrics/label_distribution.csv")
DEFAULT_MENTION_CSV_PATH = Path(
    "reports/metrics/mentioned_aspect_distribution.csv"
)

ASPECT_SOURCE_COLUMNS = dict(zip(ASPECTS, LABEL_COLUMNS, strict=True))
SENTIMENT_LABELS = frozenset({1, 2, 3})


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _require_label_columns(frame: pd.DataFrame) -> None:
    missing_columns = [
        column for column in LABEL_COLUMNS if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Cannot profile labels; missing columns: {missing_columns!r}."
        )


def _profile_aspect(
    values: pd.Series,
    *,
    aspect: str,
    source_column: str,
    total_rows: int,
) -> dict[str, Any]:
    non_null_values = values.dropna()
    invalid_values = non_null_values.loc[
        ~non_null_values.isin(VALID_LABELS)
        | ~non_null_values.mod(1).eq(0)
    ]
    if not invalid_values.empty:
        samples = sorted({float(value) for value in invalid_values.tolist()})
        raise ValueError(
            f"Cannot profile {source_column!r}; invalid labels: {samples!r}."
        )

    counts = {
        label: int(non_null_values.eq(label).sum())
        for label in sorted(VALID_LABELS)
    }
    missing_count = int(values.isna().sum())
    valid_count = int(len(non_null_values))
    mentioned_count = sum(counts[label] for label in SENTIMENT_LABELS)

    return {
        "aspect": aspect,
        "source_column": source_column,
        "total_rows": total_rows,
        "valid_label_count": valid_count,
        "missing_count": missing_count,
        "missing_proportion": _ratio(missing_count, total_rows),
        "not_mentioned_count": counts[0],
        "not_mentioned_proportion": _ratio(counts[0], total_rows),
        "mentioned_count": mentioned_count,
        "mentioned_proportion": _ratio(mentioned_count, total_rows),
        "mentioned_proportion_among_valid": _ratio(
            mentioned_count,
            valid_count,
        ),
        "labels": [
            {
                "label": label,
                "name": LABEL_NAMES[label],
                "count": counts[label],
                "proportion": _ratio(counts[label], total_rows),
                "proportion_among_valid": _ratio(counts[label], valid_count),
            }
            for label in sorted(VALID_LABELS)
        ],
    }


def profile_split(split_name: str, split: Dataset) -> dict[str, Any]:
    """Profile labels and mentioned-aspect counts for one dataset split."""
    frame = split.to_pandas()
    _require_label_columns(frame)
    total_rows = len(frame)

    aspects = [
        _profile_aspect(
            frame[source_column],
            aspect=aspect,
            source_column=source_column,
            total_rows=total_rows,
        )
        for aspect, source_column in ASPECT_SOURCE_COLUMNS.items()
    ]

    label_frame = frame.loc[:, LABEL_COLUMNS]
    mentioned_aspect_count = label_frame.isin(SENTIMENT_LABELS).sum(axis=1)
    rows_with_missing_labels = int(label_frame.isna().any(axis=1).sum())
    mention_distribution = [
        {
            "mentioned_aspect_count": count,
            "review_count": int(mentioned_aspect_count.eq(count).sum()),
            "proportion": _ratio(
                int(mentioned_aspect_count.eq(count).sum()),
                total_rows,
            ),
        }
        for count in range(len(LABEL_COLUMNS) + 1)
    ]

    return {
        "split": split_name,
        "num_rows": total_rows,
        "fingerprint": split._fingerprint,
        "aspects": aspects,
        "mentioned_aspects_per_review": {
            "mean": (
                float(mentioned_aspect_count.mean()) if total_rows else 0.0
            ),
            "rows_with_missing_labels": rows_with_missing_labels,
            "distribution": mention_distribution,
        },
    }


def profile_dataset(
    dataset: DatasetDict,
    *,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic label-distribution report."""
    if not dataset:
        raise ValueError("Cannot profile a dataset without splits.")

    return {
        "source": dict(source or {}),
        "label_mapping": {
            str(label): LABEL_NAMES[label] for label in sorted(VALID_LABELS)
        },
        "aspect_source_columns": ASPECT_SOURCE_COLUMNS,
        "splits": {
            split_name: profile_split(split_name, split)
            for split_name, split in sorted(dataset.items())
        },
    }


def _label_csv_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, split_report in report["splits"].items():
        for aspect_report in split_report["aspects"]:
            for label_report in aspect_report["labels"]:
                rows.append(
                    {
                        "split": split_name,
                        "aspect": aspect_report["aspect"],
                        "source_column": aspect_report["source_column"],
                        "label": label_report["label"],
                        "label_name": label_report["name"],
                        "count": label_report["count"],
                        "proportion": label_report["proportion"],
                        "proportion_among_valid": label_report[
                            "proportion_among_valid"
                        ],
                        "missing_count": aspect_report["missing_count"],
                        "mentioned_count": aspect_report["mentioned_count"],
                        "not_mentioned_count": aspect_report[
                            "not_mentioned_count"
                        ],
                    }
                )
    return rows


def _mention_csv_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, split_report in report["splits"].items():
        mention_report = split_report["mentioned_aspects_per_review"]
        for distribution_row in mention_report["distribution"]:
            rows.append(
                {
                    "split": split_name,
                    **distribution_row,
                    "rows_with_missing_labels": mention_report[
                        "rows_with_missing_labels"
                    ],
                }
            )
    return rows


def write_csv(
    rows: Sequence[Mapping[str, Any]],
    destination: Path,
) -> None:
    """Write deterministic UTF-8 CSV output."""
    if not rows:
        raise ValueError("Cannot write a CSV report without rows.")

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with destination.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def profile_downloaded_dataset(
    *,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    json_path: Path = DEFAULT_JSON_PATH,
    label_csv_path: Path = DEFAULT_LABEL_CSV_PATH,
    mention_csv_path: Path = DEFAULT_MENTION_CSV_PATH,
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Load the pinned dataset and write label-distribution reports."""
    dataset, source = load_dataset_from_metadata(
        metadata_path,
        local_files_only=local_files_only,
    )
    report = profile_dataset(dataset, source=source)
    write_metadata(report, Path(json_path))
    write_csv(_label_csv_rows(report), Path(label_csv_path))
    write_csv(_mention_csv_rows(report), Path(mention_csv_path))
    return report


def format_summary(report: Mapping[str, Any]) -> str:
    """Format a concise ASCII-safe console summary."""
    lines: list[str] = []
    for split_name, split_report in report["splits"].items():
        lines.append(
            f"Split {split_name}: {split_report['num_rows']} reviews"
        )
        for aspect_report in split_report["aspects"]:
            lines.append(
                "- "
                f"{aspect_report['aspect']}: "
                f"mentioned={aspect_report['mentioned_count']} "
                f"({aspect_report['mentioned_proportion']:.2%}), "
                f"not_mentioned={aspect_report['not_mentioned_count']}, "
                f"missing={aspect_report['missing_count']}"
            )
        mention_report = split_report["mentioned_aspects_per_review"]
        lines.append(
            "- mean mentioned aspects per review: "
            f"{mention_report['mean']:.4f}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Profile label counts, proportions, and mentioned aspects for the "
            "downloaded dataset."
        )
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help=f"Download metadata path (default: {DEFAULT_METADATA_PATH}).",
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=DEFAULT_JSON_PATH,
        help=f"JSON report path (default: {DEFAULT_JSON_PATH}).",
    )
    parser.add_argument(
        "--label-csv-path",
        type=Path,
        default=DEFAULT_LABEL_CSV_PATH,
        help=f"Label CSV path (default: {DEFAULT_LABEL_CSV_PATH}).",
    )
    parser.add_argument(
        "--mention-csv-path",
        type=Path,
        default=DEFAULT_MENTION_CSV_PATH,
        help=(
            "Mention-count CSV path "
            f"(default: {DEFAULT_MENTION_CSV_PATH})."
        ),
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
        report = profile_downloaded_dataset(
            metadata_path=args.metadata_path,
            json_path=args.json_path,
            label_csv_path=args.label_csv_path,
            mention_csv_path=args.mention_csv_path,
            local_files_only=not args.allow_network,
        )
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError) as error:
        logger.error("Label profiling could not run: %s", error)
        return 2

    print(format_summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
