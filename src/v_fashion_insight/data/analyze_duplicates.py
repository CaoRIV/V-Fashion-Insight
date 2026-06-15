"""Analyze exact normalized duplicate reviews without modifying the dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict

from v_fashion_insight.common.logging import configure_logging
from v_fashion_insight.data.download import (
    DEFAULT_METADATA_PATH,
    write_metadata,
)
from v_fashion_insight.data.validate import (
    ID_COLUMN,
    LABEL_COLUMNS,
    TEXT_COLUMN,
    load_dataset_from_metadata,
)

DEFAULT_JSON_PATH = Path("reports/metrics/exact_duplicate_analysis.json")
DEFAULT_GROUP_CSV_PATH = Path("reports/metrics/exact_duplicate_groups.csv")

WHITESPACE_PATTERN = re.compile(r"\s+")
ZERO_WIDTH_PATTERN = re.compile(r"[\u200b\u200c\u200d\ufeff]")
PREVIEW_LENGTH = 160
REPRESENTATIVE_GROUP_LIMIT = 20


def normalize_for_duplicate_analysis(text: str) -> str:
    """Return a conservative canonical form used only for duplicate analysis."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = ZERO_WIDTH_PATTERN.sub("", normalized)
    normalized = WHITESPACE_PATTERN.sub(" ", normalized)
    return normalized.strip().casefold()


def duplicate_group_id(normalized_text: str) -> str:
    """Create a stable identifier without exposing the complete review text."""
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def _serialize_identifier(value: Any) -> int | str | None:
    if pd.isna(value):
        return None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real) and float(value).is_integer():
        return int(value)
    return str(value)


def _preview(text: str, limit: int = PREVIEW_LENGTH) -> str:
    compact = WHITESPACE_PATTERN.sub(" ", text).strip()
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."


def _label_consistency(group: pd.DataFrame) -> dict[str, Any]:
    conflicting_columns: list[str] = []
    mixed_missing_columns: list[str] = []
    label_values: dict[str, list[int]] = {}

    for column in LABEL_COLUMNS:
        values = group[column]
        non_null_values = sorted(
            {int(value) for value in values.dropna().tolist()}
        )
        label_values[column] = non_null_values
        if len(non_null_values) > 1:
            conflicting_columns.append(column)
        if values.isna().any() and values.notna().any():
            mixed_missing_columns.append(column)

    return {
        "conflicting_label_columns": conflicting_columns,
        "mixed_missing_label_columns": mixed_missing_columns,
        "label_values": label_values,
    }


def _group_record(
    group_id: str,
    group: pd.DataFrame,
) -> dict[str, Any]:
    raw_variants = group[TEXT_COLUMN].drop_duplicates().tolist()
    consistency = _label_consistency(group)
    member_ids = [
        _serialize_identifier(value) for value in group[ID_COLUMN].tolist()
    ]
    return {
        "group_id": group_id,
        "member_count": len(group),
        "redundant_row_count": len(group) - 1,
        "raw_variant_count": len(raw_variants),
        "normalization_merged_variants": len(raw_variants) > 1,
        "has_label_conflict": bool(
            consistency["conflicting_label_columns"]
        ),
        "has_mixed_missing_labels": bool(
            consistency["mixed_missing_label_columns"]
        ),
        "conflicting_label_columns": consistency[
            "conflicting_label_columns"
        ],
        "mixed_missing_label_columns": consistency[
            "mixed_missing_label_columns"
        ],
        "label_values": consistency["label_values"],
        "member_ids": member_ids,
        "representative_preview": _preview(raw_variants[0]),
    }


def analyze_split(split_name: str, split: Dataset) -> dict[str, Any]:
    """Analyze raw and normalized exact duplicates in one split."""
    frame = split.to_pandas()
    required_columns = (ID_COLUMN, TEXT_COLUMN, *LABEL_COLUMNS)
    missing_columns = [
        column for column in required_columns if column not in frame
    ]
    if missing_columns:
        raise ValueError(
            f"Cannot analyze duplicates; missing columns: {missing_columns!r}."
        )
    if frame[TEXT_COLUMN].isna().any():
        raise ValueError("Cannot analyze duplicates with null review text.")

    working_frame = frame.loc[:, required_columns].copy()
    working_frame["_normalized_text"] = working_frame[TEXT_COLUMN].map(
        normalize_for_duplicate_analysis
    )
    if working_frame["_normalized_text"].eq("").any():
        raise ValueError("Cannot analyze empty normalized review text.")
    working_frame["_group_id"] = working_frame["_normalized_text"].map(
        duplicate_group_id
    )

    raw_sizes = working_frame.groupby(TEXT_COLUMN, sort=True).size()
    raw_duplicate_sizes = raw_sizes.loc[raw_sizes.gt(1)]

    group_records = [
        _group_record(group_id, group)
        for group_id, group in working_frame.groupby("_group_id", sort=True)
        if len(group) > 1
    ]
    group_records.sort(
        key=lambda record: (
            not record["has_label_conflict"],
            not record["has_mixed_missing_labels"],
            -record["member_count"],
            record["group_id"],
        )
    )

    duplicate_member_count = sum(
        record["member_count"] for record in group_records
    )
    conflict_group_count = sum(
        record["has_label_conflict"] for record in group_records
    )
    mixed_missing_group_count = sum(
        record["has_mixed_missing_labels"] for record in group_records
    )
    normalization_merged_group_count = sum(
        record["normalization_merged_variants"] for record in group_records
    )

    return {
        "split": split_name,
        "num_rows": len(working_frame),
        "fingerprint": split._fingerprint,
        "raw_text": {
            "unique_count": int(raw_sizes.size),
            "duplicate_group_count": int(raw_duplicate_sizes.size),
            "duplicate_member_count": int(raw_duplicate_sizes.sum()),
            "redundant_row_count": int((raw_duplicate_sizes - 1).sum()),
        },
        "normalized_text": {
            "unique_count": int(working_frame["_group_id"].nunique()),
            "duplicate_group_count": len(group_records),
            "duplicate_member_count": duplicate_member_count,
            "redundant_row_count": sum(
                record["redundant_row_count"] for record in group_records
            ),
            "normalization_merged_group_count": (
                normalization_merged_group_count
            ),
            "label_conflict_group_count": conflict_group_count,
            "mixed_missing_group_count": mixed_missing_group_count,
            "consistent_group_count": (
                len(group_records)
                - len(
                    {
                        record["group_id"]
                        for record in group_records
                        if record["has_label_conflict"]
                        or record["has_mixed_missing_labels"]
                    }
                )
            ),
        },
        "representative_groups": group_records[
            :REPRESENTATIVE_GROUP_LIMIT
        ],
        "groups": group_records,
    }


def analyze_dataset(
    dataset: DatasetDict,
    *,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic exact-duplicate analysis report."""
    if not dataset:
        raise ValueError("Cannot analyze a dataset without splits.")

    return {
        "source": dict(source or {}),
        "normalization": {
            "unicode_form": "NFKC",
            "case": "casefold",
            "whitespace": "collapse all Unicode whitespace to one space",
            "zero_width_characters": "remove U+200B/U+200C/U+200D/U+FEFF",
            "preserved": "Vietnamese diacritics, punctuation, and digits",
            "scope": "analysis only; source reviews are not modified",
        },
        "splits": {
            split_name: analyze_split(split_name, split)
            for split_name, split in sorted(dataset.items())
        },
    }


def _group_csv_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, split_report in report["splits"].items():
        for group in split_report["groups"]:
            rows.append(
                {
                    "split": split_name,
                    "group_id": group["group_id"],
                    "member_count": group["member_count"],
                    "redundant_row_count": group["redundant_row_count"],
                    "raw_variant_count": group["raw_variant_count"],
                    "normalization_merged_variants": group[
                        "normalization_merged_variants"
                    ],
                    "has_label_conflict": group["has_label_conflict"],
                    "has_mixed_missing_labels": group[
                        "has_mixed_missing_labels"
                    ],
                    "conflicting_label_columns": "|".join(
                        group["conflicting_label_columns"]
                    ),
                    "mixed_missing_label_columns": "|".join(
                        group["mixed_missing_label_columns"]
                    ),
                    "member_ids": "|".join(
                        str(member_id)
                        for member_id in group["member_ids"]
                    ),
                    "representative_preview": group[
                        "representative_preview"
                    ],
                }
            )
    return rows


def write_csv(
    rows: Sequence[Mapping[str, Any]],
    destination: Path,
) -> None:
    """Write duplicate groups as deterministic UTF-8 CSV."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "group_id",
        "member_count",
        "redundant_row_count",
        "raw_variant_count",
        "normalization_merged_variants",
        "has_label_conflict",
        "has_mixed_missing_labels",
        "conflicting_label_columns",
        "mixed_missing_label_columns",
        "member_ids",
        "representative_preview",
    ]
    with destination.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_downloaded_dataset(
    *,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    json_path: Path = DEFAULT_JSON_PATH,
    group_csv_path: Path = DEFAULT_GROUP_CSV_PATH,
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Load the pinned dataset and write exact-duplicate reports."""
    dataset, source = load_dataset_from_metadata(
        metadata_path,
        local_files_only=local_files_only,
    )
    report = analyze_dataset(dataset, source=source)
    write_metadata(report, Path(json_path))
    write_csv(_group_csv_rows(report), Path(group_csv_path))
    return report


def format_summary(report: Mapping[str, Any]) -> str:
    """Format a concise exact-duplicate summary."""
    lines: list[str] = []
    for split_name, split_report in report["splits"].items():
        raw = split_report["raw_text"]
        normalized = split_report["normalized_text"]
        lines.extend(
            [
                f"Split {split_name}: {split_report['num_rows']} reviews",
                (
                    "- raw duplicate groups="
                    f"{raw['duplicate_group_count']}, "
                    f"members={raw['duplicate_member_count']}, "
                    f"redundant={raw['redundant_row_count']}"
                ),
                (
                    "- normalized duplicate groups="
                    f"{normalized['duplicate_group_count']}, "
                    f"members={normalized['duplicate_member_count']}, "
                    f"redundant={normalized['redundant_row_count']}"
                ),
                (
                    "- normalization-merged groups="
                    f"{normalized['normalization_merged_group_count']}, "
                    f"label-conflict groups="
                    f"{normalized['label_conflict_group_count']}, "
                    f"mixed-missing groups="
                    f"{normalized['mixed_missing_group_count']}"
                ),
            ]
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze exact normalized duplicate reviews and label "
            "consistency without changing source data."
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
        "--group-csv-path",
        type=Path,
        default=DEFAULT_GROUP_CSV_PATH,
        help=f"Duplicate-group CSV path (default: {DEFAULT_GROUP_CSV_PATH}).",
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
        report = analyze_downloaded_dataset(
            metadata_path=args.metadata_path,
            json_path=args.json_path,
            group_csv_path=args.group_csv_path,
            local_files_only=not args.allow_network,
        )
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        logger.error("Exact duplicate analysis could not run: %s", error)
        return 2

    print(format_summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
