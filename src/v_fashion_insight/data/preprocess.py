"""Conservative preprocessing helpers for review text."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any, Final

import pandas as pd
import datasets.config as datasets_config
import huggingface_hub.constants as hub_constants
from datasets import DatasetDict

from v_fashion_insight.common.constants import ASPECTS
from v_fashion_insight.common.logging import configure_logging
from v_fashion_insight.data.analyze_duplicates import (
    DEFAULT_GROUP_CSV_PATH as DEFAULT_EXACT_GROUP_PATH,
)
from v_fashion_insight.data.analyze_near_duplicates import (
    DEFAULT_CLUSTER_CSV_PATH as DEFAULT_NEAR_CLUSTER_PATH,
)
from v_fashion_insight.data.download import (
    DEFAULT_METADATA_PATH,
    write_metadata,
)
from v_fashion_insight.data.grouping import (
    CONFLICT_POLICIES,
    GROUP_CONFLICTING_LABEL_COLUMNS_COLUMN,
    GROUP_CONFLICT_POLICY_COLUMN,
    GROUP_HAS_LABEL_CONFLICT_COLUMN,
    GROUP_HAS_MIXED_MISSING_COLUMN,
    GROUP_MIXED_MISSING_COLUMNS_COLUMN,
    REQUIRES_MANUAL_REVIEW_COLUMN,
    RETAIN_FOR_PROCESSING_COLUMN,
    ConflictPolicy,
    assign_duplicate_groups,
)
from v_fashion_insight.data.processed_contract import REVIEW_ID_COLUMN
from v_fashion_insight.data.validate import (
    ID_COLUMN,
    LABEL_COLUMNS,
    TEXT_COLUMN,
    load_dataset_from_metadata,
    validate_dataset,
)

_WHITESPACE_PATTERN = re.compile(r"\s+")

DEFAULT_INTERIM_PATH = Path("data/interim/reviews.csv")
DEFAULT_AUDIT_PATH = Path("reports/metrics/preprocessing_audit.json")

REVIEW_ID_CANONICAL_FIELDS: Final[tuple[str, ...]] = (
    "dataset_name",
    "dataset_revision",
    "source_split",
    "source_id",
)
REVIEW_ID_SCHEMA_VERSION: Final[str] = "v1"
REVIEW_ID_PREFIX: Final[str] = "review_"
SOURCE_SPLIT_COLUMN: Final[str] = "source_split"
SOURCE_ID_COLUMN: Final[str] = "source_id"
SOURCE_TEXT_COLUMN: Final[str] = "source_text"

INTERIM_COLUMNS: Final[tuple[str, ...]] = (
    REVIEW_ID_COLUMN,
    "group_id",
    SOURCE_SPLIT_COLUMN,
    SOURCE_ID_COLUMN,
    SOURCE_TEXT_COLUMN,
    "text",
    *ASPECTS,
    GROUP_HAS_LABEL_CONFLICT_COLUMN,
    GROUP_CONFLICTING_LABEL_COLUMNS_COLUMN,
    GROUP_HAS_MIXED_MISSING_COLUMN,
    GROUP_MIXED_MISSING_COLUMNS_COLUMN,
    GROUP_CONFLICT_POLICY_COLUMN,
    RETAIN_FOR_PROCESSING_COLUMN,
    REQUIRES_MANUAL_REVIEW_COLUMN,
)


class ReviewIdCollisionError(RuntimeError):
    """Raised when distinct canonical identities produce the same review ID."""


def normalize_review_text(text: str) -> str:
    """Normalize review text without removing sentiment-bearing content."""
    if not isinstance(text, str):
        raise TypeError("review text must be a string")

    normalized = unicodedata.normalize("NFKC", text)
    return _WHITESPACE_PATTERN.sub(" ", normalized).strip()


def _required_identity_text(field: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    return normalized


def _canonical_source_id(source_id: object) -> str:
    if isinstance(source_id, bool):
        raise TypeError("source_id must be an integer or non-empty string")
    if isinstance(source_id, Integral):
        return str(int(source_id))
    if isinstance(source_id, str):
        normalized = source_id.strip()
        if normalized:
            return normalized
        raise ValueError("source_id must not be empty")
    raise TypeError("source_id must be an integer or non-empty string")


def _canonical_review_identity(
    *,
    dataset_name: str,
    dataset_revision: str,
    source_split: str,
    source_id: object,
) -> str:
    payload = {
        "schema_version": REVIEW_ID_SCHEMA_VERSION,
        "dataset_name": _required_identity_text(
            "dataset_name",
            dataset_name,
        ),
        "dataset_revision": _required_identity_text(
            "dataset_revision",
            dataset_revision,
        ),
        "source_split": _required_identity_text(
            "source_split",
            source_split,
        ),
        "source_id": _canonical_source_id(source_id),
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _hash_canonical_identity(identity: str) -> str:
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _review_id_from_identity(identity: str) -> str:
    return f"{REVIEW_ID_PREFIX}{_hash_canonical_identity(identity)}"


def generate_review_id(
    *,
    dataset_name: str,
    dataset_revision: str,
    source_split: str,
    source_id: object,
) -> str:
    """Generate a stable ID from immutable source-row identity fields."""
    identity = _canonical_review_identity(
        dataset_name=dataset_name,
        dataset_revision=dataset_revision,
        source_split=source_split,
        source_id=source_id,
    )
    return _review_id_from_identity(identity)


def generate_review_ids(
    source_ids: Iterable[object],
    *,
    dataset_name: str,
    dataset_revision: str,
    source_split: str,
) -> list[str]:
    """Generate ordered review IDs and reject duplicates or hash collisions."""
    review_ids: list[str] = []
    identities_by_review_id: dict[str, str] = {}

    for source_id in source_ids:
        identity = _canonical_review_identity(
            dataset_name=dataset_name,
            dataset_revision=dataset_revision,
            source_split=source_split,
            source_id=source_id,
        )
        review_id = _review_id_from_identity(identity)
        previous_identity = identities_by_review_id.get(review_id)
        if previous_identity is not None:
            if previous_identity == identity:
                raise ValueError(
                    "Duplicate canonical review identity for "
                    f"source_id={source_id!r}."
                )
            raise ReviewIdCollisionError(
                "SHA-256 collision between distinct canonical review "
                f"identities for review_id={review_id!r}."
            )

        identities_by_review_id[review_id] = identity
        review_ids.append(review_id)

    return review_ids


def _ensure_output_paths_available(
    paths: Sequence[Path],
    *,
    force: bool,
) -> None:
    existing_paths = [Path(path) for path in paths if Path(path).exists()]
    if existing_paths and not force:
        formatted_paths = ", ".join(
            path.as_posix() for path in existing_paths
        )
        raise FileExistsError(
            "Refusing to overwrite existing preprocessing artifacts: "
            f"{formatted_paths}. Use --force to replace them."
        )


def _required_source_value(source: Mapping[str, Any], key: str) -> str:
    value = source.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Dataset source metadata must contain non-empty {key!r}."
        )
    return value.strip()


def _validate_raw_for_preprocessing(
    dataset: DatasetDict,
    *,
    source: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = validate_dataset(dataset, source=source)
    allowed_errors = [
        issue
        for issue in report["issues"]
        if issue["severity"] == "error" and issue["code"] == "null_label"
    ]
    blocking_errors = [
        issue
        for issue in report["issues"]
        if issue["severity"] == "error" and issue["code"] != "null_label"
    ]
    if blocking_errors:
        codes = sorted({issue["code"] for issue in blocking_errors})
        raise ValueError(
            f"Blocking raw validation errors prevent preprocessing: {codes!r}."
        )
    return report, allowed_errors


def _load_dataset_for_preprocessing(
    metadata_path: Path,
    *,
    local_files_only: bool,
) -> tuple[DatasetDict, dict[str, Any]]:
    if not local_files_only:
        return load_dataset_from_metadata(
            metadata_path,
            local_files_only=False,
        )

    previous_datasets_offline = datasets_config.HF_DATASETS_OFFLINE
    previous_hub_offline = hub_constants.HF_HUB_OFFLINE
    datasets_config.HF_DATASETS_OFFLINE = True
    hub_constants.HF_HUB_OFFLINE = True
    try:
        return load_dataset_from_metadata(
            metadata_path,
            local_files_only=True,
        )
    finally:
        datasets_config.HF_DATASETS_OFFLINE = previous_datasets_offline
        hub_constants.HF_HUB_OFFLINE = previous_hub_offline


def _prepare_interim_frame(
    dataset: DatasetDict,
    *,
    dataset_name: str,
    dataset_revision: str,
) -> pd.DataFrame:
    split_frames: list[pd.DataFrame] = []
    aspect_source_columns = dict(zip(ASPECTS, LABEL_COLUMNS, strict=True))

    for split_name, split in sorted(dataset.items()):
        raw_frame = split.to_pandas()
        source_ids = raw_frame[ID_COLUMN].tolist()
        review_ids = generate_review_ids(
            source_ids,
            dataset_name=dataset_name,
            dataset_revision=dataset_revision,
            source_split=split_name,
        )
        prepared = pd.DataFrame(
            {
                REVIEW_ID_COLUMN: review_ids,
                SOURCE_SPLIT_COLUMN: split_name,
                SOURCE_ID_COLUMN: raw_frame[ID_COLUMN].tolist(),
                SOURCE_TEXT_COLUMN: raw_frame[TEXT_COLUMN].tolist(),
                "text": raw_frame[TEXT_COLUMN].map(normalize_review_text),
            }
        )
        for aspect, source_column in aspect_source_columns.items():
            prepared[aspect] = pd.to_numeric(
                raw_frame[source_column],
                errors="raise",
            ).astype("Int64")
        split_frames.append(prepared)

    if not split_frames:
        raise ValueError("Cannot preprocess a dataset without splits.")
    return pd.concat(split_frames, ignore_index=True)


def _read_candidate_groups(
    path: Path,
    *,
    dataset_name: str,
    dataset_revision: str,
) -> list[list[str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Duplicate-group artifact does not exist: {path.as_posix()}."
        )

    groups: list[list[str]] = []
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"split", "member_ids"}
        actual_columns = set(reader.fieldnames or [])
        missing_columns = sorted(required_columns.difference(actual_columns))
        if missing_columns:
            raise ValueError(
                f"Duplicate-group artifact {path.as_posix()!r} is missing "
                f"columns: {missing_columns!r}."
            )

        for row_number, row in enumerate(reader, start=2):
            split_name = str(row["split"]).strip()
            source_ids = [
                value.strip()
                for value in str(row["member_ids"]).split("|")
                if value.strip()
            ]
            if not split_name or len(set(source_ids)) < 2:
                raise ValueError(
                    f"Invalid duplicate group at {path.as_posix()}:{row_number}."
                )
            groups.append(
                [
                    generate_review_id(
                        dataset_name=dataset_name,
                        dataset_revision=dataset_revision,
                        source_split=split_name,
                        source_id=source_id,
                    )
                    for source_id in source_ids
                ]
            )
    return groups


def _write_interim_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            frame.to_csv(
                temporary_file,
                index=False,
                lineterminator="\n",
            )
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def preprocess_downloaded_dataset(
    *,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    exact_group_path: Path = DEFAULT_EXACT_GROUP_PATH,
    near_cluster_path: Path = DEFAULT_NEAR_CLUSTER_PATH,
    interim_path: Path = DEFAULT_INTERIM_PATH,
    audit_path: Path = DEFAULT_AUDIT_PATH,
    conflict_policy: ConflictPolicy = "retain",
    force: bool = False,
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Build deterministic interim data from the pinned raw dataset."""
    interim_path = Path(interim_path)
    audit_path = Path(audit_path)
    _ensure_output_paths_available(
        (interim_path, audit_path),
        force=force,
    )

    dataset, source = _load_dataset_for_preprocessing(
        Path(metadata_path),
        local_files_only=local_files_only,
    )
    dataset_name = _required_source_value(source, "dataset_name")
    dataset_revision = _required_source_value(source, "resolved_revision")
    validation_report, allowed_errors = _validate_raw_for_preprocessing(
        dataset,
        source=source,
    )
    prepared = _prepare_interim_frame(
        dataset,
        dataset_name=dataset_name,
        dataset_revision=dataset_revision,
    )
    exact_groups = _read_candidate_groups(
        Path(exact_group_path),
        dataset_name=dataset_name,
        dataset_revision=dataset_revision,
    )
    near_groups = _read_candidate_groups(
        Path(near_cluster_path),
        dataset_name=dataset_name,
        dataset_revision=dataset_revision,
    )
    assignments, grouping_report = assign_duplicate_groups(
        prepared,
        exact_groups=exact_groups,
        high_confidence_near_groups=near_groups,
        conflict_policy=conflict_policy,
    )

    if conflict_policy == "exclude":
        written = assignments.loc[
            assignments[RETAIN_FOR_PROCESSING_COLUMN]
        ].copy()
    else:
        written = assignments.copy()
    written = written.loc[:, INTERIM_COLUMNS].reset_index(drop=True)

    missing_label_cell_count = int(
        prepared.loc[:, ASPECTS].isna().sum().sum()
    )
    held_for_manual_review_count = int(
        assignments[REQUIRES_MANUAL_REVIEW_COLUMN].sum()
    )
    removed_row_count = len(assignments) - len(written)
    noteworthy_groups = [
        group
        for group in grouping_report["groups"]
        if group["member_count"] > 1
        or group["has_label_conflict"]
        or group["has_mixed_missing_labels"]
    ]
    audit = {
        "source": dict(source),
        "inputs": {
            "metadata_path": Path(metadata_path).as_posix(),
            "exact_group_path": Path(exact_group_path).as_posix(),
            "near_cluster_path": Path(near_cluster_path).as_posix(),
        },
        "normalization": {
            "unicode_form": "NFKC",
            "whitespace": "collapse all Unicode whitespace to one space",
            "preserved": (
                "Vietnamese diacritics, negation, punctuation, digits, "
                "emoji, and product-code-like tokens"
            ),
        },
        "validation": {
            "status": (
                "passed_with_allowed_missing_labels"
                if allowed_errors
                else "passed"
            ),
            "allowed_error_codes": ["null_label"] if allowed_errors else [],
            "allowed_error_count": len(allowed_errors),
            "warning_count": validation_report["summary"]["warning_count"],
        },
        "policy": grouping_report["policy"],
        "summary": {
            "input_row_count": len(prepared),
            "written_row_count": len(written),
            "removed_row_count": removed_row_count,
            "held_for_manual_review_count": (
                held_for_manual_review_count
            ),
            "missing_label_cell_count": missing_label_cell_count,
            "exact_candidate_group_count": len(exact_groups),
            "near_candidate_group_count": len(near_groups),
        },
        "removal_reasons": {
            "conflicting_group_excluded": removed_row_count,
        },
        "hold_reasons": {
            "conflicting_group_manual_review": (
                held_for_manual_review_count
            ),
        },
        "grouping": {
            "summary": grouping_report["summary"],
            "noteworthy_groups": noteworthy_groups,
        },
        "interim_columns": list(INTERIM_COLUMNS),
    }

    _write_interim_csv(written, interim_path)
    write_metadata(audit, audit_path)

    logger = configure_logging()
    logger.info(
        "Preprocessed %d rows: wrote %d, removed %d, held %d for review.",
        len(prepared),
        len(written),
        removed_row_count,
        held_for_manual_review_count,
    )
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize the pinned raw dataset, generate stable IDs, assign "
            "duplicate groups, and write deterministic interim data."
        )
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
    )
    parser.add_argument(
        "--exact-group-path",
        type=Path,
        default=DEFAULT_EXACT_GROUP_PATH,
    )
    parser.add_argument(
        "--near-cluster-path",
        type=Path,
        default=DEFAULT_NEAR_CLUSTER_PATH,
    )
    parser.add_argument(
        "--interim-path",
        type=Path,
        default=DEFAULT_INTERIM_PATH,
    )
    parser.add_argument(
        "--audit-path",
        type=Path,
        default=DEFAULT_AUDIT_PATH,
    )
    parser.add_argument(
        "--conflict-policy",
        choices=CONFLICT_POLICIES,
        default="retain",
        help="Policy for groups with conflicting labels (default: retain).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing interim and audit artifacts.",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow dataset loading to access the network.",
    )
    return parser


def _console_safe(value: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return value.encode(encoding, errors="backslashreplace").decode(encoding)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        audit = preprocess_downloaded_dataset(
            metadata_path=args.metadata_path,
            exact_group_path=args.exact_group_path,
            near_cluster_path=args.near_cluster_path,
            interim_path=args.interim_path,
            audit_path=args.audit_path,
            conflict_policy=args.conflict_policy,
            force=args.force,
            local_files_only=not args.allow_network,
        )
    except (FileNotFoundError, FileExistsError, TypeError, ValueError) as error:
        print(_console_safe(f"Preprocessing failed: {error}"), file=sys.stderr)
        return 2

    summary = audit["summary"]
    print(
        _console_safe(
            "Preprocessing complete: "
            f"{summary['written_row_count']} rows written, "
            f"{summary['removed_row_count']} removed, "
            f"{summary['held_for_manual_review_count']} held for review."
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
