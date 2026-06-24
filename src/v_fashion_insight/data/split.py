"""Deterministic group-aware splitting for interim review data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd

from v_fashion_insight.common.constants import (
    ASPECTS,
    DEFAULT_RANDOM_SEED,
    VALID_LABELS,
)
from v_fashion_insight.common.logging import configure_logging
from v_fashion_insight.data.download import write_metadata
from v_fashion_insight.data.preprocess import DEFAULT_INTERIM_PATH
from v_fashion_insight.data.processed_contract import (
    GROUP_ID_COLUMN,
    REVIEW_ID_COLUMN,
    SPLIT_COLUMN,
    SPLIT_NAMES,
    TEXT_COLUMN,
)

DEFAULT_SPLIT_IDS_PATH = Path("data/processed/split_ids.csv")
DEFAULT_SPLIT_METADATA_PATH = Path("data/processed/metadata.json")
DEFAULT_SPLIT_RATIOS: Final[dict[str, float]] = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}
SPLIT_ID_COLUMNS: Final[tuple[str, ...]] = (
    REVIEW_ID_COLUMN,
    GROUP_ID_COLUMN,
    SPLIT_COLUMN,
)
MISSING_LABEL_TOKEN: Final[str] = "__missing__"
SPLIT_METADATA_SCHEMA_VERSION: Final[str] = "v1"


@dataclass(frozen=True)
class _GroupSummary:
    group_id: str
    review_ids: tuple[str, ...]
    row_count: int
    label_counts: dict[str, int]
    order_key: str


def _console_safe(value: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return value.encode(encoding, errors="backslashreplace").decode(encoding)


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_group_order_key(group_id: str, seed: int) -> str:
    return _stable_hash(f"{SPLIT_METADATA_SCHEMA_VERSION}:{seed}:{group_id}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataframe_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _write_bytes_atomic(payload: bytes, destination: Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(payload)
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


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
            "Refusing to overwrite existing split artifacts: "
            f"{formatted_paths}. Use --force to replace them."
        )


def _normalize_ratios(
    ratios: Mapping[str, float] | None,
) -> dict[str, float]:
    normalized = {
        split_name: float((ratios or DEFAULT_SPLIT_RATIOS)[split_name])
        for split_name in SPLIT_NAMES
    }
    if any(value <= 0 for value in normalized.values()):
        raise ValueError("Split ratios must be positive.")

    total = sum(normalized.values())
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    return {
        split_name: normalized[split_name] / total
        for split_name in SPLIT_NAMES
    }


def _feature_key(aspect: str, label: int | str) -> str:
    return f"{aspect}={label}"


def _empty_label_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for aspect in ASPECTS:
        for label in sorted(VALID_LABELS):
            counts[_feature_key(aspect, label)] = 0
        counts[_feature_key(aspect, MISSING_LABEL_TOKEN)] = 0
    return counts


def _label_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = _empty_label_counts()
    for aspect in ASPECTS:
        values = frame[aspect]
        for label in sorted(VALID_LABELS):
            counts[_feature_key(aspect, label)] = int(values.eq(label).sum())
        counts[_feature_key(aspect, MISSING_LABEL_TOKEN)] = int(
            values.isna().sum()
        )
    return counts


def _validate_input_frame(frame: pd.DataFrame) -> None:
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
    if frame.empty:
        raise ValueError("Cannot split an empty interim dataset.")
    if frame[REVIEW_ID_COLUMN].isna().any():
        raise ValueError("review_id values must not be null.")
    if frame[GROUP_ID_COLUMN].isna().any():
        raise ValueError("group_id values must not be null.")
    if frame[TEXT_COLUMN].isna().any():
        raise ValueError("text values must not be null.")
    if frame[REVIEW_ID_COLUMN].duplicated().any():
        raise ValueError("review_id values must be unique.")

    for column in (REVIEW_ID_COLUMN, GROUP_ID_COLUMN, TEXT_COLUMN):
        empty_mask = frame[column].astype("string").str.strip().eq("")
        if empty_mask.any():
            raise ValueError(f"{column} values must not be empty.")

    for aspect in ASPECTS:
        values = frame[aspect]
        non_null_values = values.dropna()
        if non_null_values.empty:
            continue
        numeric_values = pd.to_numeric(non_null_values, errors="coerce")
        if numeric_values.isna().any():
            raise ValueError(f"{aspect} contains non-numeric labels.")
        integer_mask = numeric_values.mod(1).eq(0)
        if not integer_mask.all():
            raise ValueError(f"{aspect} contains non-integer labels.")
        if not numeric_values.loc[integer_mask].isin(VALID_LABELS).all():
            raise ValueError(
                f"{aspect} contains labels outside "
                f"{sorted(VALID_LABELS)!r}."
            )


def _group_summaries(frame: pd.DataFrame, *, seed: int) -> list[_GroupSummary]:
    groups: dict[str, dict[str, Any]] = {}
    selected_columns = [REVIEW_ID_COLUMN, GROUP_ID_COLUMN, *ASPECTS]
    for row in frame.loc[:, selected_columns].itertuples(
        index=False,
        name=None,
    ):
        review_id, group_id, *labels = row
        normalized_group_id = str(group_id)
        if normalized_group_id not in groups:
            groups[normalized_group_id] = {
                "review_ids": [],
                "row_count": 0,
                "label_counts": _empty_label_counts(),
            }

        group = groups[normalized_group_id]
        group["review_ids"].append(str(review_id))
        group["row_count"] += 1
        for aspect, value in zip(ASPECTS, labels, strict=True):
            label = (
                MISSING_LABEL_TOKEN
                if pd.isna(value)
                else int(value)
            )
            group["label_counts"][_feature_key(aspect, label)] += 1

    summaries = [
        _GroupSummary(
            group_id=group_id,
            review_ids=tuple(sorted(group["review_ids"])),
            row_count=group["row_count"],
            label_counts=group["label_counts"],
            order_key=_stable_group_order_key(group_id, seed),
        )
        for group_id, group in groups.items()
    ]
    return sorted(
        summaries,
        key=lambda group: (-group.row_count, group.order_key, group.group_id),
    )


def _score_state(
    *,
    row_counts: Mapping[str, int],
    label_counts: Mapping[str, Mapping[str, int]],
    total_rows: int,
    total_label_counts: Mapping[str, int],
    ratios: Mapping[str, float],
) -> float:
    row_score = 0.0
    for split_name in SPLIT_NAMES:
        target_rows = total_rows * ratios[split_name]
        row_score += (
            (row_counts[split_name] - target_rows) / max(total_rows, 1)
        ) ** 2

    label_score = 0.0
    label_terms = 0
    for feature, total_count in total_label_counts.items():
        if total_count == 0:
            continue
        for split_name in SPLIT_NAMES:
            target_count = total_count * ratios[split_name]
            observed_count = label_counts[split_name][feature]
            label_score += abs(observed_count - target_count) / total_count
            label_terms += 1
    if label_terms:
        label_score /= label_terms

    return (4.0 * row_score) + label_score


def assign_group_aware_splits(
    frame: pd.DataFrame,
    *,
    seed: int = DEFAULT_RANDOM_SEED,
    ratios: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Assign each duplicate group to exactly one deterministic split."""
    _validate_input_frame(frame)
    normalized_ratios = _normalize_ratios(ratios)
    groups = _group_summaries(frame, seed=seed)
    total_rows = len(frame)
    total_label_counts = _label_counts(frame)

    row_counts = {split_name: 0 for split_name in SPLIT_NAMES}
    label_counts = {
        split_name: _empty_label_counts() for split_name in SPLIT_NAMES
    }
    split_by_group: dict[str, str] = {}

    for group in groups:
        scored_candidates: list[tuple[float, int, str]] = []
        for split_index, split_name in enumerate(SPLIT_NAMES):
            candidate_row_counts = dict(row_counts)
            candidate_row_counts[split_name] += group.row_count

            candidate_label_counts = {
                candidate_split: dict(counts)
                for candidate_split, counts in label_counts.items()
            }
            for feature, count in group.label_counts.items():
                candidate_label_counts[split_name][feature] += count

            scored_candidates.append(
                (
                    _score_state(
                        row_counts=candidate_row_counts,
                        label_counts=candidate_label_counts,
                        total_rows=total_rows,
                        total_label_counts=total_label_counts,
                        ratios=normalized_ratios,
                    ),
                    split_index,
                    split_name,
                )
            )

        _, _, selected_split = min(scored_candidates)
        split_by_group[group.group_id] = selected_split
        row_counts[selected_split] += group.row_count
        for feature, count in group.label_counts.items():
            label_counts[selected_split][feature] += count

    assignments = frame.loc[:, [REVIEW_ID_COLUMN, GROUP_ID_COLUMN]].copy()
    assignments[REVIEW_ID_COLUMN] = assignments[REVIEW_ID_COLUMN].astype(str)
    assignments[GROUP_ID_COLUMN] = assignments[GROUP_ID_COLUMN].astype(str)
    assignments[SPLIT_COLUMN] = assignments[GROUP_ID_COLUMN].map(
        split_by_group
    )
    if assignments[SPLIT_COLUMN].isna().any():
        raise RuntimeError("Internal split assignment error.")
    return assignments.sort_values(REVIEW_ID_COLUMN).reset_index(drop=True)


def _label_distribution(frame: pd.DataFrame) -> dict[str, Any]:
    distribution: dict[str, Any] = {}
    for aspect in ASPECTS:
        values = frame[aspect]
        distribution[aspect] = {
            "counts": {
                str(label): int(values.eq(label).sum())
                for label in sorted(VALID_LABELS)
            },
            "missing": int(values.isna().sum()),
        }
    return distribution


def build_split_metadata(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    seed: int = DEFAULT_RANDOM_SEED,
    ratios: Mapping[str, float] | None = None,
    input_path: Path | None = None,
    split_ids_path: Path | None = None,
    input_sha256: str | None = None,
    split_ids_sha256: str | None = None,
) -> dict[str, Any]:
    """Build deterministic metadata for a split assignment."""
    normalized_ratios = _normalize_ratios(ratios)
    merged = frame.merge(
        assignments,
        on=[REVIEW_ID_COLUMN, GROUP_ID_COLUMN],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(frame):
        raise ValueError("Split assignments must cover every input row.")

    leaked_groups = (
        merged.groupby(GROUP_ID_COLUMN)[SPLIT_COLUMN].nunique().gt(1)
    )
    if leaked_groups.any():
        raise ValueError("Split assignments leak group_id across splits.")

    summaries: dict[str, Any] = {}
    for split_name in SPLIT_NAMES:
        split_frame = merged.loc[merged[SPLIT_COLUMN] == split_name]
        summaries[split_name] = {
            "row_count": int(len(split_frame)),
            "group_count": int(split_frame[GROUP_ID_COLUMN].nunique()),
            "row_ratio": (
                round(len(split_frame) / len(merged), 8)
                if len(merged)
                else 0.0
            ),
            "label_distribution": _label_distribution(split_frame),
        }

    return {
        "schema_version": SPLIT_METADATA_SCHEMA_VERSION,
        "policy": {
            "method": "deterministic_group_aware_greedy",
            "group_column": GROUP_ID_COLUMN,
            "target_ratios": {
                split_name: normalized_ratios[split_name]
                for split_name in SPLIT_NAMES
            },
            "seed": int(seed),
            "label_columns": list(ASPECTS),
            "missing_label_token": MISSING_LABEL_TOKEN,
            "objective": (
                "Minimize row-count drift and five-aspect label-count drift "
                "while assigning each group_id to exactly one split."
            ),
        },
        "inputs": {
            "interim_path": Path(input_path).as_posix()
            if input_path is not None
            else None,
            "interim_sha256": input_sha256,
            "row_count": int(len(frame)),
            "group_count": int(frame[GROUP_ID_COLUMN].nunique()),
        },
        "outputs": {
            "split_ids_path": Path(split_ids_path).as_posix()
            if split_ids_path is not None
            else None,
            "split_ids_sha256": split_ids_sha256,
        },
        "summary": summaries,
    }


def split_interim_dataset(
    *,
    interim_path: Path = DEFAULT_INTERIM_PATH,
    split_ids_path: Path = DEFAULT_SPLIT_IDS_PATH,
    metadata_path: Path = DEFAULT_SPLIT_METADATA_PATH,
    seed: int = DEFAULT_RANDOM_SEED,
    ratios: Mapping[str, float] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Split interim data by group and write split IDs plus metadata."""
    interim_path = Path(interim_path)
    split_ids_path = Path(split_ids_path)
    metadata_path = Path(metadata_path)
    if not interim_path.exists():
        raise FileNotFoundError(
            f"Interim dataset does not exist: {interim_path.as_posix()}."
        )
    _ensure_output_paths_available(
        (split_ids_path, metadata_path),
        force=force,
    )

    frame = pd.read_csv(interim_path)
    assignments = assign_group_aware_splits(
        frame,
        seed=seed,
        ratios=ratios,
    )
    split_payload = _dataframe_csv_bytes(assignments.loc[:, SPLIT_ID_COLUMNS])
    input_sha256 = _file_sha256(interim_path)
    split_ids_sha256 = hashlib.sha256(split_payload).hexdigest()
    metadata = build_split_metadata(
        frame,
        assignments,
        seed=seed,
        ratios=ratios,
        input_path=interim_path,
        split_ids_path=split_ids_path,
        input_sha256=input_sha256,
        split_ids_sha256=split_ids_sha256,
    )

    _write_bytes_atomic(split_payload, split_ids_path)
    write_metadata(metadata, metadata_path)

    logger = configure_logging()
    logger.info(
        "Split %d rows across %d groups into %s.",
        len(frame),
        frame[GROUP_ID_COLUMN].nunique(),
        ", ".join(
            f"{split_name}={metadata['summary'][split_name]['row_count']}"
            for split_name in SPLIT_NAMES
        ),
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Assign deterministic train/validation/test splits by group_id "
            "and write split membership metadata."
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
        "--metadata-path",
        type=Path,
        default=DEFAULT_SPLIT_METADATA_PATH,
        help=f"Split metadata JSON (default: {DEFAULT_SPLIT_METADATA_PATH}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Shared random seed (default: {DEFAULT_RANDOM_SEED}).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_SPLIT_RATIOS["train"],
        help="Target training ratio (default: 0.70).",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=DEFAULT_SPLIT_RATIOS["validation"],
        help="Target validation ratio (default: 0.15).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_SPLIT_RATIOS["test"],
        help="Target test ratio (default: 0.15).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing split ID and metadata artifacts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ratios = {
        "train": args.train_ratio,
        "validation": args.validation_ratio,
        "test": args.test_ratio,
    }
    try:
        metadata = split_interim_dataset(
            interim_path=args.interim_path,
            split_ids_path=args.split_ids_path,
            metadata_path=args.metadata_path,
            seed=args.seed,
            ratios=ratios,
            force=args.force,
        )
    except (FileNotFoundError, FileExistsError, TypeError, ValueError) as error:
        print(_console_safe(f"Splitting failed: {error}"), file=sys.stderr)
        return 2

    summary = metadata["summary"]
    print(
        _console_safe(
            "Splitting complete: "
            f"train={summary['train']['row_count']}, "
            f"validation={summary['validation']['row_count']}, "
            f"test={summary['test']['row_count']}."
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
