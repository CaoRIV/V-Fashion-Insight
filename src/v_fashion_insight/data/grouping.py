"""Assign stable duplicate groups from exact and near-duplicate evidence."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any, Final, Literal

import pandas as pd

from v_fashion_insight.common.constants import ASPECTS, VALID_LABELS
from v_fashion_insight.data.processed_contract import REVIEW_ID_COLUMN

ConflictPolicy = Literal["retain", "exclude", "manual_review"]

CONFLICT_POLICIES: Final[tuple[ConflictPolicy, ...]] = (
    "retain",
    "exclude",
    "manual_review",
)
GROUP_ID_SCHEMA_VERSION: Final[str] = "v1"
GROUP_ID_PREFIX: Final[str] = "group_"

GROUP_HAS_LABEL_CONFLICT_COLUMN = "group_has_label_conflict"
GROUP_CONFLICTING_LABEL_COLUMNS_COLUMN = "group_conflicting_label_columns"
GROUP_HAS_MIXED_MISSING_COLUMN = "group_has_mixed_missing_labels"
GROUP_MIXED_MISSING_COLUMNS_COLUMN = "group_mixed_missing_label_columns"
GROUP_CONFLICT_POLICY_COLUMN = "group_conflict_policy"
RETAIN_FOR_PROCESSING_COLUMN = "retain_for_processing"
REQUIRES_MANUAL_REVIEW_COLUMN = "requires_manual_review"


class GroupIdCollisionError(RuntimeError):
    """Raised when distinct member sets produce the same group ID."""


class _DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: str, second: str) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        smaller_root, larger_root = sorted((first_root, second_root))
        self.parent[larger_root] = smaller_root

    def components(self) -> list[tuple[str, ...]]:
        members_by_root: dict[str, list[str]] = defaultdict(list)
        for value in sorted(self.parent):
            members_by_root[self.find(value)].append(value)
        return sorted(
            (tuple(sorted(members)) for members in members_by_root.values()),
            key=lambda members: members,
        )


def _validate_frame(frame: pd.DataFrame) -> list[str]:
    required_columns = (REVIEW_ID_COLUMN, *ASPECTS)
    missing_columns = [
        column for column in required_columns if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Cannot assign groups; missing columns: {missing_columns!r}."
        )
    if frame.empty:
        raise ValueError("Cannot assign groups to an empty dataset.")

    review_ids: list[str] = []
    for row_index, value in frame[REVIEW_ID_COLUMN].items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "review_id values must be non-empty strings; "
                f"invalid row {row_index!r}."
            )
        if value != value.strip():
            raise ValueError(
                "review_id values must not contain surrounding whitespace; "
                f"invalid row {row_index!r}."
            )
        review_ids.append(value)

    if len(set(review_ids)) != len(review_ids):
        duplicated = sorted(
            set(
                frame.loc[
                    frame.duplicated(subset=[REVIEW_ID_COLUMN], keep=False),
                    REVIEW_ID_COLUMN,
                ].tolist()
            )
        )
        raise ValueError(
            f"review_id values must be unique; duplicates: {duplicated!r}."
        )

    for column in ASPECTS:
        values = frame[column]
        non_null_values = values[values.notna()]
        if non_null_values.empty:
            continue
        if not pd.api.types.is_numeric_dtype(non_null_values.dtype):
            raise ValueError(
                f"Label column {column!r} must be numeric or missing."
            )
        integer_mask = non_null_values.mod(1).eq(0)
        if not integer_mask.all():
            invalid = non_null_values.loc[~integer_mask].tolist()
            raise ValueError(
                f"Label column {column!r} contains non-integers: {invalid!r}."
            )
        valid_mask = non_null_values.isin(VALID_LABELS)
        if not valid_mask.all():
            invalid = sorted(
                {
                    int(value)
                    for value in non_null_values.loc[~valid_mask].tolist()
                }
            )
            raise ValueError(
                f"Label column {column!r} contains invalid labels: "
                f"{invalid!r}."
            )

    return review_ids


def _normalize_candidate_groups(
    groups: Iterable[Iterable[str]],
    *,
    evidence_type: str,
    known_review_ids: set[str],
) -> list[tuple[str, ...]]:
    normalized_groups: list[tuple[str, ...]] = []
    for group_index, group in enumerate(groups):
        if isinstance(group, (str, bytes)):
            raise TypeError(
                f"{evidence_type} group {group_index} must be an iterable "
                "of review IDs, not a string."
            )
        members = list(group)
        if any(not isinstance(member, str) for member in members):
            raise TypeError(
                f"{evidence_type} group {group_index} contains a non-string "
                "review ID."
            )
        unique_members = tuple(sorted(set(members)))
        if len(unique_members) < 2:
            raise ValueError(
                f"{evidence_type} group {group_index} must contain at least "
                "two unique review IDs."
            )
        unknown_members = sorted(
            set(unique_members).difference(known_review_ids)
        )
        if unknown_members:
            raise ValueError(
                f"Unknown review_id values in {evidence_type} group "
                f"{group_index}: {unknown_members!r}."
            )
        normalized_groups.append(unique_members)
    return normalized_groups


def _hash_group_members(members: Sequence[str]) -> str:
    payload = {
        "schema_version": GROUP_ID_SCHEMA_VERSION,
        "member_review_ids": sorted(members),
    }
    canonical_payload = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()


def _group_id(members: Sequence[str]) -> str:
    return f"{GROUP_ID_PREFIX}{_hash_group_members(members)}"


def _label_summary(group: pd.DataFrame) -> dict[str, Any]:
    conflicting_columns: list[str] = []
    mixed_missing_columns: list[str] = []
    label_values: dict[str, list[int]] = {}

    for column in ASPECTS:
        values = group[column]
        unique_values = sorted(
            {int(value) for value in values.dropna().tolist()}
        )
        label_values[column] = unique_values
        if len(unique_values) > 1:
            conflicting_columns.append(column)
        if values.isna().any() and values.notna().any():
            mixed_missing_columns.append(column)

    return {
        "has_label_conflict": bool(conflicting_columns),
        "conflicting_label_columns": conflicting_columns,
        "has_mixed_missing_labels": bool(mixed_missing_columns),
        "mixed_missing_label_columns": mixed_missing_columns,
        "label_values": label_values,
    }


def _policy_flags(
    *,
    has_label_conflict: bool,
    conflict_policy: ConflictPolicy,
) -> tuple[bool, bool]:
    if not has_label_conflict:
        return True, False
    if conflict_policy == "retain":
        return True, False
    if conflict_policy == "exclude":
        return False, False
    return False, True


def assign_duplicate_groups(
    frame: pd.DataFrame,
    *,
    exact_groups: Iterable[Iterable[str]],
    high_confidence_near_groups: Iterable[Iterable[str]],
    conflict_policy: ConflictPolicy,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign deterministic group IDs and report group label conflicts."""
    if conflict_policy not in CONFLICT_POLICIES:
        raise ValueError(
            f"conflict_policy must be one of {list(CONFLICT_POLICIES)!r}."
        )

    review_ids = _validate_frame(frame)
    known_review_ids = set(review_ids)
    normalized_exact_groups = _normalize_candidate_groups(
        exact_groups,
        evidence_type="exact",
        known_review_ids=known_review_ids,
    )
    normalized_near_groups = _normalize_candidate_groups(
        high_confidence_near_groups,
        evidence_type="high_confidence_near",
        known_review_ids=known_review_ids,
    )

    disjoint_set = _DisjointSet(review_ids)
    evidence_by_review_id: dict[str, set[str]] = {
        review_id: set() for review_id in review_ids
    }
    for evidence_type, candidate_groups in (
        ("exact", normalized_exact_groups),
        ("high_confidence_near", normalized_near_groups),
    ):
        for members in candidate_groups:
            first_member = members[0]
            for member in members[1:]:
                disjoint_set.union(first_member, member)
            for member in members:
                evidence_by_review_id[member].add(evidence_type)

    frame_by_review_id = frame.set_index(REVIEW_ID_COLUMN, drop=False)
    assignment_by_review_id: dict[str, dict[str, Any]] = {}
    group_records: list[dict[str, Any]] = []
    member_sets_by_group_id: dict[str, tuple[str, ...]] = {}

    for members in disjoint_set.components():
        group_id = _group_id(members)
        previous_members = member_sets_by_group_id.get(group_id)
        if previous_members is not None and previous_members != members:
            raise GroupIdCollisionError(
                "SHA-256 collision between distinct duplicate groups for "
                f"group_id={group_id!r}."
            )
        member_sets_by_group_id[group_id] = members

        group_frame = frame_by_review_id.loc[list(members)]
        label_summary = _label_summary(group_frame)
        retain_for_processing, requires_manual_review = _policy_flags(
            has_label_conflict=label_summary["has_label_conflict"],
            conflict_policy=conflict_policy,
        )
        evidence_types = sorted(
            {
                evidence_type
                for member in members
                for evidence_type in evidence_by_review_id[member]
            }
        )
        group_record = {
            "group_id": group_id,
            "member_count": len(members),
            "member_ids": list(members),
            "evidence_types": evidence_types,
            **label_summary,
            "conflict_policy": (
                conflict_policy
                if label_summary["has_label_conflict"]
                else "not_applicable"
            ),
            "retain_for_processing": retain_for_processing,
            "requires_manual_review": requires_manual_review,
        }
        group_records.append(group_record)

        conflicting_columns = "|".join(
            label_summary["conflicting_label_columns"]
        )
        mixed_missing_columns = "|".join(
            label_summary["mixed_missing_label_columns"]
        )
        for member in members:
            assignment_by_review_id[member] = {
                "group_id": group_id,
                GROUP_HAS_LABEL_CONFLICT_COLUMN: label_summary[
                    "has_label_conflict"
                ],
                GROUP_CONFLICTING_LABEL_COLUMNS_COLUMN: conflicting_columns,
                GROUP_HAS_MIXED_MISSING_COLUMN: label_summary[
                    "has_mixed_missing_labels"
                ],
                GROUP_MIXED_MISSING_COLUMNS_COLUMN: mixed_missing_columns,
                GROUP_CONFLICT_POLICY_COLUMN: group_record[
                    "conflict_policy"
                ],
                RETAIN_FOR_PROCESSING_COLUMN: retain_for_processing,
                REQUIRES_MANUAL_REVIEW_COLUMN: requires_manual_review,
            }

    group_records.sort(key=lambda record: record["group_id"])
    assignments = frame.copy()
    assignment_columns = (
        "group_id",
        GROUP_HAS_LABEL_CONFLICT_COLUMN,
        GROUP_CONFLICTING_LABEL_COLUMNS_COLUMN,
        GROUP_HAS_MIXED_MISSING_COLUMN,
        GROUP_MIXED_MISSING_COLUMNS_COLUMN,
        GROUP_CONFLICT_POLICY_COLUMN,
        RETAIN_FOR_PROCESSING_COLUMN,
        REQUIRES_MANUAL_REVIEW_COLUMN,
    )
    for column in assignment_columns:
        assignments[column] = assignments[REVIEW_ID_COLUMN].map(
            lambda review_id, output_column=column: assignment_by_review_id[
                review_id
            ][output_column]
        )

    conflict_groups = [
        group for group in group_records if group["has_label_conflict"]
    ]
    mixed_missing_groups = [
        group
        for group in group_records
        if group["has_mixed_missing_labels"]
    ]
    report = {
        "policy": {
            "exact_duplicates": "group automatically",
            "high_confidence_near_duplicates": "group automatically",
            "needs_review_pairs": "do not group automatically",
            "conflicting_groups": conflict_policy,
        },
        "summary": {
            "row_count": len(assignments),
            "group_count": len(group_records),
            "multi_member_group_count": sum(
                group["member_count"] > 1 for group in group_records
            ),
            "conflict_group_count": len(conflict_groups),
            "mixed_missing_group_count": len(mixed_missing_groups),
            "excluded_row_count": sum(
                group["member_count"]
                for group in conflict_groups
                if not group["retain_for_processing"]
                and not group["requires_manual_review"]
            ),
            "manual_review_row_count": sum(
                group["member_count"]
                for group in conflict_groups
                if group["requires_manual_review"]
            ),
        },
        "groups": group_records,
    }
    return assignments, report
