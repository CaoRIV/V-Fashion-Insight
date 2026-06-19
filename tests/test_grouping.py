import pandas as pd
import pytest

from v_fashion_insight.data import grouping
from v_fashion_insight.data.grouping import (
    CONFLICT_POLICIES,
    GroupIdCollisionError,
    assign_duplicate_groups,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3", "r4", "r5", "r6"],
            "material": [0, 0, 0, 3, 3, 0],
            "design": [0, 0, 0, 0, 0, 0],
            "size": [0, 0, 0, 0, 0, 0],
            "price": [0, 0, 0, 0, 0, 0],
            "service": [0, 0, 0, 0, 0, 0],
        }
    )


def _group_mapping(assignments: pd.DataFrame) -> dict[str, str]:
    return assignments.set_index("review_id")["group_id"].to_dict()


def test_assign_duplicate_groups_unions_exact_and_high_confidence_near() -> None:
    assignments, report = assign_duplicate_groups(
        _frame(),
        exact_groups=[["r1", "r2"]],
        high_confidence_near_groups=[
            ["r2", "r3"],
            ["r4", "r5"],
        ],
        conflict_policy="retain",
    )
    mapping = _group_mapping(assignments)

    assert mapping["r1"] == mapping["r2"] == mapping["r3"]
    assert mapping["r1"] == (
        "group_6d49aac03d16b058af4946ad4ba4b5fc"
        "9bb6c0b95f632eab0c4b4d3235434a03"
    )
    assert mapping["r4"] == mapping["r5"]
    assert mapping["r1"] != mapping["r4"]
    assert mapping["r6"] not in {mapping["r1"], mapping["r4"]}
    assert report["summary"] == {
        "row_count": 6,
        "group_count": 3,
        "multi_member_group_count": 2,
        "conflict_group_count": 0,
        "mixed_missing_group_count": 0,
        "excluded_row_count": 0,
        "manual_review_row_count": 0,
    }
    assert (
        report["policy"]["needs_review_pairs"]
        == "do not group automatically"
    )

    merged_group = next(
        group
        for group in report["groups"]
        if group["member_ids"] == ["r1", "r2", "r3"]
    )
    assert merged_group["evidence_types"] == [
        "exact",
        "high_confidence_near",
    ]


def test_group_assignment_is_stable_across_input_order() -> None:
    first, _ = assign_duplicate_groups(
        _frame(),
        exact_groups=[["r1", "r2"], ["r4", "r5"]],
        high_confidence_near_groups=[["r2", "r3"]],
        conflict_policy="retain",
    )
    second, _ = assign_duplicate_groups(
        _frame().iloc[::-1].reset_index(drop=True),
        exact_groups=[["r5", "r4"], ["r2", "r1"]],
        high_confidence_near_groups=[["r3", "r2"]],
        conflict_policy="retain",
    )

    assert _group_mapping(first) == _group_mapping(second)


@pytest.mark.parametrize(
    (
        "policy",
        "retain_for_processing",
        "requires_manual_review",
        "excluded_row_count",
        "manual_review_row_count",
    ),
    [
        ("retain", True, False, 0, 0),
        ("exclude", False, False, 2, 0),
        ("manual_review", False, True, 0, 2),
    ],
)
def test_conflicting_groups_apply_explicit_policy(
    policy: str,
    retain_for_processing: bool,
    requires_manual_review: bool,
    excluded_row_count: int,
    manual_review_row_count: int,
) -> None:
    frame = _frame().iloc[:2].copy()
    frame["material"] = [1, 3]
    frame["design"] = [None, 2]

    assignments, report = assign_duplicate_groups(
        frame,
        exact_groups=[["r1", "r2"]],
        high_confidence_near_groups=[],
        conflict_policy=policy,
    )

    assert CONFLICT_POLICIES == ("retain", "exclude", "manual_review")
    assert assignments["group_has_label_conflict"].tolist() == [True, True]
    assert assignments["group_conflicting_label_columns"].tolist() == [
        "material",
        "material",
    ]
    assert assignments["group_has_mixed_missing_labels"].tolist() == [
        True,
        True,
    ]
    assert assignments["group_mixed_missing_label_columns"].tolist() == [
        "design",
        "design",
    ]
    assert assignments["retain_for_processing"].tolist() == [
        retain_for_processing,
        retain_for_processing,
    ]
    assert assignments["requires_manual_review"].tolist() == [
        requires_manual_review,
        requires_manual_review,
    ]
    assert report["summary"]["conflict_group_count"] == 1
    assert report["summary"]["mixed_missing_group_count"] == 1
    assert report["summary"]["excluded_row_count"] == excluded_row_count
    assert (
        report["summary"]["manual_review_row_count"]
        == manual_review_row_count
    )


def test_assign_duplicate_groups_rejects_unknown_candidate_member() -> None:
    with pytest.raises(ValueError, match="Unknown review_id"):
        assign_duplicate_groups(
            _frame(),
            exact_groups=[["r1", "missing"]],
            high_confidence_near_groups=[],
            conflict_policy="retain",
        )


def test_assign_duplicate_groups_rejects_duplicate_review_ids() -> None:
    frame = _frame()
    frame.loc[1, "review_id"] = "r1"

    with pytest.raises(ValueError, match="review_id values must be unique"):
        assign_duplicate_groups(
            frame,
            exact_groups=[],
            high_confidence_near_groups=[],
            conflict_policy="retain",
        )


def test_assign_duplicate_groups_requires_known_conflict_policy() -> None:
    with pytest.raises(ValueError, match="conflict_policy"):
        assign_duplicate_groups(
            _frame(),
            exact_groups=[],
            high_confidence_near_groups=[],
            conflict_policy="drop",  # type: ignore[arg-type]
        )


def test_assign_duplicate_groups_rejects_invalid_labels() -> None:
    frame = _frame()
    frame.loc[0, "price"] = 4

    with pytest.raises(ValueError, match="contains invalid labels"):
        assign_duplicate_groups(
            frame,
            exact_groups=[],
            high_confidence_near_groups=[],
            conflict_policy="retain",
        )


def test_assign_duplicate_groups_detects_group_id_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        grouping,
        "_hash_group_members",
        lambda _members: "0" * 64,
    )

    with pytest.raises(GroupIdCollisionError, match="SHA-256 collision"):
        assign_duplicate_groups(
            _frame().iloc[:2],
            exact_groups=[],
            high_confidence_near_groups=[],
            conflict_policy="retain",
        )
