import json
from pathlib import Path

import pandas as pd
import pytest

from v_fashion_insight.common.constants import DEFAULT_RANDOM_SEED
from v_fashion_insight.data.processed_contract import (
    GROUP_ID_COLUMN,
    REVIEW_ID_COLUMN,
    SPLIT_COLUMN,
)
from v_fashion_insight.data.split import (
    DEFAULT_SPLIT_RATIOS,
    SPLIT_ID_COLUMNS,
    assign_group_aware_splits,
    build_split_metadata,
    split_interim_dataset,
)


def _interim_frame() -> pd.DataFrame:
    rows = []
    label_pattern = [
        (3, 3, 0, 2, 0),
        (3, 0, 0, 2, 0),
        (1, 1, 3, 0, 2),
        (0, 3, 2, 3, 3),
        (2, 0, 0, 1, 0),
        (0, 2, 1, 0, 3),
        (3, 3, 3, 3, 3),
        (1, 0, 2, 1, 0),
        (0, 0, 0, 0, 0),
        (2, 2, 2, 2, 2),
    ]
    for index, labels in enumerate(label_pattern):
        rows.append(
            {
                REVIEW_ID_COLUMN: f"r{index:02d}",
                GROUP_ID_COLUMN: f"g{index:02d}",
                "text": f"review {index}",
                "material": labels[0],
                "design": labels[1],
                "size": labels[2],
                "price": labels[3],
                "service": labels[4],
            }
        )
    rows.extend(
        [
            {
                REVIEW_ID_COLUMN: "r10",
                GROUP_ID_COLUMN: "g_dup",
                "text": "duplicate variant 1",
                "material": 3,
                "design": 3,
                "size": 0,
                "price": 2,
                "service": 0,
            },
            {
                REVIEW_ID_COLUMN: "r11",
                GROUP_ID_COLUMN: "g_dup",
                "text": "duplicate variant 2",
                "material": 3,
                "design": 3,
                "size": 0,
                "price": 2,
                "service": None,
            },
        ]
    )
    return pd.DataFrame(rows)


def test_assign_group_aware_splits_is_deterministic_and_keeps_groups_together() -> None:
    frame = _interim_frame()

    first = assign_group_aware_splits(frame, seed=DEFAULT_RANDOM_SEED)
    shuffled = frame.sample(frac=1, random_state=123).reset_index(drop=True)
    second = assign_group_aware_splits(shuffled, seed=DEFAULT_RANDOM_SEED)

    assert first.equals(second)
    assert list(first.columns) == list(SPLIT_ID_COLUMNS)
    assert set(first[SPLIT_COLUMN]) == {"train", "validation", "test"}
    assert (
        first.groupby(GROUP_ID_COLUMN)[SPLIT_COLUMN].nunique().max()
        == 1
    )
    assert (
        first.loc[first[GROUP_ID_COLUMN] == "g_dup", SPLIT_COLUMN].nunique()
        == 1
    )


def test_build_split_metadata_records_ratios_seed_and_label_distributions() -> None:
    frame = _interim_frame()
    assignments = assign_group_aware_splits(frame)

    metadata = build_split_metadata(
        frame,
        assignments,
        input_path=Path("data/interim/reviews.csv"),
        split_ids_path=Path("data/processed/split_ids.csv"),
        input_sha256="input",
        split_ids_sha256="split",
    )

    assert metadata["schema_version"] == "v1"
    assert metadata["policy"]["seed"] == DEFAULT_RANDOM_SEED
    assert metadata["policy"]["target_ratios"] == DEFAULT_SPLIT_RATIOS
    assert metadata["inputs"]["row_count"] == len(frame)
    assert metadata["inputs"]["group_count"] == frame[GROUP_ID_COLUMN].nunique()
    assert metadata["outputs"]["split_ids_sha256"] == "split"
    assert set(metadata["summary"]) == {"train", "validation", "test"}
    assert (
        sum(split["row_count"] for split in metadata["summary"].values())
        == len(frame)
    )
    for split_summary in metadata["summary"].values():
        assert set(split_summary["label_distribution"]) == {
            "material",
            "design",
            "size",
            "price",
            "service",
        }


def test_split_interim_dataset_writes_deterministic_ids_and_metadata(
    tmp_path: Path,
) -> None:
    interim_path = tmp_path / "interim.csv"
    split_ids_path = tmp_path / "split_ids.csv"
    metadata_path = tmp_path / "metadata.json"
    _interim_frame().to_csv(interim_path, index=False)

    metadata = split_interim_dataset(
        interim_path=interim_path,
        split_ids_path=split_ids_path,
        metadata_path=metadata_path,
    )
    split_ids = pd.read_csv(split_ids_path)

    assert list(split_ids.columns) == list(SPLIT_ID_COLUMNS)
    assert split_ids[REVIEW_ID_COLUMN].tolist() == sorted(
        split_ids[REVIEW_ID_COLUMN].tolist()
    )
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == metadata
    assert metadata["outputs"]["split_ids_sha256"]

    first_ids = split_ids_path.read_bytes()
    first_metadata = metadata_path.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        split_interim_dataset(
            interim_path=interim_path,
            split_ids_path=split_ids_path,
            metadata_path=metadata_path,
        )

    split_interim_dataset(
        interim_path=interim_path,
        split_ids_path=split_ids_path,
        metadata_path=metadata_path,
        force=True,
    )
    assert split_ids_path.read_bytes() == first_ids
    assert metadata_path.read_bytes() == first_metadata


def test_assign_group_aware_splits_rejects_duplicate_review_ids() -> None:
    frame = _interim_frame()
    frame.loc[1, REVIEW_ID_COLUMN] = frame.loc[0, REVIEW_ID_COLUMN]

    with pytest.raises(ValueError, match="review_id values must be unique"):
        assign_group_aware_splits(frame)
