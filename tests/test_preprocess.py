import json
from pathlib import Path

import pandas as pd
import pytest
from datasets import Dataset, DatasetDict

from v_fashion_insight.data import preprocess
from v_fashion_insight.data.preprocess import (
    REVIEW_ID_CANONICAL_FIELDS,
    ReviewIdCollisionError,
    generate_review_id,
    generate_review_ids,
    normalize_review_text,
)
from v_fashion_insight.data.validate import (
    ID_COLUMN,
    LABEL_COLUMNS,
    TEXT_COLUMN,
)


def test_normalize_review_text_applies_nfkc_and_collapses_whitespace() -> None:
    text = "A\u0301o\u00a0đẹp\r\n\tkhông  bị xù"

    assert normalize_review_text(text) == "Áo đẹp không bị xù"


def test_normalize_review_text_preserves_sentiment_bearing_content() -> None:
    text = "Không tệ!!! Size M-2024 😊 áo đầm ổn, giá 199k."

    assert normalize_review_text(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "áo đẹp không bị phai màu",
        "Shop giao\nhàng\t nhanh   và tư vấn tốt.",
        "Mã SP AB12 còn nguyên, size L vừa.",
    ],
)
def test_normalize_review_text_is_idempotent(text: str) -> None:
    normalized = normalize_review_text(text)

    assert normalize_review_text(normalized) == normalized


def test_normalize_review_text_keeps_meaningful_punctuation_digits_and_emoji() -> None:
    text = "Vải mỏng quá?! Nhưng màu #12 đẹp 😊, mua 2 cái."

    assert normalize_review_text(text) == text


def test_normalize_review_text_rejects_non_string_values() -> None:
    with pytest.raises(TypeError, match="review text must be a string"):
        normalize_review_text(None)  # type: ignore[arg-type]


def test_review_id_canonical_fields_are_explicit() -> None:
    assert REVIEW_ID_CANONICAL_FIELDS == (
        "dataset_name",
        "dataset_revision",
        "source_split",
        "source_id",
    )


def test_generate_review_id_is_stable_and_uses_source_identity() -> None:
    identity = {
        "dataset_name": "vinhplaykennen/FashionReviews",
        "dataset_revision": "60abb1cef934cb248b88a2ce4c99bb1ea3129c92",
        "source_split": "train",
        "source_id": 42,
    }

    first = generate_review_id(**identity)
    second = generate_review_id(**identity)

    assert first == second
    assert first == (
        "review_f41892c17ab9e333f6cdd75efd549dc4"
        "c0fe5b310eabd13654da67a0f7b169fb"
    )
    assert generate_review_id(**{**identity, "source_id": 43}) != first
    assert (
        generate_review_id(**{**identity, "dataset_revision": "different"})
        != first
    )


def test_generate_review_ids_returns_unique_ids_in_input_order() -> None:
    review_ids = generate_review_ids(
        [10, 11, 12],
        dataset_name="owner/dataset",
        dataset_revision="abc123",
        source_split="train",
    )

    assert len(review_ids) == 3
    assert len(set(review_ids)) == 3
    assert review_ids[0] == generate_review_id(
        dataset_name="owner/dataset",
        dataset_revision="abc123",
        source_split="train",
        source_id=10,
    )


def test_generate_review_ids_rejects_duplicate_source_identity() -> None:
    with pytest.raises(ValueError, match="Duplicate canonical review identity"):
        generate_review_ids(
            [10, 10],
            dataset_name="owner/dataset",
            dataset_revision="abc123",
            source_split="train",
        )


def test_generate_review_ids_detects_hash_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preprocess,
        "_hash_canonical_identity",
        lambda _identity: "0" * 64,
    )

    with pytest.raises(ReviewIdCollisionError, match="SHA-256 collision"):
        generate_review_ids(
            [10, 11],
            dataset_name="owner/dataset",
            dataset_revision="abc123",
            source_split="train",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_name", ""),
        ("dataset_revision", " "),
        ("source_split", None),
        ("source_id", ""),
        ("source_id", 1.5),
    ],
)
def test_generate_review_id_rejects_invalid_identity_fields(
    field: str,
    value: object,
) -> None:
    identity: dict[str, object] = {
        "dataset_name": "owner/dataset",
        "dataset_revision": "abc123",
        "source_split": "train",
        "source_id": 10,
    }
    identity[field] = value

    with pytest.raises((TypeError, ValueError)):
        generate_review_id(**identity)  # type: ignore[arg-type]


def _raw_dataset(*, invalid_label: bool = False) -> DatasetDict:
    labels = {
        LABEL_COLUMNS[0]: [3, 3, 3, None],
        LABEL_COLUMNS[1]: [3, 3, 3, 0],
        LABEL_COLUMNS[2]: [0, 0, 0, 0],
        LABEL_COLUMNS[3]: [2, 2, 2, 0],
        LABEL_COLUMNS[4]: [0, 0, 0, 3],
    }
    if invalid_label:
        labels[LABEL_COLUMNS[3]][0] = 4

    return DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    ID_COLUMN: [1, 2, 3, 4],
                    TEXT_COLUMN: [
                        "  A\u0301o\nđẹp  ",
                        "Áo đẹp",
                        "Áo rất đẹp",
                        "Shop giao hàng nhanh",
                    ],
                    **labels,
                }
            )
        }
    )


def _write_group_artifacts(directory: Path) -> tuple[Path, Path]:
    exact_path = directory / "exact.csv"
    near_path = directory / "near.csv"
    pd.DataFrame(
        [{"split": "train", "member_ids": "1|2"}]
    ).to_csv(exact_path, index=False)
    pd.DataFrame(
        [{"split": "train", "member_ids": "2|3"}]
    ).to_csv(near_path, index=False)
    return exact_path, near_path


def test_preprocess_downloaded_dataset_builds_deterministic_interim_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = {
        "dataset_name": "owner/dataset",
        "resolved_revision": "abc123",
        "metadata_path": "metadata.json",
    }
    previous_offline_modes = (
        preprocess.datasets_config.HF_DATASETS_OFFLINE,
        preprocess.hub_constants.HF_HUB_OFFLINE,
    )
    observed_offline_modes: list[tuple[bool, bool]] = []

    def fake_load_dataset(
        *_args: object,
        **_kwargs: object,
    ) -> tuple[DatasetDict, dict[str, str]]:
        observed_offline_modes.append(
            (
                preprocess.datasets_config.HF_DATASETS_OFFLINE,
                preprocess.hub_constants.HF_HUB_OFFLINE,
            )
        )
        return _raw_dataset(), source

    monkeypatch.setattr(
        preprocess,
        "load_dataset_from_metadata",
        fake_load_dataset,
    )
    exact_path, near_path = _write_group_artifacts(tmp_path)
    interim_path = tmp_path / "interim.csv"
    audit_path = tmp_path / "audit.json"

    audit = preprocess.preprocess_downloaded_dataset(
        metadata_path=tmp_path / "metadata.json",
        exact_group_path=exact_path,
        near_cluster_path=near_path,
        interim_path=interim_path,
        audit_path=audit_path,
        conflict_policy="retain",
    )
    interim = pd.read_csv(interim_path)
    mapping = interim.set_index("source_id")

    assert interim["review_id"].is_unique
    assert mapping.loc[1, "text"] == "Áo đẹp"
    assert mapping.loc[1, "source_text"] == "  A\u0301o\nđẹp  "
    assert (
        mapping.loc[1, "group_id"]
        == mapping.loc[2, "group_id"]
        == mapping.loc[3, "group_id"]
    )
    assert mapping.loc[4, "group_id"] != mapping.loc[1, "group_id"]
    assert pd.isna(mapping.loc[4, "material"])
    assert audit["summary"] == {
        "input_row_count": 4,
        "written_row_count": 4,
        "removed_row_count": 0,
        "held_for_manual_review_count": 0,
        "missing_label_cell_count": 1,
        "exact_candidate_group_count": 1,
        "near_candidate_group_count": 1,
    }
    assert audit["validation"]["allowed_error_count"] == 1
    assert audit["grouping"]["summary"]["group_count"] == 2
    assert json.loads(audit_path.read_text(encoding="utf-8")) == audit

    first_csv = interim_path.read_bytes()
    first_audit = audit_path.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        preprocess.preprocess_downloaded_dataset(
            metadata_path=tmp_path / "metadata.json",
            exact_group_path=exact_path,
            near_cluster_path=near_path,
            interim_path=interim_path,
            audit_path=audit_path,
            conflict_policy="retain",
        )

    preprocess.preprocess_downloaded_dataset(
        metadata_path=tmp_path / "metadata.json",
        exact_group_path=exact_path,
        near_cluster_path=near_path,
        interim_path=interim_path,
        audit_path=audit_path,
        conflict_policy="retain",
        force=True,
    )
    assert interim_path.read_bytes() == first_csv
    assert audit_path.read_bytes() == first_audit
    assert observed_offline_modes == [(True, True), (True, True)]
    assert (
        preprocess.datasets_config.HF_DATASETS_OFFLINE,
        preprocess.hub_constants.HF_HUB_OFFLINE,
    ) == previous_offline_modes


def test_preprocess_downloaded_dataset_stops_on_invalid_raw_labels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        preprocess,
        "load_dataset_from_metadata",
        lambda *_args, **_kwargs: (
            _raw_dataset(invalid_label=True),
            {
                "dataset_name": "owner/dataset",
                "resolved_revision": "abc123",
            },
        ),
    )
    exact_path, near_path = _write_group_artifacts(tmp_path)
    interim_path = tmp_path / "interim.csv"
    audit_path = tmp_path / "audit.json"

    with pytest.raises(ValueError, match="Blocking raw validation errors"):
        preprocess.preprocess_downloaded_dataset(
            metadata_path=tmp_path / "metadata.json",
            exact_group_path=exact_path,
            near_cluster_path=near_path,
            interim_path=interim_path,
            audit_path=audit_path,
            conflict_policy="retain",
        )

    assert not interim_path.exists()
    assert not audit_path.exists()
