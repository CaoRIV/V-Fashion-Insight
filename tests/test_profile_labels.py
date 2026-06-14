import csv
import json
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict

from v_fashion_insight.data import profile_labels


def _sample_dataset() -> DatasetDict:
    return DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "STT": [1, 2, 3, 4],
                    "Nội dung review": ["A", "B", "C", "D"],
                    "Chất liệu": [0.0, 1.0, 3.0, None],
                    "Kiểu dáng": [3.0, 0.0, 2.0, 0.0],
                    "Kích cỡ": [0.0, 0.0, 0.0, 0.0],
                    "Giá cả": [2.0, 0.0, 1.0, 0.0],
                    "Dịch vụ": [0.0, 3.0, 0.0, 0.0],
                }
            )
        }
    )


def test_profile_dataset_calculates_label_and_mention_distributions() -> None:
    report = profile_labels.profile_dataset(
        _sample_dataset(),
        source={"resolved_revision": "abc123"},
    )
    split = report["splits"]["train"]
    material = split["aspects"][0]

    assert report["source"]["resolved_revision"] == "abc123"
    assert material["aspect"] == "material"
    assert material["missing_count"] == 1
    assert material["not_mentioned_count"] == 1
    assert material["mentioned_count"] == 2
    assert material["mentioned_proportion"] == pytest.approx(0.5)
    assert material["mentioned_proportion_among_valid"] == pytest.approx(2 / 3)
    assert [row["count"] for row in material["labels"]] == [1, 1, 0, 1]

    mention_report = split["mentioned_aspects_per_review"]
    assert mention_report["mean"] == pytest.approx(1.75)
    assert mention_report["rows_with_missing_labels"] == 1
    assert [
        row["review_count"] for row in mention_report["distribution"]
    ] == [1, 0, 2, 1, 0, 0]


def test_profile_dataset_rejects_missing_label_columns() -> None:
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "STT": [1],
                    "Nội dung review": ["A"],
                    "Chất liệu": [0],
                }
            )
        }
    )

    with pytest.raises(ValueError, match="missing columns"):
        profile_labels.profile_dataset(dataset)


def test_profile_dataset_rejects_invalid_labels() -> None:
    dataset = _sample_dataset()
    frame = dataset["train"].to_pandas()
    frame.loc[0, "Chất liệu"] = 4.0
    invalid_dataset = DatasetDict(
        {"train": Dataset.from_pandas(frame, preserve_index=False)}
    )

    with pytest.raises(ValueError, match="invalid labels"):
        profile_labels.profile_dataset(invalid_dataset)


def test_profile_downloaded_dataset_writes_deterministic_reports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        profile_labels,
        "load_dataset_from_metadata",
        lambda *_args, **_kwargs: (
            _sample_dataset(),
            {"resolved_revision": "abc123"},
        ),
    )
    json_path = tmp_path / "labels.json"
    label_csv_path = tmp_path / "labels.csv"
    mention_csv_path = tmp_path / "mentions.csv"

    report = profile_labels.profile_downloaded_dataset(
        metadata_path=tmp_path / "metadata.json",
        json_path=json_path,
        label_csv_path=label_csv_path,
        mention_csv_path=mention_csv_path,
    )

    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    with label_csv_path.open(encoding="utf-8", newline="") as csv_file:
        label_rows = list(csv.DictReader(csv_file))
    with mention_csv_path.open(encoding="utf-8", newline="") as csv_file:
        mention_rows = list(csv.DictReader(csv_file))

    assert len(label_rows) == 20
    assert label_rows[0]["aspect"] == "material"
    assert label_rows[0]["label_name"] == "not_mentioned"
    assert len(mention_rows) == 6
    assert mention_rows[0]["mentioned_aspect_count"] == "0"


def test_main_returns_two_when_profiling_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        profile_labels,
        "profile_downloaded_dataset",
        lambda **_: (_ for _ in ()).throw(ValueError("bad labels")),
    )

    assert profile_labels.main([]) == 2
