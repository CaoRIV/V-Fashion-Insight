import csv
import json
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict

from v_fashion_insight.data import profile_text


def _sample_dataset() -> DatasetDict:
    return DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "STT": [10, 11, 12, 13],
                    "Nội dung review": [
                        "Áo đẹp!",
                        "Vải  hơi mỏng...\nXem https://example.com",
                        "Giao nhanh 😊",
                        "Size 42\tvừa",
                    ],
                }
            )
        }
    )


def test_profile_dataset_calculates_lengths_and_patterns() -> None:
    report = profile_text.profile_dataset(
        _sample_dataset(),
        source={"resolved_revision": "abc123"},
    )
    split = report["splits"]["train"]

    assert report["source"]["resolved_revision"] == "abc123"
    assert split["num_rows"] == 4
    assert split["null_review_count"] == 0
    assert split["empty_review_count"] == 0
    assert split["lengths"]["characters"]["minimum"] == len("Áo đẹp!")
    assert split["lengths"]["characters"]["maximum"] == len(
        "Vải  hơi mỏng...\nXem https://example.com"
    )
    assert split["lengths"]["whitespace_tokens"]["minimum"] == 2
    assert split["patterns"]["url"]["review_count"] == 1
    assert split["patterns"]["emoji_like"]["review_count"] == 1
    assert split["patterns"]["tab"]["review_count"] == 1
    assert split["patterns"]["line_break"]["review_count"] == 1
    assert split["patterns"]["repeated_whitespace"]["review_count"] == 1
    assert split["patterns"]["repeated_punctuation"]["review_count"] == 1
    assert split["patterns"]["digit_character"]["occurrence_count"] == 2
    assert split["longest_reviews"][0]["review_id"] == 11


def test_profile_dataset_tracks_null_and_empty_reviews() -> None:
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "STT": [1, 2, 3],
                    "Nội dung review": [None, "   ", "Hợp lý"],
                }
            )
        }
    )

    report = profile_text.profile_dataset(dataset)
    split = report["splits"]["train"]

    assert split["null_review_count"] == 1
    assert split["empty_review_count"] == 1
    assert split["profiled_review_count"] == 1


def test_profile_dataset_rejects_missing_or_unusable_text() -> None:
    missing_text = DatasetDict(
        {"train": Dataset.from_dict({"STT": [1]})}
    )
    empty_text = DatasetDict(
        {
            "train": Dataset.from_dict(
                {"STT": [1, 2], "Nội dung review": [None, " "]}
            )
        }
    )

    with pytest.raises(ValueError, match="missing column"):
        profile_text.profile_dataset(missing_text)
    with pytest.raises(ValueError, match="non-empty reviews"):
        profile_text.profile_dataset(empty_text)


def test_profile_downloaded_dataset_writes_reports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        profile_text,
        "load_dataset_from_metadata",
        lambda *_args, **_kwargs: (
            _sample_dataset(),
            {"resolved_revision": "abc123"},
        ),
    )
    json_path = tmp_path / "text.json"
    percentile_path = tmp_path / "percentiles.csv"
    pattern_path = tmp_path / "patterns.csv"

    report = profile_text.profile_downloaded_dataset(
        metadata_path=tmp_path / "metadata.json",
        json_path=json_path,
        percentile_csv_path=percentile_path,
        pattern_csv_path=pattern_path,
    )

    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    with percentile_path.open(encoding="utf-8", newline="") as csv_file:
        percentile_rows = list(csv.DictReader(csv_file))
    with pattern_path.open(encoding="utf-8", newline="") as csv_file:
        pattern_rows = list(csv.DictReader(csv_file))

    assert len(percentile_rows) == 30
    assert percentile_rows[0]["metric"] == "characters"
    assert len(pattern_rows) == 10
    assert pattern_rows[0]["pattern"] == "url"


def test_main_returns_two_when_profiling_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        profile_text,
        "profile_downloaded_dataset",
        lambda **_: (_ for _ in ()).throw(ValueError("bad text")),
    )

    assert profile_text.main([]) == 2
