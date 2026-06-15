import csv
import json
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict

from v_fashion_insight.data import analyze_duplicates


def _dataset_with_duplicates() -> DatasetDict:
    return DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "STT": [1, 2, 3, 4, 5],
                    "Nội dung review": [
                        "Áo  ĐẸP",
                        "  áo đẹp ",
                        "Vải ổn",
                        "Vải ổn",
                        "Khác",
                    ],
                    "Chất liệu": [3.0, 3.0, 2.0, 1.0, 0.0],
                    "Kiểu dáng": [3.0, 3.0, 0.0, 0.0, 0.0],
                    "Kích cỡ": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "Giá cả": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "Dịch vụ": [0.0, 0.0, None, 0.0, 0.0],
                }
            )
        }
    )


def test_normalization_is_conservative_and_stable() -> None:
    first = analyze_duplicates.normalize_for_duplicate_analysis(
        "  ÁO\u200b\tĐẸP  "
    )
    second = analyze_duplicates.normalize_for_duplicate_analysis("áo đẹp")

    assert first == "áo đẹp"
    assert first == second
    assert analyze_duplicates.duplicate_group_id(first) == (
        analyze_duplicates.duplicate_group_id(second)
    )
    assert analyze_duplicates.normalize_for_duplicate_analysis(
        "áo đẹp!"
    ) != first


def test_analysis_detects_merged_variants_and_label_conflicts() -> None:
    report = analyze_duplicates.analyze_dataset(
        _dataset_with_duplicates(),
        source={"resolved_revision": "abc123"},
    )
    split = report["splits"]["train"]

    assert report["source"]["resolved_revision"] == "abc123"
    assert split["raw_text"] == {
        "unique_count": 4,
        "duplicate_group_count": 1,
        "duplicate_member_count": 2,
        "redundant_row_count": 1,
    }
    assert split["normalized_text"]["duplicate_group_count"] == 2
    assert split["normalized_text"]["duplicate_member_count"] == 4
    assert split["normalized_text"]["redundant_row_count"] == 2
    assert split["normalized_text"]["normalization_merged_group_count"] == 1
    assert split["normalized_text"]["label_conflict_group_count"] == 1
    assert split["normalized_text"]["mixed_missing_group_count"] == 1

    groups = {group["representative_preview"]: group for group in split["groups"]}
    assert groups["Áo ĐẸP"]["normalization_merged_variants"] is True
    assert groups["Vải ổn"]["conflicting_label_columns"] == ["Chất liệu"]
    assert groups["Vải ổn"]["mixed_missing_label_columns"] == ["Dịch vụ"]


def test_analysis_rejects_missing_or_null_text() -> None:
    missing_text = DatasetDict(
        {"train": Dataset.from_dict({"STT": [1]})}
    )
    null_text = _dataset_with_duplicates()
    frame = null_text["train"].to_pandas()
    frame.loc[0, "Nội dung review"] = None
    invalid_dataset = DatasetDict(
        {"train": Dataset.from_pandas(frame, preserve_index=False)}
    )

    with pytest.raises(ValueError, match="missing columns"):
        analyze_duplicates.analyze_dataset(missing_text)
    with pytest.raises(ValueError, match="null review text"):
        analyze_duplicates.analyze_dataset(invalid_dataset)


def test_analyze_downloaded_dataset_writes_reports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        analyze_duplicates,
        "load_dataset_from_metadata",
        lambda *_args, **_kwargs: (
            _dataset_with_duplicates(),
            {"resolved_revision": "abc123"},
        ),
    )
    json_path = tmp_path / "duplicates.json"
    csv_path = tmp_path / "duplicates.csv"

    report = analyze_duplicates.analyze_downloaded_dataset(
        metadata_path=tmp_path / "metadata.json",
        json_path=json_path,
        group_csv_path=csv_path,
    )

    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 2
    assert rows[0]["has_label_conflict"] == "True"


def test_main_returns_two_when_analysis_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        analyze_duplicates,
        "analyze_downloaded_dataset",
        lambda **_: (_ for _ in ()).throw(ValueError("bad data")),
    )

    assert analyze_duplicates.main([]) == 2
