import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict, DownloadConfig

from v_fashion_insight.data import validate


def _valid_dataset() -> DatasetDict:
    return DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "STT": [1, 2],
                    "Nội dung review": ["Áo đẹp", "Vải ổn"],
                    "Chất liệu": [0.0, 3.0],
                    "Kiểu dáng": [3.0, 0.0],
                    "Kích cỡ": [0.0, 2.0],
                    "Giá cả": [2, 0],
                    "Dịch vụ": [0.0, 3.0],
                }
            )
        }
    )


def test_validate_dataset_accepts_valid_integer_valued_labels() -> None:
    report = validate.validate_dataset(
        _valid_dataset(),
        validated_at=datetime(2026, 6, 14, 14, 0, tzinfo=UTC),
    )

    assert report["valid"] is True
    assert report["status"] == "passed"
    assert report["validated_at_utc"] == "2026-06-14T14:00:00+00:00"
    assert report["summary"] == {
        "split_count": 1,
        "total_rows": 2,
        "error_count": 0,
        "warning_count": 0,
    }


def test_validate_dataset_reports_schema_text_and_label_errors() -> None:
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "STT": [1, 2, 3],
                    "Nội dung review": ["Tốt", " ", None],
                    "Chất liệu": [0.0, 4.0, None],
                    "Kiểu dáng": [0.0, 1.5, 3.0],
                    "Kích cỡ": [0.0, 1.0, 2.0],
                    "Giá cả": [0, 1, 2],
                    "unexpected": ["x", "y", "z"],
                }
            )
        }
    )

    report = validate.validate_dataset(dataset)
    issue_codes = {issue["code"] for issue in report["issues"]}

    assert report["valid"] is False
    assert report["status"] == "failed"
    assert {
        "missing_required_columns",
        "unexpected_columns",
        "empty_review",
        "null_review",
        "null_label",
        "label_out_of_range",
        "non_integer_label",
    } <= issue_codes


def test_duplicate_reviews_are_warnings_not_errors() -> None:
    dataset = _valid_dataset()
    duplicated = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    column: list(dataset["train"][column])
                    + [dataset["train"][column][0]]
                    for column in dataset["train"].column_names
                }
            )
        }
    )

    report = validate.validate_dataset(duplicated)
    warning_codes = {
        issue["code"]
        for issue in report["issues"]
        if issue["severity"] == "warning"
    }

    assert report["valid"] is True
    assert warning_codes == {"duplicate_rows", "duplicate_review_text"}


def test_load_dataset_from_metadata_uses_pinned_revision_and_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "metadata.json"
    cache_dir = tmp_path / "cache"
    metadata_path.write_text(
        json.dumps(
            {
                "dataset_name": "owner/dataset",
                "config_name": "default",
                "resolved_revision": "abc123",
                "cache_dir": str(cache_dir),
                "downloaded_at_utc": "2026-06-14T13:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    observed: dict[str, object] = {}

    def fake_load_dataset(**kwargs: object) -> DatasetDict:
        observed.update(kwargs)
        return _valid_dataset()

    monkeypatch.setattr(validate, "load_dataset", fake_load_dataset)

    _, source = validate.load_dataset_from_metadata(metadata_path)

    assert observed["path"] == "owner/dataset"
    assert observed["name"] == "default"
    assert observed["revision"] == "abc123"
    assert observed["cache_dir"] == str(cache_dir)
    assert isinstance(observed["download_config"], DownloadConfig)
    assert observed["download_config"].local_files_only is True  # type: ignore[union-attr]
    assert source["resolved_revision"] == "abc123"


def test_validate_downloaded_dataset_writes_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        validate,
        "load_dataset_from_metadata",
        lambda *_args, **_kwargs: (_valid_dataset(), {"dataset_name": "test"}),
    )
    report_path = tmp_path / "report.json"

    report = validate.validate_downloaded_dataset(
        metadata_path=tmp_path / "metadata.json",
        report_path=report_path,
    )

    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_main_returns_one_for_invalid_data_and_writes_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = {
        "status": "failed",
        "valid": False,
        "summary": {
            "total_rows": 3,
            "error_count": 1,
            "warning_count": 0,
        },
        "issues": [
            {
                "code": "null_label",
                "severity": "error",
                "split": "train",
                "column": "Chất liệu",
                "count": 1,
                "message": "Aspect labels must not be null.",
            }
        ],
    }
    monkeypatch.setattr(
        validate,
        "validate_downloaded_dataset",
        lambda **_: report,
    )

    exit_code = validate.main([])

    assert exit_code == 1
    assert "Validation FAILED" in capsys.readouterr().out


def test_main_returns_two_when_validation_cannot_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        validate,
        "validate_downloaded_dataset",
        lambda **_: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )

    assert validate.main([]) == 2
