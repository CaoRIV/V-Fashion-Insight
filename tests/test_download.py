import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import Dataset, DatasetDict, DownloadMode

from v_fashion_insight.data import download


def _sample_dataset() -> DatasetDict:
    return DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "review": ["Áo đẹp", "Vải hơi mỏng"],
                    "material": [0, 1],
                }
            ),
            "validation": Dataset.from_dict(
                {
                    "review": ["Giao hàng nhanh"],
                    "material": [0],
                }
            ),
        }
    )


def test_download_dataset_resolves_revision_and_writes_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = _sample_dataset()
    load_calls: list[dict[str, object]] = []
    fixed_time = datetime(2026, 6, 14, 13, 0, tzinfo=UTC)

    class FakeApi:
        def dataset_info(self, **kwargs: object) -> SimpleNamespace:
            assert kwargs == {
                "repo_id": "owner/dataset",
                "revision": "release",
                "token": None,
            }
            return SimpleNamespace(sha="abc123")

    def fake_load_dataset(**kwargs: object) -> DatasetDict:
        load_calls.append(kwargs)
        return dataset

    monkeypatch.setattr(download, "HfApi", FakeApi)
    monkeypatch.setattr(download, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(download, "_utc_now", lambda: fixed_time)

    metadata_path = tmp_path / "metadata" / "dataset.json"
    cache_dir = tmp_path / "cache"
    metadata = download.download_dataset(
        dataset_name="owner/dataset",
        revision="release",
        cache_dir=cache_dir,
        metadata_path=metadata_path,
    )

    assert load_calls == [
        {
            "path": "owner/dataset",
            "name": None,
            "cache_dir": str(cache_dir),
            "revision": "abc123",
            "token": None,
            "download_mode": DownloadMode.REUSE_DATASET_IF_EXISTS,
        }
    ]
    assert metadata["resolved_revision"] == "abc123"
    assert metadata["requested_revision"] == "release"
    assert metadata["downloaded_at_utc"] == "2026-06-14T13:00:00+00:00"
    assert metadata["total_rows"] == 3
    assert metadata["splits"]["train"]["num_rows"] == 2
    assert metadata["splits"]["train"]["column_names"] == [
        "review",
        "material",
    ]
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == metadata


def test_force_redownload_uses_force_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed_mode: DownloadMode | None = None

    class FakeApi:
        def dataset_info(self, **_: object) -> SimpleNamespace:
            return SimpleNamespace(sha="abc123")

    def fake_load_dataset(**kwargs: object) -> DatasetDict:
        nonlocal observed_mode
        observed_mode = kwargs["download_mode"]  # type: ignore[assignment]
        return _sample_dataset()

    monkeypatch.setattr(download, "HfApi", FakeApi)
    monkeypatch.setattr(download, "load_dataset", fake_load_dataset)

    download.download_dataset(
        cache_dir=tmp_path / "cache",
        metadata_path=tmp_path / "metadata.json",
        force_redownload=True,
    )

    assert observed_mode is DownloadMode.FORCE_REDOWNLOAD


def test_download_rejects_single_split_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeApi:
        def dataset_info(self, **_: object) -> SimpleNamespace:
            return SimpleNamespace(sha="abc123")

    monkeypatch.setattr(download, "HfApi", FakeApi)
    monkeypatch.setattr(
        download,
        "load_dataset",
        lambda **_: Dataset.from_dict({"review": ["Áo đẹp"]}),
    )

    with pytest.raises(TypeError, match="DatasetDict"):
        download.download_dataset(
            cache_dir=tmp_path / "cache",
            metadata_path=tmp_path / "metadata.json",
        )

    assert not (tmp_path / "metadata.json").exists()


def test_resolve_revision_rejects_missing_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeApi:
        def dataset_info(self, **_: object) -> SimpleNamespace:
            return SimpleNamespace(sha=None)

    monkeypatch.setattr(download, "HfApi", FakeApi)

    with pytest.raises(RuntimeError, match="commit SHA"):
        download.resolve_dataset_revision("owner/dataset", "main")


def test_main_prints_ascii_safe_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata = {"feature": "Nội dung"}
    monkeypatch.setattr(download, "download_dataset", lambda **_: metadata)

    exit_code = download.main([])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.isascii()
    assert json.loads(captured.out) == metadata
