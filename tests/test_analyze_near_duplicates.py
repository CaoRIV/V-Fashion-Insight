import csv
import json
from pathlib import Path

import pandas as pd
import pytest
from datasets import Dataset, DatasetDict

from v_fashion_insight.data import analyze_near_duplicates
from v_fashion_insight.data.analyze_duplicates import (
    normalize_for_duplicate_analysis,
)


def _review_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "STT": [1, 2, 3, 4],
            "Nội dung review": [
                (
                    "Quần bị chật phần eo không ních vào nổi. Giá cả thì đắt "
                    "đỏ lố bịch so với thị trường. Shop hỗ trợ đổi size nhanh."
                ),
                (
                    "Quần bị chật ở phần eo không mặc vào nổi. Giá cả đắt đỏ "
                    "so với thị trường. Shop hỗ trợ đổi kích cỡ rất nhanh."
                ),
                (
                    "Quần bị chật phần eo không ních vào nổi. Giá cả thì đắt "
                    "đỏ lố bịch so với thị trường. Shop hỗ trợ đổi size nhanh."
                ),
                (
                    "Áo có chất liệu cotton mềm mại, thiết kế đẹp và giao "
                    "hàng đúng hẹn nên mình khá hài lòng."
                ),
            ],
            "Chất liệu": [0.0, 0.0, 0.0, 3.0],
            "Kiểu dáng": [0.0, 0.0, 0.0, 3.0],
            "Kích cỡ": [1.0, 2.0, 1.0, 0.0],
            "Giá cả": [1.0, 1.0, 1.0, 0.0],
            "Dịch vụ": [3.0, 3.0, 3.0, 3.0],
        }
    )


def _signature(text: str) -> int:
    normalized = normalize_for_duplicate_analysis(text)
    tokens = analyze_near_duplicates.tokenize_for_similarity(normalized)
    return analyze_near_duplicates.calculate_simhash(
        analyze_near_duplicates.simhash_features(tokens)
    )


def test_simhash_and_similarity_primitives_are_deterministic() -> None:
    text = "áo cotton mềm và đẹp"
    first = _signature(text)
    second = _signature(text)

    assert first == second
    assert analyze_near_duplicates.hamming_distance(first, second) == 0
    assert analyze_near_duplicates.character_ngrams("abcd") == {
        "abc",
        "bcd",
    }
    assert analyze_near_duplicates.jaccard_similarity(
        {"abc", "bcd"},
        {"abc", "cde"},
    ) == pytest.approx(1 / 3)


def test_candidate_verification_excludes_exact_and_unrelated_reviews() -> None:
    frame = _review_frame()
    normalized = {
        index: normalize_for_duplicate_analysis(text)
        for index, text in frame["Nội dung review"].items()
    }
    signatures = {
        index: _signature(text)
        for index, text in frame["Nội dung review"].items()
    }

    pairs = analyze_near_duplicates._verify_candidates(
        frame,
        normalized,
        signatures,
        {(0, 1), (0, 2), (0, 3)},
    )

    assert len(pairs) == 1
    assert pairs[0]["first_review_id"] == 1
    assert pairs[0]["second_review_id"] == 2
    assert pairs[0]["classification"] in {
        "high_confidence",
        "needs_review",
    }
    assert pairs[0]["conflicting_label_columns"] == ["Kích cỡ"]


def test_high_confidence_pairs_form_clusters() -> None:
    frame = _review_frame()
    pairs = [
        {
            "classification": "high_confidence",
            "first_row_index": 0,
            "second_row_index": 1,
            "character_trigram_jaccard": 0.8,
            "sequence_ratio": 0.9,
        },
        {
            "classification": "high_confidence",
            "first_row_index": 1,
            "second_row_index": 2,
            "character_trigram_jaccard": 0.85,
            "sequence_ratio": 0.91,
        },
    ]

    clusters = analyze_near_duplicates._build_clusters(frame, pairs)

    assert len(clusters) == 1
    assert clusters[0]["member_count"] == 3
    assert clusters[0]["edge_count"] == 2
    assert clusters[0]["has_label_conflict"] is True
    assert clusters[0]["conflicting_label_columns"] == ["Kích cỡ"]


def test_deterministic_sampling_is_evenly_spaced() -> None:
    assert analyze_near_duplicates._sample_indices(10, 4) == [0, 2, 5, 7]
    assert analyze_near_duplicates._sample_indices(3, None) == [0, 1, 2]
    with pytest.raises(ValueError, match="positive"):
        analyze_near_duplicates._sample_indices(10, 0)


def test_analyze_downloaded_dataset_writes_reports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = DatasetDict(
        {"train": Dataset.from_pandas(_review_frame(), preserve_index=False)}
    )
    monkeypatch.setattr(
        analyze_near_duplicates,
        "load_dataset_from_metadata",
        lambda *_args, **_kwargs: (
            dataset,
            {"resolved_revision": "abc123"},
        ),
    )
    json_path = tmp_path / "near.json"
    pair_path = tmp_path / "pairs.csv"
    cluster_path = tmp_path / "clusters.csv"

    report = analyze_near_duplicates.analyze_downloaded_dataset(
        metadata_path=tmp_path / "metadata.json",
        json_path=json_path,
        pair_csv_path=pair_path,
        cluster_csv_path=cluster_path,
    )

    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    with pair_path.open(encoding="utf-8", newline="") as csv_file:
        pair_rows = list(csv.DictReader(csv_file))
    with cluster_path.open(encoding="utf-8", newline="") as csv_file:
        cluster_rows = list(csv.DictReader(csv_file))
    assert pair_path.read_text(encoding="utf-8").startswith("split,pair_id")
    assert cluster_path.read_text(encoding="utf-8").startswith(
        "split,cluster_id"
    )
    assert isinstance(pair_rows, list)
    assert isinstance(cluster_rows, list)


def test_main_returns_two_when_analysis_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        analyze_near_duplicates,
        "analyze_downloaded_dataset",
        lambda **_: (_ for _ in ()).throw(ValueError("bad data")),
    )

    assert analyze_near_duplicates.main([]) == 2
