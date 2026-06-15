import json
from pathlib import Path

import pandas as pd
import pytest

from v_fashion_insight.data import eda_report


def _reports() -> dict:
    fingerprint = "fingerprint"
    return {
        "validation": {
            "status": "failed",
            "source": {
                "dataset_name": "example/fashion",
                "resolved_revision": "abc123",
            },
            "splits": {
                "train": {
                    "num_rows": 4,
                    "fingerprint": fingerprint,
                    "column_names": [
                        "STT",
                        "Nội dung review",
                        "Chất liệu",
                        "Kiểu dáng",
                        "Kích cỡ",
                        "Giá cả",
                        "Dịch vụ",
                    ],
                    "dtypes": {
                        "STT": "int64",
                        "Nội dung review": "str",
                        "Chất liệu": "float64",
                        "Kiểu dáng": "float64",
                        "Kích cỡ": "float64",
                        "Giá cả": "float64",
                        "Dịch vụ": "float64",
                    },
                }
            },
            "issues": [
                {
                    "code": "null_label",
                    "column": "Dịch vụ",
                    "count": 1,
                }
            ],
        },
        "labels": {
            "label_mapping": {
                "0": "not_mentioned",
                "1": "negative",
                "2": "neutral",
                "3": "positive",
            },
            "splits": {
                "train": {
                    "num_rows": 4,
                    "fingerprint": fingerprint,
                    "aspects": [
                        {
                            "aspect": aspect,
                            "source_column": source_column,
                            "missing_count": 0,
                            "mentioned_count": 3,
                            "mentioned_proportion": 0.75,
                            "labels": [
                                {
                                    "label": label,
                                    "name": name,
                                    "count": 1,
                                    "proportion": 0.25,
                                    "proportion_among_valid": 0.25,
                                }
                                for label, name in (
                                    (0, "not_mentioned"),
                                    (1, "negative"),
                                    (2, "neutral"),
                                    (3, "positive"),
                                )
                            ],
                        }
                        for aspect, source_column in (
                            ("material", "Chất liệu"),
                            ("design", "Kiểu dáng"),
                            ("size", "Kích cỡ"),
                            ("price", "Giá cả"),
                            ("service", "Dịch vụ"),
                        )
                    ],
                    "mentioned_aspects_per_review": {
                        "mean": 3.0,
                        "distribution": [],
                    },
                }
            },
        },
        "text": {
            "splits": {
                "train": {
                    "num_rows": 4,
                    "fingerprint": fingerprint,
                    "null_review_count": 0,
                    "empty_review_count": 0,
                    "lengths": {
                        metric: {
                            "minimum": 1,
                            "maximum": 10,
                            "upper_outlier_count": 1,
                            "percentiles": {
                                "p50": 5.0,
                                "p95": 9.0,
                                "p99": 10.0,
                            },
                        }
                        for metric in (
                            "characters",
                            "whitespace_tokens",
                            "lines",
                        )
                    },
                    "threshold_counts": {},
                    "patterns": {
                        name: {"review_count": 0}
                        for name in (
                            "line_break",
                            "repeated_punctuation",
                            "digit_character",
                            "url",
                            "emoji_like",
                        )
                    },
                }
            }
        },
        "exact_duplicates": {
            "splits": {
                "train": {
                    "num_rows": 4,
                    "fingerprint": fingerprint,
                    "normalized_text": {
                        "duplicate_group_count": 1,
                        "duplicate_member_count": 2,
                        "label_conflict_group_count": 0,
                    },
                }
            }
        },
        "near_duplicates": {
            "recommended_grouping_policy": {
                "split_grouping": "Group connected reviews.",
            },
            "splits": {
                "train": {
                    "num_rows": 4,
                    "fingerprint": fingerprint,
                    "results": {
                        "verified_pair_count": 1,
                        "high_confidence_pair_count": 1,
                        "needs_review_pair_count": 0,
                        "high_confidence_cluster_count": 1,
                        "clustered_review_count": 2,
                        "label_conflict_pair_count": 0,
                        "label_conflict_cluster_count": 0,
                    },
                }
            },
        },
    }


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "STT": [1, 2, 3, 4],
            "Nội dung review": [
                "Áo mã AB-12 đẹp và vừa.",
                "Chất vải mỏng nhưng kiểu đẹp.",
                "Size chật, giá cao, giao hàng nhanh.",
                "Sản phẩm bình thường.",
            ],
            "Chất liệu": [0, 1, 2, 3],
            "Kiểu dáng": [1, 2, 3, 0],
            "Kích cỡ": [2, 3, 0, 1],
            "Giá cả": [3, 0, 1, 2],
            "Dịch vụ": [0, 2, 3, 1],
        }
    )


def test_report_alignment_rejects_mismatched_fingerprint() -> None:
    reports = _reports()
    reports["text"]["splits"]["train"]["fingerprint"] = "different"

    with pytest.raises(ValueError, match="do not describe the same"):
        eda_report.validate_report_alignment(reports)


def test_samples_are_deterministic_and_compact() -> None:
    first = eda_report.sample_representative_reviews(_frame())
    second = eda_report.sample_representative_reviews(_frame())

    assert first == second
    assert len(first["aspect_label_samples"]) == 20
    assert first["product_code_like_review_count"] == 1
    assert first["multi_aspect_mixed_sentiment_samples"][0]["review_id"] == 2


def test_summary_and_markdown_cover_phase_one_decisions() -> None:
    reports = _reports()
    samples = eda_report.sample_representative_reviews(_frame())

    summary = eda_report.build_eda_summary(reports, samples)
    markdown = eda_report.render_markdown(summary)

    assert summary["dataset"]["fingerprint"] == "fingerprint"
    assert summary["quality"]["null_label_cell_count"] == 1
    assert summary["quality"]["source_records_modified"] is False
    assert "Never replace missing labels with label 0" in json.dumps(summary)
    assert "leakage-resistant" in markdown
    assert "Recommended Phase 2 Policy" in markdown


def test_plot_functions_write_non_empty_png_files(tmp_path: Path) -> None:
    summary = eda_report.build_eda_summary(
        _reports(),
        eda_report.sample_representative_reviews(_frame()),
    )
    destinations = [
        tmp_path / "labels.png",
        tmp_path / "lengths.png",
        tmp_path / "duplicates.png",
    ]

    eda_report.plot_label_distribution(summary, destinations[0])
    eda_report.plot_review_lengths(_frame(), destinations[1])
    eda_report.plot_duplicate_summary(summary, destinations[2])

    assert all(path.stat().st_size > 0 for path in destinations)


def test_notebook_is_small_and_uses_reusable_builder() -> None:
    notebook_path = Path("notebooks/01_eda.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert notebook["nbformat"] == 4
    assert "build_downloaded_eda" in code
    assert notebook_path.stat().st_size < 100_000
    assert all(
        not cell.get("outputs")
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
