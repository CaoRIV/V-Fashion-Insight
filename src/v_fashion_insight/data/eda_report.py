"""Build the Phase 1 EDA report from reusable analysis artifacts."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "v-fashion-insight-matplotlib"),
)

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from v_fashion_insight.common.constants import ASPECTS, LABEL_NAMES
from v_fashion_insight.common.logging import configure_logging
from v_fashion_insight.data.download import DEFAULT_METADATA_PATH, write_metadata
from v_fashion_insight.data.profile_labels import ASPECT_SOURCE_COLUMNS
from v_fashion_insight.data.validate import (
    ID_COLUMN,
    LABEL_COLUMNS,
    TEXT_COLUMN,
    load_dataset_from_metadata,
)

DEFAULT_METRICS_DIR = Path("reports/metrics")
DEFAULT_FIGURES_DIR = Path("reports/figures")
DEFAULT_REPORT_PATH = Path("reports/eda_report.md")
DEFAULT_SUMMARY_PATH = DEFAULT_METRICS_DIR / "eda_summary.json"

REPORT_FILES = {
    "validation": "data_validation.json",
    "labels": "label_distribution.json",
    "text": "text_profile.json",
    "exact_duplicates": "exact_duplicate_analysis.json",
    "near_duplicates": "near_duplicate_analysis.json",
}

COLUMN_DEFINITIONS = {
    ID_COLUMN: "Source review identifier.",
    TEXT_COLUMN: "Vietnamese fashion-review text.",
    **{
        source_column: (
            f"{aspect.title()} aspect label: 0=not mentioned, 1=negative, "
            "2=neutral, 3=positive."
        )
        for aspect, source_column in ASPECT_SOURCE_COLUMNS.items()
    },
}

PRODUCT_CODE_PATTERN = re.compile(
    r"\b(?=[A-Z0-9-]*[A-Z])(?=[A-Z0-9-]*\d)"
    r"[A-Z0-9]+(?:-[A-Z0-9]+)*\b"
)


def _read_json(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as report_file:
        value = json.load(report_file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def load_analysis_reports(
    metrics_dir: Path = DEFAULT_METRICS_DIR,
) -> dict[str, dict[str, Any]]:
    """Load all Phase 1 analysis reports required by the EDA."""
    metrics_dir = Path(metrics_dir)
    return {
        name: _read_json(metrics_dir / filename)
        for name, filename in REPORT_FILES.items()
    }


def _only_split(report: Mapping[str, Any], report_name: str) -> tuple[str, Any]:
    splits = report.get("splits")
    if not isinstance(splits, Mapping) or len(splits) != 1:
        raise ValueError(
            f"{report_name} must contain exactly one analyzed split."
        )
    return next(iter(splits.items()))


def validate_report_alignment(
    reports: Mapping[str, Mapping[str, Any]],
) -> tuple[str, int, str]:
    """Ensure every report describes the same split and dataset fingerprint."""
    split_names: set[str] = set()
    row_counts: set[int] = set()
    fingerprints: set[str] = set()

    for name in REPORT_FILES:
        if name not in reports:
            raise ValueError(f"Missing required report: {name}.")
        split_name, split = _only_split(reports[name], name)
        split_names.add(split_name)
        row_counts.add(int(split["num_rows"]))
        fingerprints.add(str(split["fingerprint"]))

    if len(split_names) != 1 or len(row_counts) != 1 or len(fingerprints) != 1:
        raise ValueError(
            "Phase 1 reports do not describe the same split, row count, and "
            "dataset fingerprint."
        )
    return (
        next(iter(split_names)),
        next(iter(row_counts)),
        next(iter(fingerprints)),
    )


def _preview(text: str, limit: int = 180) -> str:
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


def sample_representative_reviews(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """Select deterministic, compact examples without changing source rows."""
    required = [ID_COLUMN, TEXT_COLUMN, *LABEL_COLUMNS]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Cannot sample reviews; missing columns: {missing!r}.")

    working = frame.loc[:, required].copy()
    working["_length"] = working[TEXT_COLUMN].astype(str).str.len()
    aspect_label_samples: list[dict[str, Any]] = []

    for aspect in ASPECTS:
        source_column = ASPECT_SOURCE_COLUMNS[aspect]
        for label in sorted(LABEL_NAMES):
            candidates = working.loc[working[source_column].eq(label)].copy()
            if candidates.empty:
                continue
            median_length = float(candidates["_length"].median())
            candidates["_median_distance"] = (
                candidates["_length"] - median_length
            ).abs()
            selected = candidates.sort_values(
                ["_median_distance", ID_COLUMN],
                kind="stable",
            ).iloc[0]
            aspect_label_samples.append(
                {
                    "aspect": aspect,
                    "label": label,
                    "label_name": LABEL_NAMES[label],
                    "review_id": int(selected[ID_COLUMN]),
                    "character_count": int(selected["_length"]),
                    "preview": _preview(selected[TEXT_COLUMN]),
                }
            )

    label_frame = working.loc[:, LABEL_COLUMNS]
    mentioned_count = label_frame.isin({1, 2, 3}).sum(axis=1)
    sentiment_variety = label_frame.apply(
        lambda row: len({int(value) for value in row if value in {1, 2, 3}}),
        axis=1,
    )
    multi_candidates = working.loc[
        mentioned_count.ge(3) & sentiment_variety.ge(2)
    ].copy()
    multi_candidates["_mentioned_count"] = mentioned_count.loc[
        multi_candidates.index
    ]
    multi_candidates["_sentiment_variety"] = sentiment_variety.loc[
        multi_candidates.index
    ]
    multi_candidates = multi_candidates.sort_values(
        ["_mentioned_count", "_sentiment_variety", ID_COLUMN],
        ascending=[False, False, True],
        kind="stable",
    ).head(5)

    multi_aspect_samples = []
    for _, row in multi_candidates.iterrows():
        multi_aspect_samples.append(
            {
                "review_id": int(row[ID_COLUMN]),
                "mentioned_aspect_count": int(row["_mentioned_count"]),
                "distinct_sentiment_count": int(row["_sentiment_variety"]),
                "labels": {
                    aspect: (
                        None
                        if pd.isna(row[ASPECT_SOURCE_COLUMNS[aspect]])
                        else int(row[ASPECT_SOURCE_COLUMNS[aspect]])
                    )
                    for aspect in ASPECTS
                },
                "preview": _preview(row[TEXT_COLUMN]),
            }
        )

    product_code_mask = working[TEXT_COLUMN].astype(str).str.contains(
        PRODUCT_CODE_PATTERN,
        regex=True,
    )
    product_code_rows = working.loc[product_code_mask].head(10)
    product_code_samples = [
        {
            "review_id": int(row[ID_COLUMN]),
            "preview": _preview(row[TEXT_COLUMN]),
        }
        for _, row in product_code_rows.iterrows()
    ]

    return {
        "aspect_label_samples": aspect_label_samples,
        "multi_aspect_mixed_sentiment_samples": multi_aspect_samples,
        "product_code_like_review_count": int(product_code_mask.sum()),
        "product_code_like_samples": product_code_samples,
    }


def build_eda_summary(
    reports: Mapping[str, Mapping[str, Any]],
    samples: Mapping[str, Any],
) -> dict[str, Any]:
    """Consolidate Phase 1 findings into a compact machine-readable summary."""
    split_name, row_count, fingerprint = validate_report_alignment(reports)
    validation = reports["validation"]
    label_split = reports["labels"]["splits"][split_name]
    text_split = reports["text"]["splits"][split_name]
    exact_split = reports["exact_duplicates"]["splits"][split_name]
    near_split = reports["near_duplicates"]["splits"][split_name]

    null_labels = {
        issue["column"]: int(issue["count"])
        for issue in validation["issues"]
        if issue["code"] == "null_label"
    }
    total_null_labels = sum(null_labels.values())
    invalid_label_codes = {
        "invalid_label_type",
        "non_integer_label",
        "label_out_of_range",
    }
    invalid_label_count = sum(
        int(issue.get("count", 1))
        for issue in validation["issues"]
        if issue["code"] in invalid_label_codes
    )

    aspect_summaries = []
    for aspect in label_split["aspects"]:
        valid_labels = aspect["labels"]
        largest = max(valid_labels, key=lambda item: item["count"])
        nonzero_counts = [
            item["count"] for item in valid_labels if item["count"] > 0
        ]
        smallest_count = min(nonzero_counts)
        aspect_summaries.append(
            {
                "aspect": aspect["aspect"],
                "source_column": aspect["source_column"],
                "missing_count": aspect["missing_count"],
                "mentioned_count": aspect["mentioned_count"],
                "mentioned_proportion": aspect["mentioned_proportion"],
                "dominant_label": largest["label"],
                "dominant_label_name": largest["name"],
                "dominant_label_count": largest["count"],
                "largest_to_smallest_class_ratio": (
                    largest["count"] / smallest_count
                ),
                "labels": valid_labels,
            }
        )

    schema_split = validation["splits"][split_name]
    schema = [
        {
            "column": column,
            "dtype": schema_split["dtypes"][column],
            "definition": COLUMN_DEFINITIONS[column],
        }
        for column in schema_split["column_names"]
    ]

    return {
        "dataset": {
            "name": validation["source"]["dataset_name"],
            "revision": validation["source"]["resolved_revision"],
            "split": split_name,
            "row_count": row_count,
            "fingerprint": fingerprint,
        },
        "schema": schema,
        "label_mapping": reports["labels"]["label_mapping"],
        "quality": {
            "validation_status": validation["status"],
            "null_review_count": text_split["null_review_count"],
            "empty_review_count": text_split["empty_review_count"],
            "null_label_cell_count": total_null_labels,
            "null_labels_by_column": null_labels,
            "invalid_label_count": invalid_label_count,
            "source_records_modified": False,
        },
        "labels": {
            "aspects": aspect_summaries,
            "mean_mentioned_aspects_per_review": label_split[
                "mentioned_aspects_per_review"
            ]["mean"],
            "mentioned_aspect_distribution": label_split[
                "mentioned_aspects_per_review"
            ]["distribution"],
        },
        "text": {
            "lengths": text_split["lengths"],
            "threshold_counts": text_split["threshold_counts"],
            "patterns": text_split["patterns"],
            "product_code_like_review_count": samples[
                "product_code_like_review_count"
            ],
        },
        "duplicates": {
            "exact": exact_split["normalized_text"],
            "near": near_split["results"],
            "source_group_information_present": False,
        },
        "samples": dict(samples),
        "recommended_policy": {
            "text_cleaning": [
                "Normalize Unicode to NFKC and collapse whitespace in the "
                "processed text while retaining the untouched source text.",
                "Preserve Vietnamese diacritics, negation, punctuation, "
                "digits, emoji, and product-code-like tokens unless later "
                "experiments demonstrate that a signal is harmful.",
                "Do not remove length outliers automatically; inspect them "
                "and use model truncation based on measured percentiles.",
            ],
            "missing_labels": (
                "Never replace missing labels with label 0. For independent "
                "aspect models, exclude a row only from the affected aspect; "
                "for a future multi-head model, use a masked loss or an "
                "explicitly documented complete-case policy."
            ),
            "grouping": reports["near_duplicates"][
                "recommended_grouping_policy"
            ],
            "split_strategy": (
                "Build a stable group key from the union of exact-duplicate "
                "groups and connected high-confidence near-duplicate "
                "clusters. Assign unmatched reviews individual stable group "
                "IDs, then perform a deterministic grouped split with "
                "multilabel distribution checks. Never allow a group to cross "
                "train, validation, or test boundaries."
            ),
            "label_conflicts": (
                "Keep conflicting duplicate members in one split, retain all "
                "original labels, and require manual adjudication before any "
                "deduplication or label overwrite."
            ),
        },
        "limitations": [
            "Near-duplicate LSH is a scalable heuristic and does not guarantee "
            "recall of every semantic paraphrase.",
            "Product-code detection is a conservative uppercase letter-and-"
            "digit heuristic, not a domain lexicon.",
            "The source dataset provides one train split and no original-"
            "review group identifier.",
        ],
    }


def _save_figure(figure: plt.Figure, destination: Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_label_distribution(
    summary: Mapping[str, Any],
    destination: Path,
) -> None:
    rows = []
    for aspect in summary["labels"]["aspects"]:
        for label in aspect["labels"]:
            rows.append(
                {
                    "aspect": aspect["aspect"],
                    "label": f"{label['label']}: {label['name']}",
                    "proportion": label["proportion_among_valid"],
                }
            )
    frame = pd.DataFrame(rows)
    figure, axis = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=frame,
        x="aspect",
        y="proportion",
        hue="label",
        ax=axis,
    )
    axis.set(
        title="Aspect label distribution among valid labels",
        xlabel="Aspect",
        ylabel="Proportion",
    )
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.legend(title="Label", bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    _save_figure(figure, destination)


def plot_review_lengths(frame: pd.DataFrame, destination: Path) -> None:
    texts = frame[TEXT_COLUMN].dropna().astype(str)
    lengths = pd.DataFrame(
        {
            "characters": texts.str.len(),
            "whitespace_tokens": texts.str.count(r"\S+"),
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, column, title in (
        (axes[0], "characters", "Character count"),
        (axes[1], "whitespace_tokens", "Whitespace-token count"),
    ):
        upper = float(lengths[column].quantile(0.99))
        sns.histplot(
            lengths.loc[lengths[column].le(upper), column],
            bins=50,
            ax=axis,
            color="#4472C4",
        )
        axis.axvline(
            lengths[column].median(),
            color="#C00000",
            linestyle="--",
            label="median",
        )
        axis.set(title=f"{title} (up to p99)", xlabel=title, ylabel="Reviews")
        axis.legend()
    figure.tight_layout()
    _save_figure(figure, destination)


def plot_duplicate_summary(
    summary: Mapping[str, Any],
    destination: Path,
) -> None:
    exact = summary["duplicates"]["exact"]
    near = summary["duplicates"]["near"]
    rows = pd.DataFrame(
        [
            {"metric": "Exact groups", "count": exact["duplicate_group_count"]},
            {
                "metric": "Exact members",
                "count": exact["duplicate_member_count"],
            },
            {
                "metric": "Near pairs (high)",
                "count": near["high_confidence_pair_count"],
            },
            {
                "metric": "Near pairs (review)",
                "count": near["needs_review_pair_count"],
            },
            {
                "metric": "Near clusters",
                "count": near["high_confidence_cluster_count"],
            },
            {
                "metric": "Clustered reviews",
                "count": near["clustered_review_count"],
            },
        ]
    )
    figure, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(data=rows, x="count", y="metric", color="#70AD47", ax=axis)
    axis.set(
        title="Exact and near-duplicate analysis",
        xlabel="Count",
        ylabel="",
    )
    for container in axis.containers:
        axis.bar_label(container, padding=3)
    figure.tight_layout()
    _save_figure(figure, destination)


def _escape_table(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def render_markdown(summary: Mapping[str, Any]) -> str:
    """Render the human-readable Phase 1 decision report."""
    dataset = summary["dataset"]
    quality = summary["quality"]
    text = summary["text"]
    exact = summary["duplicates"]["exact"]
    near = summary["duplicates"]["near"]
    lines = [
        "# Phase 1 Dataset Exploration Report",
        "",
        "## Dataset",
        "",
        f"- Dataset: `{dataset['name']}`",
        f"- Pinned revision: `{dataset['revision']}`",
        f"- Split: `{dataset['split']}`",
        f"- Rows: {dataset['row_count']:,}",
        f"- Fingerprint: `{dataset['fingerprint']}`",
        "- Source records modified: no",
        "",
        "## Schema and Labels",
        "",
        "| Column | Type | Definition |",
        "|---|---|---|",
    ]
    for column in summary["schema"]:
        lines.append(
            f"| {_escape_table(column['column'])} | `{column['dtype']}` | "
            f"{_escape_table(column['definition'])} |"
        )

    lines.extend(
        [
            "",
            "Labels are fixed as `0=not mentioned`, `1=negative`, "
            "`2=neutral`, and `3=positive`.",
            "",
            "## Data Quality",
            "",
            f"- Null or empty reviews: {quality['null_review_count']} null, "
            f"{quality['empty_review_count']} empty.",
            f"- Invalid labels outside 0-3: {quality['invalid_label_count']}.",
            f"- Missing label cells: {quality['null_label_cell_count']}.",
        ]
    )
    for column, count in quality["null_labels_by_column"].items():
        lines.append(f"- Missing `{column}` labels: {count}.")

    lines.extend(
        [
            "",
            "## Label Distribution",
            "",
            "| Aspect | Mentioned | Missing | Dominant class | "
            "Largest/smallest ratio |",
            "|---|---:|---:|---|---:|",
        ]
    )
    for aspect in summary["labels"]["aspects"]:
        lines.append(
            f"| {aspect['aspect']} | "
            f"{aspect['mentioned_proportion']:.2%} | "
            f"{aspect['missing_count']} | "
            f"{aspect['dominant_label']}: "
            f"{aspect['dominant_label_name']} | "
            f"{aspect['largest_to_smallest_class_ratio']:.2f} |"
        )
    lines.extend(
        [
            "",
            "![Label distribution](figures/label_distribution.png)",
            "",
            "## Review Text",
            "",
        ]
    )
    characters = text["lengths"]["characters"]
    tokens = text["lengths"]["whitespace_tokens"]
    patterns = text["patterns"]
    lines.extend(
        [
            f"- Characters: median {characters['percentiles']['p50']:.0f}, "
            f"p95 {characters['percentiles']['p95']:.0f}, "
            f"p99 {characters['percentiles']['p99']:.0f}, "
            f"maximum {characters['maximum']}.",
            f"- Whitespace tokens: median "
            f"{tokens['percentiles']['p50']:.0f}, "
            f"p95 {tokens['percentiles']['p95']:.0f}, "
            f"p99 {tokens['percentiles']['p99']:.0f}, "
            f"maximum {tokens['maximum']}.",
            f"- IQR upper outliers: {characters['upper_outlier_count']:,} by "
            f"characters and {tokens['upper_outlier_count']:,} by tokens.",
            f"- Multiline reviews: "
            f"{patterns['line_break']['review_count']:,}; repeated "
            f"punctuation: {patterns['repeated_punctuation']['review_count']:,}; "
            f"digit-containing reviews: "
            f"{patterns['digit_character']['review_count']:,}.",
            f"- URL heuristic: {patterns['url']['review_count']}; emoji-like "
            f"heuristic: {patterns['emoji_like']['review_count']}; "
            f"product-code-like heuristic: "
            f"{text['product_code_like_review_count']:,}.",
            "",
            "Length outliers are retained. The observed p95/p99 values should "
            "inform later vectorizer and tokenizer limits.",
            "",
            "![Review lengths](figures/review_length_distribution.png)",
            "",
            "## Duplicate and Augmentation Findings",
            "",
            f"- Exact normalized groups: "
            f"{exact['duplicate_group_count']:,} groups containing "
            f"{exact['duplicate_member_count']:,} reviews.",
            f"- Exact groups with label conflicts: "
            f"{exact['label_conflict_group_count']:,}.",
            f"- Verified near-duplicate pairs: "
            f"{near['verified_pair_count']:,}, including "
            f"{near['high_confidence_pair_count']:,} high-confidence and "
            f"{near['needs_review_pair_count']:,} needs-review pairs.",
            f"- High-confidence near clusters: "
            f"{near['high_confidence_cluster_count']:,}, containing "
            f"{near['clustered_review_count']:,} reviews.",
            f"- Near clusters with label conflicts: "
            f"{near['label_conflict_cluster_count']:,}.",
            "- The source data does not provide an original-review group ID.",
            "",
            "![Duplicate summary](figures/duplicate_summary.png)",
            "",
            "## Representative Review Samples",
            "",
            "The notebook and `eda_summary.json` contain one deterministic "
            "median-length sample for every available aspect/label pair. "
            "They also contain five multi-aspect reviews with mixed sentiment "
            "labels. Only IDs and short previews are stored.",
            "",
            "| Aspect | Label | Review ID | Characters | Preview |",
            "|---|---|---:|---:|---|",
        ]
    )
    for sample in summary["samples"]["aspect_label_samples"]:
        lines.append(
            f"| {sample['aspect']} | {sample['label']}: "
            f"{sample['label_name']} | {sample['review_id']} | "
            f"{sample['character_count']} | "
            f"{_escape_table(sample['preview'])} |"
        )

    lines.extend(
        [
            "",
            "## Recommended Phase 2 Policy",
            "",
            "1. Keep the source dataset immutable. Write all normalized data "
            "to interim or processed artifacts.",
        ]
    )
    lines.extend(
        f"- {rule}" for rule in summary["recommended_policy"]["text_cleaning"]
    )
    lines.extend(
        [
            f"- Missing labels: "
            f"{summary['recommended_policy']['missing_labels']}",
            f"- Grouping: "
            f"{summary['recommended_policy']['split_strategy']}",
            f"- Label conflicts: "
            f"{summary['recommended_policy']['label_conflicts']}",
            "- Exact duplicates and high-confidence near-duplicate clusters "
            "must be merged into the same grouping graph before splitting.",
            "- `needs_review` pairs are not grouped automatically.",
            "",
            "This strategy is leakage-resistant because all known variants of "
            "the same review are assigned atomically to one split.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in summary["limitations"])
    return "\n".join(lines) + "\n"


def build_downloaded_eda(
    *,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    metrics_dir: Path = DEFAULT_METRICS_DIR,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
    local_files_only: bool = True,
) -> dict[str, str]:
    """Generate the complete Phase 1 EDA from the pinned local dataset."""
    reports = load_analysis_reports(metrics_dir)
    split_name, _, expected_fingerprint = validate_report_alignment(reports)
    dataset, _ = load_dataset_from_metadata(
        metadata_path,
        local_files_only=local_files_only,
    )
    if split_name not in dataset:
        raise ValueError(f"Dataset does not contain split {split_name!r}.")
    if dataset[split_name]._fingerprint != expected_fingerprint:
        raise ValueError("Loaded dataset fingerprint does not match reports.")

    frame = dataset[split_name].to_pandas()
    samples = sample_representative_reviews(frame)
    summary = build_eda_summary(reports, samples)

    figures_dir = Path(figures_dir)
    label_figure = figures_dir / "label_distribution.png"
    length_figure = figures_dir / "review_length_distribution.png"
    duplicate_figure = figures_dir / "duplicate_summary.png"
    plot_label_distribution(summary, label_figure)
    plot_review_lengths(frame, length_figure)
    plot_duplicate_summary(summary, duplicate_figure)

    write_metadata(summary, Path(summary_path))
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(summary), encoding="utf-8")

    return {
        "summary": Path(summary_path).as_posix(),
        "report": report_path.as_posix(),
        "label_figure": label_figure.as_posix(),
        "length_figure": length_figure.as_posix(),
        "duplicate_figure": duplicate_figure.as_posix(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the Phase 1 EDA notebook artifacts and report."
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=DEFAULT_METRICS_DIR,
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow network access if the pinned dataset is not cached.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = configure_logging()
    try:
        artifacts = build_downloaded_eda(
            metadata_path=args.metadata_path,
            metrics_dir=args.metrics_dir,
            figures_dir=args.figures_dir,
            report_path=args.report_path,
            summary_path=args.summary_path,
            local_files_only=not args.allow_network,
        )
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        logger.error("EDA report could not be built: %s", error)
        return 2

    for name, path in artifacts.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
