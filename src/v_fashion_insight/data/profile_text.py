"""Profile review lengths and text patterns without modifying source text."""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict

from v_fashion_insight.common.logging import configure_logging
from v_fashion_insight.data.download import (
    DEFAULT_METADATA_PATH,
    write_metadata,
)
from v_fashion_insight.data.validate import (
    ID_COLUMN,
    TEXT_COLUMN,
    load_dataset_from_metadata,
)

DEFAULT_JSON_PATH = Path("reports/metrics/text_profile.json")
DEFAULT_PERCENTILE_CSV_PATH = Path(
    "reports/metrics/text_length_percentiles.csv"
)
DEFAULT_PATTERN_CSV_PATH = Path("reports/metrics/text_pattern_counts.csv")

PERCENTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
URL_PATTERN = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
REPEATED_WHITESPACE_PATTERN = re.compile(r"[ \t]{2,}")
REPEATED_PUNCTUATION_PATTERN = re.compile(r"[!?.,;:]{2,}")
LINE_BREAK_PATTERN = re.compile(r"\r\n|\r|\n")
ZERO_WIDTH_CHARACTERS = frozenset({"\u200b", "\u200c", "\u200d", "\ufeff"})
NON_BREAKING_SPACES = frozenset({"\u00a0", "\u202f"})


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _percentile_name(percentile: float) -> str:
    return f"p{round(percentile * 100):02d}"


def _length_statistics(values: pd.Series) -> dict[str, Any]:
    numeric_values = values.astype("int64")
    quantiles = numeric_values.quantile(PERCENTILES, interpolation="linear")
    first_quartile = float(quantiles.loc[0.25])
    third_quartile = float(quantiles.loc[0.75])
    interquartile_range = third_quartile - first_quartile
    lower_fence = max(0.0, first_quartile - 1.5 * interquartile_range)
    upper_fence = third_quartile + 1.5 * interquartile_range

    return {
        "minimum": int(numeric_values.min()),
        "maximum": int(numeric_values.max()),
        "mean": float(numeric_values.mean()),
        "standard_deviation": float(numeric_values.std(ddof=0)),
        "percentiles": {
            _percentile_name(percentile): float(quantiles.loc[percentile])
            for percentile in PERCENTILES
        },
        "iqr": interquartile_range,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "lower_outlier_count": int((numeric_values < lower_fence).sum()),
        "upper_outlier_count": int((numeric_values > upper_fence).sum()),
    }


def _is_emoji_like(character: str) -> bool:
    code_point = ord(character)
    return (
        0x1F1E6 <= code_point <= 0x1F1FF
        or 0x1F300 <= code_point <= 0x1FAFF
        or 0x2600 <= code_point <= 0x27BF
        or code_point in {0xFE0F, 0x20E3}
    )


def _count_characters(text: str, predicate: Callable[[str], bool]) -> int:
    return sum(predicate(character) for character in text)


def _count_regex(text: str, pattern: re.Pattern[str]) -> int:
    return sum(1 for _ in pattern.finditer(text))


def _signal_summary(counts: pd.Series, total_rows: int) -> dict[str, Any]:
    row_count = int(counts.gt(0).sum())
    return {
        "review_count": row_count,
        "review_proportion": _ratio(row_count, total_rows),
        "occurrence_count": int(counts.sum()),
    }


def _pattern_signals(texts: pd.Series) -> dict[str, dict[str, Any]]:
    total_rows = len(texts)
    counters: dict[str, pd.Series] = {
        "url": texts.map(lambda text: _count_regex(text, URL_PATTERN)),
        "emoji_like": texts.map(
            lambda text: _count_characters(text, _is_emoji_like)
        ),
        "tab": texts.str.count("\t"),
        "line_break": texts.map(
            lambda text: _count_regex(text, LINE_BREAK_PATTERN)
        ),
        "repeated_whitespace": texts.map(
            lambda text: _count_regex(text, REPEATED_WHITESPACE_PATTERN)
        ),
        "non_breaking_space": texts.map(
            lambda text: _count_characters(
                text,
                lambda character: character in NON_BREAKING_SPACES,
            )
        ),
        "zero_width_character": texts.map(
            lambda text: _count_characters(
                text,
                lambda character: character in ZERO_WIDTH_CHARACTERS,
            )
        ),
        "punctuation_character": texts.map(
            lambda text: _count_characters(
                text,
                lambda character: unicodedata.category(character).startswith(
                    "P"
                ),
            )
        ),
        "repeated_punctuation": texts.map(
            lambda text: _count_regex(text, REPEATED_PUNCTUATION_PATTERN)
        ),
        "digit_character": texts.map(
            lambda text: _count_characters(text, str.isdigit)
        ),
    }
    return {
        name: _signal_summary(counts, total_rows)
        for name, counts in counters.items()
    }


def _longest_reviews(
    frame: pd.DataFrame,
    character_lengths: pd.Series,
    token_lengths: pd.Series,
    limit: int = 10,
) -> list[dict[str, Any]]:
    ranking = pd.DataFrame(
        {
            "row_index": frame.index,
            "review_id": (
                frame[ID_COLUMN]
                if ID_COLUMN in frame
                else pd.Series(frame.index, index=frame.index)
            ),
            "character_count": character_lengths,
            "token_count": token_lengths,
        }
    )
    ranking = ranking.sort_values(
        by=["character_count", "token_count", "row_index"],
        ascending=[False, False, True],
    ).head(limit)

    return [
        {
            "row_index": int(row["row_index"]),
            "review_id": (
                int(row["review_id"])
                if pd.notna(row["review_id"])
                else None
            ),
            "character_count": int(row["character_count"]),
            "token_count": int(row["token_count"]),
        }
        for _, row in ranking.iterrows()
    ]


def profile_split(split_name: str, split: Dataset) -> dict[str, Any]:
    """Profile review lengths and text patterns for one dataset split."""
    frame = split.to_pandas()
    if TEXT_COLUMN not in frame:
        raise ValueError(
            f"Cannot profile text; missing column {TEXT_COLUMN!r}."
        )

    raw_texts = frame[TEXT_COLUMN]
    null_count = int(raw_texts.isna().sum())
    texts = raw_texts.dropna().astype(str)
    empty_count = int(texts.str.strip().eq("").sum())
    non_empty_texts = texts.loc[~texts.str.strip().eq("")]
    if non_empty_texts.empty:
        raise ValueError("Cannot profile text without non-empty reviews.")

    character_lengths = non_empty_texts.str.len()
    token_lengths = non_empty_texts.str.count(r"\S+")
    line_counts = non_empty_texts.map(
        lambda text: _count_regex(text, LINE_BREAK_PATTERN) + 1
    )
    character_statistics = _length_statistics(character_lengths)
    token_statistics = _length_statistics(token_lengths)

    source_frame = frame.loc[non_empty_texts.index]
    return {
        "split": split_name,
        "num_rows": len(frame),
        "fingerprint": split._fingerprint,
        "null_review_count": null_count,
        "empty_review_count": empty_count,
        "profiled_review_count": len(non_empty_texts),
        "lengths": {
            "characters": character_statistics,
            "whitespace_tokens": token_statistics,
            "lines": _length_statistics(line_counts),
        },
        "threshold_counts": {
            "characters_lte_10": int(character_lengths.le(10).sum()),
            "tokens_lte_3": int(token_lengths.le(3).sum()),
            "characters_gte_p99": int(
                character_lengths.ge(
                    character_statistics["percentiles"]["p99"]
                ).sum()
            ),
            "tokens_gte_p99": int(
                token_lengths.ge(
                    token_statistics["percentiles"]["p99"]
                ).sum()
            ),
        },
        "patterns": _pattern_signals(non_empty_texts),
        "longest_reviews": _longest_reviews(
            source_frame,
            character_lengths,
            token_lengths,
        ),
    }


def profile_dataset(
    dataset: DatasetDict,
    *,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic text-characteristics report."""
    if not dataset:
        raise ValueError("Cannot profile a dataset without splits.")

    return {
        "source": dict(source or {}),
        "definitions": {
            "whitespace_token": "A maximal sequence matching \\S+.",
            "emoji_like": (
                "Characters in selected Unicode emoji/symbol ranges; this is "
                "an analysis heuristic, not full emoji grapheme parsing."
            ),
            "outlier": "A value outside the 1.5 * IQR fences.",
        },
        "splits": {
            split_name: profile_split(split_name, split)
            for split_name, split in sorted(dataset.items())
        },
    }


def _percentile_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, split_report in report["splits"].items():
        for metric, statistics in split_report["lengths"].items():
            for percentile, value in statistics["percentiles"].items():
                rows.append(
                    {
                        "split": split_name,
                        "metric": metric,
                        "percentile": percentile,
                        "value": value,
                    }
                )
    return rows


def _pattern_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, split_report in report["splits"].items():
        for pattern, summary in split_report["patterns"].items():
            rows.append(
                {
                    "split": split_name,
                    "pattern": pattern,
                    **summary,
                }
            )
    return rows


def write_csv(
    rows: Sequence[Mapping[str, Any]],
    destination: Path,
) -> None:
    """Write a deterministic UTF-8 CSV file."""
    if not rows:
        raise ValueError("Cannot write a CSV report without rows.")

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def profile_downloaded_dataset(
    *,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    json_path: Path = DEFAULT_JSON_PATH,
    percentile_csv_path: Path = DEFAULT_PERCENTILE_CSV_PATH,
    pattern_csv_path: Path = DEFAULT_PATTERN_CSV_PATH,
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Load the pinned dataset and write text-characteristics reports."""
    dataset, source = load_dataset_from_metadata(
        metadata_path,
        local_files_only=local_files_only,
    )
    report = profile_dataset(dataset, source=source)
    write_metadata(report, Path(json_path))
    write_csv(_percentile_rows(report), Path(percentile_csv_path))
    write_csv(_pattern_rows(report), Path(pattern_csv_path))
    return report


def format_summary(report: Mapping[str, Any]) -> str:
    """Format a concise console summary."""
    lines: list[str] = []
    for split_name, split_report in report["splits"].items():
        character_stats = split_report["lengths"]["characters"]
        token_stats = split_report["lengths"]["whitespace_tokens"]
        lines.extend(
            [
                f"Split {split_name}: {split_report['num_rows']} reviews",
                (
                    "- characters: "
                    f"median={character_stats['percentiles']['p50']:.1f}, "
                    f"p95={character_stats['percentiles']['p95']:.1f}, "
                    f"p99={character_stats['percentiles']['p99']:.1f}, "
                    f"max={character_stats['maximum']}"
                ),
                (
                    "- whitespace tokens: "
                    f"median={token_stats['percentiles']['p50']:.1f}, "
                    f"p95={token_stats['percentiles']['p95']:.1f}, "
                    f"p99={token_stats['percentiles']['p99']:.1f}, "
                    f"max={token_stats['maximum']}"
                ),
                (
                    "- pattern review counts: "
                    f"url={split_report['patterns']['url']['review_count']}, "
                    "emoji_like="
                    f"{split_report['patterns']['emoji_like']['review_count']}, "
                    "repeated_whitespace="
                    f"{split_report['patterns']['repeated_whitespace']['review_count']}, "
                    "repeated_punctuation="
                    f"{split_report['patterns']['repeated_punctuation']['review_count']}"
                ),
            ]
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Profile review lengths, outliers, URLs, Unicode symbols, "
            "whitespace, and punctuation patterns."
        )
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help=f"Download metadata path (default: {DEFAULT_METADATA_PATH}).",
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=DEFAULT_JSON_PATH,
        help=f"JSON report path (default: {DEFAULT_JSON_PATH}).",
    )
    parser.add_argument(
        "--percentile-csv-path",
        type=Path,
        default=DEFAULT_PERCENTILE_CSV_PATH,
        help=(
            "Length percentile CSV path "
            f"(default: {DEFAULT_PERCENTILE_CSV_PATH})."
        ),
    )
    parser.add_argument(
        "--pattern-csv-path",
        type=Path,
        default=DEFAULT_PATTERN_CSV_PATH,
        help=f"Pattern CSV path (default: {DEFAULT_PATTERN_CSV_PATH}).",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow Hugging Face network access if the local cache is missing.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = configure_logging()
    try:
        report = profile_downloaded_dataset(
            metadata_path=args.metadata_path,
            json_path=args.json_path,
            percentile_csv_path=args.percentile_csv_path,
            pattern_csv_path=args.pattern_csv_path,
            local_files_only=not args.allow_network,
        )
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError) as error:
        logger.error("Text profiling could not run: %s", error)
        return 2

    print(format_summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
