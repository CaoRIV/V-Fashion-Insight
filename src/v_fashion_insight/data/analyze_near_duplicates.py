"""Find likely augmented review variants with scalable similarity blocking."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict

from v_fashion_insight.common.logging import configure_logging
from v_fashion_insight.data.analyze_duplicates import (
    _label_consistency,
    _preview,
    _serialize_identifier,
    duplicate_group_id,
    normalize_for_duplicate_analysis,
)
from v_fashion_insight.data.download import (
    DEFAULT_METADATA_PATH,
    write_metadata,
)
from v_fashion_insight.data.validate import (
    ID_COLUMN,
    LABEL_COLUMNS,
    TEXT_COLUMN,
    load_dataset_from_metadata,
)

DEFAULT_JSON_PATH = Path("reports/metrics/near_duplicate_analysis.json")
DEFAULT_PAIR_CSV_PATH = Path("reports/metrics/near_duplicate_pairs.csv")
DEFAULT_CLUSTER_CSV_PATH = Path(
    "reports/metrics/near_duplicate_clusters.csv"
)

WORD_PATTERN = re.compile(r"\w+", re.UNICODE)
SIMHASH_BITS = 64
SIMHASH_BANDS = 4
SIMHASH_BAND_BITS = SIMHASH_BITS // SIMHASH_BANDS
MAX_HAMMING_DISTANCE = 16
MIN_CHARACTER_COUNT = 30
MIN_TOKEN_COUNT = 6
REVIEW_JACCARD_THRESHOLD = 0.60
REVIEW_SEQUENCE_THRESHOLD = 0.75
HIGH_JACCARD_THRESHOLD = 0.72
HIGH_SEQUENCE_THRESHOLD = 0.82
MIN_LENGTH_RATIO = 0.65
REPRESENTATIVE_PAIR_LIMIT = 30
REPRESENTATIVE_CLUSTER_LIMIT = 20


def tokenize_for_similarity(normalized_text: str) -> list[str]:
    """Extract Unicode word tokens from normalized review text."""
    return WORD_PATTERN.findall(normalized_text)


def simhash_features(tokens: Sequence[str]) -> set[str]:
    """Use unique word unigrams and adjacent bigrams as SimHash features."""
    features = set(tokens)
    features.update(
        f"{tokens[index]}\u241f{tokens[index + 1]}"
        for index in range(len(tokens) - 1)
    )
    return features


@lru_cache(maxsize=500_000)
def _feature_hash(feature: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(),
        "big",
    )


def calculate_simhash(features: Sequence[str] | set[str]) -> int:
    """Calculate a deterministic 64-bit SimHash signature."""
    vector = [0] * SIMHASH_BITS
    for feature in features:
        feature_hash = _feature_hash(feature)
        for bit in range(SIMHASH_BITS):
            vector[bit] += 1 if feature_hash & (1 << bit) else -1
    return sum(
        1 << bit for bit, weight in enumerate(vector) if weight >= 0
    )


def hamming_distance(first: int, second: int) -> int:
    return (first ^ second).bit_count()


def character_ngrams(text: str, size: int = 3) -> set[str]:
    if len(text) < size:
        return {text} if text else set()
    return {text[index : index + size] for index in range(len(text) - size + 1)}


def jaccard_similarity(first: set[str], second: set[str]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 0.0


def _sample_indices(total_rows: int, sample_size: int | None) -> list[int]:
    if sample_size is None or sample_size >= total_rows:
        return list(range(total_rows))
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")

    step = total_rows / sample_size
    return [min(int(index * step), total_rows - 1) for index in range(sample_size)]


def _candidate_pairs(signatures: Mapping[int, int]) -> tuple[set[tuple[int, int]], int]:
    buckets: list[dict[int, list[int]]] = [
        defaultdict(list) for _ in range(SIMHASH_BANDS)
    ]
    candidates: set[tuple[int, int]] = set()
    raw_band_matches = 0
    mask = (1 << SIMHASH_BAND_BITS) - 1

    for row_index in sorted(signatures):
        signature = signatures[row_index]
        for band in range(SIMHASH_BANDS):
            key = (signature >> (band * SIMHASH_BAND_BITS)) & mask
            for other_index in buckets[band][key]:
                raw_band_matches += 1
                pair = (other_index, row_index)
                if pair not in candidates and hamming_distance(
                    signatures[other_index],
                    signature,
                ) <= MAX_HAMMING_DISTANCE:
                    candidates.add(pair)
            buckets[band][key].append(row_index)

    return candidates, raw_band_matches


def _pair_label_consistency(
    frame: pd.DataFrame,
    first_index: int,
    second_index: int,
) -> dict[str, Any]:
    return _label_consistency(frame.loc[[first_index, second_index]])


def _classify_pair(
    *,
    char_jaccard: float,
    sequence_ratio: float,
    length_ratio: float,
) -> str | None:
    if (
        char_jaccard >= HIGH_JACCARD_THRESHOLD
        and sequence_ratio >= HIGH_SEQUENCE_THRESHOLD
        and length_ratio >= MIN_LENGTH_RATIO
    ):
        return "high_confidence"
    if (
        char_jaccard >= REVIEW_JACCARD_THRESHOLD
        and sequence_ratio >= REVIEW_SEQUENCE_THRESHOLD
        and length_ratio >= MIN_LENGTH_RATIO
    ):
        return "needs_review"
    return None


def _verify_candidates(
    frame: pd.DataFrame,
    normalized_texts: Mapping[int, str],
    signatures: Mapping[int, int],
    candidates: set[tuple[int, int]],
) -> list[dict[str, Any]]:
    ngram_cache: dict[int, set[str]] = {}
    verified_pairs: list[dict[str, Any]] = []

    for first_index, second_index in sorted(candidates):
        first_text = normalized_texts[first_index]
        second_text = normalized_texts[second_index]
        if first_text == second_text:
            continue

        first_length = len(first_text)
        second_length = len(second_text)
        length_ratio = min(first_length, second_length) / max(
            first_length,
            second_length,
        )
        if length_ratio < MIN_LENGTH_RATIO:
            continue

        first_ngrams = ngram_cache.setdefault(
            first_index,
            character_ngrams(first_text),
        )
        second_ngrams = ngram_cache.setdefault(
            second_index,
            character_ngrams(second_text),
        )
        char_jaccard = jaccard_similarity(first_ngrams, second_ngrams)
        if char_jaccard < REVIEW_JACCARD_THRESHOLD:
            continue

        sequence_ratio = SequenceMatcher(
            None,
            first_text,
            second_text,
            autojunk=False,
        ).ratio()
        classification = _classify_pair(
            char_jaccard=char_jaccard,
            sequence_ratio=sequence_ratio,
            length_ratio=length_ratio,
        )
        if classification is None:
            continue

        consistency = _pair_label_consistency(
            frame,
            first_index,
            second_index,
        )
        first_id = _serialize_identifier(frame.at[first_index, ID_COLUMN])
        second_id = _serialize_identifier(frame.at[second_index, ID_COLUMN])
        verified_pairs.append(
            {
                "pair_id": hashlib.sha256(
                    (
                        f"{duplicate_group_id(first_text)}:"
                        f"{duplicate_group_id(second_text)}"
                    ).encode("ascii")
                ).hexdigest(),
                "classification": classification,
                "first_row_index": first_index,
                "second_row_index": second_index,
                "first_review_id": first_id,
                "second_review_id": second_id,
                "hamming_distance": hamming_distance(
                    signatures[first_index],
                    signatures[second_index],
                ),
                "length_ratio": length_ratio,
                "character_trigram_jaccard": char_jaccard,
                "sequence_ratio": sequence_ratio,
                "has_label_conflict": bool(
                    consistency["conflicting_label_columns"]
                ),
                "has_mixed_missing_labels": bool(
                    consistency["mixed_missing_label_columns"]
                ),
                "conflicting_label_columns": consistency[
                    "conflicting_label_columns"
                ],
                "mixed_missing_label_columns": consistency[
                    "mixed_missing_label_columns"
                ],
                "first_preview": _preview(frame.at[first_index, TEXT_COLUMN]),
                "second_preview": _preview(frame.at[second_index, TEXT_COLUMN]),
            }
        )

    verified_pairs.sort(
        key=lambda pair: (
            pair["classification"] != "high_confidence",
            -min(
                pair["character_trigram_jaccard"],
                pair["sequence_ratio"],
            ),
            pair["first_row_index"],
            pair["second_row_index"],
        )
    )
    return verified_pairs


class _DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, value: int) -> int:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[max(first_root, second_root)] = min(
                first_root,
                second_root,
            )


def _build_clusters(
    frame: pd.DataFrame,
    pairs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    disjoint_set = _DisjointSet()
    high_confidence_pairs = [
        pair for pair in pairs if pair["classification"] == "high_confidence"
    ]
    for pair in high_confidence_pairs:
        disjoint_set.union(
            int(pair["first_row_index"]),
            int(pair["second_row_index"]),
        )

    members_by_root: dict[int, set[int]] = defaultdict(set)
    for value in disjoint_set.parent:
        members_by_root[disjoint_set.find(value)].add(value)

    clusters: list[dict[str, Any]] = []
    for members in members_by_root.values():
        sorted_members = sorted(members)
        group = frame.loc[sorted_members]
        consistency = _label_consistency(group)
        member_ids = [
            _serialize_identifier(value) for value in group[ID_COLUMN]
        ]
        cluster_id = hashlib.sha256(
            "|".join(str(member_id) for member_id in member_ids).encode(
                "utf-8"
            )
        ).hexdigest()
        cluster_pairs = [
            pair
            for pair in high_confidence_pairs
            if int(pair["first_row_index"]) in members
            and int(pair["second_row_index"]) in members
        ]
        clusters.append(
            {
                "cluster_id": cluster_id,
                "member_count": len(sorted_members),
                "member_ids": member_ids,
                "edge_count": len(cluster_pairs),
                "minimum_character_trigram_jaccard": min(
                    pair["character_trigram_jaccard"]
                    for pair in cluster_pairs
                ),
                "minimum_sequence_ratio": min(
                    pair["sequence_ratio"] for pair in cluster_pairs
                ),
                "has_label_conflict": bool(
                    consistency["conflicting_label_columns"]
                ),
                "has_mixed_missing_labels": bool(
                    consistency["mixed_missing_label_columns"]
                ),
                "conflicting_label_columns": consistency[
                    "conflicting_label_columns"
                ],
                "mixed_missing_label_columns": consistency[
                    "mixed_missing_label_columns"
                ],
                "representative_preview": _preview(
                    group.iloc[0][TEXT_COLUMN]
                ),
            }
        )

    clusters.sort(
        key=lambda cluster: (
            not cluster["has_label_conflict"],
            -cluster["member_count"],
            cluster["cluster_id"],
        )
    )
    return clusters


def analyze_split(
    split_name: str,
    split: Dataset,
    *,
    sample_size: int | None = None,
) -> dict[str, Any]:
    """Analyze near-duplicate candidates in one split."""
    frame = split.to_pandas()
    required_columns = (ID_COLUMN, TEXT_COLUMN, *LABEL_COLUMNS)
    missing_columns = [
        column for column in required_columns if column not in frame
    ]
    if missing_columns:
        raise ValueError(
            f"Cannot analyze near-duplicates; missing columns: "
            f"{missing_columns!r}."
        )
    if frame[TEXT_COLUMN].isna().any():
        raise ValueError(
            "Cannot analyze near-duplicates with null review text."
        )

    selected_indices = _sample_indices(len(frame), sample_size)
    normalized_texts: dict[int, str] = {}
    signatures: dict[int, int] = {}
    excluded_short_count = 0

    for row_index in selected_indices:
        normalized_text = normalize_for_duplicate_analysis(
            frame.at[row_index, TEXT_COLUMN]
        )
        tokens = tokenize_for_similarity(normalized_text)
        if (
            len(normalized_text) < MIN_CHARACTER_COUNT
            or len(tokens) < MIN_TOKEN_COUNT
        ):
            excluded_short_count += 1
            continue
        normalized_texts[row_index] = normalized_text
        signatures[row_index] = calculate_simhash(simhash_features(tokens))

    candidates, raw_band_matches = _candidate_pairs(signatures)
    verified_pairs = _verify_candidates(
        frame,
        normalized_texts,
        signatures,
        candidates,
    )
    clusters = _build_clusters(frame, verified_pairs)
    high_confidence_pairs = [
        pair
        for pair in verified_pairs
        if pair["classification"] == "high_confidence"
    ]
    needs_review_pairs = [
        pair
        for pair in verified_pairs
        if pair["classification"] == "needs_review"
    ]

    return {
        "split": split_name,
        "num_rows": len(frame),
        "analyzed_row_count": len(selected_indices),
        "eligible_row_count": len(signatures),
        "excluded_short_count": excluded_short_count,
        "sample_size": sample_size,
        "fingerprint": split._fingerprint,
        "candidate_generation": {
            "raw_band_match_count": raw_band_matches,
            "hamming_filtered_pair_count": len(candidates),
        },
        "results": {
            "verified_pair_count": len(verified_pairs),
            "high_confidence_pair_count": len(high_confidence_pairs),
            "needs_review_pair_count": len(needs_review_pairs),
            "high_confidence_cluster_count": len(clusters),
            "clustered_review_count": len(
                {
                    member_id
                    for cluster in clusters
                    for member_id in cluster["member_ids"]
                }
            ),
            "label_conflict_pair_count": sum(
                pair["has_label_conflict"] for pair in verified_pairs
            ),
            "label_conflict_cluster_count": sum(
                cluster["has_label_conflict"] for cluster in clusters
            ),
        },
        "representative_pairs": verified_pairs[
            :REPRESENTATIVE_PAIR_LIMIT
        ],
        "representative_clusters": clusters[
            :REPRESENTATIVE_CLUSTER_LIMIT
        ],
        "pairs": verified_pairs,
        "clusters": clusters,
    }


def analyze_dataset(
    dataset: DatasetDict,
    *,
    source: Mapping[str, Any] | None = None,
    sample_size: int | None = None,
) -> dict[str, Any]:
    """Build a deterministic near-duplicate analysis report."""
    if not dataset:
        raise ValueError("Cannot analyze a dataset without splits.")

    return {
        "source": dict(source or {}),
        "method": {
            "normalization": (
                "Reuse exact-duplicate NFKC/casefold/whitespace normalization."
            ),
            "blocking": (
                "64-bit SimHash over unique Unicode word unigrams and adjacent "
                "bigrams, blocked by four 16-bit bands."
            ),
            "candidate_filter": (
                f"Hamming distance <= {MAX_HAMMING_DISTANCE}; reviews shorter "
                f"than {MIN_CHARACTER_COUNT} characters or {MIN_TOKEN_COUNT} "
                "tokens are excluded."
            ),
            "verification": (
                "Character trigram Jaccard plus SequenceMatcher ratio and "
                "normalized length ratio."
            ),
            "high_confidence_thresholds": {
                "character_trigram_jaccard": HIGH_JACCARD_THRESHOLD,
                "sequence_ratio": HIGH_SEQUENCE_THRESHOLD,
                "length_ratio": MIN_LENGTH_RATIO,
            },
            "needs_review_thresholds": {
                "character_trigram_jaccard": REVIEW_JACCARD_THRESHOLD,
                "sequence_ratio": REVIEW_SEQUENCE_THRESHOLD,
                "length_ratio": MIN_LENGTH_RATIO,
            },
            "grouping_policy": (
                "Only high-confidence edges form candidate clusters. Exact "
                "normalized duplicates are excluded from near-duplicate pairs."
            ),
            "limitations": (
                "LSH is a candidate generator and does not guarantee recall. "
                "Semantic paraphrases without lexical overlap may be missed; "
                "needs-review pairs must not be auto-grouped."
            ),
        },
        "recommended_grouping_policy": {
            "split_grouping": (
                "Assign one group_id to every connected high-confidence "
                "cluster and keep all members in the same data split."
            ),
            "exact_duplicates": (
                "Merge high-confidence cluster membership with exact "
                "duplicate groups before splitting."
            ),
            "label_conflicts": (
                "Keep conflicting members grouped for splitting, but do not "
                "deduplicate, overwrite labels, or use majority labels "
                "without manual adjudication."
            ),
            "needs_review": (
                "Do not auto-group needs-review pairs. Review them manually "
                "or leave them as independent groups."
            ),
            "unmatched_reviews": (
                "Assign each unmatched review its own stable group_id."
            ),
        },
        "splits": {
            split_name: analyze_split(
                split_name,
                split,
                sample_size=sample_size,
            )
            for split_name, split in sorted(dataset.items())
        },
    }


def _pair_csv_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, split_report in report["splits"].items():
        for pair in split_report["pairs"]:
            rows.append(
                {
                    "split": split_name,
                    **{
                        key: value
                        for key, value in pair.items()
                        if key
                        not in {
                            "conflicting_label_columns",
                            "mixed_missing_label_columns",
                        }
                    },
                    "conflicting_label_columns": "|".join(
                        pair["conflicting_label_columns"]
                    ),
                    "mixed_missing_label_columns": "|".join(
                        pair["mixed_missing_label_columns"]
                    ),
                }
            )
    return rows


def _cluster_csv_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, split_report in report["splits"].items():
        for cluster in split_report["clusters"]:
            rows.append(
                {
                    "split": split_name,
                    **{
                        key: value
                        for key, value in cluster.items()
                        if key
                        not in {
                            "member_ids",
                            "conflicting_label_columns",
                            "mixed_missing_label_columns",
                        }
                    },
                    "member_ids": "|".join(
                        str(member_id)
                        for member_id in cluster["member_ids"]
                    ),
                    "conflicting_label_columns": "|".join(
                        cluster["conflicting_label_columns"]
                    ),
                    "mixed_missing_label_columns": "|".join(
                        cluster["mixed_missing_label_columns"]
                    ),
                }
            )
    return rows


def write_csv(
    rows: Sequence[Mapping[str, Any]],
    destination: Path,
    fieldnames: Sequence[str],
) -> None:
    """Write deterministic UTF-8 CSV, including headers for empty results."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_downloaded_dataset(
    *,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    json_path: Path = DEFAULT_JSON_PATH,
    pair_csv_path: Path = DEFAULT_PAIR_CSV_PATH,
    cluster_csv_path: Path = DEFAULT_CLUSTER_CSV_PATH,
    local_files_only: bool = True,
    sample_size: int | None = None,
) -> dict[str, Any]:
    """Load the pinned dataset and write near-duplicate candidate reports."""
    dataset, source = load_dataset_from_metadata(
        metadata_path,
        local_files_only=local_files_only,
    )
    report = analyze_dataset(
        dataset,
        source=source,
        sample_size=sample_size,
    )
    write_metadata(report, Path(json_path))
    pair_rows = _pair_csv_rows(report)
    cluster_rows = _cluster_csv_rows(report)
    write_csv(
        pair_rows,
        Path(pair_csv_path),
        (
            "split",
            "pair_id",
            "classification",
            "first_row_index",
            "second_row_index",
            "first_review_id",
            "second_review_id",
            "hamming_distance",
            "length_ratio",
            "character_trigram_jaccard",
            "sequence_ratio",
            "has_label_conflict",
            "has_mixed_missing_labels",
            "first_preview",
            "second_preview",
            "conflicting_label_columns",
            "mixed_missing_label_columns",
        ),
    )
    write_csv(
        cluster_rows,
        Path(cluster_csv_path),
        (
            "split",
            "cluster_id",
            "member_count",
            "edge_count",
            "minimum_character_trigram_jaccard",
            "minimum_sequence_ratio",
            "has_label_conflict",
            "has_mixed_missing_labels",
            "representative_preview",
            "member_ids",
            "conflicting_label_columns",
            "mixed_missing_label_columns",
        ),
    )
    return report


def format_summary(report: Mapping[str, Any]) -> str:
    """Format a concise near-duplicate summary."""
    lines: list[str] = []
    for split_name, split_report in report["splits"].items():
        results = split_report["results"]
        lines.extend(
            [
                (
                    f"Split {split_name}: analyzed "
                    f"{split_report['analyzed_row_count']} reviews"
                ),
                (
                    "- candidate pairs after Hamming filter="
                    f"{split_report['candidate_generation']['hamming_filtered_pair_count']}"
                ),
                (
                    "- verified pairs="
                    f"{results['verified_pair_count']}, "
                    f"high_confidence={results['high_confidence_pair_count']}, "
                    f"needs_review={results['needs_review_pair_count']}"
                ),
                (
                    "- high-confidence clusters="
                    f"{results['high_confidence_cluster_count']}, "
                    f"clustered reviews={results['clustered_review_count']}, "
                    f"label-conflict clusters="
                    f"{results['label_conflict_cluster_count']}"
                ),
            ]
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Find likely augmented review variants using SimHash blocking and "
            "character-level verification."
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
        "--pair-csv-path",
        type=Path,
        default=DEFAULT_PAIR_CSV_PATH,
        help=f"Pair CSV path (default: {DEFAULT_PAIR_CSV_PATH}).",
    )
    parser.add_argument(
        "--cluster-csv-path",
        type=Path,
        default=DEFAULT_CLUSTER_CSV_PATH,
        help=f"Cluster CSV path (default: {DEFAULT_CLUSTER_CSV_PATH}).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help=(
            "Analyze an evenly spaced deterministic sample instead of all "
            "rows."
        ),
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
        report = analyze_downloaded_dataset(
            metadata_path=args.metadata_path,
            json_path=args.json_path,
            pair_csv_path=args.pair_csv_path,
            cluster_csv_path=args.cluster_csv_path,
            local_files_only=not args.allow_network,
            sample_size=args.sample_size,
        )
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        logger.error("Near-duplicate analysis could not run: %s", error)
        return 2

    print(format_summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
