"""Find and quantify validation errors for every aspect and target label."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import pandas as pd

from v_fashion_insight.analysis.validation_predictions import (
    DEFAULT_INTERIM_PATH,
    DEFAULT_PREDICTIONS_PATH,
    DEFAULT_RECORDED_METRICS_PATH,
    DEFAULT_RUN_NAME,
    DEFAULT_SPLIT_IDS_PATH,
    assert_metrics_match_recorded,
    load_enriched_validation_predictions,
)
from v_fashion_insight.common.constants import ASPECTS, LABEL_NAMES, VALID_LABELS
from v_fashion_insight.data.processed_contract import (
    GROUP_ID_COLUMN,
    REVIEW_ID_COLUMN,
    TEXT_COLUMN,
)
from v_fashion_insight.models.baseline import file_sha256, write_json_atomic
from v_fashion_insight.models.metrics import compute_multioutput_metrics

DEFAULT_SUMMARY_PATH = Path(
    "reports/metrics/combined_svc_validation_aspect_label_analysis.json"
)
DEFAULT_LABEL_ERRORS_PATH = Path(
    "reports/metrics/combined_svc_validation_label_errors.csv"
)
DEFAULT_DETAILED_ERRORS_PATH = Path(
    "reports/analysis/combined_svc_validation_aspect_errors_detailed.csv"
)
DEFAULT_ASPECT_REPORT_DIR = Path(
    "reports/analysis/combined_svc_aspects"
)
LABEL_ORDER: Final[tuple[int, ...]] = tuple(sorted(VALID_LABELS))
ERROR_FAMILIES: Final[tuple[str, ...]] = (
    "missed_mention",
    "false_mention",
    "sentiment_confusion",
)


def _console_safe(value: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return value.encode(
        encoding,
        errors="backslashreplace",
    ).decode(encoding)


def classify_error_family(true_label: int, predicted_label: int) -> str:
    """Classify one incorrect aspect prediction by its structural failure."""
    if true_label == predicted_label:
        raise ValueError("Error-family classification requires unequal labels.")
    if true_label == 0:
        return "false_mention"
    if predicted_label == 0:
        return "missed_mention"
    return "sentiment_confusion"


def build_detailed_aspect_errors(enriched: pd.DataFrame) -> pd.DataFrame:
    """Convert wide predictions into one traceable row per aspect error."""
    rows: list[dict[str, Any]] = []
    for aspect in ASPECTS:
        true_column = f"true_{aspect}"
        predicted_column = f"predicted_{aspect}"
        errors = enriched.loc[
            enriched[true_column].notna()
            & enriched[true_column].ne(enriched[predicted_column])
        ]
        for row in errors.itertuples(index=False):
            true_label = int(getattr(row, true_column))
            predicted_label = int(getattr(row, predicted_column))
            rows.append(
                {
                    REVIEW_ID_COLUMN: getattr(row, REVIEW_ID_COLUMN),
                    GROUP_ID_COLUMN: getattr(row, GROUP_ID_COLUMN),
                    "aspect": aspect,
                    "true_label": true_label,
                    "true_label_name": LABEL_NAMES[true_label],
                    "predicted_label": predicted_label,
                    "predicted_label_name": LABEL_NAMES[predicted_label],
                    "error_family": classify_error_family(
                        true_label,
                        predicted_label,
                    ),
                    "review_error_aspect_count": int(
                        getattr(row, "error_aspect_count")
                    ),
                    "text_character_count": int(
                        getattr(row, "text_character_count")
                    ),
                    "text_token_count": int(
                        getattr(row, "text_token_count")
                    ),
                    TEXT_COLUMN: getattr(row, TEXT_COLUMN),
                }
            )
    columns = [
        REVIEW_ID_COLUMN,
        GROUP_ID_COLUMN,
        "aspect",
        "true_label",
        "true_label_name",
        "predicted_label",
        "predicted_label_name",
        "error_family",
        "review_error_aspect_count",
        "text_character_count",
        "text_token_count",
        TEXT_COLUMN,
    ]
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(
            [
                "aspect",
                "true_label",
                "predicted_label",
                REVIEW_ID_COLUMN,
            ]
        )
        .reset_index(drop=True)
    )


def _dominant_label(
    values: pd.Series,
) -> tuple[int | None, str | None, int]:
    if values.empty:
        return None, None, 0
    counts = values.astype(int).value_counts().sort_index()
    largest_count = int(counts.max())
    label = int(
        min(
            index
            for index, count in counts.items()
            if int(count) == largest_count
        )
    )
    return label, LABEL_NAMES[label], largest_count


def _metric_frames(
    enriched: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    true = pd.DataFrame(
        {
            aspect: enriched[f"true_{aspect}"]
            for aspect in ASPECTS
        }
    )
    predicted = pd.DataFrame(
        {
            aspect: enriched[f"predicted_{aspect}"]
            for aspect in ASPECTS
        }
    )
    return true, predicted


def build_aspect_label_analysis(
    enriched: pd.DataFrame,
    recorded_report: dict[str, Any],
    *,
    inputs: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Build per-aspect and per-label error diagnostics."""
    true, predicted = _metric_frames(enriched)
    metrics = compute_multioutput_metrics(true, predicted)
    assert_metrics_match_recorded(metrics, recorded_report)
    detailed = build_detailed_aspect_errors(enriched)

    label_rows: list[dict[str, Any]] = []
    aspect_summaries: dict[str, Any] = {}
    for aspect in ASPECTS:
        true_column = f"true_{aspect}"
        predicted_column = f"predicted_{aspect}"
        labeled = enriched.loc[enriched[true_column].notna()].copy()
        labeled[true_column] = labeled[true_column].astype(int)
        aspect_errors = detailed.loc[detailed["aspect"].eq(aspect)]
        family_counts = (
            aspect_errors["error_family"]
            .value_counts()
            .reindex(ERROR_FAMILIES, fill_value=0)
            .astype(int)
            .to_dict()
        )
        error_count = int(len(aspect_errors))
        confusions: list[dict[str, Any]] = []
        confusion_counts = (
            aspect_errors.groupby(
                ["true_label", "predicted_label"],
                sort=True,
            )
            .size()
            .sort_values(ascending=False)
        )
        for (true_label, predicted_label), count in confusion_counts.items():
            confusions.append(
                {
                    "true_label": int(true_label),
                    "true_label_name": LABEL_NAMES[int(true_label)],
                    "predicted_label": int(predicted_label),
                    "predicted_label_name": LABEL_NAMES[
                        int(predicted_label)
                    ],
                    "count": int(count),
                    "share_of_aspect_errors": (
                        int(count) / error_count if error_count else 0.0
                    ),
                }
            )

        aspect_label_rows: list[dict[str, Any]] = []
        for label in LABEL_ORDER:
            true_mask = labeled[true_column].eq(label)
            predicted_mask = labeled[predicted_column].eq(label)
            correct_mask = true_mask & predicted_mask
            support = int(true_mask.sum())
            predicted_count = int(predicted_mask.sum())
            correct_count = int(correct_mask.sum())
            false_negative_count = support - correct_count
            false_positive_count = predicted_count - correct_count
            outgoing = labeled.loc[
                true_mask & ~predicted_mask,
                predicted_column,
            ]
            incoming = labeled.loc[
                ~true_mask & predicted_mask,
                true_column,
            ]
            (
                dominant_wrong_label,
                dominant_wrong_name,
                dominant_wrong_count,
            ) = _dominant_label(outgoing)
            (
                dominant_source_label,
                dominant_source_name,
                dominant_source_count,
            ) = _dominant_label(incoming)
            class_metrics = metrics["aspects"][aspect]["classes"][
                str(label)
            ]
            row = {
                "aspect": aspect,
                "label": label,
                "label_name": LABEL_NAMES[label],
                "support": support,
                "predicted_count": predicted_count,
                "prediction_minus_support": predicted_count - support,
                "correct_count": correct_count,
                "error_count": false_negative_count,
                "error_rate": (
                    false_negative_count / support if support else 0.0
                ),
                "false_negative_count": false_negative_count,
                "false_positive_count": false_positive_count,
                "precision": class_metrics["precision"],
                "recall": class_metrics["recall"],
                "f1": class_metrics["f1"],
                "dominant_wrong_prediction": dominant_wrong_label,
                "dominant_wrong_prediction_name": dominant_wrong_name,
                "dominant_wrong_prediction_count": dominant_wrong_count,
                "dominant_false_positive_source": dominant_source_label,
                "dominant_false_positive_source_name": (
                    dominant_source_name
                ),
                "dominant_false_positive_source_count": (
                    dominant_source_count
                ),
            }
            label_rows.append(row)
            aspect_label_rows.append(row)

        weakest = min(
            aspect_label_rows,
            key=lambda row: (row["f1"], row["label"]),
        )
        highest_error = min(
            aspect_label_rows,
            key=lambda row: (-row["error_rate"], row["label"]),
        )
        aspect_metrics = metrics["aspects"][aspect]
        aspect_summaries[aspect] = {
            "evaluated_count": aspect_metrics["evaluated_count"],
            "correct_count": (
                aspect_metrics["evaluated_count"] - error_count
            ),
            "error_count": error_count,
            "error_rate": (
                error_count / aspect_metrics["evaluated_count"]
            ),
            "accuracy": aspect_metrics["accuracy"],
            "macro_f1": aspect_metrics["macro_f1"],
            "weighted_f1": aspect_metrics["weighted_f1"],
            "error_family_counts": family_counts,
            "error_family_shares": {
                family: (
                    count / error_count if error_count else 0.0
                )
                for family, count in family_counts.items()
            },
            "weakest_label_by_f1": {
                "label": weakest["label"],
                "name": weakest["label_name"],
                "f1": weakest["f1"],
            },
            "highest_error_rate_label": {
                "label": highest_error["label"],
                "name": highest_error["label_name"],
                "error_rate": highest_error["error_rate"],
            },
            "top_confusions": confusions,
            "labels": aspect_label_rows,
        }

    label_table = pd.DataFrame(label_rows)
    global_weakest = label_table.sort_values(
        ["f1", "aspect", "label"]
    ).iloc[0]
    summary = {
        "schema_version": "v1",
        "run_name": DEFAULT_RUN_NAME,
        "split": "validation",
        "test_set_accessed": False,
        "model_loaded": False,
        "inference_run": False,
        "metrics_match_recorded_report": True,
        "inputs": inputs,
        "summary": {
            "validation_row_count": int(len(enriched)),
            "total_aspect_error_count": int(len(detailed)),
            "error_family_counts": {
                family: int(
                    detailed["error_family"].eq(family).sum()
                )
                for family in ERROR_FAMILIES
            },
            "global_weakest_aspect_label": {
                "aspect": str(global_weakest["aspect"]),
                "label": int(global_weakest["label"]),
                "name": str(global_weakest["label_name"]),
                "f1": float(global_weakest["f1"]),
                "error_rate": float(global_weakest["error_rate"]),
            },
        },
        "aspects": aspect_summaries,
    }
    return summary, label_table, detailed


def render_aspect_report(
    aspect: str,
    aspect_summary: dict[str, Any],
    label_table: pd.DataFrame,
) -> str:
    """Render one self-contained Markdown report for an aspect."""
    weakest = aspect_summary["weakest_label_by_f1"]
    highest_error = aspect_summary["highest_error_rate_label"]
    families = aspect_summary["error_family_counts"]
    shares = aspect_summary["error_family_shares"]
    lines = [
        f"# {aspect.title()} Validation Errors",
        "",
        "Analysis source: saved `combined_svc` validation predictions. "
        "No model inference and no test-set access.",
        "",
        "## Summary",
        "",
        f"- Evaluated targets: {aspect_summary['evaluated_count']:,}.",
        f"- Errors: {aspect_summary['error_count']:,} "
        f"({aspect_summary['error_rate']:.2%}).",
        f"- Macro F1: {aspect_summary['macro_f1']:.6f}.",
        f"- Weakest label by F1: `{weakest['name']}` "
        f"({weakest['f1']:.4f}).",
        f"- Highest target error rate: `{highest_error['name']}` "
        f"({highest_error['error_rate']:.2%}).",
        "",
        "## Error Families",
        "",
        "| Family | Count | Share of aspect errors |",
        "|---|---:|---:|",
    ]
    for family in ERROR_FAMILIES:
        lines.append(
            f"| {family} | {families[family]:,} | "
            f"{shares[family]:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Label Diagnostics",
            "",
            "| Label | Support | Predicted | Correct | FN | FP | "
            "Error rate | Precision | Recall | F1 | "
            "Most often predicted as |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in label_table.sort_values("label").itertuples(index=False):
        dominant = (
            f"{row.dominant_wrong_prediction_name} "
            f"({row.dominant_wrong_prediction_count:,})"
            if row.dominant_wrong_prediction_name
            else "none"
        )
        lines.append(
            f"| {row.label_name} | {row.support:,} | "
            f"{row.predicted_count:,} | {row.correct_count:,} | "
            f"{row.false_negative_count:,} | "
            f"{row.false_positive_count:,} | "
            f"{row.error_rate:.2%} | {row.precision:.4f} | "
            f"{row.recall:.4f} | {row.f1:.4f} | {dominant} |"
        )

    lines.extend(
        [
            "",
            "## Confusion Directions",
            "",
            "| True | Predicted | Count | Share of aspect errors |",
            "|---|---|---:|---:|",
        ]
    )
    for confusion in aspect_summary["top_confusions"]:
        lines.append(
            f"| {confusion['true_label_name']} | "
            f"{confusion['predicted_label_name']} | "
            f"{confusion['count']:,} | "
            f"{confusion['share_of_aspect_errors']:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Evidence-based Priority",
            "",
            f"Start manual review with `{highest_error['name']}` targets, "
            f"then inspect the largest confusion direction above. "
            "The detailed error CSV retains stable IDs and review text for "
            "every row.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_aspect_errors(
    *,
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
    recorded_metrics_path: Path = DEFAULT_RECORDED_METRICS_PATH,
    interim_path: Path = DEFAULT_INTERIM_PATH,
    split_ids_path: Path = DEFAULT_SPLIT_IDS_PATH,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
    label_errors_path: Path = DEFAULT_LABEL_ERRORS_PATH,
    detailed_errors_path: Path = DEFAULT_DETAILED_ERRORS_PATH,
    aspect_report_dir: Path = DEFAULT_ASPECT_REPORT_DIR,
    force: bool = False,
) -> dict[str, Any]:
    """Write reproducible per-aspect and per-label error analysis outputs."""
    input_paths = [
        Path(predictions_path),
        Path(recorded_metrics_path),
        Path(interim_path),
        Path(split_ids_path),
    ]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required analysis input does not exist: {path.as_posix()}."
            )
    aspect_report_dir = Path(aspect_report_dir)
    report_paths = [
        aspect_report_dir / f"{aspect}.md"
        for aspect in ASPECTS
    ]
    output_paths = [
        Path(summary_path),
        Path(label_errors_path),
        Path(detailed_errors_path),
        *report_paths,
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Refusing to overwrite aspect error outputs: "
            + ", ".join(path.as_posix() for path in existing)
            + ". Use --force to replace them."
        )

    enriched = load_enriched_validation_predictions(
        predictions_path,
        interim_path,
        split_ids_path,
    )
    recorded_report = json.loads(
        Path(recorded_metrics_path).read_text(encoding="utf-8")
    )
    inputs = {
        "predictions_path": Path(predictions_path).as_posix(),
        "predictions_sha256": file_sha256(predictions_path),
        "recorded_metrics_path": Path(
            recorded_metrics_path
        ).as_posix(),
        "recorded_metrics_sha256": file_sha256(
            recorded_metrics_path
        ),
        "interim_path": Path(interim_path).as_posix(),
        "interim_sha256": file_sha256(interim_path),
        "split_ids_path": Path(split_ids_path).as_posix(),
        "split_ids_sha256": file_sha256(split_ids_path),
    }
    summary, label_table, detailed = build_aspect_label_analysis(
        enriched,
        recorded_report,
        inputs=inputs,
    )

    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(summary, summary_path)
    label_table.to_csv(
        label_errors_path,
        index=False,
        lineterminator="\n",
    )
    detailed.to_csv(
        detailed_errors_path,
        index=False,
        lineterminator="\n",
    )
    for aspect, report_path in zip(
        ASPECTS,
        report_paths,
        strict=True,
    ):
        aspect_labels = label_table.loc[
            label_table["aspect"].eq(aspect)
        ]
        report_path.write_text(
            render_aspect_report(
                aspect,
                summary["aspects"][aspect],
                aspect_labels,
            ),
            encoding="utf-8",
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Find validation errors for every combined_svc aspect and label "
            "without running the model."
        )
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
    )
    parser.add_argument(
        "--recorded-metrics-path",
        type=Path,
        default=DEFAULT_RECORDED_METRICS_PATH,
    )
    parser.add_argument(
        "--interim-path",
        type=Path,
        default=DEFAULT_INTERIM_PATH,
    )
    parser.add_argument(
        "--split-ids-path",
        type=Path,
        default=DEFAULT_SPLIT_IDS_PATH,
    )
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument(
        "--label-errors-path",
        type=Path,
        default=DEFAULT_LABEL_ERRORS_PATH,
    )
    parser.add_argument(
        "--detailed-errors-path",
        type=Path,
        default=DEFAULT_DETAILED_ERRORS_PATH,
    )
    parser.add_argument(
        "--aspect-report-dir",
        type=Path,
        default=DEFAULT_ASPECT_REPORT_DIR,
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = analyze_aspect_errors(
            predictions_path=args.predictions_path,
            recorded_metrics_path=args.recorded_metrics_path,
            interim_path=args.interim_path,
            split_ids_path=args.split_ids_path,
            summary_path=args.summary_path,
            label_errors_path=args.label_errors_path,
            detailed_errors_path=args.detailed_errors_path,
            aspect_report_dir=args.aspect_report_dir,
            force=args.force,
        )
    except (
        FileNotFoundError,
        FileExistsError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as error:
        print(
            _console_safe(f"Aspect error analysis failed: {error}"),
            file=sys.stderr,
        )
        return 2
    weakest = summary["summary"]["global_weakest_aspect_label"]
    print(
        _console_safe(
            "Aspect error analysis complete: "
            f"errors={summary['summary']['total_aspect_error_count']}, "
            f"weakest={weakest['aspect']}:{weakest['name']}, "
            f"f1={weakest['f1']:.6f}."
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
