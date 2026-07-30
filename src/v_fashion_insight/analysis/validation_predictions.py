"""Analyze saved validation predictions without loading or running a model."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from v_fashion_insight.common.constants import ASPECTS, LABEL_NAMES, VALID_LABELS
from v_fashion_insight.data.processed_contract import (
    GROUP_ID_COLUMN,
    REVIEW_ID_COLUMN,
    SPLIT_COLUMN,
    TEXT_COLUMN,
)
from v_fashion_insight.models.baseline import file_sha256, write_json_atomic
from v_fashion_insight.models.metrics import compute_multioutput_metrics

DEFAULT_RUN_NAME: Final[str] = "combined_svc"
DEFAULT_PREDICTIONS_PATH = Path(
    "models/baseline/combined_svc/validation_predictions.csv"
)
DEFAULT_RECORDED_METRICS_PATH = Path(
    "models/baseline/combined_svc/validation_metrics.json"
)
DEFAULT_INTERIM_PATH = Path("data/interim/reviews.csv")
DEFAULT_SPLIT_IDS_PATH = Path("data/processed/split_ids.csv")
DEFAULT_SUMMARY_PATH = Path(
    "reports/metrics/combined_svc_validation_prediction_analysis.json"
)
DEFAULT_ASPECT_ERRORS_PATH = Path(
    "reports/metrics/combined_svc_validation_aspect_errors.csv"
)
DEFAULT_CONFUSIONS_PATH = Path(
    "reports/metrics/combined_svc_validation_confusions.csv"
)
DEFAULT_EXAMPLES_PATH = Path(
    "reports/analysis/combined_svc_validation_error_examples.csv"
)
DEFAULT_REPORT_PATH = Path(
    "reports/analysis/combined_svc_validation_prediction_analysis.md"
)
LABEL_ORDER: Final[tuple[int, ...]] = tuple(sorted(VALID_LABELS))


def _console_safe(value: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return value.encode(
        encoding,
        errors="backslashreplace",
    ).decode(encoding)


def _required_prediction_columns() -> tuple[str, ...]:
    columns = [REVIEW_ID_COLUMN, GROUP_ID_COLUMN]
    for aspect in ASPECTS:
        columns.extend((f"true_{aspect}", f"predicted_{aspect}"))
    return tuple(columns)


def _validate_prediction_labels(frame: pd.DataFrame) -> None:
    for aspect in ASPECTS:
        true_column = f"true_{aspect}"
        predicted_column = f"predicted_{aspect}"
        true_values = pd.to_numeric(frame[true_column], errors="coerce")
        unexpected_null = frame[true_column].notna() & true_values.isna()
        if unexpected_null.any():
            raise ValueError(
                f"{true_column} contains non-numeric target values."
            )
        non_null_true = true_values.dropna()
        if (
            not non_null_true.mod(1).eq(0).all()
            or not non_null_true.isin(VALID_LABELS).all()
        ):
            raise ValueError(f"{true_column} contains invalid target labels.")

        predicted_values = pd.to_numeric(
            frame[predicted_column],
            errors="coerce",
        )
        if (
            predicted_values.isna().any()
            or not predicted_values.mod(1).eq(0).all()
            or not predicted_values.isin(VALID_LABELS).all()
        ):
            raise ValueError(
                f"{predicted_column} must contain labels 0-3 for every row."
            )
        frame[true_column] = true_values
        frame[predicted_column] = predicted_values.astype(int)


def load_enriched_validation_predictions(
    predictions_path: Path,
    interim_path: Path,
    split_ids_path: Path,
) -> pd.DataFrame:
    """Validate prediction coverage and join public review text by stable ID."""
    paths = [
        Path(predictions_path),
        Path(interim_path),
        Path(split_ids_path),
    ]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required analysis input does not exist: {path.as_posix()}."
            )

    predictions = pd.read_csv(predictions_path)
    required = _required_prediction_columns()
    missing = [
        column for column in required if column not in predictions.columns
    ]
    if missing:
        raise ValueError(
            f"Validation predictions are missing columns: {missing!r}."
        )
    predictions = predictions.loc[:, required].copy()
    if predictions.empty:
        raise ValueError("Validation predictions must not be empty.")
    if predictions[REVIEW_ID_COLUMN].duplicated().any():
        raise ValueError("Validation prediction review_id values must be unique.")
    for column in (REVIEW_ID_COLUMN, GROUP_ID_COLUMN):
        if (
            predictions[column].isna().any()
            or predictions[column].astype("string").str.strip().eq("").any()
        ):
            raise ValueError(f"{column} values must not be null or empty.")
    _validate_prediction_labels(predictions)

    split_ids = pd.read_csv(split_ids_path)
    split_required = [
        REVIEW_ID_COLUMN,
        GROUP_ID_COLUMN,
        SPLIT_COLUMN,
    ]
    missing_split = [
        column for column in split_required if column not in split_ids.columns
    ]
    if missing_split:
        raise ValueError(f"Split IDs are missing columns: {missing_split!r}.")
    validation_ids = split_ids.loc[
        split_ids[SPLIT_COLUMN].eq("validation"),
        split_required,
    ].copy()
    if validation_ids[REVIEW_ID_COLUMN].duplicated().any():
        raise ValueError("Validation split review_id values must be unique.")
    prediction_keys = set(
        zip(
            predictions[REVIEW_ID_COLUMN],
            predictions[GROUP_ID_COLUMN],
            strict=True,
        )
    )
    validation_keys = set(
        zip(
            validation_ids[REVIEW_ID_COLUMN],
            validation_ids[GROUP_ID_COLUMN],
            strict=True,
        )
    )
    if prediction_keys != validation_keys:
        missing_count = len(validation_keys.difference(prediction_keys))
        unknown_count = len(prediction_keys.difference(validation_keys))
        raise ValueError(
            "Predictions must exactly cover the validation split; "
            f"missing={missing_count}, unknown={unknown_count}."
        )

    interim = pd.read_csv(
        interim_path,
        usecols=[
            REVIEW_ID_COLUMN,
            GROUP_ID_COLUMN,
            TEXT_COLUMN,
        ],
    )
    if interim[REVIEW_ID_COLUMN].duplicated().any():
        raise ValueError("Interim review_id values must be unique.")
    enriched = predictions.merge(
        interim,
        on=[REVIEW_ID_COLUMN, GROUP_ID_COLUMN],
        how="left",
        validate="one_to_one",
    )
    if (
        enriched[TEXT_COLUMN].isna().any()
        or enriched[TEXT_COLUMN].astype("string").str.strip().eq("").any()
    ):
        raise ValueError(
            "Every validation prediction must resolve to non-empty review text."
        )

    enriched["text_character_count"] = (
        enriched[TEXT_COLUMN].astype(str).str.len().astype(int)
    )
    enriched["text_token_count"] = (
        enriched[TEXT_COLUMN]
        .astype(str)
        .str.split()
        .map(len)
        .astype(int)
    )
    error_columns: list[str] = []
    for aspect in ASPECTS:
        true_column = f"true_{aspect}"
        predicted_column = f"predicted_{aspect}"
        error_column = f"error_{aspect}"
        error_columns.append(error_column)
        enriched[error_column] = (
            enriched[true_column].notna()
            & enriched[true_column].ne(enriched[predicted_column])
        )
    enriched["error_aspect_count"] = (
        enriched[error_columns].sum(axis=1).astype(int)
    )
    enriched["complete_target"] = enriched[
        [f"true_{aspect}" for aspect in ASPECTS]
    ].notna().all(axis=1)
    enriched["exact_match"] = (
        enriched["complete_target"]
        & enriched["error_aspect_count"].eq(0)
    )
    return enriched.sort_values(REVIEW_ID_COLUMN).reset_index(drop=True)


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


def _aspect_error_rows(
    enriched: pd.DataFrame,
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for aspect in ASPECTS:
        true_column = f"true_{aspect}"
        predicted_column = f"predicted_{aspect}"
        labeled = enriched.loc[enriched[true_column].notna()]
        errors = labeled.loc[
            labeled[true_column].ne(labeled[predicted_column])
        ]
        missed = errors.loc[
            errors[true_column].ne(0)
            & errors[predicted_column].eq(0)
        ]
        false_mention = errors.loc[
            errors[true_column].eq(0)
            & errors[predicted_column].ne(0)
        ]
        sentiment = errors.loc[
            errors[true_column].ne(0)
            & errors[predicted_column].ne(0)
        ]
        aspect_metrics = metrics["aspects"][aspect]
        weakest_label = min(
            LABEL_ORDER,
            key=lambda label: (
                aspect_metrics["classes"][str(label)]["f1"],
                label,
            ),
        )
        rows.append(
            {
                "aspect": aspect,
                "evaluated_count": int(len(labeled)),
                "correct_count": int(len(labeled) - len(errors)),
                "error_count": int(len(errors)),
                "error_rate": float(len(errors) / len(labeled)),
                "macro_f1": aspect_metrics["macro_f1"],
                "weighted_f1": aspect_metrics["weighted_f1"],
                "accuracy": aspect_metrics["accuracy"],
                "missed_mention_count": int(len(missed)),
                "false_mention_count": int(len(false_mention)),
                "sentiment_confusion_count": int(len(sentiment)),
                "weakest_label": weakest_label,
                "weakest_label_name": LABEL_NAMES[weakest_label],
                "weakest_label_f1": aspect_metrics["classes"][
                    str(weakest_label)
                ]["f1"],
            }
        )
    return rows


def _confusion_rows(enriched: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for aspect in ASPECTS:
        true_column = f"true_{aspect}"
        predicted_column = f"predicted_{aspect}"
        labeled = enriched.loc[enriched[true_column].notna()].copy()
        labeled[true_column] = labeled[true_column].astype(int)
        aspect_error_count = int(
            labeled[true_column].ne(labeled[predicted_column]).sum()
        )
        true_counts = labeled[true_column].value_counts().to_dict()
        counts = (
            labeled.groupby(
                [true_column, predicted_column],
                sort=True,
            )
            .size()
            .to_dict()
        )
        for true_label in LABEL_ORDER:
            for predicted_label in LABEL_ORDER:
                count = int(
                    counts.get((true_label, predicted_label), 0)
                )
                is_error = true_label != predicted_label
                rows.append(
                    {
                        "aspect": aspect,
                        "true_label": true_label,
                        "true_label_name": LABEL_NAMES[true_label],
                        "predicted_label": predicted_label,
                        "predicted_label_name": LABEL_NAMES[
                            predicted_label
                        ],
                        "is_error": is_error,
                        "count": count,
                        "proportion_of_true_label": (
                            count / true_counts.get(true_label, 1)
                        ),
                        "proportion_of_aspect_errors": (
                            count / aspect_error_count
                            if is_error and aspect_error_count
                            else 0.0
                        ),
                    }
                )
    return rows


def _error_examples(
    enriched: pd.DataFrame,
    *,
    examples_per_confusion: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for aspect in ASPECTS:
        true_column = f"true_{aspect}"
        predicted_column = f"predicted_{aspect}"
        errors = enriched.loc[
            enriched[true_column].notna()
            & enriched[true_column].ne(enriched[predicted_column])
        ]
        for (true_label, predicted_label), group in errors.groupby(
            [true_column, predicted_column],
            sort=True,
        ):
            for _, row in group.sort_values(REVIEW_ID_COLUMN).head(
                examples_per_confusion
            ).iterrows():
                rows.append(
                    {
                        REVIEW_ID_COLUMN: row[REVIEW_ID_COLUMN],
                        GROUP_ID_COLUMN: row[GROUP_ID_COLUMN],
                        "aspect": aspect,
                        "true_label": int(true_label),
                        "true_label_name": LABEL_NAMES[int(true_label)],
                        "predicted_label": int(predicted_label),
                        "predicted_label_name": LABEL_NAMES[
                            int(predicted_label)
                        ],
                        "text_character_count": int(
                            row["text_character_count"]
                        ),
                        "text_token_count": int(
                            row["text_token_count"]
                        ),
                        TEXT_COLUMN: row[TEXT_COLUMN],
                    }
                )
    return pd.DataFrame(rows)


def assert_metrics_match_recorded(
    metrics: dict[str, Any],
    recorded_report: dict[str, Any],
) -> None:
    if recorded_report.get("run_name") != DEFAULT_RUN_NAME:
        raise ValueError(
            "Recorded metrics do not belong to combined_svc."
        )
    if recorded_report.get("selection_split") != "validation":
        raise ValueError("Recorded metrics are not for validation.")
    if recorded_report.get("test_set_accessed") is not False:
        raise ValueError("Recorded report does not prove test isolation.")
    recorded = recorded_report.get("metrics")
    if not isinstance(recorded, dict):
        raise ValueError("Recorded report is missing metrics.")
    summary_keys = (
        "row_count",
        "evaluated_target_count",
        "complete_target_row_count",
        "mean_macro_f1",
        "mean_weighted_f1",
        "exact_match_ratio",
    )
    for key in summary_keys:
        if not np.isclose(
            metrics["summary"][key],
            recorded["summary"][key],
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError(
                f"Recomputed metric {key!r} differs from the saved report."
            )
    for aspect in ASPECTS:
        if metrics["aspects"][aspect]["confusion_matrix"] != (
            recorded["aspects"][aspect]["confusion_matrix"]
        ):
            raise ValueError(
                f"Recomputed confusion matrix differs for {aspect!r}."
            )


def build_prediction_analysis(
    enriched: pd.DataFrame,
    *,
    run_name: str,
    inputs: dict[str, Any],
    recorded_report: dict[str, Any],
    examples_per_confusion: int = 5,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Build summary, aspect, confusion, and deterministic example outputs."""
    if examples_per_confusion <= 0:
        raise ValueError("examples_per_confusion must be positive.")
    true, predicted = _metric_frames(enriched)
    metrics = compute_multioutput_metrics(true, predicted)
    assert_metrics_match_recorded(metrics, recorded_report)
    aspect_errors = pd.DataFrame(
        _aspect_error_rows(enriched, metrics)
    )
    confusions = pd.DataFrame(_confusion_rows(enriched))
    examples = _error_examples(
        enriched,
        examples_per_confusion=examples_per_confusion,
    )
    error_distribution = (
        enriched["error_aspect_count"]
        .value_counts()
        .reindex(range(len(ASPECTS) + 1), fill_value=0)
        .sort_index()
    )
    weakest_aspect_row = aspect_errors.sort_values(
        ["macro_f1", "aspect"],
        ascending=[True, True],
    ).iloc[0]
    error_confusions = confusions.loc[
        confusions["is_error"] & confusions["count"].gt(0)
    ].sort_values(
        ["count", "aspect", "true_label", "predicted_label"],
        ascending=[False, True, True, True],
    )
    top_confusion = error_confusions.iloc[0]
    any_error_count = int(enriched["error_aspect_count"].gt(0).sum())
    complete = enriched.loc[enriched["complete_target"]]
    summary = {
        "schema_version": "v1",
        "run_name": run_name,
        "split": "validation",
        "test_set_accessed": False,
        "metrics_match_recorded_report": True,
        "scope": {
            "model_loaded": False,
            "inference_run": False,
            "decision_scores_available": False,
            "confidence_analysis_performed": False,
            "confidence_analysis_note": (
                "The saved prediction CSV contains labels but no Linear SVM "
                "decision scores. Confidence analysis would require a new "
                "inference export and is intentionally deferred."
            ),
        },
        "inputs": inputs,
        "summary": {
            **metrics["summary"],
            "row_with_any_error_count": any_error_count,
            "row_with_any_error_rate": any_error_count / len(enriched),
            "complete_row_exact_match_count": int(
                complete["exact_match"].sum()
            ),
            "total_aspect_error_count": int(
                aspect_errors["error_count"].sum()
            ),
            "mean_errors_per_row": float(
                enriched["error_aspect_count"].mean()
            ),
            "error_aspect_count_distribution": {
                str(index): int(value)
                for index, value in error_distribution.items()
            },
        },
        "key_findings": {
            "weakest_aspect": str(weakest_aspect_row["aspect"]),
            "weakest_aspect_macro_f1": float(
                weakest_aspect_row["macro_f1"]
            ),
            "weakest_aspect_label": int(
                weakest_aspect_row["weakest_label"]
            ),
            "weakest_aspect_label_name": str(
                weakest_aspect_row["weakest_label_name"]
            ),
            "weakest_aspect_label_f1": float(
                weakest_aspect_row["weakest_label_f1"]
            ),
            "largest_confusion": {
                "aspect": str(top_confusion["aspect"]),
                "true_label": int(top_confusion["true_label"]),
                "true_label_name": str(
                    top_confusion["true_label_name"]
                ),
                "predicted_label": int(
                    top_confusion["predicted_label"]
                ),
                "predicted_label_name": str(
                    top_confusion["predicted_label_name"]
                ),
                "count": int(top_confusion["count"]),
            },
        },
        "metrics": metrics,
    }
    return summary, aspect_errors, confusions, examples


def render_analysis_report(
    summary: dict[str, Any],
    aspect_errors: pd.DataFrame,
    confusions: pd.DataFrame,
) -> str:
    """Render a compact, evidence-based Markdown analysis report."""
    values = summary["summary"]
    findings = summary["key_findings"]
    lines = [
        "# combined_svc Validation Prediction Analysis",
        "",
        "This report analyzes saved validation predictions only. "
        "No model was loaded, no inference was run, and the test split "
        "was not accessed.",
        "",
        "The saved CSV contains predicted labels but no Linear SVM decision "
        "scores, so this stage does not make confidence-based claims.",
        "",
        "## Executive Summary",
        "",
        f"- Validation rows: {values['row_count']:,}.",
        f"- Mean Macro F1: {values['mean_macro_f1']:.6f}.",
        f"- Exact-match ratio on complete targets: "
        f"{values['exact_match_ratio']:.6f}.",
        f"- Rows with at least one aspect error: "
        f"{values['row_with_any_error_count']:,} "
        f"({values['row_with_any_error_rate']:.2%}).",
        f"- Total aspect-level errors: "
        f"{values['total_aspect_error_count']:,}.",
        f"- Weakest aspect: `{findings['weakest_aspect']}` "
        f"(Macro F1 {findings['weakest_aspect_macro_f1']:.6f}).",
        "",
        "## Aspect Results",
        "",
        "| Aspect | Evaluated | Errors | Error rate | Macro F1 | "
        "Missed mention | False mention | Sentiment confusion | "
        "Weakest label |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in aspect_errors.itertuples(index=False):
        lines.append(
            f"| {row.aspect} | {row.evaluated_count:,} | "
            f"{row.error_count:,} | {row.error_rate:.2%} | "
            f"{row.macro_f1:.6f} | {row.missed_mention_count:,} | "
            f"{row.false_mention_count:,} | "
            f"{row.sentiment_confusion_count:,} | "
            f"{row.weakest_label_name} ({row.weakest_label_f1:.4f}) |"
        )

    lines.extend(
        [
            "",
            "## Largest Confusions",
            "",
            "| Aspect | True | Predicted | Count | "
            "Share of true label | Share of aspect errors |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    largest = (
        confusions.loc[
            confusions["is_error"] & confusions["count"].gt(0)
        ]
        .sort_values(
            ["count", "aspect", "true_label", "predicted_label"],
            ascending=[False, True, True, True],
        )
        .head(15)
    )
    for row in largest.itertuples(index=False):
        lines.append(
            f"| {row.aspect} | {row.true_label_name} | "
            f"{row.predicted_label_name} | {row.count:,} | "
            f"{row.proportion_of_true_label:.2%} | "
            f"{row.proportion_of_aspect_errors:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Errors per Review",
            "",
            "| Incorrect aspects | Reviews |",
            "|---:|---:|",
        ]
    )
    for error_count, count in values[
        "error_aspect_count_distribution"
    ].items():
        lines.append(f"| {error_count} | {count:,} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `size` is the first priority for manual review, especially "
            "neutral size targets.",
            "- Missed/false mention errors separate aspect detection problems "
            "from sentiment-polarity problems.",
            "- The deterministic examples CSV contains review text for every "
            "populated off-diagonal confusion cell and is the input to the "
            "next manual taxonomy step.",
            "- These results reproduce the frozen validation metrics exactly.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_validation_predictions(
    *,
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
    recorded_metrics_path: Path = DEFAULT_RECORDED_METRICS_PATH,
    interim_path: Path = DEFAULT_INTERIM_PATH,
    split_ids_path: Path = DEFAULT_SPLIT_IDS_PATH,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
    aspect_errors_path: Path = DEFAULT_ASPECT_ERRORS_PATH,
    confusions_path: Path = DEFAULT_CONFUSIONS_PATH,
    examples_path: Path = DEFAULT_EXAMPLES_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    examples_per_confusion: int = 5,
    force: bool = False,
) -> dict[str, Any]:
    """Analyze saved combined_svc validation predictions and write reports."""
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
    output_paths = [
        Path(summary_path),
        Path(aspect_errors_path),
        Path(confusions_path),
        Path(examples_path),
        Path(report_path),
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Refusing to overwrite validation analysis outputs: "
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
        "recorded_metrics_path": Path(recorded_metrics_path).as_posix(),
        "recorded_metrics_sha256": file_sha256(
            recorded_metrics_path
        ),
        "interim_path": Path(interim_path).as_posix(),
        "interim_sha256": file_sha256(interim_path),
        "split_ids_path": Path(split_ids_path).as_posix(),
        "split_ids_sha256": file_sha256(split_ids_path),
    }
    summary, aspect_errors, confusions, examples = (
        build_prediction_analysis(
            enriched,
            run_name=DEFAULT_RUN_NAME,
            inputs=inputs,
            recorded_report=recorded_report,
            examples_per_confusion=examples_per_confusion,
        )
    )

    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(summary, summary_path)
    aspect_errors.to_csv(
        aspect_errors_path,
        index=False,
        lineterminator="\n",
    )
    confusions.to_csv(
        confusions_path,
        index=False,
        lineterminator="\n",
    )
    examples.to_csv(
        examples_path,
        index=False,
        lineterminator="\n",
    )
    Path(report_path).write_text(
        render_analysis_report(summary, aspect_errors, confusions),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze saved combined_svc validation predictions without "
            "loading or running a model."
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
        "--aspect-errors-path",
        type=Path,
        default=DEFAULT_ASPECT_ERRORS_PATH,
    )
    parser.add_argument(
        "--confusions-path",
        type=Path,
        default=DEFAULT_CONFUSIONS_PATH,
    )
    parser.add_argument(
        "--examples-path",
        type=Path,
        default=DEFAULT_EXAMPLES_PATH,
    )
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--examples-per-confusion",
        type=int,
        default=5,
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = analyze_validation_predictions(
            predictions_path=args.predictions_path,
            recorded_metrics_path=args.recorded_metrics_path,
            interim_path=args.interim_path,
            split_ids_path=args.split_ids_path,
            summary_path=args.summary_path,
            aspect_errors_path=args.aspect_errors_path,
            confusions_path=args.confusions_path,
            examples_path=args.examples_path,
            report_path=args.report_path,
            examples_per_confusion=args.examples_per_confusion,
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
            _console_safe(f"Validation prediction analysis failed: {error}"),
            file=sys.stderr,
        )
        return 2
    values = summary["summary"]
    print(
        _console_safe(
            "Validation prediction analysis complete: "
            f"rows={values['row_count']}, "
            f"rows_with_errors={values['row_with_any_error_count']}, "
            f"mean_macro_f1={values['mean_macro_f1']:.6f}."
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
