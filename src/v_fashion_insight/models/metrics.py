"""Shared metrics for five-aspect sentiment models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from v_fashion_insight.common.constants import ASPECTS, LABEL_NAMES, VALID_LABELS

LABEL_ORDER = tuple(sorted(VALID_LABELS))


def _as_frame(values: pd.DataFrame | dict[str, Any], name: str) -> pd.DataFrame:
    frame = values.copy() if isinstance(values, pd.DataFrame) else pd.DataFrame(values)
    missing = [aspect for aspect in ASPECTS if aspect not in frame]
    if missing:
        raise ValueError(f"{name} is missing aspect columns: {missing!r}.")
    return frame.loc[:, ASPECTS]


def compute_multioutput_metrics(
    y_true: pd.DataFrame | dict[str, Any],
    y_pred: pd.DataFrame | dict[str, Any],
) -> dict[str, Any]:
    """Compute deterministic metrics while excluding missing target cells."""
    true = _as_frame(y_true, "y_true").reset_index(drop=True)
    pred = _as_frame(y_pred, "y_pred").reset_index(drop=True)
    if len(true) != len(pred):
        raise ValueError("y_true and y_pred must contain the same number of rows.")

    aspect_metrics: dict[str, Any] = {}
    macro_scores: list[float] = []
    weighted_scores: list[float] = []
    total_evaluated = 0
    for aspect in ASPECTS:
        mask = true[aspect].notna()
        evaluated = int(mask.sum())
        missing = int((~mask).sum())
        if evaluated == 0:
            raise ValueError(f"No labeled rows are available for aspect {aspect!r}.")
        actual = true.loc[mask, aspect].astype(int).to_numpy()
        predicted = pred.loc[mask, aspect].astype(int).to_numpy()
        invalid = sorted(set(predicted).difference(LABEL_ORDER))
        if invalid:
            raise ValueError(f"Predictions for {aspect!r} contain invalid labels {invalid!r}.")

        precision, recall, class_f1, support = precision_recall_fscore_support(
            actual,
            predicted,
            labels=LABEL_ORDER,
            zero_division=0,
        )
        macro_f1 = float(f1_score(actual, predicted, labels=LABEL_ORDER, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(actual, predicted, labels=LABEL_ORDER, average="weighted", zero_division=0))
        macro_scores.append(macro_f1)
        weighted_scores.append(weighted_f1)
        total_evaluated += evaluated
        aspect_metrics[aspect] = {
            "evaluated_count": evaluated,
            "missing_target_count": missing,
            "accuracy": float(accuracy_score(actual, predicted)),
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "classes": {
                str(label): {
                    "name": LABEL_NAMES[label],
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(class_f1[index]),
                    "support": int(support[index]),
                }
                for index, label in enumerate(LABEL_ORDER)
            },
            "confusion_matrix": confusion_matrix(
                actual, predicted, labels=LABEL_ORDER
            ).astype(int).tolist(),
        }

    complete_mask = true.notna().all(axis=1)
    complete_count = int(complete_mask.sum())
    exact_match = (
        float(
            np.all(
                true.loc[complete_mask].astype(int).to_numpy()
                == pred.loc[complete_mask].astype(int).to_numpy(),
                axis=1,
            ).mean()
        )
        if complete_count
        else None
    )
    return {
        "schema_version": "v1",
        "aspect_order": list(ASPECTS),
        "label_order": list(LABEL_ORDER),
        "missing_target_policy": "exclude each missing target cell from that aspect's metrics",
        "summary": {
            "row_count": int(len(true)),
            "evaluated_target_count": total_evaluated,
            "complete_target_row_count": complete_count,
            "mean_macro_f1": float(np.mean(macro_scores)),
            "mean_weighted_f1": float(np.mean(weighted_scores)),
            "exact_match_ratio": exact_match,
        },
        "aspects": aspect_metrics,
    }


def metrics_rows(metrics: dict[str, Any], run_name: str, split: str) -> list[dict[str, Any]]:
    """Flatten summary and per-aspect metrics for CSV comparison."""
    rows: list[dict[str, Any]] = []
    for aspect in ASPECTS:
        values = metrics["aspects"][aspect]
        rows.append(
            {
                "run_name": run_name,
                "split": split,
                "aspect": aspect,
                "macro_f1": values["macro_f1"],
                "weighted_f1": values["weighted_f1"],
                "accuracy": values["accuracy"],
                "evaluated_count": values["evaluated_count"],
            }
        )
    return rows
