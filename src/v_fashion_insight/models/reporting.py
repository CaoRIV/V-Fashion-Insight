"""Report and visualization helpers for baseline experiments."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "v-fashion-insight-matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from v_fashion_insight.common.constants import ASPECTS, LABEL_NAMES


def write_metrics_csv(reports: list[dict[str, Any]], path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for report in reports:
        summary = report["metrics"]["summary"]
        common = {
            "run_name": report["run_name"],
            "analyzer": report["experiment"]["analyzer"],
            "classifier": report["experiment"]["classifier"],
            "class_weight": report["experiment"]["class_weight"],
            "mean_macro_f1": summary["mean_macro_f1"],
            "mean_weighted_f1": summary["mean_weighted_f1"],
            "exact_match_ratio": summary["exact_match_ratio"],
            "feature_count": report["training"]["feature_count"],
            "fit_seconds": report["training"]["total_fit_seconds"],
            "inference_ms_per_review": report["training"]["mean_inference_milliseconds_per_review"],
            "artifact_size_bytes": report["artifact"]["size_bytes"],
        }
        for aspect in ASPECTS:
            common[f"{aspect}_macro_f1"] = report["metrics"]["aspects"][aspect]["macro_f1"]
        rows.append(common)
    table = pd.DataFrame(rows).sort_values(
        ["mean_macro_f1", "run_name"], ascending=[False, True]
    ).reset_index(drop=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, lineterminator="\n")
    return table


def plot_confusion_matrices(metrics: dict[str, Any], path: Path, *, title: str) -> None:
    labels = [LABEL_NAMES[index] for index in metrics["label_order"]]
    figure, axes = plt.subplots(2, 3, figsize=(18, 11))
    for axis, aspect in zip(axes.flat, ASPECTS, strict=False):
        matrix = metrics["aspects"][aspect]["confusion_matrix"]
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=labels,
            yticklabels=labels,
            ax=axis,
        )
        axis.set_title(aspect)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
    axes.flat[-1].axis("off")
    figure.suptitle(title, fontsize=16)
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def render_selection_markdown(registry: dict[str, Any], reports: list[dict[str, Any]]) -> str:
    selected = registry["selected_run"]
    lines = [
        "# Frozen TF-IDF Baseline",
        "",
        f"Selected run: `{selected}`.",
        "",
        "Selection used validation mean Macro F1 only. The test split was not accessed.",
        "",
        "| Run | Features | Classifier | Class weight | Mean Macro F1 | Exact match | Latency ms/review | Size MB |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    ordered = sorted(
        reports,
        key=lambda report: (-report["metrics"]["summary"]["mean_macro_f1"], report["run_name"]),
    )
    for report in ordered:
        summary = report["metrics"]["summary"]
        training = report["training"]
        experiment = report["experiment"]
        exact = summary["exact_match_ratio"]
        lines.append(
            "| {name} | {analyzer} | {classifier} | {weight} | {macro:.6f} | {exact:.6f} | {latency:.4f} | {size:.2f} |".format(
                name=report["run_name"],
                analyzer=experiment["analyzer"],
                classifier=experiment["classifier"],
                weight=experiment["class_weight"] or "none",
                macro=summary["mean_macro_f1"],
                exact=exact if exact is not None else float("nan"),
                latency=training["mean_inference_milliseconds_per_review"],
                size=report["artifact"]["size_bytes"] / (1024 * 1024),
            )
        )
    lines.extend(
        [
            "",
            "## Selection rationale",
            "",
            registry["selection_rationale"],
            "",
            "The registry stores the artifact checksum so later loading can verify immutability.",
            "",
        ]
    )
    return "\n".join(lines)
