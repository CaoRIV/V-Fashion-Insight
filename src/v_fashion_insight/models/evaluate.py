"""CLI for evaluating a frozen baseline on an explicitly selected split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from v_fashion_insight.common.constants import ASPECTS
from v_fashion_insight.data.processed_contract import GROUP_ID_COLUMN, REVIEW_ID_COLUMN, SPLIT_COLUMN, TEXT_COLUMN
from v_fashion_insight.models.baseline import file_sha256, load_artifact, load_modeling_frame, write_json_atomic
from v_fashion_insight.models.metrics import compute_multioutput_metrics
from v_fashion_insight.models.predict import DEFAULT_REGISTRY_PATH, resolve_selected_artifact
from v_fashion_insight.models.reporting import plot_confusion_matrices

DEFAULT_INTERIM_PATH = Path("data/interim/reviews.csv")
DEFAULT_SPLIT_IDS_PATH = Path("data/processed/split_ids.csv")
DEFAULT_REPORT_DIR = Path("reports/metrics/baseline_evaluation")
DEFAULT_FIGURE_DIR = Path("reports/figures/baseline_evaluation")
TEST_UNLOCK_TOKEN = "I_UNDERSTAND_TEST_IS_FINAL"


def evaluate_selected_baseline(
    *,
    split: str,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    interim_path: Path = DEFAULT_INTERIM_PATH,
    split_ids_path: Path = DEFAULT_SPLIT_IDS_PATH,
    report_dir: Path = DEFAULT_REPORT_DIR,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
    test_unlock: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test.")
    if split == "test" and test_unlock != TEST_UNLOCK_TOKEN:
        raise PermissionError(
            "Test evaluation is locked. Pass the exact --unlock-test token only for final evaluation."
        )
    artifact_path, checksum = resolve_selected_artifact(registry_path)
    artifact = load_artifact(artifact_path, expected_sha256=checksum)
    expected_interim = artifact.dataset.get("interim_sha256")
    expected_splits = artifact.dataset.get("split_ids_sha256")
    if expected_interim and file_sha256(interim_path) != expected_interim:
        raise ValueError("Interim dataset checksum differs from the training artifact.")
    if expected_splits and file_sha256(split_ids_path) != expected_splits:
        raise ValueError("Split ID checksum differs from the training artifact.")
    frame = load_modeling_frame(
        interim_path,
        split_ids_path,
        include_splits=(split,),
    )
    selected = frame.reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"Selected split {split!r} is empty.")

    report_path = Path(report_dir) / f"{artifact.run_name}_{split}_metrics.json"
    predictions_path = Path(report_dir) / f"{artifact.run_name}_{split}_predictions.csv"
    figure_path = Path(figure_dir) / f"{artifact.run_name}_{split}_confusion_matrices.png"
    existing = [path for path in (report_path, predictions_path, figure_path) if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Refusing to overwrite evaluation outputs: "
            + ", ".join(path.as_posix() for path in existing)
            + ". Use --force to replace them."
        )
    started = time.perf_counter()
    predicted = artifact.predict(selected[TEXT_COLUMN].astype(str).tolist())
    elapsed = time.perf_counter() - started
    metrics = compute_multioutput_metrics(selected.loc[:, ASPECTS], predicted)
    report = {
        "schema_version": "v1",
        "run_name": artifact.run_name,
        "split": split,
        "test_set_accessed": split == "test",
        "artifact": {"path": artifact_path.as_posix(), "sha256": checksum},
        "timing": {
            "total_inference_seconds": elapsed,
            "mean_inference_milliseconds_per_review": 1000 * elapsed / len(selected),
        },
        "metrics": metrics,
    }
    output = selected.loc[:, [REVIEW_ID_COLUMN, GROUP_ID_COLUMN]].copy()
    for aspect in ASPECTS:
        output[f"true_{aspect}"] = selected[aspect]
        output[f"predicted_{aspect}"] = predicted[aspect]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(predictions_path, index=False, lineterminator="\n")
    write_json_atomic(report, report_path)
    plot_confusion_matrices(metrics, figure_path, title=f"{artifact.run_name} — {split} confusion matrices")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the frozen baseline on train or validation; test is explicitly locked.")
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--interim-path", type=Path, default=DEFAULT_INTERIM_PATH)
    parser.add_argument("--split-ids-path", type=Path, default=DEFAULT_SPLIT_IDS_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--unlock-test", metavar="TOKEN")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = evaluate_selected_baseline(
            split=args.split,
            registry_path=args.registry,
            interim_path=args.interim_path,
            split_ids_path=args.split_ids_path,
            report_dir=args.report_dir,
            figure_dir=args.figure_dir,
            test_unlock=args.unlock_test,
            force=args.force,
        )
    except (FileNotFoundError, FileExistsError, PermissionError, ValueError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 2
    print(
        f"{report['run_name']} {report['split']} mean Macro F1: "
        f"{report['metrics']['summary']['mean_macro_f1']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
