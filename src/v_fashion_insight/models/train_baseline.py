"""CLI for training and selecting reproducible TF-IDF baselines."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from v_fashion_insight.common.logging import configure_logging
from v_fashion_insight.common.reproducibility import seed_everything
from v_fashion_insight.models.baseline import (
    file_sha256,
    load_modeling_frame,
    save_artifact,
    train_experiment,
    write_json_atomic,
)
from v_fashion_insight.models.config import load_baseline_config
from v_fashion_insight.models.reporting import (
    plot_confusion_matrices,
    render_selection_markdown,
    write_metrics_csv,
)

DEFAULT_CONFIG_PATH = Path("configs/baseline.yaml")


def _console_safe(value: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return value.encode(encoding, errors="backslashreplace").decode(encoding)


def _ensure_available(paths: list[Path], *, force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Refusing to overwrite baseline outputs: "
            + ", ".join(path.as_posix() for path in existing)
            + ". Use --force to replace them."
        )


def _select_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    best = min(
        reports,
        key=lambda report: (
            -report["metrics"]["summary"]["mean_macro_f1"],
            report["training"]["mean_inference_milliseconds_per_review"],
            report["artifact"]["size_bytes"],
            report["run_name"],
        ),
    )
    return {
        "schema_version": "v1",
        "selected_run": best["run_name"],
        "selection_split": "validation",
        "primary_metric": "mean_macro_f1",
        "primary_metric_value": best["metrics"]["summary"]["mean_macro_f1"],
        "test_set_accessed": False,
        "artifact": best["artifact"],
        "selection_rationale": (
            "Selected the highest validation mean Macro F1 across the five aspects; "
            "latency, artifact size, and run name are deterministic tie-breakers."
        ),
        "immutable": True,
    }


def train_baselines(
    config_path: Path = DEFAULT_CONFIG_PATH,
    *,
    run_names: Sequence[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Train declared runs, compare validation metrics, and freeze the best."""
    config_path = Path(config_path)
    config = load_baseline_config(config_path)
    seed_everything(config["seed"])
    experiments = config["experiments"]
    if run_names:
        requested = set(run_names)
        unknown = sorted(requested.difference(item["name"] for item in experiments))
        if unknown:
            raise ValueError(f"Unknown baseline runs: {unknown!r}.")
        experiments = [item for item in experiments if item["name"] in requested]

    model_dir = Path(config["output"]["model_dir"])
    report_dir = Path(config["output"]["report_dir"])
    figure_dir = Path(config["output"]["figure_dir"])
    run_paths = [model_dir / experiment["name"] for experiment in experiments]
    shared_paths = [
        report_dir / "baseline_comparison.csv",
        model_dir / "selected.json",
        model_dir / "README.md",
    ]
    _ensure_available(run_paths + shared_paths, force=force)

    interim_path = Path(config["data"]["interim_path"])
    split_ids_path = Path(config["data"]["split_ids_path"])
    frame = load_modeling_frame(
        interim_path,
        split_ids_path,
        include_splits=("train", "validation"),
    )
    dataset = {
        "interim_path": interim_path.as_posix(),
        "interim_sha256": file_sha256(interim_path),
        "split_ids_path": split_ids_path.as_posix(),
        "split_ids_sha256": file_sha256(split_ids_path),
        "modeling_row_count": int(len(frame)),
        "included_splits": ["train", "validation"],
        "test_rows_loaded_for_modeling": False,
        "split_counts": {str(key): int(value) for key, value in frame["split"].value_counts().sort_index().items()},
    }
    logger = configure_logging()
    reports: list[dict[str, Any]] = []
    for experiment in experiments:
        name = experiment["name"]
        logger.info("Training baseline run %s.", name)
        artifact, report, predictions = train_experiment(
            frame, experiment, seed=config["seed"], dataset=dataset
        )
        run_dir = model_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        artifact_info = save_artifact(artifact, run_dir / "artifact.joblib")
        artifact_info["path"] = (run_dir / "artifact.joblib").as_posix()
        report["artifact"] = artifact_info
        write_json_atomic(report, run_dir / "validation_metrics.json")
        write_json_atomic(
            {
                "schema_version": "v1",
                "run_name": name,
                "artifact": artifact_info,
                "dataset": dataset,
                "experiment": experiment,
                "package_versions": artifact.package_versions,
            },
            run_dir / "metadata.json",
        )
        predictions.to_csv(run_dir / "validation_predictions.csv", index=False, lineterminator="\n")
        plot_confusion_matrices(
            report["metrics"],
            figure_dir / f"{name}_validation_confusion_matrices.png",
            title=f"{name} — validation confusion matrices",
        )
        reports.append(report)
        logger.info(
            "Completed %s: mean_macro_f1=%.6f.",
            name,
            report["metrics"]["summary"]["mean_macro_f1"],
        )

    comparison_path = report_dir / "baseline_comparison.csv"
    write_metrics_csv(reports, comparison_path)
    registry = _select_reports(reports)
    registry["config_path"] = config_path.as_posix()
    registry["config_sha256"] = file_sha256(config_path)
    registry["comparison_path"] = comparison_path.as_posix()
    write_json_atomic(registry, model_dir / "selected.json")
    (model_dir / "README.md").write_text(
        render_selection_markdown(registry, reports), encoding="utf-8"
    )
    return registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train, compare, and freeze TF-IDF baseline experiments without test access."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--run", action="append", dest="run_names", help="Train only this declared run; repeat as needed.")
    parser.add_argument("--force", action="store_true", help="Replace existing baseline outputs.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        registry = train_baselines(
            args.config, run_names=args.run_names, force=args.force
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(_console_safe(f"Baseline training failed: {error}"), file=sys.stderr)
        return 2
    print(
        _console_safe(
            f"Selected baseline {registry['selected_run']} with validation mean Macro F1 "
            f"{registry['primary_metric_value']:.6f}. Test split was not accessed."
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
