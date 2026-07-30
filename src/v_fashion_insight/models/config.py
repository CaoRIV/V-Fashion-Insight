"""Configuration contract for reproducible linear baseline experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import yaml

from v_fashion_insight.common.constants import ASPECTS, DEFAULT_RANDOM_SEED

SUPPORTED_ANALYZERS: Final[frozenset[str]] = frozenset(
    {"word", "char", "char_wb", "combined"}
)
SUPPORTED_CLASSIFIERS: Final[frozenset[str]] = frozenset(
    {"logistic_regression", "linear_svc"}
)
REQUIRED_EXPERIMENT_KEYS: Final[frozenset[str]] = frozenset(
    {"name", "analyzer", "classifier"}
)


def _positive_int(value: Any, name: str, *, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _ngram_range(value: Any, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or value[0] <= 0
        or value[0] > value[1]
    ):
        raise ValueError(f"{name} must contain two positive ordered integers.")
    return int(value[0]), int(value[1])


def _validate_feature_section(section: dict[str, Any], name: str) -> dict[str, Any]:
    ngram_range = _ngram_range(section.get("ngram_range"), f"{name}.ngram_range")
    min_df = section.get("min_df", 1)
    max_df = section.get("max_df", 1.0)
    if not isinstance(min_df, (int, float)) or isinstance(min_df, bool) or min_df <= 0:
        raise ValueError(f"{name}.min_df must be positive.")
    if not isinstance(max_df, (int, float)) or isinstance(max_df, bool) or max_df <= 0:
        raise ValueError(f"{name}.max_df must be positive.")
    max_features = _positive_int(
        section.get("max_features"), f"{name}.max_features", allow_none=True
    )
    return {
        "ngram_range": list(ngram_range),
        "min_df": min_df,
        "max_df": max_df,
        "max_features": max_features,
        "sublinear_tf": bool(section.get("sublinear_tf", True)),
        "strip_accents": section.get("strip_accents"),
    }


def validate_baseline_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the baseline YAML configuration."""
    if not isinstance(raw, dict):
        raise ValueError("Baseline configuration must be a mapping.")
    if raw.get("schema_version") != "v1":
        raise ValueError("schema_version must be 'v1'.")

    data = raw.get("data")
    output = raw.get("output")
    experiments = raw.get("experiments")
    if not isinstance(data, dict):
        raise ValueError("data must be a mapping.")
    if not isinstance(output, dict):
        raise ValueError("output must be a mapping.")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("experiments must be a non-empty list.")

    interim_path = data.get("interim_path")
    split_ids_path = data.get("split_ids_path")
    if not isinstance(interim_path, str) or not interim_path.strip():
        raise ValueError("data.interim_path must be a non-empty path.")
    if not isinstance(split_ids_path, str) or not split_ids_path.strip():
        raise ValueError("data.split_ids_path must be a non-empty path.")
    model_dir = output.get("model_dir")
    report_dir = output.get("report_dir")
    figure_dir = output.get("figure_dir")
    for value, name in (
        (model_dir, "output.model_dir"),
        (report_dir, "output.report_dir"),
        (figure_dir, "output.figure_dir"),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty path.")

    normalized_experiments: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, experiment in enumerate(experiments):
        if not isinstance(experiment, dict):
            raise ValueError(f"experiments[{index}] must be a mapping.")
        missing = sorted(REQUIRED_EXPERIMENT_KEYS.difference(experiment))
        if missing:
            raise ValueError(f"experiments[{index}] is missing {missing!r}.")
        name = experiment["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"experiments[{index}].name must be non-empty.")
        if name in names:
            raise ValueError(f"Duplicate experiment name: {name!r}.")
        names.add(name)

        analyzer = experiment["analyzer"]
        classifier = experiment["classifier"]
        if analyzer not in SUPPORTED_ANALYZERS:
            raise ValueError(f"Unsupported analyzer {analyzer!r} for {name!r}.")
        if classifier not in SUPPORTED_CLASSIFIERS:
            raise ValueError(f"Unsupported classifier {classifier!r} for {name!r}.")
        class_weight = experiment.get("class_weight")
        if class_weight not in (None, "balanced"):
            raise ValueError(f"{name}.class_weight must be null or 'balanced'.")
        c_value = experiment.get("C", 1.0)
        if not isinstance(c_value, (int, float)) or c_value <= 0:
            raise ValueError(f"{name}.C must be positive.")

        normalized: dict[str, Any] = {
            "name": name,
            "description": str(experiment.get("description", "")),
            "analyzer": analyzer,
            "classifier": classifier,
            "class_weight": class_weight,
            "C": float(c_value),
            "max_iter": _positive_int(experiment.get("max_iter", 1000), f"{name}.max_iter"),
        }
        if analyzer in {"word", "combined"}:
            section = experiment.get("word")
            if not isinstance(section, dict):
                raise ValueError(f"{name}.word must be a mapping.")
            normalized["word"] = _validate_feature_section(section, f"{name}.word")
        if analyzer in {"char", "char_wb", "combined"}:
            section = experiment.get("char")
            if not isinstance(section, dict):
                raise ValueError(f"{name}.char must be a mapping.")
            normalized["char"] = _validate_feature_section(section, f"{name}.char")
        normalized_experiments.append(normalized)

    configured_aspects = raw.get("aspects", list(ASPECTS))
    if tuple(configured_aspects) != ASPECTS:
        raise ValueError(f"aspects must preserve the stable order {list(ASPECTS)!r}.")
    seed = raw.get("seed", DEFAULT_RANDOM_SEED)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer.")

    return {
        "schema_version": "v1",
        "seed": seed,
        "aspects": list(ASPECTS),
        "selection_split": "validation",
        "data": {
            "interim_path": interim_path,
            "split_ids_path": split_ids_path,
        },
        "output": {
            "model_dir": model_dir,
            "report_dir": report_dir,
            "figure_dir": figure_dir,
        },
        "experiments": normalized_experiments,
    }


def load_baseline_config(path: Path) -> dict[str, Any]:
    """Read and validate a baseline configuration file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration does not exist: {path.as_posix()}.")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return validate_baseline_config(raw)
