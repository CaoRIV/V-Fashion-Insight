"""Training, artifact, prediction, and data helpers for linear baselines."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from v_fashion_insight.common.constants import ASPECTS, LABEL_NAMES, VALID_LABELS
from v_fashion_insight.data.processed_contract import (
    GROUP_ID_COLUMN,
    REVIEW_ID_COLUMN,
    SPLIT_COLUMN,
    TEXT_COLUMN,
)
from v_fashion_insight.models.metrics import compute_multioutput_metrics

ARTIFACT_SCHEMA_VERSION: Final[str] = "v1"
REQUIRED_SPLIT_COLUMNS: Final[tuple[str, ...]] = (
    REVIEW_ID_COLUMN,
    GROUP_ID_COLUMN,
    SPLIT_COLUMN,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(payload: dict[str, Any], destination: Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=f".{destination.name}.",
            suffix=".tmp", delete=False
        ) as handle:
            handle.write(encoded)
            temporary = Path(handle.name)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_modeling_frame(
    interim_path: Path,
    split_ids_path: Path,
    *,
    include_splits: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Load and join immutable review content to frozen split membership."""
    interim_path = Path(interim_path)
    split_ids_path = Path(split_ids_path)
    if not interim_path.exists():
        raise FileNotFoundError(f"Interim dataset does not exist: {interim_path.as_posix()}.")
    if not split_ids_path.exists():
        raise FileNotFoundError(f"Split IDs do not exist: {split_ids_path.as_posix()}.")
    interim = pd.read_csv(interim_path)
    split_ids = pd.read_csv(split_ids_path)
    required_interim = [REVIEW_ID_COLUMN, GROUP_ID_COLUMN, TEXT_COLUMN, *ASPECTS]
    missing_interim = [column for column in required_interim if column not in interim]
    missing_split = [column for column in REQUIRED_SPLIT_COLUMNS if column not in split_ids]
    if missing_interim:
        raise ValueError(f"Interim dataset is missing columns: {missing_interim!r}.")
    if missing_split:
        raise ValueError(f"Split IDs are missing columns: {missing_split!r}.")
    if interim[REVIEW_ID_COLUMN].duplicated().any():
        raise ValueError("Interim review_id values must be unique.")
    if split_ids[REVIEW_ID_COLUMN].duplicated().any():
        raise ValueError("Split review_id values must be unique.")
    invalid_splits = sorted(set(split_ids[SPLIT_COLUMN]).difference({"train", "validation", "test"}))
    if invalid_splits:
        raise ValueError(f"Invalid split values: {invalid_splits!r}.")
    selected_split_ids = split_ids
    if include_splits is not None:
        invalid_requested = sorted(set(include_splits).difference({"train", "validation", "test"}))
        if invalid_requested:
            raise ValueError(f"Invalid requested splits: {invalid_requested!r}.")
        selected_split_ids = split_ids.loc[split_ids[SPLIT_COLUMN].isin(include_splits)]
        if selected_split_ids.empty:
            raise ValueError("Requested modeling splits contain no rows.")
    merged = interim.loc[:, required_interim].merge(
        selected_split_ids.loc[:, REQUIRED_SPLIT_COLUMNS],
        on=[REVIEW_ID_COLUMN, GROUP_ID_COLUMN],
        how="inner",
        validate="one_to_one",
    )
    expected_rows = len(interim) if include_splits is None else len(selected_split_ids)
    if len(merged) != expected_rows:
        raise ValueError("Requested split IDs must have exact one-to-one interim coverage.")
    if merged[TEXT_COLUMN].isna().any() or merged[TEXT_COLUMN].astype("string").str.strip().eq("").any():
        raise ValueError("Modeling text must not be null or empty.")
    for aspect in ASPECTS:
        values = merged[aspect].dropna()
        if not values.mod(1).eq(0).all() or not values.isin(VALID_LABELS).all():
            raise ValueError(f"Invalid labels in aspect {aspect!r}.")
    return merged.sort_values(REVIEW_ID_COLUMN).reset_index(drop=True)


def _vectorizer(options: dict[str, Any], *, analyzer: str) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=tuple(options["ngram_range"]),
        min_df=options["min_df"],
        max_df=options["max_df"],
        max_features=options["max_features"],
        sublinear_tf=options["sublinear_tf"],
        strip_accents=options["strip_accents"],
        dtype=np.float32,
        lowercase=True,
    )


def build_feature_transformer(experiment: dict[str, Any]) -> BaseEstimator:
    """Build the declared TF-IDF feature transformer."""
    analyzer = experiment["analyzer"]
    if analyzer == "word":
        return _vectorizer(experiment["word"], analyzer="word")
    if analyzer in {"char", "char_wb"}:
        return _vectorizer(experiment["char"], analyzer=analyzer)
    if analyzer == "combined":
        return FeatureUnion(
            [
                ("word", _vectorizer(experiment["word"], analyzer="word")),
                ("char", _vectorizer(experiment["char"], analyzer="char_wb")),
            ]
        )
    raise ValueError(f"Unsupported analyzer: {analyzer!r}.")


def build_classifier(experiment: dict[str, Any], *, seed: int) -> BaseEstimator:
    """Build a deterministic linear classifier from normalized config."""
    common = {"C": experiment["C"], "class_weight": experiment["class_weight"]}
    if experiment["classifier"] == "logistic_regression":
        return LogisticRegression(
            **common,
            max_iter=experiment["max_iter"],
            solver="lbfgs",
            random_state=seed,
        )
    if experiment["classifier"] == "linear_svc":
        return LinearSVC(**common, max_iter=experiment["max_iter"], random_state=seed)
    raise ValueError(f"Unsupported classifier: {experiment['classifier']!r}.")


@dataclass
class BaselineArtifact:
    """Versioned inference bundle containing one shared transformer and five heads."""

    schema_version: str
    run_name: str
    transformer: BaseEstimator
    classifiers: dict[str, BaseEstimator]
    experiment: dict[str, Any]
    seed: int
    dataset: dict[str, Any]
    package_versions: dict[str, str]

    def validate(self) -> None:
        if self.schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported artifact schema: {self.schema_version!r}.")
        if tuple(self.classifiers) != ASPECTS:
            raise ValueError(f"Artifact classifiers must preserve aspect order {list(ASPECTS)!r}.")
        if not hasattr(self.transformer, "transform"):
            raise ValueError("Artifact transformer is not fitted.")
        for aspect, classifier in self.classifiers.items():
            classes = {int(value) for value in getattr(classifier, "classes_", [])}
            if not classes or not classes.issubset(VALID_LABELS):
                raise ValueError(f"Classifier {aspect!r} has invalid classes {sorted(classes)!r}.")

    def predict(self, texts: list[str] | tuple[str, ...]) -> pd.DataFrame:
        normalized = validate_prediction_texts(texts)
        features = self.transformer.transform(normalized)
        predictions = {
            aspect: classifier.predict(features).astype(int)
            for aspect, classifier in self.classifiers.items()
        }
        return pd.DataFrame(predictions, columns=ASPECTS)

    def predict_with_scores(self, texts: list[str] | tuple[str, ...]) -> list[dict[str, Any]]:
        normalized = validate_prediction_texts(texts)
        features = self.transformer.transform(normalized)
        predicted = self.predict(normalized)
        rows: list[dict[str, Any]] = []
        for row_index, text in enumerate(normalized):
            aspects: dict[str, Any] = {}
            for aspect in ASPECTS:
                classifier = self.classifiers[aspect]
                label = int(predicted.loc[row_index, aspect])
                entry: dict[str, Any] = {"label": label, "name": LABEL_NAMES[label]}
                if hasattr(classifier, "predict_proba"):
                    probabilities = classifier.predict_proba(features[row_index])[0]
                    entry["probabilities"] = {
                        str(int(cls)): float(score)
                        for cls, score in zip(classifier.classes_, probabilities, strict=True)
                    }
                elif hasattr(classifier, "decision_function"):
                    scores = np.asarray(classifier.decision_function(features[row_index])).reshape(-1)
                    entry["decision_scores"] = {
                        str(int(cls)): float(score)
                        for cls, score in zip(classifier.classes_, scores, strict=True)
                    }
                aspects[aspect] = entry
            rows.append({"text": text, "aspects": aspects})
        return rows


def validate_prediction_texts(texts: list[str] | tuple[str, ...]) -> list[str]:
    if not isinstance(texts, (list, tuple)) or not texts:
        raise ValueError("At least one review text is required.")
    normalized: list[str] = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Review text must be a non-empty string.")
        normalized.append(text.strip())
    return normalized


def save_artifact(artifact: BaselineArtifact, path: Path) -> dict[str, Any]:
    artifact.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        joblib.dump(artifact, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"path": path.as_posix(), "sha256": file_sha256(path), "size_bytes": path.stat().st_size}


def load_artifact(path: Path, *, expected_sha256: str | None = None) -> BaselineArtifact:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Baseline artifact does not exist: {path.as_posix()}.")
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise ValueError("Baseline artifact checksum does not match registry metadata.")
    artifact = joblib.load(path)
    if not isinstance(artifact, BaselineArtifact):
        raise ValueError("File does not contain a BaselineArtifact.")
    artifact.validate()
    return artifact


def _matrix_bytes(matrix: Any) -> int:
    if sparse.issparse(matrix):
        return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
    return int(np.asarray(matrix).nbytes)


def train_experiment(
    frame: pd.DataFrame,
    experiment: dict[str, Any],
    *,
    seed: int,
    dataset: dict[str, Any],
) -> tuple[BaselineArtifact, dict[str, Any], pd.DataFrame]:
    """Fit only on training rows and evaluate only on validation rows."""
    train = frame.loc[frame[SPLIT_COLUMN] == "train"].reset_index(drop=True)
    validation = frame.loc[frame[SPLIT_COLUMN] == "validation"].reset_index(drop=True)
    if train.empty or validation.empty:
        raise ValueError("Both train and validation splits must contain rows.")

    transformer = build_feature_transformer(experiment)
    started = time.perf_counter()
    train_features = transformer.fit_transform(train[TEXT_COLUMN].astype(str).tolist())
    feature_seconds = time.perf_counter() - started
    validation_features = transformer.transform(validation[TEXT_COLUMN].astype(str).tolist())

    classifiers: dict[str, BaseEstimator] = {}
    fit_seconds: dict[str, float] = {}
    label_counts: dict[str, dict[str, int]] = {}
    for aspect in ASPECTS:
        mask = train[aspect].notna().to_numpy()
        labels = train.loc[mask, aspect].astype(int).to_numpy()
        observed = sorted(set(labels))
        if len(observed) < 2:
            raise ValueError(f"Aspect {aspect!r} requires at least two observed training classes.")
        classifier = build_classifier(experiment, seed=seed)
        aspect_started = time.perf_counter()
        classifier.fit(train_features[mask], labels)
        fit_seconds[aspect] = time.perf_counter() - aspect_started
        classifiers[aspect] = classifier
        label_counts[aspect] = {
            str(label): int(np.sum(labels == label)) for label in sorted(VALID_LABELS)
        }

    artifact = BaselineArtifact(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        run_name=experiment["name"],
        transformer=transformer,
        classifiers=classifiers,
        experiment=experiment,
        seed=seed,
        dataset=dataset,
        package_versions={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    )
    artifact.validate()

    inference_started = time.perf_counter()
    predictions = {
        aspect: classifier.predict(validation_features).astype(int)
        for aspect, classifier in classifiers.items()
    }
    inference_seconds = time.perf_counter() - inference_started
    prediction_frame = pd.DataFrame(predictions, columns=ASPECTS)
    metrics = compute_multioutput_metrics(validation.loc[:, ASPECTS], prediction_frame)
    output_predictions = validation.loc[:, [REVIEW_ID_COLUMN, GROUP_ID_COLUMN]].copy()
    for aspect in ASPECTS:
        output_predictions[f"true_{aspect}"] = validation[aspect]
        output_predictions[f"predicted_{aspect}"] = prediction_frame[aspect]

    report = {
        "schema_version": "v1",
        "run_name": experiment["name"],
        "selection_split": "validation",
        "test_set_accessed": False,
        "experiment": experiment,
        "dataset": dataset,
        "training": {
            "train_row_count": int(len(train)),
            "validation_row_count": int(len(validation)),
            "feature_count": int(train_features.shape[1]),
            "train_feature_matrix_bytes": _matrix_bytes(train_features),
            "feature_fit_seconds": feature_seconds,
            "classifier_fit_seconds": fit_seconds,
            "total_fit_seconds": feature_seconds + sum(fit_seconds.values()),
            "validation_inference_seconds": inference_seconds,
            "mean_inference_milliseconds_per_review": 1000 * inference_seconds / len(validation),
            "labeled_train_counts": {aspect: int(train[aspect].notna().sum()) for aspect in ASPECTS},
            "missing_train_counts": {aspect: int(train[aspect].isna().sum()) for aspect in ASPECTS},
            "class_counts": label_counts,
        },
        "metrics": metrics,
    }
    return artifact, report, output_predictions
