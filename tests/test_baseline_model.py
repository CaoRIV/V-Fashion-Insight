import json
from pathlib import Path

import pandas as pd
import pytest

from v_fashion_insight.common.constants import ASPECTS
from v_fashion_insight.models.baseline import (
    load_artifact,
    load_modeling_frame,
    save_artifact,
    train_experiment,
)
from v_fashion_insight.models.predict import predict_reviews
from v_fashion_insight.models.predict import main as predict_main
from v_fashion_insight.models.train_baseline import train_baselines


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    reviews = []
    split_ids = []
    for index in range(32):
        label = index % 4
        split = "train" if index < 24 else "validation"
        review_id = f"r{index:02d}"
        group_id = f"g{index:02d}"
        reviews.append(
            {
                "review_id": review_id,
                "group_id": group_id,
                "text": f"san pham label{label} mau{label} review {index}",
                **{aspect: label for aspect in ASPECTS},
            }
        )
        split_ids.append({"review_id": review_id, "group_id": group_id, "split": split})
    reviews[3]["material"] = None
    return pd.DataFrame(reviews), pd.DataFrame(split_ids)


def _experiment() -> dict:
    return {
        "name": "tiny_word_lr",
        "description": "test",
        "analyzer": "word",
        "classifier": "logistic_regression",
        "class_weight": None,
        "C": 5.0,
        "max_iter": 200,
        "word": {
            "ngram_range": [1, 2],
            "min_df": 1,
            "max_df": 1.0,
            "max_features": 100,
            "sublinear_tf": True,
            "strip_accents": None,
        },
    }


def test_load_modeling_frame_and_artifact_round_trip(tmp_path: Path) -> None:
    reviews, split_ids = _frames()
    interim_path = tmp_path / "reviews.csv"
    split_path = tmp_path / "splits.csv"
    reviews.to_csv(interim_path, index=False)
    split_ids.to_csv(split_path, index=False)
    frame = load_modeling_frame(interim_path, split_path)
    artifact, report, predictions = train_experiment(
        frame, _experiment(), seed=42, dataset={"fingerprint": "test"}
    )
    assert report["test_set_accessed"] is False
    assert report["training"]["missing_train_counts"]["material"] == 1
    assert report["metrics"]["summary"]["mean_macro_f1"] == pytest.approx(1.0)
    assert len(predictions) == 8

    path = tmp_path / "artifact.joblib"
    identity = save_artifact(artifact, path)
    loaded = load_artifact(path, expected_sha256=identity["sha256"])
    texts = ["san pham label2 mau2"]
    assert loaded.predict(texts).equals(artifact.predict(texts))
    with pytest.raises(ValueError, match="checksum"):
        load_artifact(path, expected_sha256="wrong")


def test_train_command_freezes_registry_and_prediction_uses_checksum(tmp_path: Path) -> None:
    reviews, split_ids = _frames()
    interim_path = tmp_path / "reviews.csv"
    split_path = tmp_path / "splits.csv"
    reviews.to_csv(interim_path, index=False)
    split_ids.to_csv(split_path, index=False)
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        "\n".join(
            [
                "schema_version: v1",
                "seed: 42",
                "aspects: [material, design, size, price, service]",
                "data:",
                f"  interim_path: {interim_path.as_posix()}",
                f"  split_ids_path: {split_path.as_posix()}",
                "output:",
                f"  model_dir: {(tmp_path / 'models').as_posix()}",
                f"  report_dir: {(tmp_path / 'reports').as_posix()}",
                f"  figure_dir: {(tmp_path / 'figures').as_posix()}",
                "experiments:",
                "  - name: tiny_word_lr",
                "    analyzer: word",
                "    classifier: logistic_regression",
                "    class_weight: null",
                "    C: 5.0",
                "    max_iter: 200",
                "    word: {ngram_range: [1, 2], min_df: 1, max_df: 1.0, max_features: 100, sublinear_tf: true, strip_accents: null}",
            ]
        ),
        encoding="utf-8",
    )
    registry = train_baselines(config_path)
    assert registry["selected_run"] == "tiny_word_lr"
    assert registry["test_set_accessed"] is False
    registry_path = tmp_path / "models" / "selected.json"
    assert json.loads(registry_path.read_text(encoding="utf-8"))["immutable"] is True
    output = predict_reviews(["san pham label1 mau1"], registry_path=registry_path)
    assert list(output["predictions"][0]["aspects"]) == list(ASPECTS)
    assert output["predictions"][0]["aspects"]["material"]["label"] == 1
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        train_baselines(config_path)


def test_prediction_cli_is_console_safe_for_vietnamese(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    class _Artifact:
        run_name = "stub"

        def predict_with_scores(self, texts):
            return [{"text": texts[0], "aspects": {}}]

    monkeypatch.setattr(
        "v_fashion_insight.models.predict.resolve_selected_artifact",
        lambda path: (tmp_path / "artifact", "hash"),
    )
    monkeypatch.setattr(
        "v_fashion_insight.models.predict.load_artifact",
        lambda path, expected_sha256: _Artifact(),
    )
    assert predict_main(["--text", "Áo đẹp", "--pretty"]) == 0
    assert "predictions" in capsys.readouterr().out
