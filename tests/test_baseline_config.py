from pathlib import Path

import pytest

from v_fashion_insight.models.config import load_baseline_config, validate_baseline_config


def _config() -> dict:
    return {
        "schema_version": "v1",
        "seed": 42,
        "aspects": ["material", "design", "size", "price", "service"],
        "data": {"interim_path": "interim.csv", "split_ids_path": "splits.csv"},
        "output": {"model_dir": "models", "report_dir": "reports", "figure_dir": "figures"},
        "experiments": [
            {
                "name": "word_lr",
                "analyzer": "word",
                "classifier": "logistic_regression",
                "C": 1.0,
                "max_iter": 100,
                "class_weight": None,
                "word": {
                    "ngram_range": [1, 2],
                    "min_df": 1,
                    "max_df": 1.0,
                    "max_features": 100,
                    "sublinear_tf": True,
                    "strip_accents": None,
                },
            }
        ],
    }


def test_validate_baseline_config_normalizes_contract() -> None:
    config = validate_baseline_config(_config())
    assert config["selection_split"] == "validation"
    assert config["experiments"][0]["word"]["ngram_range"] == [1, 2]
    assert config["experiments"][0]["class_weight"] is None


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(experiments=[]), "non-empty"),
        (lambda value: value["experiments"][0].update(analyzer="semantic"), "Unsupported analyzer"),
        (lambda value: value["experiments"][0].update(class_weight="auto"), "class_weight"),
        (lambda value: value.update(aspects=list(reversed(value["aspects"]))), "stable order"),
    ],
)
def test_invalid_baseline_config_fails_clearly(mutation, message: str) -> None:
    raw = _config()
    mutation(raw)
    with pytest.raises(ValueError, match=message):
        validate_baseline_config(raw)


def test_project_baseline_config_is_valid() -> None:
    config = load_baseline_config(Path("configs/baseline.yaml"))
    assert [item["name"] for item in config["experiments"]] == [
        "word_logreg",
        "char_svc",
        "combined_logreg",
        "combined_svc",
        "combined_svc_balanced",
    ]
