"""CLI for predicting all five aspects with a saved baseline."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from v_fashion_insight.models.baseline import load_artifact

DEFAULT_REGISTRY_PATH = Path("models/baseline/selected.json")


def _console_safe(value: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return value.encode(encoding, errors="backslashreplace").decode(encoding)


def resolve_selected_artifact(registry_path: Path) -> tuple[Path, str]:
    registry_path = Path(registry_path)
    if not registry_path.exists():
        raise FileNotFoundError(f"Selected baseline registry does not exist: {registry_path.as_posix()}.")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != "v1" or not registry.get("immutable"):
        raise ValueError("Selected baseline registry is invalid or not frozen.")
    artifact = registry.get("artifact")
    if not isinstance(artifact, dict) or not artifact.get("path") or not artifact.get("sha256"):
        raise ValueError("Selected baseline registry is missing artifact identity.")
    return Path(artifact["path"]), str(artifact["sha256"])


def predict_reviews(
    texts: list[str],
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any]:
    artifact_path, checksum = resolve_selected_artifact(registry_path)
    artifact = load_artifact(artifact_path, expected_sha256=checksum)
    return {
        "schema_version": "v1",
        "model": {"run_name": artifact.run_name, "artifact_sha256": checksum},
        "predictions": artifact.predict_with_scores(texts),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict five aspect sentiment labels for Vietnamese fashion reviews.")
    parser.add_argument("--text", action="append", required=True, help="Review text; repeat for batch prediction.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = predict_reviews(args.text, registry_path=args.registry)
    except (FileNotFoundError, ValueError) as error:
        print(_console_safe(f"Prediction failed: {error}"), file=sys.stderr)
        return 2
    print(
        _console_safe(
            json.dumps(
                output,
                ensure_ascii=False,
                indent=2 if args.pretty else None,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
