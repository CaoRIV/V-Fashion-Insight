"""Download a pinned Hugging Face dataset and record reproducibility metadata."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import datasets
from datasets import Dataset, DatasetDict, DownloadMode, load_dataset
from huggingface_hub import HfApi

from v_fashion_insight.common.logging import configure_logging

DEFAULT_DATASET_NAME = "vinhplaykennen/FashionReviews"
DEFAULT_REVISION = "main"
DEFAULT_CACHE_DIR = Path("data/raw/huggingface")
DEFAULT_METADATA_PATH = Path("data/raw/metadata.json")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def resolve_dataset_revision(
    dataset_name: str,
    revision: str,
    *,
    token: bool | str | None = None,
) -> str:
    """Resolve a branch, tag, or commit to an immutable Hub commit SHA."""
    dataset_info = HfApi().dataset_info(
        repo_id=dataset_name,
        revision=revision,
        token=token,
    )
    if not dataset_info.sha:
        raise RuntimeError(
            f"Hugging Face did not return a commit SHA for {dataset_name!r}."
        )
    return dataset_info.sha


def _serialize_split(split: Dataset) -> dict[str, Any]:
    return {
        "num_rows": len(split),
        "column_names": list(split.column_names),
        "features": split.features.to_dict(),
        "fingerprint": split._fingerprint,
    }


def build_metadata(
    dataset: DatasetDict,
    *,
    dataset_name: str,
    config_name: str | None,
    requested_revision: str,
    resolved_revision: str,
    cache_dir: Path,
    downloaded_at: datetime,
) -> dict[str, Any]:
    """Create serializable metadata for a downloaded dataset."""
    return {
        "dataset_name": dataset_name,
        "config_name": config_name,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "downloaded_at_utc": downloaded_at.astimezone(UTC).isoformat(),
        "cache_dir": cache_dir.as_posix(),
        "datasets_version": datasets.__version__,
        "total_rows": sum(len(split) for split in dataset.values()),
        "splits": {
            split_name: _serialize_split(split)
            for split_name, split in sorted(dataset.items())
        },
    }


def write_metadata(metadata: Mapping[str, Any], destination: Path) -> None:
    """Write JSON metadata atomically to avoid leaving a partial file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            json.dump(metadata, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.write("\n")
            temporary_path = Path(temporary_file.name)

        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def download_dataset(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    revision: str = DEFAULT_REVISION,
    config_name: str | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    force_redownload: bool = False,
    token: bool | str | None = None,
) -> dict[str, Any]:
    """Download a dataset at a resolved revision and write its metadata."""
    logger = configure_logging()
    cache_dir = Path(cache_dir)
    metadata_path = Path(metadata_path)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Resolving dataset revision for %s@%s", dataset_name, revision)
    resolved_revision = resolve_dataset_revision(
        dataset_name,
        revision,
        token=token,
    )

    logger.info(
        "Loading dataset %s at revision %s",
        dataset_name,
        resolved_revision,
    )
    loaded_dataset = load_dataset(
        path=dataset_name,
        name=config_name,
        cache_dir=str(cache_dir),
        revision=resolved_revision,
        token=token,
        download_mode=(
            DownloadMode.FORCE_REDOWNLOAD
            if force_redownload
            else DownloadMode.REUSE_DATASET_IF_EXISTS
        ),
    )
    if not isinstance(loaded_dataset, DatasetDict):
        raise TypeError(
            "Expected load_dataset() to return a DatasetDict when no split is "
            f"requested, got {type(loaded_dataset).__name__}."
        )

    metadata = build_metadata(
        loaded_dataset,
        dataset_name=dataset_name,
        config_name=config_name,
        requested_revision=revision,
        resolved_revision=resolved_revision,
        cache_dir=cache_dir,
        downloaded_at=_utc_now(),
    )
    write_metadata(metadata, metadata_path)
    logger.info(
        "Downloaded %d rows across %d split(s); metadata written to %s",
        metadata["total_rows"],
        len(metadata["splits"]),
        metadata_path,
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download a pinned Hugging Face dataset and write reproducibility "
            "metadata."
        )
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_NAME,
        help=f"Hugging Face dataset ID (default: {DEFAULT_DATASET_NAME}).",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help=f"Hub branch, tag, or commit (default: {DEFAULT_REVISION}).",
    )
    parser.add_argument(
        "--config-name",
        help="Optional Hugging Face dataset configuration name.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Dataset cache directory (default: {DEFAULT_CACHE_DIR}).",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help=f"Metadata JSON destination (default: {DEFAULT_METADATA_PATH}).",
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Ignore reusable cached data and download the dataset again.",
    )
    parser.add_argument(
        "--use-auth-token",
        action="store_true",
        help="Use the token saved by Hugging Face CLI.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata = download_dataset(
        dataset_name=args.dataset,
        revision=args.revision,
        config_name=args.config_name,
        cache_dir=args.cache_dir,
        metadata_path=args.metadata_path,
        force_redownload=args.force_redownload,
        token=True if args.use_auth_token else None,
    )
    # ASCII escaping keeps JSON output portable across Windows console code pages.
    print(json.dumps(metadata, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
