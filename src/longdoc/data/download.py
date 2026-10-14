from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from longdoc.training.operations import atomic_write_json

from .catalog import CATALOG, DatasetInfo, get_dataset_info


def _normalise_text(value: str | list[str]) -> str:
    return "\n\n".join(value) if isinstance(value, list) else str(value)


def _parquet_data_files(info: DatasetInfo) -> dict[str, str]:
    """Return immutable Hub paths for a dataset's generated Parquet snapshot.

    Loading the source repository directly would execute its legacy dataset
    script. Hugging Face Datasets 4+ intentionally removed that capability,
    while the Hub's generated Parquet snapshots remain supported.
    """
    if not info.revision:
        raise ValueError(f"{info.name} must pin a Parquet conversion revision")
    config = info.parquet_config or info.hf_config
    if not config:
        raise ValueError(f"{info.name} must specify its Parquet config directory")
    root = f"hf://datasets/{info.hf_path}@{info.revision}/{config}"
    return {split: f"{root}/{split}/*.parquet" for split in info.splits}


def _normalise_dataset(raw: Any, info: DatasetInfo) -> Any:
    from datasets import DatasetDict

    result = DatasetDict()
    for split, dataset in raw.items():
        def convert(row: dict[str, Any], index: int) -> dict[str, Any]:
            output: dict[str, Any] = {
                "id": f"{info.name}:{split}:{index}",
                "text": _normalise_text(row[info.text_field]),
            }
            if info.task_type == "multilabel":
                output["labels"] = list(row[info.label_field])
            else:
                output["label"] = int(row[info.label_field])
            return output

        columns = ["id", "text", "labels" if info.task_type == "multilabel" else "label"]
        result[split] = dataset.map(convert, with_indices=True, remove_columns=dataset.column_names)
        result[split] = result[split].select_columns(columns)
    return result


def _add_imdb_validation(raw: Any, seed: int, validation_size: int) -> Any:
    from datasets import DatasetDict

    if "validation" in raw:
        return raw
    split = raw["train"].train_test_split(
        test_size=validation_size,
        seed=seed,
        stratify_by_column="label",
    )
    return DatasetDict(train=split["train"], validation=split["test"], test=raw["test"])


def _text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _deduplicate_imdb_training(raw: Any) -> tuple[Any, dict[str, int]]:
    """Remove documents that would leak into IMDB validation or test data.

    The published IMDB splits contain repeated reviews both within training
    and across train/test. Keep the official test set intact, and retain only
    the first occurrence of every remaining training document before carving
    validation out of that pool.
    """
    from datasets import DatasetDict

    test_digests = {_text_digest(text) for text in raw["test"]["text"]}
    seen: set[str] = set()
    keep_indices: list[int] = []
    removed_test_overlaps = 0
    removed_train_duplicates = 0
    for index, text in enumerate(raw["train"]["text"]):
        digest = _text_digest(text)
        if digest in test_digests:
            removed_test_overlaps += 1
        elif digest in seen:
            removed_train_duplicates += 1
        else:
            seen.add(digest)
            keep_indices.append(index)
    result = DatasetDict(train=raw["train"].select(keep_indices), test=raw["test"])
    stats = {
        "original_train_rows": len(raw["train"]),
        "removed_train_test_overlaps": removed_test_overlaps,
        "removed_train_duplicates": removed_train_duplicates,
        "deduplicated_train_rows": len(keep_indices),
    }
    return result, stats


def _validate_split_isolation(dataset: Any, name: str) -> dict[str, int]:
    seen: dict[str, str] = {}
    duplicates = 0
    for split, rows in dataset.items():
        for text in rows["text"]:
            digest = _text_digest(text)
            previous = seen.setdefault(digest, split)
            if previous != split:
                duplicates += 1
    if duplicates:
        raise RuntimeError(
            f"{name} has {duplicates} exact documents shared across splits. "
            "Refusing to create a leakage-prone dataset."
        )
    return {split: len(rows) for split, rows in dataset.items()}


def download_one(
    name: str,
    data_dir: Path,
    seed: int = 17,
    imdb_validation_size: int = 2_500,
    force: bool = False,
) -> dict[str, Any]:
    from datasets import load_dataset

    info = get_dataset_info(name)
    destination = data_dir / "datasets" / name
    if destination.exists() and not force:
        manifest = destination / "manifest.json"
        if manifest.exists():
            return json.loads(manifest.read_text())
        raise FileExistsError(f"{destination} exists without a manifest; pass --force to replace it")

    raw = load_dataset(
        "parquet",
        data_files=_parquet_data_files(info),
        cache_dir=str(data_dir / "cache" / "huggingface"),
    )
    preparation: dict[str, Any] = {}
    if name == "imdb":
        raw, preparation = _deduplicate_imdb_training(raw)
        raw = _add_imdb_validation(raw, seed, imdb_validation_size)
    normalised = _normalise_dataset(raw, info)
    split_sizes = _validate_split_isolation(normalised, name)

    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": name,
        "source": info.hf_path,
        "source_config": info.hf_config,
        "source_revision": info.revision,
        "source_format": "parquet",
        "preparation": preparation,
        "task_type": info.task_type,
        "num_labels": info.num_labels,
        "splits": split_sizes,
        "split_fingerprints": {
            split: getattr(rows, "_fingerprint", None) for split, rows in normalised.items()
        },
        "train_only_for_tokenizer_and_mlm": True,
        "seed": seed,
    }
    # Prepare fully in a sibling staging directory so an interrupted --force
    # never destroys the last usable copy before the replacement is ready.
    staging_root = Path(tempfile.mkdtemp(prefix=f".{name}-staging-", dir=destination.parent))
    staged = staging_root / "dataset"
    backup = staging_root / "previous"
    try:
        normalised.save_to_disk(str(staged))
        atomic_write_json(staged / "manifest.json", manifest)
        if destination.exists():
            destination.rename(backup)
        try:
            staged.rename(destination)
        except BaseException:
            if backup.exists() and not destination.exists():
                backup.rename(destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare long-document datasets")
    parser.add_argument("--data-dir", type=Path, required=True, help="Root directory for all data")
    parser.add_argument(
        "--datasets", nargs="+", default=list(CATALOG), choices=sorted(CATALOG),
        help="Datasets to download (default: all)",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--imdb-validation-size", type=int, default=2_500)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-tokenizer", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=32_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.imdb_validation_size <= 0:
        raise ValueError("--imdb-validation-size must be positive")
    if args.vocab_size <= 5:
        raise ValueError("--vocab-size is too small for the tokenizer special tokens")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        print(f"Preparing {name} ...", flush=True)
        download_one(name, args.data_dir, args.seed, args.imdb_validation_size, args.force)
    tokenizer_vocab_size = None
    if not args.skip_tokenizer:
        from .tokenizer import train_tokenizer
        from tokenizers import Tokenizer

        print("Training shared byte-level BPE from training documents only ...", flush=True)
        tokenizer_path = train_tokenizer(
            args.data_dir, args.datasets, args.vocab_size, args.force
        )
        tokenizer_vocab_size = Tokenizer.from_file(str(tokenizer_path)).get_vocab_size()
    else:
        tokenizer_path = args.data_dir / "tokenizer" / "tokenizer.json"
        if tokenizer_path.exists():
            from tokenizers import Tokenizer

            tokenizer_vocab_size = Tokenizer.from_file(str(tokenizer_path)).get_vocab_size()
    # Inventory every prepared corpus, not only the subset touched by this
    # invocation, so incremental preparation cannot erase provenance.
    manifests = []
    for name in CATALOG:
        manifest_path = args.data_dir / "datasets" / name / "manifest.json"
        if manifest_path.exists():
            manifests.append(json.loads(manifest_path.read_text()))
    root_manifest = {
        "datasets": manifests,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "requested_tokenizer_vocab_size": None if args.skip_tokenizer else args.vocab_size,
    }
    atomic_write_json(args.data_dir / "manifest.json", root_manifest)
    print(f"Data is ready under {args.data_dir.resolve()}")


if __name__ == "__main__":
    main()
