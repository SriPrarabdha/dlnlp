from longdoc.data.catalog import CATALOG
from longdoc.data.download import (
    _deduplicate_imdb_training,
    _parquet_data_files,
    _remove_cross_split_overlaps,
    _validate_split_isolation,
)


def test_every_dataset_uses_an_immutable_parquet_snapshot():
    for info in CATALOG.values():
        files = _parquet_data_files(info)
        assert set(files) == set(info.splits)
        assert all(path.startswith(f"hf://datasets/{info.hf_path}@") for path in files.values())
        assert all(info.revision in path for path in files.values())
        assert all(path.endswith("/*.parquet") for path in files.values())


def test_imdb_does_not_download_the_unlabelled_split():
    assert set(_parquet_data_files(CATALOG["imdb"])) == {"train", "test"}


def test_imdb_training_is_deduplicated_without_changing_test():
    from datasets import Dataset, DatasetDict

    raw = DatasetDict({
        "train": Dataset.from_dict({
            "text": ["train", "duplicate", "duplicate", "overlap"],
            "label": [0, 1, 1, 0],
        }),
        "test": Dataset.from_dict({"text": ["overlap", "test"], "label": [0, 1]}),
    })

    result, stats = _deduplicate_imdb_training(raw)

    assert result["train"]["text"] == ["train", "duplicate"]
    assert result["test"]["text"] == raw["test"]["text"]
    assert stats == {
        "original_train_rows": 4,
        "removed_train_test_overlaps": 1,
        "removed_train_duplicates": 1,
        "deduplicated_train_rows": 2,
    }


def test_cross_split_cleanup_preserves_evaluation_priority_and_in_split_rows():
    from datasets import Dataset, DatasetDict

    raw = DatasetDict({
        "train": Dataset.from_dict({
            "text": ["validation overlap", "train duplicate", "train duplicate", "train"],
            "label": [0, 0, 1, 0],
        }),
        "validation": Dataset.from_dict({
            "text": ["test overlap", "validation overlap", "validation"],
            "label": [0, 1, 0],
        }),
        "test": Dataset.from_dict({"text": ["test overlap", "test"], "label": [1, 0]}),
    })

    result, stats = _remove_cross_split_overlaps(raw)

    assert result["test"]["text"] == ["test overlap", "test"]
    assert result["validation"]["text"] == ["validation overlap", "validation"]
    assert result["train"]["text"] == ["train duplicate", "train duplicate", "train"]
    assert stats == {
        "priority": ["test", "validation", "train"],
        "removed_by_split": {"test": 0, "validation": 1, "train": 1},
        "total_removed": 2,
    }
    assert _validate_split_isolation(result, "fixture") == {
        "train": 3,
        "validation": 2,
        "test": 2,
    }


def test_cross_split_cleanup_supports_paragraph_lists():
    from datasets import Dataset, DatasetDict

    raw = DatasetDict({
        "train": Dataset.from_dict({"text": [["same", "body"], ["train"]], "labels": [[0], [1]]}),
        "test": Dataset.from_dict({"text": [["same", "body"], ["test"]], "labels": [[0], [1]]}),
    })

    result, stats = _remove_cross_split_overlaps(raw)

    assert result["train"]["text"] == [["train"]]
    assert stats["removed_by_split"] == {"test": 0, "train": 1}
