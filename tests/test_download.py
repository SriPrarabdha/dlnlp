from longdoc.data.catalog import CATALOG
from longdoc.data.download import _deduplicate_imdb_training, _parquet_data_files


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
