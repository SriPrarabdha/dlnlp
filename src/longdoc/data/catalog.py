from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    hf_path: str
    hf_config: str | None
    task_type: str
    num_labels: int
    text_field: str = "text"
    label_field: str = "label"
    revision: str | None = None
    parquet_config: str | None = None
    splits: tuple[str, ...] = ("train", "validation", "test")


CATALOG: dict[str, DatasetInfo] = {
    "arxiv": DatasetInfo(
        name="arxiv", hf_path="ccdv/arxiv-classification", hf_config="no_ref",
        task_type="multiclass", num_labels=11,
        revision="fef8d228a72cf418c01eccc4607e6b8580e1ef0e",
    ),
    "scotus": DatasetInfo(
        name="scotus", hf_path="coastalcph/lex_glue", hf_config="scotus",
        task_type="multiclass", num_labels=14,
        revision="823338615adfc9049aadc2841a9b2d2bd531363c",
    ),
    "ecthr_a": DatasetInfo(
        name="ecthr_a", hf_path="coastalcph/lex_glue", hf_config="ecthr_a",
        task_type="multilabel", num_labels=10, label_field="labels",
        revision="823338615adfc9049aadc2841a9b2d2bd531363c",
    ),
    "imdb": DatasetInfo(
        name="imdb", hf_path="stanfordnlp/imdb", hf_config=None,
        task_type="binary", num_labels=2,
        revision="ffbabfbb5831d7fea34dd335eee7f5b28bc80961",
        parquet_config="plain_text",
        splits=("train", "test"),
    ),
}


def get_dataset_info(name: str) -> DatasetInfo:
    if name not in CATALOG:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(CATALOG)}")
    return CATALOG[name]
