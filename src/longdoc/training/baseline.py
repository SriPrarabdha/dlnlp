from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from longdoc.data.catalog import get_dataset_info

from .metrics import bootstrap_confidence_intervals, compute_metrics
from .operations import atomic_write_json


def run_tfidf_baseline(data_dir: str, task: str, output: Path, seed: int = 17) -> dict:
    from datasets import load_from_disk

    if task.startswith(("synthetic_", "stress_")):
        raise ValueError("TF-IDF baseline is intended for the four real classification tasks")
    info = get_dataset_info(task)
    dataset = load_from_disk(str(Path(data_dir) / "datasets" / task))
    label_key = "labels" if info.task_type == "multilabel" else "label"
    train_labels = dataset["train"][label_key]
    test_labels = dataset["test"][label_key]
    if info.task_type == "multilabel":
        def expand(rows):
            matrix = np.zeros((len(rows), info.num_labels), dtype=np.int64)
            for index, labels in enumerate(rows):
                matrix[index, labels] = 1
            return matrix
        train_labels, test_labels = expand(train_labels), expand(test_labels)
        estimator = OneVsRestClassifier(LogisticRegression(max_iter=200, random_state=seed, n_jobs=1))
    else:
        train_labels, test_labels = np.asarray(train_labels), np.asarray(test_labels)
        estimator = LogisticRegression(max_iter=300, random_state=seed, n_jobs=-1)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=250_000, sublinear_tf=True)),
        ("classifier", estimator),
    ])
    pipeline.fit(dataset["train"]["text"], train_labels)
    if info.task_type == "multilabel":
        probabilities = pipeline.predict_proba(dataset["test"]["text"])
        logits = np.log(np.clip(probabilities, 1e-7, 1 - 1e-7) / np.clip(1 - probabilities, 1e-7, 1))
    else:
        decision = pipeline.decision_function(dataset["test"]["text"])
        logits = np.stack((-decision, decision), axis=-1) if decision.ndim == 1 else decision
    metrics = compute_metrics(logits, test_labels, info.task_type)
    metrics["confidence_95"] = bootstrap_confidence_intervals(logits, test_labels, info.task_type, seed=seed)
    atomic_write_json(output, metrics)
    return metrics
