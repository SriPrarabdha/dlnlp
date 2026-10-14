from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def predictions_from_logits(logits: np.ndarray, task_type: str) -> np.ndarray:
    if task_type == "multilabel":
        return (logits >= 0.0).astype(np.int64)
    return logits.argmax(axis=-1)


def compute_metrics(logits: np.ndarray, labels: np.ndarray, task_type: str) -> dict[str, float]:
    predictions = predictions_from_logits(logits, task_type)
    if task_type == "multilabel":
        return {
            "micro_f1": float(f1_score(labels, predictions, average="micro", zero_division=0)),
            "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        }
    result = {"accuracy": float(accuracy_score(labels, predictions))}
    if task_type == "multiclass":
        result["macro_f1"] = float(f1_score(
            labels,
            predictions,
            labels=np.arange(logits.shape[-1]),
            average="macro",
            zero_division=0,
        ))
    return result


def bootstrap_confidence_intervals(
    logits: np.ndarray,
    labels: np.ndarray,
    task_type: str,
    samples: int = 1_000,
    seed: int = 17,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    count = len(labels)
    if count == 0:
        raise ValueError("Cannot bootstrap an empty evaluation set")
    if samples <= 0:
        raise ValueError("Bootstrap samples must be positive")
    predictions = predictions_from_logits(logits, task_type)
    collected: dict[str, list[float]] = {}
    for _ in range(samples):
        indices = rng.integers(0, count, size=count)
        sampled_predictions = predictions[indices]
        sampled_labels = labels[indices]
        if task_type == "multilabel":
            true_positive = np.logical_and(
                sampled_predictions == 1, sampled_labels == 1
            ).sum(axis=0)
            predicted_positive = (sampled_predictions == 1).sum(axis=0)
            actual_positive = (sampled_labels == 1).sum(axis=0)
            denominator = predicted_positive + actual_positive
            per_label = np.divide(
                2 * true_positive,
                denominator,
                out=np.zeros_like(denominator, dtype=np.float64),
                where=denominator != 0,
            )
            total_denominator = denominator.sum()
            metrics = {
                "micro_f1": (
                    float(2 * true_positive.sum() / total_denominator)
                    if total_denominator else 0.0
                ),
                "macro_f1": float(per_label.mean()),
            }
        else:
            metrics = {
                "accuracy": float(np.mean(sampled_predictions == sampled_labels))
            }
            if task_type == "multiclass":
                classes = logits.shape[-1]
                correct = sampled_predictions == sampled_labels
                true_positive = np.bincount(
                    sampled_labels[correct], minlength=classes
                )
                predicted_count = np.bincount(sampled_predictions, minlength=classes)
                actual_count = np.bincount(sampled_labels, minlength=classes)
                denominator = predicted_count + actual_count
                per_class = np.divide(
                    2 * true_positive,
                    denominator,
                    out=np.zeros(classes, dtype=np.float64),
                    where=denominator != 0,
                )
                metrics["macro_f1"] = float(per_class.mean())
        for key, value in metrics.items():
            collected.setdefault(key, []).append(value)
    return {
        key: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
        for key, values in collected.items()
    }


def evidence_breakdown(
    logits: np.ndarray,
    labels: np.ndarray,
    positions: list,
    lengths: list[int],
) -> dict[str, float]:
    predictions = logits.argmax(-1)
    cells: dict[str, list[bool]] = {}
    for predicted, label, item_positions, length in zip(predictions, labels, positions, lengths):
        if item_positions is None:
            continue
        if len(item_positions) == 1:
            position_key = f"{float(item_positions[0]):.2f}"
        else:
            position_key = "distant_pair"
        cells.setdefault(f"length_{length}/position_{position_key}", []).append(bool(predicted == label))
    return {key: float(np.mean(values)) for key, values in cells.items()}


def length_bucket_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    lengths: list[int],
    task_type: str,
) -> dict[str, dict[str, float]]:
    boundaries = [(0, 512), (513, 1_024), (1_025, 2_048), (2_049, 4_096), (4_097, 10**12)]
    length_array = np.asarray(lengths)
    result = {}
    for lower, upper in boundaries:
        selected = (length_array >= lower) & (length_array <= upper)
        if selected.any():
            label = f"{lower}-{upper}" if upper < 10**12 else "4097+"
            result[label] = compute_metrics(logits[selected], labels[selected], task_type)
    return result


def stress_breakdown(
    logits: np.ndarray,
    labels: np.ndarray,
    positions: list,
    lengths: list[int],
    challenges: list[str | None],
    distractor_counts: list[int],
) -> dict[str, float]:
    predictions = logits.argmax(-1)
    cells: dict[str, list[bool]] = {}
    for predicted, label, spans, length, challenge, distractors in zip(
        predictions, labels, positions, lengths, challenges, distractor_counts
    ):
        correct = bool(predicted == label)
        base_keys = [f"length_{length}"]
        if spans:
            primary_position = float(spans[0])
            base_keys.append(f"position_{primary_position:.2f}")
            region = ("oldest" if primary_position < 0.25 else "middle" if primary_position < 0.5
                      else "recent" if primary_position < 0.75 else "current")
            base_keys.append(f"region_{region}")
        if challenge == "distract":
            base_keys.append(f"distractors_{distractors}")
        for key in base_keys:
            cells.setdefault(key, []).append(correct)
        if spans:
            cells.setdefault(
                f"length_{length}/position_{float(spans[0]):.2f}", []
            ).append(correct)
            if len(spans) >= 2:
                first_segment = min(int(float(spans[0]) * length) // 512, max(0, length // 512 - 1))
                second_segment = min(int(float(spans[1]) * length) // 512, max(0, length // 512 - 1))
                cells.setdefault(
                    f"length_{length}/segment_pair_{first_segment}-{second_segment}", []
                ).append(correct)
        if challenge == "distract":
            cells.setdefault(f"length_{length}/distractors_{distractors}", []).append(correct)
            if spans:
                cells.setdefault(
                    f"position_{float(spans[0]):.2f}/distractors_{distractors}", []
                ).append(correct)
    return {key: float(np.mean(values)) for key, values in cells.items()}
