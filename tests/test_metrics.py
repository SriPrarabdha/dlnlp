import numpy as np

from longdoc.training.metrics import bootstrap_confidence_intervals


def test_bootstrap_metrics_are_reproducible_and_finite():
    logits = np.array([[3.0, 1.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
    labels = np.array([0, 1, 1, 1])
    first = bootstrap_confidence_intervals(
        logits, labels, "multiclass", samples=40, seed=9
    )
    second = bootstrap_confidence_intervals(
        logits, labels, "multiclass", samples=40, seed=9
    )
    assert first == second
    assert set(first) == {"accuracy", "macro_f1"}
    assert all(np.isfinite(bound) for interval in first.values() for bound in interval)
