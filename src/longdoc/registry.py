from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    architecture: str
    model_overrides: dict[str, Any] = field(default_factory=dict)
    pretraining: str = "fresh"
    reuse_from: str | None = None
    primary_tasks: tuple[str, ...] = ()
    description: str = ""


def _spec(name: str, architecture: str, **kwargs: Any) -> ExperimentSpec:
    return ExperimentSpec(name=name, architecture=architecture, **kwargs)


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "exact.baseline": _spec("exact.baseline", "exact", primary_tasks=("arxiv", "scotus", "ecthr_a", "imdb")),
    "exact.truncated512": _spec("exact.truncated512", "exact", model_overrides={"max_length": 512}),
    "longformer.baseline": _spec("longformer.baseline", "longformer", primary_tasks=("arxiv",)),
    "longformer.learned_global": _spec(
        "longformer.learned_global", "longformer",
        model_overrides={"global_selection": "learned"}, pretraining="reuse",
        reuse_from="longformer.baseline", primary_tasks=("arxiv", "stress_retrieve", "stress_distract"),
        description="Straight-through previous-layer top-k global-token selection.",
    ),
    "longformer.random_global": _spec(
        "longformer.random_global", "longformer", model_overrides={"global_selection": "random"},
        pretraining="reuse", reuse_from="longformer.baseline",
    ),
    "longformer.even_global": _spec(
        "longformer.even_global", "longformer", model_overrides={"global_selection": "even"},
        pretraining="reuse", reuse_from="longformer.baseline",
    ),
    "longformer.sentence_global": _spec(
        "longformer.sentence_global", "longformer", model_overrides={"global_selection": "sentence"},
        pretraining="reuse", reuse_from="longformer.baseline",
    ),
    "linformer.fixed": _spec("linformer.fixed", "linformer", model_overrides={"projection_type": "fixed"}),
    "linformer.fixed_exact_cls": _spec(
        "linformer.fixed_exact_cls", "linformer",
        model_overrides={"projection_type": "fixed", "exact_cls": True},
        pretraining="reuse", reuse_from="linformer.fixed",
    ),
    "linformer.resampled": _spec(
        "linformer.resampled", "linformer", model_overrides={"projection_type": "resampled"},
        primary_tasks=("imdb", "stress_retrieve", "stress_combine"),
    ),
    "linformer.resampled_exact_cls": _spec(
        "linformer.resampled_exact_cls", "linformer",
        model_overrides={"projection_type": "resampled", "exact_cls": True},
        pretraining="reuse", reuse_from="linformer.resampled",
    ),
    "longshort.baseline": _spec(
        "longshort.baseline", "longshort", model_overrides={"projection_rank": 16}
    ),
    "longshort.segment_summaries": _spec(
        "longshort.segment_summaries", "longshort",
        model_overrides={"summary_type": "segmented", "summaries_per_segment": 2, "projection_rank": 16},
        primary_tasks=("arxiv", "stress_combine", "stress_order"),
    ),
    "compressive.baseline": _spec("compressive.baseline", "compressive"),
    "compressive.salient": _spec(
        "compressive.salient", "compressive", model_overrides={"compressor": "salient"},
    ),
    "compressive.dual_channel": _spec(
        "compressive.dual_channel", "compressive", model_overrides={"compressor": "dual"},
        primary_tasks=("ecthr_a", "stress_retrieve"),
    ),
    "compressive.dual_channel_full_bptt": _spec(
        "compressive.dual_channel_full_bptt", "compressive",
        model_overrides={"compressor": "dual", "detach_memory": False},
        primary_tasks=("ecthr_a", "stress_retrieve"),
    ),
    "roformer.baseline": _spec(
        "roformer.baseline", "roformer", model_overrides={"position_encoding": "rope"},
        primary_tasks=("scotus",),
    ),
    "roformer.frequency_gates": _spec(
        "roformer.frequency_gates", "roformer",
        model_overrides={"position_encoding": "rope", "rope_gated_bands": True},
        pretraining="reuse", reuse_from="roformer.baseline",
        primary_tasks=("scotus", "stress_retrieve_extrapolation", "stress_order"),
    ),
}


def get_experiment(name: str) -> ExperimentSpec:
    try:
        return EXPERIMENTS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(EXPERIMENTS))
        raise ValueError(f"Unknown experiment {name!r}. Available: {choices}") from exc


def apply_experiment(config: Any, name: str) -> ExperimentSpec:
    spec = get_experiment(name)
    config.experiment = name
    config.model.architecture = spec.architecture
    for key, value in spec.model_overrides.items():
        setattr(config.model, key, value)
    config.train.lengths = [length for length in config.train.lengths if length <= config.model.max_length]
    config.validate()
    return spec
