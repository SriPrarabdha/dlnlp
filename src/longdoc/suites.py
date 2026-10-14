from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

PAPERS = ("longformer", "linformer", "longshort", "compressive", "roformer")
STUDIES = ("head_to_head", "stress_test")
SUITES = (*STUDIES, *PAPERS)
REAL_TASKS = ("arxiv", "scotus", "ecthr_a", "imdb")
STRESS_TASKS = ("stress_retrieve", "stress_combine", "stress_order", "stress_distract")
HEADLINE_CELLS = (
    "exact.baseline",
    "longformer.baseline",
    "linformer.fixed",
    "longshort.baseline",
    "compressive.baseline",
    "roformer.baseline",
)


@dataclass(frozen=True)
class Analysis:
    name: str
    cell: str
    task: str
    kind: str
    value: int = 0
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PaperSuite:
    paper: str
    baseline: str
    cells: tuple[str, ...]
    tasks: dict[str, tuple[str, ...]]
    analyses: tuple[Analysis, ...]
    curve_tasks: tuple[str, ...] = REAL_TASKS
    curve_cells: tuple[str, ...] = ()
    tfidf_tasks: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_suite(paper: str, max_length: int = 4_096) -> PaperSuite:
    common = (*REAL_TASKS, *STRESS_TASKS)
    suites = {
        "head_to_head": PaperSuite(
            paper="head_to_head",
            baseline="exact.baseline",
            cells=(*HEADLINE_CELLS, "exact.truncated512"),
            tasks={
                **{cell: REAL_TASKS for cell in HEADLINE_CELLS},
                "exact.truncated512": REAL_TASKS,
            },
            analyses=tuple(
                Analysis(f"{task}_truncation", cell, task, "truncation")
                for cell in HEADLINE_CELLS
                for task in REAL_TASKS
            ),
            curve_tasks=REAL_TASKS,
            curve_cells=HEADLINE_CELLS,
            tfidf_tasks=REAL_TASKS,
        ),
        "stress_test": PaperSuite(
            paper="stress_test",
            baseline="exact.baseline",
            cells=HEADLINE_CELLS,
            tasks={
                cell: STRESS_TASKS + (
                    ("stress_retrieve_extrapolation",)
                    if cell == "roformer.baseline" and max_length >= 4_096
                    else ()
                )
                for cell in HEADLINE_CELLS
            },
            analyses=tuple(
                Analysis(f"{task}_chunk_shuffle", cell, task, "chunk_shuffle", 512)
                for cell in HEADLINE_CELLS
                for task in ("stress_combine", "stress_order")
            ),
            curve_tasks=(),
        ),
        "longformer": PaperSuite(
            paper="longformer",
            baseline="longformer.baseline",
            cells=(
                "longformer.baseline", "longformer.learned_global", "longformer.random_global",
                "longformer.even_global", "longformer.sentence_global",
            ),
            tasks={
                "longformer.baseline": common,
                "longformer.learned_global": ("arxiv", "stress_retrieve", "stress_distract"),
                "longformer.random_global": ("arxiv", "stress_retrieve", "stress_distract"),
                "longformer.even_global": ("arxiv", "stress_retrieve", "stress_distract"),
                "longformer.sentence_global": ("arxiv", "stress_retrieve", "stress_distract"),
            },
            analyses=(
                Analysis("selected_positions", "longformer.learned_global", "stress_retrieve", "selection_diagnostics", 100),
                Analysis("truncation", "longformer.learned_global", "arxiv", "truncation"),
                Analysis("lowest_scoring", "longformer.learned_global", "stress_retrieve", "ablation", overrides={"model.global_selection": "lowest"}),
                Analysis("random_inference", "longformer.learned_global", "stress_retrieve", "ablation", overrides={"model.global_selection": "random"}),
                Analysis("even_inference", "longformer.learned_global", "stress_retrieve", "ablation", overrides={"model.global_selection": "even"}),
            ),
        ),
        "linformer": PaperSuite(
            paper="linformer",
            baseline="linformer.fixed",
            cells=("linformer.fixed", "linformer.fixed_exact_cls", "linformer.resampled", "linformer.resampled_exact_cls"),
            tasks={
                "linformer.fixed": common,
                "linformer.fixed_exact_cls": ("imdb", "stress_retrieve"),
                "linformer.resampled": ("imdb", "stress_retrieve"),
                "linformer.resampled_exact_cls": ("imdb", "stress_retrieve"),
            },
            analyses=(
                Analysis("prepend_128", "linformer.resampled_exact_cls", "imdb", "prepend", 128),
                Analysis("prepend_512", "linformer.resampled_exact_cls", "imdb", "prepend", 512),
                Analysis("projection_spectrum", "linformer.resampled", "imdb", "projection_spectrum"),
                Analysis("truncation", "linformer.resampled_exact_cls", "imdb", "truncation"),
            ),
        ),
        "longshort": PaperSuite(
            paper="longshort",
            baseline="longshort.baseline",
            cells=("longshort.baseline", "longshort.segment_summaries"),
            tasks={
                "longshort.baseline": common,
                "longshort.segment_summaries": ("arxiv", "stress_combine", "stress_order"),
            },
            analyses=(
                Analysis("truncation", "longshort.segment_summaries", "arxiv", "truncation"),
                Analysis("summaries_disabled", "longshort.segment_summaries", "stress_combine", "ablation", overrides={"model.summary_type": "disabled"}),
                Analysis("summaries_shuffled", "longshort.segment_summaries", "stress_order", "ablation", overrides={"model.summary_ablation": "shuffle"}),
                Analysis("segments_shuffled", "longshort.segment_summaries", "stress_order", "chunk_shuffle", 512),
            ),
        ),
        "compressive": PaperSuite(
            paper="compressive",
            baseline="compressive.baseline",
            cells=(
                "compressive.baseline", "compressive.salient", "compressive.dual_channel",
                "compressive.dual_channel_full_bptt",
            ),
            tasks={
                "compressive.baseline": common,
                "compressive.salient": ("ecthr_a", "stress_retrieve", "stress_distract"),
                "compressive.dual_channel": ("ecthr_a", "stress_retrieve", "stress_distract"),
                "compressive.dual_channel_full_bptt": ("ecthr_a", "stress_retrieve"),
            },
            analyses=tuple(
                Analysis(f"mask_{part}", "compressive.dual_channel", "stress_retrieve", "ablation", overrides={"model.memory_ablation": part})
                for part in ("regular", "compressed", "coverage", "salience", "current")
            ) + (Analysis("truncation", "compressive.dual_channel", "ecthr_a", "truncation"),),
        ),
        "roformer": PaperSuite(
            paper="roformer",
            baseline="roformer.baseline",
            cells=("roformer.baseline", "roformer.frequency_gates"),
            tasks={
                "roformer.baseline": common + (("stress_retrieve_extrapolation",) if max_length >= 4_096 else ()),
                "roformer.frequency_gates": (
                    "scotus", "stress_order",
                    *(("stress_retrieve_extrapolation",) if max_length >= 4_096 else ()),
                ),
            },
            analyses=(
                Analysis("truncation", "roformer.frequency_gates", "scotus", "truncation"),
                Analysis("zero_low_band", "roformer.frequency_gates", "stress_order", "ablation", overrides={"model.rope_ablate_band": 0}),
                Analysis("zero_middle_band", "roformer.frequency_gates", "stress_order", "ablation", overrides={"model.rope_ablate_band": 1}),
                Analysis("zero_high_band", "roformer.frequency_gates", "stress_order", "ablation", overrides={"model.rope_ablate_band": 2}),
            ),
        ),
    }
    try:
        return suites[paper]
    except KeyError as error:
        raise ValueError(f"Unknown suite {paper!r}; choose one of {', '.join(SUITES)}") from error
