from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal


FILLER = (
    "amber river quiet cedar stone paper orbit lantern meadow copper winter "
    "garden silver canvas harbor gentle valley cloud timber violet ".split()
)
CHALLENGES = ("retrieve", "combine", "order", "distract")


@dataclass(frozen=True)
class SyntheticExample:
    text: str
    label: int
    positions: tuple[float, ...]
    length: int
    markers: tuple[str, ...]
    distractor_count: int = 0


def _bit_templates(split: str) -> tuple[list[str], list[str]]:
    """Exact marker phrases are disjoint across splits; class semantics remain learnable."""
    offsets = {"train": 100, "validation": 200, "test": 300}
    prefix = {"train": "archive flag", "validation": "sealed memorandum", "test": "verdict token"}[split]
    start = offsets[split]
    zero = [f"{prefix} cobalt {start + index}" for index in range(24)]
    one = [f"{prefix} saffron {start + index}" for index in range(24, 48)]
    return zero, one


def _order_templates(split: str) -> tuple[list[str], list[str]]:
    prefix = {"train": "archive", "validation": "sealed", "test": "verdict"}[split]
    first = [f"{prefix} alpha beacon {index}" for index in range(12)]
    second = [f"{prefix} beta beacon {index + 20}" for index in range(12)]
    return first, second


def _distractor_templates(split: str) -> list[str]:
    prefix = {"train": "noise ledger", "validation": "irrelevant docket", "test": "decoy record"}[split]
    colors = ("teal", "ochre", "plum", "ivory", "maroon", "azure", "pearl", "bronze")
    return [f"{prefix} {color} {index}" for index, color in enumerate(colors)]


def _render(length: int, rng: random.Random, markers: list[str], positions: list[float]) -> str:
    words = [rng.choice(FILLER) for _ in range(length)]
    for phrase, relative in zip(markers, positions):
        marker = phrase.split()
        index = min(length - len(marker), max(0, int(relative * length)))
        words[index : index + len(marker)] = marker
    return " ".join(words)


def generate_example(
    task: Literal["retrieve", "retrieval", "combine", "xor", "order", "distract"],
    length: int,
    split: Literal["train", "validation", "test"],
    rng: random.Random,
    position: float | None = None,
    distractor_count: int | None = None,
    position_pair: tuple[float, float] | None = None,
    render_text: bool = True,
) -> SyntheticExample:
    task = {"retrieval": "retrieve", "xor": "combine"}.get(task, task)
    zero, one = _bit_templates(split)

    if task == "retrieve":
        label = rng.randrange(2)
        position = rng.random() if position is None else position
        marker = rng.choice(one if label else zero)
        text = _render(length, rng, [marker], [position]) if render_text else ""
        return SyntheticExample(text, label, (position,), length, (marker,))

    if task == "combine":
        first_bit, second_bit = rng.randrange(2), rng.randrange(2)
        label = first_bit ^ second_bit
        first_position = rng.uniform(0.03, 0.24)
        second_position = rng.uniform(first_position + 0.5, 0.97)
        markers = [rng.choice(one if bit else zero) for bit in (first_bit, second_bit)]
        positions = [first_position, second_position]
        text = _render(length, rng, markers, positions) if render_text else ""
        return SyntheticExample(text, label, tuple(positions), length, tuple(markers))

    if task == "order":
        label = rng.randrange(2)  # one iff alpha occurs before beta
        early, late = position_pair or (rng.uniform(0.05, 0.25), rng.uniform(0.75, 0.95))
        alpha, beta = rng.choice(_order_templates(split)[0]), rng.choice(_order_templates(split)[1])
        markers = [alpha, beta] if label else [beta, alpha]
        positions = [early, late]
        text = _render(length, rng, markers, positions) if render_text else ""
        return SyntheticExample(text, label, tuple(positions), length, tuple(markers))

    if task != "distract":
        raise ValueError(f"Unknown synthetic challenge: {task}")
    label = rng.randrange(2)
    position = rng.random() if position is None else position
    label_marker = rng.choice(one if label else zero)
    count = rng.choice((0, 8, 32)) if distractor_count is None else distractor_count
    distractor_bank = _distractor_templates(split)
    distractors = [f"{rng.choice(distractor_bank)} instance {index}" for index in range(count)]
    distractor_positions = []
    for _ in distractors:
        candidate = rng.uniform(0.02, 0.98)
        while abs(candidate - position) < 0.025 or any(
            abs(candidate - existing) < 0.012 for existing in distractor_positions
        ):
            candidate = rng.uniform(0.02, 0.98)
        distractor_positions.append(candidate)
    markers = [label_marker, *distractors]
    positions = [position, *distractor_positions]
    text = _render(length, rng, markers, positions) if render_text else ""
    return SyntheticExample(text, label, tuple(positions), length, tuple(markers), count)


class SyntheticDataset:
    """Deterministic shared stress chamber at exact target token lengths."""

    def __init__(self, task: str, split: str, size: int, lengths: list[int], seed: int = 17):
        self.task = {"retrieval": "retrieve", "xor": "combine"}.get(task, task)
        if self.task not in CHALLENGES:
            raise ValueError(f"Unknown challenge {task!r}; choose from {CHALLENGES}")
        self.split = split
        self.size = size
        self.lengths = lengths
        self.seed = seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict:
        split_offset = {"train": 0, "validation": 1_000_000, "test": 2_000_000}[self.split]
        rng = random.Random(self.seed + split_offset + index)
        length = self.lengths[index % len(self.lengths)]
        position = None
        distractors = None
        position_pair = None
        if self.split == "test" and self.task in {"retrieve", "distract"}:
            positions = [0.05, 0.25, 0.5, 0.75, 0.95]
            position = positions[(index // len(self.lengths)) % len(positions)]
        if self.split == "test" and self.task == "distract":
            counts = [0, 8, 32]
            distractors = counts[(index // (len(self.lengths) * 5)) % len(counts)]
        if self.split == "test" and self.task == "order":
            pairs = [(0.05, 0.55), (0.15, 0.65), (0.30, 0.80), (0.45, 0.95)]
            position_pair = pairs[(index // len(self.lengths)) % len(pairs)]
        item = generate_example(
            self.task, length, self.split, rng, position, distractors,
            position_pair=position_pair, render_text=False,
        )
        return {
            "id": f"stress:{self.task}:{self.split}:{index}",
            "challenge": self.task,
            "text": item.text,
            "label": item.label,
            "positions": item.positions,
            "source_length": item.length,
            "markers": item.markers,
            "distractor_count": item.distractor_count,
        }
