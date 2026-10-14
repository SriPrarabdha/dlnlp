import random

import torch

from longdoc.data.batching import ClassificationCollator, PooledMLMBatches, crop_tokens
from longdoc.data.synthetic import FILLER, SyntheticDataset, generate_example


def test_head_tail_crop_is_deterministic_and_keeps_both_ends():
    tokens = list(range(100))
    cropped = crop_tokens(tokens, 16)
    assert cropped == list(range(8)) + list(range(92, 100))


def test_retrieval_positions_and_labels_are_balanced():
    examples = [
        generate_example("retrieval", 256, "test", random.Random(index), position=0.95)
        for index in range(200)
    ]
    assert {example.positions for example in examples} == {(0.95,)}
    assert 0.35 < sum(example.label for example in examples) / len(examples) < 0.65
    filler = set(FILLER)
    assert all(any(word not in filler for word in example.text.split()) for example in examples)


def test_xor_markers_are_far_apart_and_balanced():
    examples = [generate_example("xor", 1_024, "train", random.Random(index)) for index in range(400)]
    assert all(second - first >= 0.5 for first, second in (item.positions for item in examples))
    assert 0.4 < sum(example.label for example in examples) / len(examples) < 0.6


def test_stress_chamber_surface_forms_are_split_disjoint():
    marker_sets = {}
    for split in ("train", "validation", "test"):
        marker_sets[split] = {
            marker
            for index in range(200)
            for marker in generate_example("retrieve", 128, split, random.Random(index)).markers
        }
    assert marker_sets["train"].isdisjoint(marker_sets["validation"])
    assert marker_sets["train"].isdisjoint(marker_sets["test"])
    assert marker_sets["validation"].isdisjoint(marker_sets["test"])


def test_order_rule_and_distractor_grid():
    order_examples = [generate_example("order", 512, "test", random.Random(index)) for index in range(200)]
    assert 0.4 < sum(item.label for item in order_examples) / len(order_examples) < 0.6
    distract = SyntheticDataset("distract", "test", 1_000, [1_024, 2_048, 4_096])
    assert {distract[index]["distractor_count"] for index in range(100)} == {0, 8, 32}


def test_collator_never_pads_past_model_capacity():
    collator = ClassificationCollator(pad_id=0, pad_to_multiple=8, max_length=10)
    rows = [{
        "id": "x", "input_ids": list(range(10)), "labels": 1, "coverage": 1.0,
        "positions": None, "source_length": 10, "challenge": None,
        "distractor_count": 0,
    }]
    batch = collator(rows)
    assert batch["input_ids"].shape == (1, 10)
    assert batch["attention_mask"].dtype == torch.bool


def test_mlm_step_planning_accounts_for_non_divisible_lengths():
    batches = object.__new__(PooledMLMBatches)
    batches.total_tokens = 101
    batches.batch_tokens = 20
    batches.lengths = [6, 7]
    batches.boundaries = [50, 101]
    # Steps process 18, 14, 18, then boundary-clipped 0? The exact simulator
    # must agree with a direct rendering of its own scheduling rules.
    processed = step = boundary = 0
    while processed < batches.total_tokens:
        length = batches.lengths[step % len(batches.lengths)]
        remaining = batches.boundaries[boundary] - processed
        processed += min(max(1, batches.batch_tokens // length) * length, remaining)
        if processed == batches.boundaries[boundary]:
            boundary += 1
        step += 1
    assert batches.planned_steps() == step
    assert batches.planned_steps(max_steps=3) == 3
