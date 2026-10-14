from __future__ import annotations

import math
import queue
import random
import threading
from pathlib import Path
from typing import Any, Iterator

import torch

from .catalog import CATALOG, get_dataset_info
from .synthetic import FILLER, SyntheticDataset
from .tokenizer import load_tokenizer


def _special_ids(tokenizer: Any) -> dict[str, int]:
    result = {name: tokenizer.token_to_id(token) for name, token in {
        "pad": "<pad>", "cls": "<s>", "sep": "</s>", "unk": "<unk>", "mask": "<mask>"
    }.items()}
    missing = [name for name, value in result.items() if value is None]
    if missing:
        raise ValueError(f"Tokenizer is missing required special tokens: {missing}")
    return result


def crop_tokens(ids: list[int], max_length: int, policy: str = "head_tail") -> list[int]:
    """One truncation policy shared by all architectures and real tasks."""
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    if len(ids) <= max_length:
        return ids
    if policy == "head":
        return ids[:max_length]
    if policy != "head_tail":
        raise ValueError(f"Unknown truncation policy: {policy}")
    left = (max_length + 1) // 2
    return ids[:left] + ids[-(max_length - left) :]


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        task: str,
        split: str,
        max_length: int,
        synthetic_size: int = 20_000,
        synthetic_lengths: list[int] | None = None,
        seed: int = 17,
    ):
        self.data_dir = Path(data_dir)
        self.task = task
        self.split = split
        self.max_length = max_length
        self.tokenizer = load_tokenizer(data_dir)
        self.ids = _special_ids(self.tokenizer)
        self.filler_ids = self.tokenizer.encode(
            " " + " ".join(FILLER), add_special_tokens=False
        ).ids
        if not self.filler_ids:
            raise ValueError("Tokenizer produced no tokens for the synthetic filler vocabulary")
        if task.startswith(("synthetic_", "stress_")):
            kind = task.split("_", 1)[1]
            kind = {"retrieve_extrapolation": "retrieve"}.get(kind, kind)
            size = synthetic_size if split == "train" else max(1_000, synthetic_size // 10)
            self.data = SyntheticDataset(kind, split, size, synthetic_lengths or [1_024, 2_048, 4_096], seed)
            self.task_type = "binary"
            self.num_labels = 2
        else:
            from datasets import load_from_disk

            info = get_dataset_info(task)
            self.data = load_from_disk(str(self.data_dir / "datasets" / task))[split]
            self.task_type = info.task_type
            self.num_labels = info.num_labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.data[index]
        if self.task.startswith(("synthetic_", "stress_")):
            target = min(int(row["source_length"]), self.max_length) - 2
            # Build at token level so 1K/2K/4K means model tokens, not an
            # unpredictable number of BPE pieces, while retaining text templates.
            repeats = (target + len(self.filler_ids) - 1) // len(self.filler_ids)
            body = (self.filler_ids * repeats)[:target]
            for phrase, position in zip(row["markers"], row["positions"]):
                marker_ids = self.tokenizer.encode(" " + phrase, add_special_tokens=False).ids
                if len(marker_ids) > target:
                    raise ValueError(
                        f"Synthetic marker needs {len(marker_ids)} tokens but only {target} are available"
                    )
                marker_index = min(target - len(marker_ids), max(0, int(position * target)))
                body[marker_index : marker_index + len(marker_ids)] = marker_ids
            input_ids = [self.ids["cls"], *body, self.ids["sep"]]
            original_length = int(row["source_length"])
        else:
            encoded = self.tokenizer.encode(row["text"], add_special_tokens=True)
            original_length = len(encoded.ids)
            input_ids = crop_tokens(encoded.ids, self.max_length)
        result: dict[str, Any] = {
            "id": row["id"],
            "input_ids": input_ids,
            "original_length": original_length,
            "coverage": min(1.0, self.max_length / max(1, original_length)),
            "source_length": int(row.get("source_length", original_length)),
        }
        if self.task_type == "multilabel":
            target = torch.zeros(self.num_labels, dtype=torch.float32)
            target[row["labels"]] = 1.0
            result["labels"] = target
        else:
            result["labels"] = int(row["label"])
        if "positions" in row:
            result["positions"] = row["positions"]
        result["challenge"] = row.get("challenge")
        result["distractor_count"] = int(row.get("distractor_count", 0))
        return result


class ClassificationCollator:
    def __init__(self, pad_id: int, pad_to_multiple: int = 8, max_length: int | None = None):
        self.pad_id = pad_id
        self.pad_to_multiple = pad_to_multiple
        self.max_length = max_length

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        max_length = max(len(row["input_ids"]) for row in rows)
        padded = int(math.ceil(max_length / self.pad_to_multiple) * self.pad_to_multiple)
        max_length = min(padded, self.max_length) if self.max_length is not None else padded
        input_ids = torch.full((len(rows), max_length), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), max_length), dtype=torch.bool)
        for index, row in enumerate(rows):
            length = len(row["input_ids"])
            input_ids[index, :length] = torch.tensor(row["input_ids"], dtype=torch.long)
            attention_mask[index, :length] = True
        labels = torch.stack([row["labels"] for row in rows]) if torch.is_tensor(rows[0]["labels"]) else torch.tensor(
            [row["labels"] for row in rows], dtype=torch.long
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "ids": [row["id"] for row in rows],
            "coverage": torch.tensor([row["coverage"] for row in rows]),
            "positions": [row.get("positions") for row in rows],
            "source_lengths": [row["source_length"] for row in rows],
            "challenges": [row["challenge"] for row in rows],
            "distractor_counts": [row["distractor_count"] for row in rows],
        }


class SyntheticLengthBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Keep 1K/2K/4K examples separate so padding does not waste accelerator time."""

    def __init__(self, dataset: ClassificationDataset, batch_tokens: int, shuffle: bool, seed: int):
        self.dataset = dataset
        self.batch_tokens = batch_tokens
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.start_batch = 0

    def _batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        lengths = self.dataset.data.lengths
        groups = [[index for index in range(len(self.dataset)) if index % len(lengths) == offset]
                  for offset in range(len(lengths))]
        batches = []
        for length, indices in zip(lengths, groups):
            if self.shuffle:
                rng.shuffle(indices)
            batch_size = max(1, self.batch_tokens // min(length, self.dataset.max_length))
            batches.extend(indices[start : start + batch_size] for start in range(0, len(indices), batch_size))
        if self.shuffle:
            rng.shuffle(batches)
        return batches[self.start_batch :]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_start_batch(self, start_batch: int) -> None:
        self.start_batch = max(0, start_batch)

    def __iter__(self):
        yield from self._batches()

    def __len__(self) -> int:
        total = 0
        count = len(self.dataset.data.lengths)
        for offset, length in enumerate(self.dataset.data.lengths):
            group_size = max(0, (len(self.dataset) - offset + count - 1) // count)
            total += math.ceil(group_size / max(1, self.batch_tokens // min(length, self.dataset.max_length)))
        return total


class _CorpusPacker:
    def __init__(self, texts: list[str], tokenizer: Any, seed: int, sep_id: int):
        if len(texts) == 0:
            raise ValueError("Cannot build an MLM token packer from an empty corpus")
        self.texts = texts
        self.tokenizer = tokenizer
        self.rng = random.Random(seed)
        self.order = list(range(len(texts)))
        self.rng.shuffle(self.order)
        self.cursor = 0
        self.buffer: list[int] = []
        self.sep_id = sep_id

    def _refill(self) -> None:
        documents = []
        for _ in range(min(32, len(self.order))):
            if self.cursor == len(self.order):
                self.rng.shuffle(self.order)
                self.cursor = 0
            documents.append(self.texts[self.order[self.cursor]])
            self.cursor += 1
        for encoded in self.tokenizer.encode_batch(documents, add_special_tokens=False):
            self.buffer.extend(encoded.ids)
            self.buffer.append(self.sep_id)

    def take(self, length: int) -> list[int]:
        while len(self.buffer) < length:
            self._refill()
        result, self.buffer = self.buffer[:length], self.buffer[length:]
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "cursor": self.cursor,
            "buffer": self.buffer,
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.order = list(state["order"])
        self.cursor = int(state["cursor"])
        self.buffer = list(state["buffer"])
        self.rng.setstate(state["rng_state"])


class PooledMLMBatches:
    """Balanced, fully-packed MLM batches with equal token budgets per length.

    Every consecutive group of batches visits each requested length once. Within a
    batch, examples cycle across corpora, preventing the large arXiv set from
    dominating. Sequences contain no padding before the optional final partial step.
    """

    def __init__(
        self,
        data_dir: str | Path,
        lengths: list[int],
        batch_tokens: int,
        total_tokens: int,
        seed: int,
        mask_probability: float = 0.15,
        checkpoint_tokens: list[int] | None = None,
    ):
        from datasets import load_from_disk

        self.tokenizer = load_tokenizer(data_dir)
        self.special = _special_ids(self.tokenizer)
        self.lengths = lengths
        self.batch_tokens = batch_tokens
        self.total_tokens = total_tokens
        self.mask_probability = mask_probability
        self.boundaries = sorted({
            token for token in (checkpoint_tokens or []) if 0 < token < total_tokens
        } | {total_tokens})
        self.generator = torch.Generator().manual_seed(seed)
        self.packers = []
        for offset, name in enumerate(CATALOG):
            path = Path(data_dir) / "datasets" / name
            if path.exists():
                texts = load_from_disk(str(path))["train"]["text"]
                self.packers.append(_CorpusPacker(texts, self.tokenizer, seed + offset, self.special["sep"]))
        if not self.packers:
            raise FileNotFoundError("No prepared datasets found; run download_data.py")
        self.processed = 0
        self.step = 0
        self.corpus_cursor = 0
        self.boundary_index = 0

    def planned_steps(self, max_steps: int | None = None) -> int:
        """Exact number of iterator steps from the initial state.

        This mirrors the variable-length cycling and milestone clipping used by
        ``__iter__``; a simple ``ceil(tokens / batch_tokens)`` is wrong when a
        configured length does not divide the logical token batch.
        """
        processed = 0
        step = 0
        boundary_index = 0
        while processed < self.total_tokens and (max_steps is None or step < max_steps):
            length = self.lengths[step % len(self.lengths)]
            remaining = self.boundaries[boundary_index] - processed
            rows = max(1, self.batch_tokens // length)
            processed += min(rows * length, remaining)
            if processed == self.boundaries[boundary_index]:
                boundary_index += 1
            step += 1
        return step

    def state_dict(self) -> dict[str, Any]:
        return {
            "processed": self.processed,
            "step": self.step,
            "corpus_cursor": self.corpus_cursor,
            "boundary_index": self.boundary_index,
            "generator_state": self.generator.get_state(),
            "packers": [packer.state_dict() for packer in self.packers],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.processed = int(state["processed"])
        self.step = int(state["step"])
        self.corpus_cursor = int(state["corpus_cursor"])
        self.boundary_index = int(state["boundary_index"])
        self.generator.set_state(state["generator_state"])
        if len(state["packers"]) != len(self.packers):
            raise ValueError("Prepared corpus set changed since this checkpoint was saved")
        for packer, saved in zip(self.packers, state["packers"]):
            packer.load_state_dict(saved)

    def _mask(self, clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        labels = clean.clone()
        probability = torch.rand(clean.shape, generator=self.generator)
        special = (clean == self.special["cls"]) | (clean == self.special["sep"])
        selected = (probability < self.mask_probability) & ~special
        labels[~selected] = -100
        replace = torch.rand(clean.shape, generator=self.generator)
        masked = clean.clone()
        masked[selected & (replace < 0.8)] = self.special["mask"]
        random_mask = selected & (replace >= 0.8) & (replace < 0.9)
        random_words = torch.randint(
            low=len(self.special), high=self.tokenizer.get_vocab_size(), size=clean.shape,
            generator=self.generator,
        )
        masked[random_mask] = random_words[random_mask]
        return masked, labels

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        while self.processed < self.total_tokens:
            length = self.lengths[self.step % len(self.lengths)]
            boundary = self.boundaries[self.boundary_index]
            remaining = boundary - self.processed
            batch_size = min(
                max(1, self.batch_tokens // length),
                max(1, math.ceil(remaining / length)),
            )
            rows = []
            for _ in range(batch_size):
                packer = self.packers[self.corpus_cursor % len(self.packers)]
                self.corpus_cursor += 1
                rows.append([self.special["cls"]] + packer.take(length - 2) + [self.special["sep"]])
            clean = torch.tensor(rows, dtype=torch.long)
            input_ids, labels = self._mask(clean)
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            if clean.numel() > remaining:
                attention_mask.view(-1)[remaining:] = False
                input_ids.view(-1)[remaining:] = self.special["pad"]
                labels.view(-1)[remaining:] = -100
            if not (labels != -100).any():
                eligible = attention_mask & (clean != self.special["cls"]) & (
                    clean != self.special["sep"]
                )
                candidates = eligible.flatten().nonzero()
                if candidates.numel():
                    flat_index = int(candidates[0])
                    labels.view(-1)[flat_index] = clean.view(-1)[flat_index]
                    input_ids.view(-1)[flat_index] = self.special["mask"]
            self.processed += int(attention_mask.sum())
            if self.processed == boundary:
                self.boundary_index += 1
            self.step += 1
            yield {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class EpochBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Deterministic epoch-specific batches, allowing exact batch-index resume."""

    def __init__(self, size: int, batch_size: int, shuffle: bool, seed: int):
        self.size = size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.start_batch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_start_batch(self, start_batch: int) -> None:
        self.start_batch = max(0, start_batch)

    def __iter__(self):
        indices = list(range(self.size))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)
        batches = [
            indices[start : start + self.batch_size]
            for start in range(0, self.size, self.batch_size)
        ]
        yield from batches[self.start_batch :]

    def __len__(self) -> int:
        return math.ceil(self.size / self.batch_size)


def task_dataloader(
    data_dir: str | Path,
    task: str,
    split: str,
    max_length: int,
    batch_tokens: int,
    workers: int,
    seed: int,
    shuffle: bool,
    synthetic_lengths: list[int] | None = None,
    dataset_seed: int | None = None,
) -> torch.utils.data.DataLoader:
    dataset = ClassificationDataset(
        data_dir, task, split, max_length, seed=seed if dataset_seed is None else dataset_seed,
        synthetic_lengths=synthetic_lengths,
    )
    batch_size = max(1, batch_tokens // max_length)
    collator = ClassificationCollator(
        _special_ids(dataset.tokenizer)["pad"], max_length=max_length
    )
    if task.startswith(("synthetic_", "stress_")):
        batch_sampler = SyntheticLengthBatchSampler(dataset, batch_tokens, shuffle, seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0,
        )
    batch_sampler = EpochBatchSampler(len(dataset), batch_size, shuffle, seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collator,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def prefetch(iterator: Iterator[dict[str, torch.Tensor]], depth: int = 2) -> Iterator[dict[str, torch.Tensor]]:
    """Overlap Rust tokenization/masking with accelerator compute."""
    items: queue.Queue = queue.Queue(maxsize=depth)
    sentinel = object()

    def produce() -> None:
        try:
            for item in iterator:
                items.put(item)
        except BaseException as error:  # re-raised in the training thread
            items.put(error)
        finally:
            items.put(sentinel)

    threading.Thread(target=produce, daemon=True).start()
    while True:
        item = items.get()
        if item is sentinel:
            return
        if isinstance(item, BaseException):
            raise item
        yield item
