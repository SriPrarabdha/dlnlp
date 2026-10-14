from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch

from longdoc.config import RunConfig
from longdoc.data.batching import task_dataloader
from longdoc.data.catalog import get_dataset_info
from longdoc.data.tokenizer import configure_token_dependent_fields, load_tokenizer
from longdoc.models.modeling import build_model

from .metrics import evidence_breakdown, stress_breakdown
from .operations import atomic_write_json
from .runtime import (
    amp_dtype,
    autocast_context,
    configure_accelerator,
    move_batch,
    resolve_device,
    seed_everything,
)
from .trainer import evaluate


def _task_info(task: str) -> tuple[str, int]:
    if task.startswith(("synthetic_", "stress_")):
        return "binary", 2
    info = get_dataset_info(task)
    return info.task_type, info.num_labels


class TransformLoader:
    def __init__(self, loader: Iterable[dict[str, Any]], transform: Callable[[dict], dict]):
        self.loader = loader
        self.transform = transform

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for batch in self.loader:
            yield self.transform(batch)


def _prepend(
    batch: dict[str, Any], count: int, token_id: int = 3, max_length: int | None = None
) -> dict[str, Any]:
    if count <= 0:
        return batch
    ids, mask = batch["input_ids"], batch["attention_mask"]
    prefix = torch.full((ids.shape[0], count), token_id, dtype=ids.dtype)
    prefix_mask = torch.ones_like(prefix, dtype=torch.bool)
    limit = max_length or ids.shape[1] + count
    batch["input_ids"] = torch.cat((ids[:, :1], prefix, ids[:, 1:]), 1)[:, :limit]
    batch["attention_mask"] = torch.cat((mask[:, :1], prefix_mask, mask[:, 1:]), 1)[:, :limit]
    return batch


def _shuffle_chunks(batch: dict[str, Any], chunk_size: int = 64) -> dict[str, Any]:
    ids = batch["input_ids"].clone()
    for row in range(ids.shape[0]):
        length = int(batch["attention_mask"][row].sum())
        content = ids[row, 1 : length - 1]
        chunks = list(content.split(chunk_size))
        ids[row, 1 : length - 1] = torch.cat(list(reversed(chunks)))
    batch["input_ids"] = ids
    return batch


def projection_singular_values(model: torch.nn.Module) -> dict[str, list[float]]:
    result = {}
    for name, parameter in model.named_parameters():
        if "projection" in name and parameter.ndim == 2:
            result[name] = torch.linalg.svdvals(parameter.float()).detach().cpu().tolist()
    return result


@torch.inference_mode()
def selection_diagnostics(model, loader, device, dtype, max_batches: int) -> dict:
    positions, scores = [], []
    model.eval()
    completed = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        moved = move_batch(batch, device)
        completed += 1
        with autocast_context(device, dtype):
            output = model(moved["input_ids"], moved["attention_mask"])
        valid_lengths = moved["attention_mask"].sum(-1).cpu().numpy()
        for key, value in output.diagnostics.items():
            if key.endswith("global_indices"):
                indices = value.cpu().numpy()
                valid_key = key.removesuffix("global_indices") + "global_valid"
                selected_valid = output.diagnostics.get(valid_key)
                valid_rows = (
                    selected_valid.cpu().numpy()
                    if selected_valid is not None
                    else np.ones_like(indices, dtype=bool)
                )
                for row, row_valid, length in zip(indices, valid_rows, valid_lengths):
                    row = row[row_valid]
                    positions.extend((row / max(1, length - 1)).clip(0, 1).tolist())
            elif key.endswith("global_scores"):
                valid_scores = value.masked_select(moved["attention_mask"])
                scores.extend(valid_scores[torch.isfinite(valid_scores)].float().cpu().tolist())
    histogram, edges = np.histogram(positions, bins=20, range=(0, 1))
    return {
        "normalized_position_histogram": histogram.tolist(),
        "position_bin_edges": edges.tolist(),
        "score_quantiles": np.quantile(scores, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]).tolist() if scores else [],
        "batches": completed,
    }


def analyze_checkpoint(
    config: RunConfig,
    checkpoint: Path,
    output: Path,
    analysis: str,
    device_name: str = "auto",
    value: int = 0,
) -> dict:
    seed_everything(config.seed)
    task_type, num_labels = _task_info(config.task)
    device = resolve_device(device_name)
    configure_accelerator(device)
    dtype = amp_dtype(device, config.train.amp)
    configure_token_dependent_fields(config.model, config.data_dir)
    model = build_model(config.model, "finetune", num_labels, task_type)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    tolerated = ("scorer", "band_logits")
    bad = [name for name in [*missing, *unexpected] if not any(key in name for key in tolerated)]
    if bad:
        raise RuntimeError(f"Analysis config is incompatible with checkpoint: {bad}")
    model.to(device)
    if analysis == "projection_spectrum":
        result = {"singular_values": projection_singular_values(model)}
    elif analysis == "selection_diagnostics":
        loader = task_dataloader(
            config.data_dir, config.task, "test", config.model.max_length,
            config.train.eval_batch_tokens, config.train.num_workers, config.seed, False,
        )
        result = selection_diagnostics(model, loader, device, dtype, value or 100)
    else:
        lengths = [512, 1_024, 2_048, 4_096] if analysis == "truncation" else [config.model.max_length]
        result = {}
        for length in lengths:
            if length > config.model.max_length:
                continue
            loader: Iterable[dict[str, Any]] = task_dataloader(
                config.data_dir, config.task, "test", length,
                config.train.eval_batch_tokens, config.train.num_workers, config.seed, False,
            )
            if analysis == "prepend":
                tokenizer = load_tokenizer(config.data_dir)
                neutral_ids = tokenizer.encode(" the", add_special_tokens=False).ids
                neutral_id = neutral_ids[0] if neutral_ids else 3
                loader = TransformLoader(
                    loader,
                    lambda batch, prefix_id=neutral_id: _prepend(
                        batch, value or 128, prefix_id, config.model.max_length
                    ),
                )
            elif analysis == "chunk_shuffle":
                loader = TransformLoader(loader, lambda batch: _shuffle_chunks(batch, value or 64))
            elif analysis not in {"truncation", "ablation"}:
                raise ValueError(f"Unknown analysis: {analysis}")
            metrics, raw = evaluate(model, loader, task_type, device, dtype)
            if config.task.startswith(("synthetic_", "stress_")):
                metrics["evidence_heatmap"] = evidence_breakdown(
                    raw["logits"], raw["labels"], raw["positions"], raw["source_lengths"]
                )
                metrics["stress_breakdown"] = stress_breakdown(
                    raw["logits"], raw["labels"], raw["positions"], raw["source_lengths"],
                    raw["challenges"], raw["distractor_counts"],
                )
            result[str(length)] = metrics
    atomic_write_json(output, result)
    return result
