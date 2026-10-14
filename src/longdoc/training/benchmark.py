from __future__ import annotations

import math
import time
from pathlib import Path

import torch

from longdoc.config import RunConfig
from longdoc.models.modeling import build_model

from .operations import atomic_write_json
from .runtime import (
    amp_dtype,
    autocast_context,
    configure_accelerator,
    resolve_device,
    seed_everything,
)


def estimate_attention_flops(config, length: int, batch: int) -> int:
    h, d, layers = config.num_heads, config.hidden_size // config.num_heads, config.num_layers
    if config.architecture in {"exact", "roformer"}:
        pairs = length * length
    elif config.architecture == "longformer":
        pairs = length * (config.attention_window + 2 * max(1, length // config.global_stride))
    elif config.architecture == "linformer":
        pairs = length * config.projection_rank
    elif config.architecture == "longshort":
        if config.summary_type == "disabled":
            summaries = 0
        elif config.summary_type == "global":
            summaries = config.projection_rank
        else:
            summaries = math.ceil(length / config.segment_size) * config.summaries_per_segment
        pairs = length * (config.attention_window + summaries)
    else:
        segments = math.ceil(length / config.segment_size)
        pairs = segments * config.segment_size * (config.segment_size + config.memory_size + config.compressed_memory_size)
    return int(batch * layers * 4 * h * d * pairs)


def benchmark(config: RunConfig, output: Path, device_name: str = "auto", repeats: int = 10) -> dict:
    if repeats <= 0:
        raise ValueError("Benchmark repeats must be positive")
    seed_everything(config.seed)
    device = resolve_device(device_name)
    configure_accelerator(device)
    dtype = amp_dtype(device, config.train.amp)
    model = build_model(config.model, "finetune", 2, "binary").to(device).eval()
    if config.train.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    results = {"parameters": sum(parameter.numel() for parameter in model.parameters()), "lengths": {}}
    for length in config.train.lengths:
        batch = max(1, config.train.eval_batch_tokens // length)
        while True:
            try:
                input_ids = torch.randint(5, config.model.vocab_size, (batch, length), device=device)
                mask = torch.ones_like(input_ids, dtype=torch.bool)
                with torch.inference_mode(), autocast_context(device, dtype):
                    for _ in range(2):
                        model(input_ids, mask)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    for _ in range(repeats):
                        model(input_ids, mask)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                break
            except torch.OutOfMemoryError:
                if device.type != "cuda" or batch == 1:
                    raise
                batch = max(1, batch // 2)
                input_ids = None
                mask = None
                torch.cuda.empty_cache()
        results["lengths"][str(length)] = {
            "documents_per_second": batch * repeats / elapsed,
            "milliseconds_per_batch": elapsed * 1_000 / repeats,
            "peak_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
            "estimated_attention_flops": estimate_attention_flops(config.model, length, batch),
            "batch_size": batch,
        }
    atomic_write_json(output, results)
    return results
