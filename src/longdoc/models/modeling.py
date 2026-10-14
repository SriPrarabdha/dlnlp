from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from longdoc.config import ModelConfig

from .attention import (
    AttentionOutput,
    ExactAttention,
    LinformerAttention,
    LongformerAttention,
    LongShortAttention,
    MemoryAttention,
)


@dataclass
class EncoderOutput:
    hidden: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


@dataclass
class ModelOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, attention: nn.Module):
        super().__init__()
        self.attention = attention
        self.attention_norm = nn.LayerNorm(config.hidden_size)
        self.ffn_norm = nn.LayerNorm(config.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Any) -> AttentionOutput:
        attended = self.attention(self.attention_norm(x), attention_mask, **kwargs)
        x = x + self.dropout(attended.hidden)
        x = x + self.ffn(self.ffn_norm(x))
        return AttentionOutput(x, attended.diagnostics)


def _attention_for(config: ModelConfig) -> nn.Module:
    if config.architecture == "exact":
        return ExactAttention(config)
    if config.architecture == "roformer":
        return ExactAttention(config, rope=True)
    if config.architecture == "longformer":
        return LongformerAttention(config)
    if config.architecture == "linformer":
        return LinformerAttention(config)
    if config.architecture == "longshort":
        return LongShortAttention(config)
    raise ValueError(f"No standard attention for {config.architecture}")


class StandardEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = (
            nn.Embedding(config.max_length, config.hidden_size)
            if config.position_encoding == "learned" and config.architecture != "roformer"
            else None
        )
        self.embedding_norm = nn.LayerNorm(config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            TransformerBlock(config, _attention_for(config)) for _ in range(config.num_layers)
        )
        if config.architecture == "longformer" and config.global_selection in {"learned", "lowest"}:
            shared_scorer = self.layers[0].attention.scorer
            for layer in self.layers[1:]:
                layer.attention.scorer = shared_scorer
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.gradient_checkpointing = False
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> EncoderOutput:
        length = input_ids.shape[1]
        if length > self.config.max_length:
            raise ValueError(
                f"Input length {length} exceeds model.max_length={self.config.max_length}"
            )
        x = self.token_embeddings(input_ids)
        if self.position_embeddings is not None:
            positions = torch.arange(length, device=input_ids.device)
            x = x + self.position_embeddings(positions)[None]
        x = self.embedding_dropout(self.embedding_norm(x))
        diagnostics: dict[str, torch.Tensor] = {}
        for layer_index, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                x = checkpoint(
                    lambda hidden, mask, ids, module=layer: module(hidden, mask, input_ids=ids).hidden,
                    x,
                    attention_mask,
                    input_ids,
                    use_reentrant=False,
                )
            else:
                result = layer(x, attention_mask, input_ids=input_ids)
                x = result.hidden
                diagnostics.update({f"layer_{layer_index}/{k}": v for k, v in result.diagnostics.items()})
        return EncoderOutput(self.final_norm(x), diagnostics)


class DualChannelCompressor(nn.Module):
    def __init__(self, hidden_size: int, rate: int, mode: str, max_slots: int):
        super().__init__()
        self.rate = rate
        self.mode = mode
        self.salience = nn.Linear(hidden_size, 1)
        self.salience_queries = nn.Parameter(torch.randn(max_slots, hidden_size) * 0.02)

    @staticmethod
    def _mean_pool(
        x: torch.Tensor, mask: torch.Tensor, slots: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = mask[:, None].to(x.dtype)
        numerator = F.adaptive_avg_pool1d((x.transpose(1, 2) * weights), slots)
        denominator = F.adaptive_avg_pool1d(weights, slots)
        valid = denominator.squeeze(1) > 0
        pooled = (numerator / denominator.clamp_min(1e-12)).transpose(1, 2)
        return pooled.masked_fill(~valid[:, :, None], 0), valid

    def _salient_pool(
        self, x: torch.Tensor, mask: torch.Tensor, slots: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        queries = self.salience_queries[:slots]
        scores = torch.einsum("bnd,sd->bns", x, queries) / math.sqrt(x.shape[-1])
        # A shared content salience term helps all slots notice a small rare fact;
        # slot queries keep the summaries diverse.
        scores = scores + self.salience(x)
        valid_rows = mask.any(-1)
        scores = scores.masked_fill(~mask[:, :, None], -torch.inf)
        scores = torch.where(valid_rows[:, None, None], scores, torch.zeros_like(scores))
        weights = torch.softmax(scores, dim=1) * mask[:, :, None]
        valid = valid_rows[:, None].expand(-1, slots)
        pooled = torch.einsum("bns,bnd->bsd", weights, x)
        return pooled.masked_fill(~valid[:, :, None], 0), valid

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[
        torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
    ]:
        slots = max(1, math.ceil(x.shape[1] / self.rate))
        if self.mode == "uniform":
            coverage, coverage_mask = self._mean_pool(x, mask, slots)
            return coverage, None, coverage_mask, None
        if self.mode == "salient":
            salience, salience_mask = self._salient_pool(x, mask, slots)
            return None, salience, None, salience_mask
        if self.mode != "dual":
            raise ValueError(f"Unknown compressor mode: {self.mode}")
        uniform_slots = slots // 2
        salient_slots = slots - uniform_slots
        coverage, coverage_mask = (
            self._mean_pool(x, mask, uniform_slots) if uniform_slots else (None, None)
        )
        salience, salience_mask = (
            self._salient_pool(x, mask, salient_slots) if salient_slots else (None, None)
        )
        return coverage, salience, coverage_mask, salience_mask


class CompressiveEncoder(nn.Module):
    """Segment-recurrent bidirectional encoder with per-layer two-tier memory."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_length, config.hidden_size)
        self.embedding_norm = nn.LayerNorm(config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            TransformerBlock(config, MemoryAttention(config)) for _ in range(config.num_layers)
        )
        self.compressors = nn.ModuleList(
            DualChannelCompressor(
                config.hidden_size, config.compression_rate, config.compressor,
                config.compressed_memory_size,
            )
            for _ in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.gradient_checkpointing = False
        self.apply(StandardEncoder._init_weights)

    def _update_memory(
        self,
        old_memory: torch.Tensor | None,
        old_memory_mask: torch.Tensor | None,
        old_coverage: torch.Tensor | None,
        old_coverage_mask: torch.Tensor | None,
        old_salience: torch.Tensor | None,
        old_salience_mask: torch.Tensor | None,
        current: torch.Tensor,
        current_mask: torch.Tensor,
        layer_index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if self.config.detach_memory:
            current = current.detach()
        memory = current if old_memory is None else torch.cat((old_memory, current), dim=1)
        memory_mask = (
            current_mask
            if old_memory_mask is None
            else torch.cat((old_memory_mask, current_mask), dim=1)
        )
        coverage, salience = old_coverage, old_salience
        coverage_mask, salience_mask = old_coverage_mask, old_salience_mask
        overflow = max(0, memory.shape[1] - self.config.memory_size)
        if overflow:
            evicted, memory = memory[:, :overflow], memory[:, overflow:]
            evicted_mask, memory_mask = memory_mask[:, :overflow], memory_mask[:, overflow:]
            new_coverage, new_salience, new_coverage_mask, new_salience_mask = (
                self.compressors[layer_index](evicted, evicted_mask)
            )
            if new_coverage is not None:
                coverage = new_coverage if coverage is None else torch.cat((coverage, new_coverage), dim=1)
                coverage_mask = (
                    new_coverage_mask
                    if coverage_mask is None
                    else torch.cat((coverage_mask, new_coverage_mask), dim=1)
                )
            if new_salience is not None:
                salience = new_salience if salience is None else torch.cat((salience, new_salience), dim=1)
                salience_mask = (
                    new_salience_mask
                    if salience_mask is None
                    else torch.cat((salience_mask, new_salience_mask), dim=1)
                )
            if self.config.compressor == "dual":
                coverage_cap = self.config.compressed_memory_size // 2
                salience_cap = self.config.compressed_memory_size - coverage_cap
            else:
                coverage_cap = self.config.compressed_memory_size
                salience_cap = self.config.compressed_memory_size
            if coverage is not None:
                coverage = coverage[:, -coverage_cap:] if coverage_cap else None
                coverage_mask = coverage_mask[:, -coverage_cap:] if coverage_cap else None
            if salience is not None:
                salience = salience[:, -salience_cap:] if salience_cap else None
                salience_mask = salience_mask[:, -salience_cap:] if salience_cap else None
        return memory, memory_mask, coverage, coverage_mask, salience, salience_mask

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> EncoderOutput:
        total_length = input_ids.shape[1]
        if total_length > self.config.max_length:
            raise ValueError(
                f"Input length {total_length} exceeds model.max_length={self.config.max_length}"
            )
        positions = torch.arange(total_length, device=input_ids.device)
        embeddings = self.token_embeddings(input_ids) + self.position_embeddings(positions)[None]
        embeddings = self.embedding_dropout(self.embedding_norm(embeddings))
        memories: list[torch.Tensor | None] = [None] * len(self.layers)
        memory_masks: list[torch.Tensor | None] = [None] * len(self.layers)
        coverage_memory: list[torch.Tensor | None] = [None] * len(self.layers)
        coverage_masks: list[torch.Tensor | None] = [None] * len(self.layers)
        salience_memory: list[torch.Tensor | None] = [None] * len(self.layers)
        salience_masks: list[torch.Tensor | None] = [None] * len(self.layers)
        output_segments = []
        diagnostics: dict[str, torch.Tensor] = {}
        for start in range(0, total_length, self.config.segment_size):
            end = min(total_length, start + self.config.segment_size)
            x = embeddings[:, start:end]
            segment_mask = attention_mask[:, start:end]
            for layer_index, layer in enumerate(self.layers):
                layer_input = x
                layer_memory = memories[layer_index]
                layer_memory_mask = memory_masks[layer_index]
                layer_coverage = coverage_memory[layer_index]
                layer_coverage_mask = coverage_masks[layer_index]
                layer_salience = salience_memory[layer_index]
                layer_salience_mask = salience_masks[layer_index]
                if self.config.memory_ablation in {"recent", "regular"}:
                    layer_memory = None
                    layer_memory_mask = None
                elif self.config.memory_ablation == "compressed":
                    layer_coverage, layer_salience = None, None
                    layer_coverage_mask, layer_salience_mask = None, None
                elif self.config.memory_ablation in {"coverage", "uniform"}:
                    layer_coverage = None
                    layer_coverage_mask = None
                elif self.config.memory_ablation in {"salience", "salient"}:
                    layer_salience = None
                    layer_salience_mask = None
                compressed_parts = [
                    item for item in (layer_coverage, layer_salience) if item is not None
                ]
                compressed_mask_parts = [
                    item for item in (layer_coverage_mask, layer_salience_mask)
                    if item is not None
                ]
                layer_compressed = torch.cat(compressed_parts, dim=1) if compressed_parts else None
                layer_compressed_mask = (
                    torch.cat(compressed_mask_parts, dim=1) if compressed_mask_parts else None
                )
                include_current = not (
                    self.config.memory_ablation == "current" and end == total_length
                )
                if self.gradient_checkpointing and self.training:
                    hidden = checkpoint(
                        lambda current, mask, module=layer, recurrent=layer_memory,
                        recurrent_mask=layer_memory_mask, compact=layer_compressed,
                        compact_mask=layer_compressed_mask,
                        current_visible=include_current: module(
                            current, mask, memory=recurrent, memory_mask=recurrent_mask,
                            compressed=compact, compressed_mask=compact_mask,
                            include_current=current_visible,
                        ).hidden,
                        x,
                        segment_mask,
                        use_reentrant=False,
                    )
                    result = AttentionOutput(hidden, {})
                else:
                    result = layer(
                        x,
                        segment_mask,
                        memory=layer_memory,
                        memory_mask=layer_memory_mask,
                        compressed=layer_compressed,
                        compressed_mask=layer_compressed_mask,
                        include_current=include_current,
                    )
                x = result.hidden
                (
                    memories[layer_index], memory_masks[layer_index],
                    coverage_memory[layer_index], coverage_masks[layer_index],
                    salience_memory[layer_index], salience_masks[layer_index],
                ) = self._update_memory(
                    memories[layer_index], memory_masks[layer_index],
                    coverage_memory[layer_index], coverage_masks[layer_index],
                    salience_memory[layer_index], salience_masks[layer_index],
                    layer_input, segment_mask, layer_index,
                )
                result.diagnostics["coverage_tokens"] = torch.tensor(
                    0.0, device=x.device
                ) if layer_coverage_mask is None else layer_coverage_mask.sum(-1).float().mean()
                result.diagnostics["salience_tokens"] = torch.tensor(
                    0.0, device=x.device
                ) if layer_salience_mask is None else layer_salience_mask.sum(-1).float().mean()
                diagnostics.update(
                    {f"layer_{layer_index}/{key}": value for key, value in result.diagnostics.items()}
                )
            output_segments.append(x)
        return EncoderOutput(self.final_norm(torch.cat(output_segments, dim=1)), diagnostics)


class LongDocumentForMLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = CompressiveEncoder(config) if config.architecture == "compressive" else StandardEncoder(config)
        self.mlm_norm = nn.LayerNorm(config.hidden_size)
        # Deliberately untied: 32K x 512 makes the controlled pretraining models ~45M parameters.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        nn.init.normal_(self.decoder.weight, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor | None = None
    ) -> ModelOutput:
        encoded = self.encoder(input_ids, attention_mask)
        logits = self.decoder(self.mlm_norm(encoded.hidden))
        loss = None
        if labels is not None:
            loss = (
                F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
                )
                if (labels != -100).any()
                else logits.sum() * 0.0
            )
        return ModelOutput(loss, logits, encoded.diagnostics)


class LongDocumentForClassification(nn.Module):
    def __init__(self, config: ModelConfig, num_labels: int, task_type: str):
        super().__init__()
        self.config = config
        self.task_type = task_type
        self.encoder = CompressiveEncoder(config) if config.architecture == "compressive" else StandardEncoder(config)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor | None = None
    ) -> ModelOutput:
        if self.config.architecture == "compressive":
            # Segment recurrence is left-to-right across segments. Put a copy
            # of the document's CLS token at the final real position so the
            # pooled query can attend the entire final segment and both memory
            # tiers, rather than using SEP as an accidental classifier token.
            input_ids = input_ids.clone()
            final_index = attention_mask.sum(-1).sub(1).clamp_min(0)
            input_ids[
                torch.arange(input_ids.shape[0], device=input_ids.device), final_index
            ] = input_ids[:, 0]
        encoded = self.encoder(input_ids, attention_mask)
        if self.config.architecture == "compressive":
            pooled = encoded.hidden[
                torch.arange(input_ids.shape[0], device=input_ids.device), final_index
            ]
        else:
            pooled = encoded.hidden[:, 0]
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels) if self.task_type == "multilabel" else F.cross_entropy(logits, labels)
        return ModelOutput(loss, logits, encoded.diagnostics)


def build_model(config: ModelConfig, stage: str, num_labels: int = 2, task_type: str = "binary") -> nn.Module:
    return LongDocumentForMLM(config) if stage == "pretrain" else LongDocumentForClassification(
        config, num_labels, task_type
    )


def load_encoder_checkpoint(model: nn.Module, path: str) -> tuple[list[str], list[str]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    encoder_state = {
        key.removeprefix("encoder."): value for key, value in state.items() if key.startswith("encoder.")
    }
    return model.encoder.load_state_dict(encoder_state, strict=False)
