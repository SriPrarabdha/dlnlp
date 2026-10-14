from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from longdoc.config import ModelConfig


@dataclass
class AttentionOutput:
    hidden: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


def _split_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    batch, length, width = x.shape
    return x.view(batch, length, heads, width // heads).transpose(1, 2)


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch, heads, length, width = x.shape
    return x.transpose(1, 2).contiguous().view(batch, length, heads * width)


def _masked_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_mask: torch.Tensor,
    dropout: float,
    training: bool,
) -> torch.Tensor:
    mask = key_mask[:, None, None, :]
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=dropout if training else 0.0
    )


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float, gated_bands: bool, heads: int):
        super().__init__()
        frequencies = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.head_dim = head_dim
        self.heads = heads
        if gated_bands:
            # Identity at sigmoid(logit) * 2 == 1.
            self.band_logits = nn.Parameter(torch.zeros(heads, 3))
        else:
            self.band_logits = None
        self.register_buffer("ablate_bands", torch.ones(3), persistent=False)

    def set_band_ablation(self, band: int | None) -> None:
        self.ablate_bands.fill_(1)
        if band is not None:
            self.ablate_bands[band] = 0

    def _angles(self, positions: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        angles = positions.float()[:, None] * self.frequencies[None, :]
        pair_count = self.frequencies.numel()
        boundaries = torch.linspace(0, pair_count, 4, device=positions.device).long()
        if self.band_logits is not None:
            gates = 2 * self.band_logits.sigmoid() * self.ablate_bands
            expanded = torch.empty(self.heads, pair_count, device=positions.device)
            # inv_freq is stored fastest-to-slowest, while public band indices
            # are semantic: 0=low/slow and 2=high/fast.
            for physical_band in range(3):
                semantic_band = 2 - physical_band
                expanded[:, boundaries[physical_band] : boundaries[physical_band + 1]] = (
                    gates[:, semantic_band : semantic_band + 1]
                )
            return angles[None, :, :] * expanded[:, None, :]
        expanded = torch.empty(pair_count, device=positions.device)
        for physical_band in range(3):
            semantic_band = 2 - physical_band
            expanded[boundaries[physical_band] : boundaries[physical_band + 1]] = (
                self.ablate_bands[semantic_band]
            )
        return angles[None, :, :] * expanded[None, :]

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # x: batch, heads, length, head_dim
        angle = self._angles(positions, x.dtype)
        cos = angle.cos().to(dtype=x.dtype)[None, ...]
        sin = angle.sin().to(dtype=x.dtype)[None, ...]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class ExactAttention(nn.Module):
    def __init__(self, config: ModelConfig, rope: bool = False):
        super().__init__()
        self.heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = config.dropout
        self.rope = RotaryEmbedding(
            self.head_dim, config.rope_base, config.rope_gated_bands, config.num_heads
        ) if rope else None
        if self.rope is not None:
            self.rope.set_band_ablation(config.rope_ablate_band)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor, **_: Any) -> AttentionOutput:
        q, k, v = (_split_heads(part, self.heads) for part in self.qkv(x).chunk(3, dim=-1))
        if self.rope is not None:
            positions = torch.arange(x.shape[1], device=x.device)
            q, k = self.rope(q, positions), self.rope(k, positions)
        output = _masked_sdpa(q, k, v, attention_mask, self.dropout, self.training)
        return AttentionOutput(self.out(_merge_heads(output)), {})


def _local_windows(tensor: torch.Tensor, radius: int) -> torch.Tensor:
    padded = F.pad(tensor, (0, 0, radius, radius))
    # unfold appends the window dimension after feature dimension.
    return padded.unfold(2, 2 * radius + 1, 1).permute(0, 1, 2, 4, 3)


def _local_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    return F.pad(mask, (radius, radius), value=False).unfold(1, 2 * radius + 1, 1)


def _segment_indices(length: int, span: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Indices for the segment-wise window used by Transformer-LS.

    Queries in one home segment share the home segment plus balanced context
    on either side. ``span`` is the total local key/value budget.
    """
    home = max(1, span // 2)
    left = (span - home) // 2
    positions = torch.arange(length, device=device)
    starts = (positions // home) * home - left
    indices = starts[:, None] + torch.arange(span, device=device)[None, :]
    valid = (indices >= 0) & (indices < length)
    return indices.clamp(0, max(0, length - 1)), valid


def _segment_windows(tensor: torch.Tensor, span: int) -> tuple[torch.Tensor, torch.Tensor]:
    indices, in_bounds = _segment_indices(tensor.shape[2], span, tensor.device)
    return tensor[:, :, indices, :], in_bounds


def _segment_mask(mask: torch.Tensor, span: int) -> torch.Tensor:
    indices, in_bounds = _segment_indices(mask.shape[1], span, mask.device)
    return mask[:, indices] & in_bounds[None]


class LongformerAttention(nn.Module):
    """Linear-memory sliding-window attention with sparse global rows/columns."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.radius = max(1, config.attention_window // 2)
        self.stride = config.global_stride
        self.selection = config.global_selection
        self.sentence_boundary_token_ids = config.sentence_boundary_token_ids
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        # Sharing keeps the controlled models within the matched parameter budget.
        # Set separate_global_projections=true to reproduce original Longformer.
        self.global_qkv = (
            nn.Linear(config.hidden_size, 3 * config.hidden_size)
            if config.separate_global_projections else self.qkv
        )
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.scorer = nn.Linear(config.hidden_size, 1) if self.selection in {"learned", "lowest"} else None
        self.dropout = nn.Dropout(config.dropout)

    def _global_indices(
        self, x: torch.Tensor, mask: torch.Tensor, input_ids: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch, length, _ = x.shape
        real_lengths = mask.sum(-1)
        additional = torch.ceil(real_lengths.float() / self.stride).long().clamp(min=1)
        additional = torch.minimum(additional, (real_lengths - 1).clamp_min(0))
        counts = torch.ones_like(additional) if self.selection == "cls" else additional + 1
        real_length_values = [int(value) for value in real_lengths.tolist()]
        additional_values = [int(value) for value in additional.tolist()]
        count = max(int(value) for value in counts.tolist())
        valid_scores = torch.full((batch, length), -torch.inf, device=x.device)
        learned_scores = None

        def even_scores() -> torch.Tensor:
            result = torch.zeros((batch, length), device=x.device)
            for row, (real_length, slots) in enumerate(
                zip(real_length_values, additional_values)
            ):
                if slots == 0:
                    continue
                targets = torch.linspace(1, max(1, real_length - 1), slots, device=x.device).round().long()
                result[row, targets] = 1.0
            result -= torch.arange(length, device=x.device)[None].float() / max(1, length * 10)
            return result.masked_fill(~mask, -torch.inf)

        if self.selection in {"learned", "lowest"}:
            learned_scores = self.scorer(x).squeeze(-1)
            chosen = learned_scores if self.selection == "learned" else -learned_scores
            valid_scores = chosen.masked_fill(~mask, -torch.inf)
        elif self.selection == "random":
            valid_scores = torch.rand((batch, length), device=x.device).masked_fill(~mask, -torch.inf)
        elif self.selection == "even":
            valid_scores = even_scores()
        elif self.selection == "sentence":
            boundaries = torch.zeros((batch, length), dtype=torch.bool, device=x.device)
            if input_ids is not None and self.sentence_boundary_token_ids:
                boundary_ids = torch.tensor(self.sentence_boundary_token_ids, device=x.device)
                boundaries[:, 1:] = (input_ids[:, :-1, None] == boundary_ids).any(-1)
            # Boundary positions win; evenly spaced positions fill any shortfall.
            valid_scores = even_scores() + 2 * boundaries.float()
            valid_scores = valid_scores.masked_fill(~mask, -torch.inf)
        else:  # [CLS] only baseline
            count = 1
        valid_scores[:, 0] = torch.inf
        indices = valid_scores.topk(count, dim=-1).indices
        selected_valid = torch.arange(count, device=x.device)[None, :] < counts[:, None]
        indices[:, 0] = 0
        return indices, selected_valid, learned_scores

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor,
        input_ids: torch.Tensor | None = None, **_: Any,
    ) -> AttentionOutput:
        q, k, v = (_split_heads(part, self.heads) for part in self.qkv(x).chunk(3, dim=-1))
        if self.global_qkv is self.qkv:
            gq, gk, gv = q, k, v
        else:
            gq, gk, gv = (
                _split_heads(part, self.heads) for part in self.global_qkv(x).chunk(3, dim=-1)
            )
        k_window, v_window = _local_windows(k, self.radius), _local_windows(v, self.radius)
        window_mask = _local_mask(attention_mask, self.radius)

        indices, global_valid, scores = self._global_indices(x, attention_mask, input_ids)
        global_tokens = torch.zeros_like(attention_mask)
        global_tokens.scatter_(1, indices, global_valid)
        # Global columns are concatenated explicitly below; exclude them from
        # the local band so nearby global tokens are not counted twice.
        window_mask = window_mask & ~_local_mask(global_tokens, self.radius)
        gather = indices[:, None, :, None].expand(-1, self.heads, -1, self.head_dim)
        global_k = gk.gather(2, gather)
        global_v = gv.gather(2, gather)
        gate = torch.ones_like(global_valid, dtype=x.dtype)
        if scores is not None:
            chosen_scores = scores.gather(1, indices)
            soft = chosen_scores.sigmoid()
            gate = torch.ones_like(soft) + soft - soft.detach()  # straight-through top-k
            gate = torch.cat((torch.ones_like(gate[:, :1]), gate[:, 1:]), dim=1)
        global_k = global_k * gate[:, None, :, None]
        global_v = global_v * gate[:, None, :, None]

        scale = self.head_dim ** -0.5
        local_logits = torch.einsum("bhnd,bhnwd->bhnw", q, k_window) * scale
        local_logits = local_logits.masked_fill(~window_mask[:, None], -torch.inf)
        global_logits = torch.einsum("bhnd,bhgd->bhng", q, global_k) * scale
        global_logits = global_logits.masked_fill(~global_valid[:, None, None, :], -torch.inf)
        weights = torch.softmax(torch.cat((local_logits, global_logits), dim=-1), dim=-1)
        weights = self.dropout(weights)
        local_weight, global_weight = weights.split((local_logits.shape[-1], global_logits.shape[-1]), -1)
        output = torch.einsum("bhnw,bhnwd->bhnd", local_weight, v_window)
        output = output + torch.einsum("bhng,bhgd->bhnd", global_weight, global_v)

        # Selected global queries get a true all-token row using separate projections.
        global_q = gq.gather(2, gather)
        global_output = _masked_sdpa(
            global_q, gk, gv, attention_mask, self.dropout.p, self.training
        )
        global_output = global_output * gate[:, None, :, None]
        previous = output.gather(2, gather)
        global_output = torch.where(
            global_valid[:, None, :, None], global_output, previous
        ).to(output.dtype)
        output = output.scatter(2, gather, global_output)
        output = output * attention_mask[:, None, :, None]
        valid_query_count = attention_mask.sum().clamp_min(1) * self.heads
        diagnostics = {
            "global_indices": indices.detach(),
            "global_valid": global_valid.detach(),
            "global_attention_mass": (
                global_weight.sum(-1) * attention_mask[:, None]
            ).sum().div(valid_query_count).detach(),
        }
        if scores is not None:
            diagnostics["global_scores"] = scores.detach()
        return AttentionOutput(self.out(_merge_heads(output)), diagnostics)


class LinformerAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.rank = config.projection_rank
        self.max_length = config.max_length
        self.projection_type = config.projection_type
        self.exact_cls = config.exact_cls
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.key_projection = nn.Parameter(torch.empty(self.rank, self.max_length))
        self.value_projection = nn.Parameter(torch.empty(self.rank, self.max_length))
        nn.init.normal_(self.key_projection, std=1 / math.sqrt(self.max_length))
        nn.init.normal_(self.value_projection, std=1 / math.sqrt(self.max_length))
        self.dropout = config.dropout

    def _matrix(self, parameter: torch.Tensor, length: int, valid: int) -> torch.Tensor:
        if self.projection_type == "fixed":
            matrix = parameter[:, :length]
        elif self.projection_type == "resampled":
            matrix = F.interpolate(
                parameter[None], size=valid, mode="linear", align_corners=False
            )[0]
            target_norm = parameter.norm(dim=-1, keepdim=True)
            matrix = matrix * (target_norm / matrix.norm(dim=-1, keepdim=True).clamp_min(1e-6))
            matrix = F.pad(matrix, (0, length - valid))
        elif self.projection_type == "mean":
            bins = torch.arange(valid, device=parameter.device) * self.rank // max(valid, 1)
            matrix = F.one_hot(bins, self.rank).T.to(parameter.dtype)
            matrix = matrix / matrix.sum(-1, keepdim=True).clamp_min(1)
            matrix = F.pad(matrix, (0, length - valid))
        else:
            raise ValueError(f"Unknown Linformer projection: {self.projection_type}")
        return matrix

    def _project(
        self,
        tensor: torch.Tensor,
        parameter: torch.Tensor,
        mask: torch.Tensor,
        valid_lengths: list[int] | None = None,
    ) -> torch.Tensor:
        length = tensor.shape[2]
        tensor = tensor * mask[:, None, :, None]
        if self.projection_type == "fixed":
            return torch.einsum("kn,bhnd->bhkd", parameter[:, :length], tensor)
        # Transfer all lengths in one synchronization and project the complete
        # batch in one contraction instead of launching one einsum per sample.
        valid_lengths = valid_lengths or [int(value) for value in mask.sum(-1).tolist()]
        matrices = torch.stack([
            self._matrix(parameter, length, int(valid)) for valid in valid_lengths
        ])
        return torch.einsum("bkn,bhnd->bhkd", matrices, tensor)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor, **_: Any) -> AttentionOutput:
        q, k, v = (_split_heads(part, self.heads) for part in self.qkv(x).chunk(3, dim=-1))
        valid_lengths = (
            None
            if self.projection_type == "fixed"
            else [int(value) for value in attention_mask.sum(-1).tolist()]
        )
        projected_k = self._project(k, self.key_projection, attention_mask, valid_lengths)
        projected_v = self._project(v, self.value_projection, attention_mask, valid_lengths)
        output = F.scaled_dot_product_attention(
            q, projected_k, projected_v, dropout_p=self.dropout if self.training else 0.0
        )
        if self.exact_cls:
            cls_output = _masked_sdpa(
                q[:, :, :1], k, v, attention_mask, self.dropout, self.training
            )
            output = torch.cat((cls_output, output[:, :, 1:]), dim=2)
        output = output * attention_mask[:, None, :, None]
        return AttentionOutput(self.out(_merge_heads(output)), {})


class LongShortAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.local_span = config.attention_window
        self.rank = config.projection_rank
        self.segment_size = config.segment_size
        self.total_slots = config.total_summary_slots
        self.slots_per_segment = config.summaries_per_segment
        self.summary_type = config.summary_type
        self.summary_ablation = config.summary_ablation
        self.dual_ln = config.dual_layer_norm
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.dynamic_query = nn.Parameter(torch.randn(self.heads, self.rank, self.head_dim) * 0.02)
        self.segment_query = nn.Parameter(
            torch.randn(self.heads, self.total_slots, self.head_dim) * 0.02
        )
        self.summary_position = nn.Parameter(
            torch.randn(self.heads, self.total_slots, self.head_dim) * 0.02
        )
        self.local_norm = nn.LayerNorm(self.head_dim) if self.dual_ln else nn.Identity()
        self.global_norm = nn.LayerNorm(self.head_dim) if self.dual_ln else nn.Identity()
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def _summaries(
        self, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.summary_type == "disabled":
            empty = k[:, :, :0]
            return empty, empty, mask[:, :0]
        if self.summary_type == "global":
            score = torch.einsum("bhnd,hrd->bhnr", k, self.dynamic_query)
            score = score.masked_fill(~mask[:, None, :, None], -torch.inf)
            weight = torch.softmax(score, dim=2)
            valid = mask.any(-1, keepdim=True).expand(-1, self.rank)
            return (
                torch.einsum("bhnr,bhnd->bhrd", weight, k),
                torch.einsum("bhnr,bhnd->bhrd", weight, v),
                valid,
            )

        batch_keys, batch_values, counts = [], [], []
        real_lengths = [max(1, int(value)) for value in mask.sum(-1).tolist()]
        for batch_index, real_length in enumerate(real_lengths):
            segments = math.ceil(real_length / self.segment_size)
            count_for_row = segments * self.slots_per_segment
            if count_for_row > self.total_slots:
                raise ValueError(
                    f"Segmented summaries need {count_for_row} slots, but only "
                    f"{self.total_slots} were configured"
                )
            summary_keys, summary_values = [], []
            cursor = 0
            for segment in range(segments):
                count = self.slots_per_segment
                start = segment * self.segment_size
                end = min(real_length, start + self.segment_size)
                segment_k = k[batch_index, :, start:end]
                segment_v = v[batch_index, :, start:end]
                queries = self.segment_query[:, cursor : cursor + count]
                score = torch.einsum("hnd,hqd->hnq", segment_k, queries)
                weight = torch.softmax(score, dim=1)
                summary_keys.append(torch.einsum("hnq,hnd->hqd", weight, segment_k))
                summary_values.append(torch.einsum("hnq,hnd->hqd", weight, segment_v))
                cursor += count
            batch_keys.append(torch.cat(summary_keys, dim=1))
            batch_values.append(torch.cat(summary_values, dim=1))
            counts.append(count_for_row)
        max_count = max(counts)
        summary_k = torch.stack([
            F.pad(row, (0, 0, 0, max_count - row.shape[1])) for row in batch_keys
        ])
        summary_v = torch.stack([
            F.pad(row, (0, 0, 0, max_count - row.shape[1])) for row in batch_values
        ])
        summary_valid = (
            torch.arange(max_count, device=mask.device)[None, :]
            < torch.tensor(counts, device=mask.device)[:, None]
        )
        position = self.summary_position
        if self.summary_ablation == "shuffle":
            position = position.roll(shifts=max(1, self.total_slots // 2), dims=1)
        summary_k = summary_k + position[None, :, :max_count]
        return summary_k, summary_v, summary_valid

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor, **_: Any) -> AttentionOutput:
        q, k, v = (_split_heads(part, self.heads) for part in self.qkv(x).chunk(3, dim=-1))
        normalized_k, normalized_v = self.local_norm(k), self.local_norm(v)
        local_k, _ = _segment_windows(normalized_k, self.local_span)
        local_v, _ = _segment_windows(normalized_v, self.local_span)
        summary_k, summary_v, summary_valid = self._summaries(
            normalized_k, normalized_v, attention_mask
        )
        summary_k, summary_v = self.global_norm(summary_k), self.global_norm(summary_v)
        local_score = torch.einsum("bhnd,bhnwd->bhnw", q, local_k) / math.sqrt(self.head_dim)
        local_score = local_score.masked_fill(
            ~_segment_mask(attention_mask, self.local_span)[:, None], -torch.inf
        )
        summary_score = torch.einsum("bhnd,bhkd->bhnk", q, summary_k) / math.sqrt(self.head_dim)
        summary_score = summary_score.masked_fill(~summary_valid[:, None, None, :], -torch.inf)
        weights = self.dropout(torch.softmax(torch.cat((local_score, summary_score), -1), -1))
        local_weight, summary_weight = weights.split((local_score.shape[-1], summary_score.shape[-1]), -1)
        output = torch.einsum("bhnw,bhnwd->bhnd", local_weight, local_v)
        output += torch.einsum("bhnk,bhkd->bhnd", summary_weight, summary_v)
        output *= attention_mask[:, None, :, None]
        valid_query_count = attention_mask.sum().clamp_min(1) * self.heads
        return AttentionOutput(
            self.out(_merge_heads(output)),
            {"summary_attention_mass": (
                summary_weight.sum(-1) * attention_mask[:, None]
            ).sum().div(valid_query_count).detach()},
        )


class MemoryAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.num_heads
        self.q = nn.Linear(config.hidden_size, config.hidden_size)
        self.kv = nn.Linear(config.hidden_size, 2 * config.hidden_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = config.dropout

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        memory: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        compressed: torch.Tensor | None = None,
        compressed_mask: torch.Tensor | None = None,
        include_current: bool = True,
        **_: Any,
    ) -> AttentionOutput:
        candidates = (
            ((compressed, compressed_mask), (memory, memory_mask), (x, attention_mask))
            if include_current
            else ((compressed, compressed_mask), (memory, memory_mask))
        )
        sources = [
            (item, item_mask)
            for item, item_mask in candidates
            if item is not None and item.shape[1]
        ]
        fallback_current = not sources
        if fallback_current:  # short-document safety for the current-segment ablation
            sources = [(x, attention_mask)]
        context = torch.cat([item for item, _ in sources], dim=1)
        context_mask = torch.cat([
            item_mask if item_mask is not None else torch.ones(
                (x.shape[0], item.shape[1]), dtype=torch.bool, device=x.device
            )
            for item, item_mask in sources
        ], dim=1)
        current_in_context = include_current or fallback_current
        prefix = context.shape[1] - (x.shape[1] if current_in_context else 0)
        q = _split_heads(self.q(x), self.heads)
        k, v = (_split_heads(part, self.heads) for part in self.kv(context).chunk(2, -1))
        output = _masked_sdpa(q, k, v, context_mask, self.dropout, self.training)
        output = output * attention_mask[:, None, :, None]
        prefix_mask = context_mask[:, :prefix]
        diagnostics = {
            "memory_tokens": prefix_mask.sum(-1).float().mean()
            if prefix else torch.tensor(0.0, device=x.device)
        }
        return AttentionOutput(self.out(_merge_heads(output)), diagnostics)
