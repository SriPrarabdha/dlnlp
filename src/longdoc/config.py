from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    architecture: str = "exact"
    vocab_size: int = 32_000
    hidden_size: int = 512
    intermediate_size: int = 2_048
    num_layers: int = 4
    num_heads: int = 8
    max_length: int = 4_096
    dropout: float = 0.1
    attention_window: int = 256
    projection_rank: int = 64
    projection_type: str = "fixed"
    exact_cls: bool = False
    position_encoding: str = "learned"
    rope_base: float = 10_000.0
    rope_gated_bands: bool = False
    rope_ablate_band: int | None = None
    global_selection: str = "cls"
    global_stride: int = 256
    separate_global_projections: bool = False
    sentence_boundary_token_ids: list[int] = field(default_factory=list)
    summaries_per_segment: int = 2
    total_summary_slots: int = 16
    summary_type: str = "global"
    summary_ablation: str = "none"
    dual_layer_norm: bool = True
    segment_size: int = 512
    memory_size: int = 512
    compressed_memory_size: int = 128
    compression_rate: int = 4
    compressor: str = "uniform"
    detach_memory: bool = True
    memory_ablation: str = "none"

    def validate(self) -> None:
        if self.hidden_size <= 0 or self.intermediate_size <= 0:
            raise ValueError("hidden_size and intermediate_size must be positive")
        if self.num_layers <= 0 or self.num_heads <= 0:
            raise ValueError("num_layers and num_heads must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.architecture == "roformer" and (self.hidden_size // self.num_heads) % 2:
            raise ValueError("RoFormer requires an even attention head dimension")
        if self.max_length < 2:
            raise ValueError("max_length must be at least 2")
        if self.vocab_size <= 5:
            raise ValueError("vocab_size must leave room for special and content tokens")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.attention_window <= 0:
            raise ValueError("attention_window must be positive")
        if self.projection_rank <= 0:
            raise ValueError("projection_rank must be positive")
        if self.segment_size <= 0:
            raise ValueError("segment_size must be positive")
        if self.memory_size <= 0 or self.compressed_memory_size <= 0:
            raise ValueError("memory sizes must be positive")
        if self.compression_rate <= 0:
            raise ValueError("compression_rate must be positive")
        required_compressed_slots = math.ceil(self.segment_size / self.compression_rate)
        if (
            self.architecture == "compressive"
            and self.compressed_memory_size < required_compressed_slots
        ):
            raise ValueError(
                "compressed_memory_size must hold one fully compressed segment "
                f"(need at least {required_compressed_slots})"
            )
        if self.global_stride <= 0:
            raise ValueError("global_stride must be positive")
        if self.summaries_per_segment <= 0 or self.total_summary_slots <= 0:
            raise ValueError("summary slot counts must be positive")
        required_summary_slots = math.ceil(self.max_length / self.segment_size) * self.summaries_per_segment
        if self.summary_type == "segmented" and self.total_summary_slots < required_summary_slots:
            raise ValueError(
                "total_summary_slots must cover summaries_per_segment at model.max_length "
                f"(need at least {required_summary_slots})"
            )
        if self.rope_base <= 0:
            raise ValueError("rope_base must be positive")
        if self.rope_ablate_band not in {None, 0, 1, 2}:
            raise ValueError("rope_ablate_band must be null, 0, 1, or 2")
        if self.projection_type not in {"fixed", "resampled", "mean"}:
            raise ValueError(f"Unknown Linformer projection: {self.projection_type}")
        if self.global_selection not in {"cls", "learned", "lowest", "random", "even", "sentence"}:
            raise ValueError(f"Unknown global selection: {self.global_selection}")
        if self.summary_type not in {"disabled", "global", "segmented"}:
            raise ValueError(f"Unknown summary type: {self.summary_type}")
        if self.summary_ablation not in {"none", "shuffle"}:
            raise ValueError(f"Unknown summary ablation: {self.summary_ablation}")
        if self.compressor not in {"uniform", "salient", "dual"}:
            raise ValueError(f"Unknown compressor: {self.compressor}")
        if self.memory_ablation not in {
            "none", "recent", "regular", "compressed", "coverage", "uniform",
            "salience", "salient", "current",
        }:
            raise ValueError(f"Unknown memory ablation: {self.memory_ablation}")
        if self.position_encoding not in {"learned", "rope"}:
            raise ValueError(f"Unknown position encoding: {self.position_encoding}")
        if self.architecture not in {
            "exact", "longformer", "linformer", "longshort", "compressive", "roformer"
        }:
            raise ValueError(f"Unknown architecture: {self.architecture}")


@dataclass
class TrainConfig:
    lengths: list[int] = field(default_factory=lambda: [512, 1_024, 2_048, 4_096])
    mlm_tokens: int = 200_000_000
    batch_tokens: int = 16_384
    eval_batch_tokens: int = 32_768
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    epochs: int = 3
    max_steps: int | None = None
    gradient_checkpointing: bool = True
    compile: bool = False
    num_workers: int = 4
    log_every: int = 20
    save_every: int = 1_000
    checkpoint_tokens: list[int] = field(
        default_factory=lambda: [25_000_000, 50_000_000, 100_000_000, 200_000_000]
    )
    amp: str = "auto"
    finetune_seeds: list[int] = field(default_factory=lambda: [17, 29, 43])
    mask_probability: float = 0.15
    finetune_train_length: int | None = None
    finetune_validation_length: int | None = None
    finetune_eval_length: int | None = None


@dataclass
class MonitoringConfig:
    enabled: bool = True
    project: str = "longdoc-five-paper"
    entity: str | None = None
    mode: str = "online"
    required: bool = False
    log_every_seconds: float = 15.0
    tags: list[str] = field(default_factory=list)


@dataclass
class RuntimeConfig:
    resume: str = "auto"
    checkpoint_every_minutes: float = 15.0
    keep_last_checkpoints: int = 3
    heartbeat_seconds: float = 60.0
    min_free_disk_gb: float = 2.0
    fail_fast: bool = False


@dataclass
class RunConfig:
    seed: int = 17
    data_dir: str = "./data"
    output_dir: str = "./runs"
    experiment: str = "exact.baseline"
    task: str = "arxiv"
    stage: str = "finetune"
    checkpoint: str | None = None
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> None:
        self.model.validate()
        if not self.train.lengths or any(length < 2 for length in self.train.lengths):
            raise ValueError("train.lengths must contain lengths of at least 2")
        if any(length > self.model.max_length for length in self.train.lengths):
            raise ValueError("All training lengths must be <= model.max_length")
        for name in ("finetune_train_length", "finetune_validation_length", "finetune_eval_length"):
            value = getattr(self.train, name)
            if value is not None and not 2 <= value <= self.model.max_length:
                raise ValueError(f"train.{name} must be between 2 and model.max_length")
        if self.train.mlm_tokens <= 0:
            raise ValueError("train.mlm_tokens must be positive")
        if self.train.batch_tokens <= 0 or self.train.eval_batch_tokens <= 0:
            raise ValueError("training and evaluation token budgets must be positive")
        if self.train.learning_rate <= 0 or self.train.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if not 0 <= self.train.warmup_ratio <= 1:
            raise ValueError("warmup_ratio must be in [0, 1]")
        if self.train.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.train.max_steps is not None and self.train.max_steps <= 0:
            raise ValueError("max_steps must be positive when set")
        if self.train.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.train.log_every <= 0 or self.train.save_every <= 0:
            raise ValueError("log_every and save_every must be positive")
        if self.train.amp not in {"auto", "none", "fp16", "bf16"}:
            raise ValueError("amp must be auto, none, fp16, or bf16")
        if not 0 < self.train.mask_probability <= 1:
            raise ValueError("mask_probability must be in (0, 1]")
        if not self.train.finetune_seeds:
            raise ValueError("finetune_seeds cannot be empty")
        if len(set(self.train.finetune_seeds)) != len(self.train.finetune_seeds):
            raise ValueError("finetune_seeds must be unique")
        if any(value <= 0 for value in self.train.checkpoint_tokens):
            raise ValueError("checkpoint token milestones must be positive")
        if self.runtime.resume not in {"auto", "never", "required"}:
            raise ValueError("runtime.resume must be auto, never, or required")
        if self.monitoring.mode not in {"online", "offline", "disabled"}:
            raise ValueError("monitoring.mode must be online, offline, or disabled")
        if self.monitoring.log_every_seconds <= 0:
            raise ValueError("monitoring.log_every_seconds must be positive")
        if self.runtime.checkpoint_every_minutes <= 0:
            raise ValueError("runtime.checkpoint_every_minutes must be positive")
        if self.runtime.keep_last_checkpoints < 0:
            raise ValueError("runtime.keep_last_checkpoints cannot be negative")
        if self.runtime.min_free_disk_gb < 0:
            raise ValueError("runtime.min_free_disk_gb cannot be negative")
        if self.runtime.heartbeat_seconds <= 0:
            raise ValueError("runtime.heartbeat_seconds must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        temporary.replace(path)


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _set_dotted(config: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"Override must be key=value, got: {expression}")
    dotted, raw = expression.split("=", 1)
    value = yaml.safe_load(raw)
    cursor = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> RunConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    for expression in overrides or []:
        _set_dotted(raw, expression)
    known = {field.name for field in dataclasses.fields(RunConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown top-level config fields: {sorted(unknown)}")
    raw["model"] = ModelConfig(**raw.get("model", {}))
    raw["train"] = TrainConfig(**raw.get("train", {}))
    raw["monitoring"] = MonitoringConfig(**raw.get("monitoring", {}))
    raw["runtime"] = RuntimeConfig(**raw.get("runtime", {}))
    config = RunConfig(**raw)
    config.validate()
    return config


def stable_config_id(config: RunConfig) -> str:
    import hashlib

    identity = config.to_dict()
    # Operational choices must not fork scientific identity or make a preempted
    # server job impossible to find when it is resumed with stricter flags.
    for key in ("output_dir", "stage", "monitoring", "runtime"):
        identity.pop(key, None)
    identity.get("train", {}).pop("gradient_checkpointing", None)
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
