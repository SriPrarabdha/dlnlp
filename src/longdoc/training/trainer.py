from __future__ import annotations

import json
import math
import os
import shutil
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from longdoc.config import RunConfig
from longdoc.data.batching import PooledMLMBatches, task_dataloader
from longdoc.data.catalog import get_dataset_info
from longdoc.data.tokenizer import configure_token_dependent_fields
from longdoc.models.modeling import build_model, load_encoder_checkpoint

from .metrics import (
    bootstrap_confidence_intervals,
    compute_metrics,
    evidence_breakdown,
    length_bucket_metrics,
    stress_breakdown,
)
from .operations import (
    ETA,
    GracefulStop,
    SignalController,
    atomic_write_json,
    format_duration,
    tracking_run,
    write_heartbeat,
)
from .runtime import (
    amp_dtype,
    atomic_hardlink,
    atomic_torch_save,
    autocast_context,
    capture_rng_state,
    configure_accelerator,
    move_batch,
    resolve_device,
    restore_rng_state,
    seed_everything,
)


def _optimizer(model: torch.nn.Module, config: RunConfig) -> AdamW:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if parameter.ndim < 2 or name.endswith("bias") else decay).append(parameter)
    return AdamW(
        [{"params": decay, "weight_decay": config.train.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=config.train.learning_rate,
        fused=next(model.parameters()).is_cuda,
    )


def _scheduler(optimizer: AdamW, total_steps: int, warmup_ratio: float) -> LambdaLR:
    warmup = int(total_steps * warmup_ratio)

    def schedule(step: int) -> float:
        if warmup and step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, schedule)


def _scaler(device: torch.device, dtype: torch.dtype | None):
    enabled = device.type == "cuda" and dtype == torch.float16
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # PyTorch 2.2 compatibility
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _step(
    model: torch.nn.Module,
    batch: dict[str, Any],
    optimizer: AdamW,
    scaler: Any,
    device: torch.device,
    dtype: torch.dtype | None,
) -> float:
    batch = move_batch(batch, device)
    full_size = batch["input_ids"].shape[0]
    micro_size = full_size
    while True:
        optimizer.zero_grad(set_to_none=True)
        weighted_loss = 0.0
        try:
            for start in range(0, full_size, micro_size):
                end = min(full_size, start + micro_size)
                with autocast_context(device, dtype):
                    output = model(
                        batch["input_ids"][start:end],
                        batch["attention_mask"][start:end],
                        batch["labels"][start:end],
                    )
                    if output.loss is None or not torch.isfinite(output.loss):
                        raise FloatingPointError(f"Non-finite training loss: {output.loss}")
                    scaled_loss = output.loss * ((end - start) / full_size)
                scaler.scale(scaled_loss).backward()
                weighted_loss += float(output.loss.detach()) * (end - start)
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm) and not scaler.is_enabled():
                raise FloatingPointError(f"Non-finite gradient norm: {gradient_norm}")
            scaler.step(optimizer)
            scaler.update()
            return weighted_loss / full_size
        except torch.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            if device.type != "cuda" or micro_size == 1:
                raise
            micro_size = max(1, micro_size // 2)
            output = None
            scaled_loss = None
            torch.cuda.empty_cache()
            print(f"CUDA OOM: retrying logical batch with micro-batch size {micro_size}", flush=True)


def _checkpoint_payload(
    model: torch.nn.Module,
    config: RunConfig,
    step: int,
    processed_tokens: int | None = None,
    optimizer: AdamW | None = None,
    scheduler: LambdaLR | None = None,
    scaler: Any | None = None,
    data_state: dict[str, Any] | None = None,
    epoch: int = 0,
    batch_in_epoch: int = 0,
    best_score: float = -math.inf,
    status: str = "running",
    training_seconds: float = 0.0,
) -> dict[str, Any]:
    unwrapped = getattr(model, "_orig_mod", model)
    payload = {
        "format_version": 2,
        "model": unwrapped.state_dict(), "config": config.to_dict(), "step": step,
        "processed_tokens": processed_tokens, "epoch": epoch,
        "batch_in_epoch": batch_in_epoch, "best_score": best_score,
        "status": status, "training_seconds": training_seconds,
        "rng_state": capture_rng_state(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if data_state is not None:
        payload["data_state"] = data_state
    return payload


def _validate_saved_config(saved_config: dict[str, Any], config: RunConfig, source: Path) -> None:
    saved_identity = dict(saved_config)
    current_identity = config.to_dict()
    # These fields may safely change when relocating or operating a run;
    # scientific/model/training fields may not.
    for key in ("output_dir", "stage", "monitoring", "runtime"):
        saved_identity.pop(key, None)
        current_identity.pop(key, None)
    if saved_identity != current_identity:
        raise ValueError(
            f"Saved configuration does not match the current run: {source}. "
            "Use the original configuration or a new output directory."
        )


def _resume_checkpoint(config: RunConfig, run_dir: Path) -> Path | None:
    if config.runtime.resume == "never":
        return None
    final = run_dir / "checkpoint_last.pt"
    if final.exists():
        try:
            state = torch.load(final, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, ValueError, EOFError):
            state = None  # A corrupt final falls back to the atomic latest checkpoint.
        if state is not None:
            if state.get("config") is not None:
                _validate_saved_config(state["config"], config, final)
            if state.get("status") in {None, "completed"}:
                return final
    latest = run_dir / "checkpoint_latest.pt"
    candidate = latest if latest.exists() else None
    if config.runtime.resume == "required" and candidate is None:
        raise FileNotFoundError(f"Resume required, but no checkpoint exists in {run_dir}")
    return candidate


def _restore_training_state(
    checkpoint: Path,
    model: torch.nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: Any,
    config: RunConfig,
) -> dict[str, Any]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_config = state.get("config")
    if saved_config is not None:
        _validate_saved_config(saved_config, config, checkpoint)
    getattr(model, "_orig_mod", model).load_state_dict(state["model"])
    if "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    restore_rng_state(state.get("rng_state"))
    return state


def _prune_checkpoints(run_dir: Path, keep: int) -> None:
    paths = sorted(
        run_dir.glob("checkpoint_step_*.pt"), key=lambda path: int(path.stem.rsplit("_", 1)[1])
    )
    stale = paths[:-keep] if keep > 0 else paths
    for path in stale:
        path.unlink(missing_ok=True)


def _ensure_checkpoint_space(config: RunConfig, path: Path) -> None:
    free = shutil.disk_usage(path).free
    required = int(config.runtime.min_free_disk_gb * 1024 ** 3)
    if free < required:
        raise OSError(
            f"Checkpoint paused: {free / 1024 ** 3:.2f} GiB free at {path}; "
            f"runtime.min_free_disk_gb={config.runtime.min_free_disk_gb}"
        )


def pretrain(
    config: RunConfig,
    run_dir: Path,
    device_name: str = "auto",
    tracking_group: str | None = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    completed = _resume_checkpoint(config, run_dir)
    if completed == run_dir / "checkpoint_last.pt":
        return completed
    seed_everything(config.seed)
    device = resolve_device(device_name)
    configure_accelerator(device)
    dtype = amp_dtype(device, config.train.amp)
    configure_token_dependent_fields(config.model, config.data_dir)
    model = build_model(config.model, "pretrain").to(device)
    model.encoder.gradient_checkpointing = config.train.gradient_checkpointing
    if config.train.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    batches = PooledMLMBatches(
        config.data_dir, config.train.lengths, config.train.batch_tokens,
        config.train.mlm_tokens, config.seed, config.train.mask_probability,
        config.train.checkpoint_tokens,
    )
    total_steps = batches.planned_steps(config.train.max_steps)
    optimizer = _optimizer(model, config)
    scheduler = _scheduler(optimizer, total_steps, config.train.warmup_ratio)
    scaler = _scaler(device, dtype)
    config.save(run_dir / "config.yaml")
    step = 0
    processed_tokens = 0
    prior_training_seconds = 0.0
    if completed is not None:
        state = _restore_training_state(completed, model, optimizer, scheduler, scaler, config)
        step = int(state.get("step", 0))
        processed_tokens = int(state.get("processed_tokens", 0))
        prior_training_seconds = float(state.get("training_seconds", 0.0))
        if state.get("data_state"):
            batches.load_state_dict(state["data_state"])
    milestone_values = sorted(set(config.train.checkpoint_tokens))
    milestone_index = sum(value <= processed_tokens for value in milestone_values)
    start = time.perf_counter()
    session_start_tokens = processed_tokens
    last_checkpoint = time.monotonic()
    last_heartbeat = 0.0
    eta = ETA(total_steps, step)
    group = tracking_group or f"standalone/{config.experiment}"
    with tracking_run(config, run_dir, f"{config.experiment}-pretrain", group, "pretrain") as (logger, tracker, events):
        if completed:
            logger.info("Resumed pretraining from %s at step %d (%d tokens)", completed, step, processed_tokens)
        events.write("pretrain_started", step=step, tokens=processed_tokens, total_steps=total_steps)
        progress = tqdm(total=total_steps, initial=step, desc="MLM", dynamic_ncols=True)
        with SignalController() as signals:
            batch_iterator = iter(batches)
            while step < total_steps:
                batch = next(batch_iterator)
                step += 1
                loss = _step(model, batch, optimizer, scaler, device, dtype)
                processed_tokens = batches.processed
                scheduler.step()
                progress.update(1)
                eta_state = eta.update(step)
                elapsed = time.perf_counter() - start
                if step % config.train.log_every == 0 or step == 1:
                    cumulative_seconds = prior_training_seconds + elapsed
                    record = {
                        "step": step, "loss": loss, "tokens": processed_tokens,
                        "training_seconds": cumulative_seconds,
                        "gpu_hours": cumulative_seconds / 3_600,
                        "gpu_hours_this_session": elapsed / 3_600, "seconds_this_session": elapsed,
                        "tokens_per_second": (processed_tokens - session_start_tokens) / max(1e-9, elapsed),
                        "eta_seconds": eta_state["eta_seconds"],
                        "learning_rate": scheduler.get_last_lr()[0],
                    }
                    progress.set_postfix(loss=f"{loss:.4f}", eta=format_duration(eta_state["eta_seconds"]))
                    with (run_dir / "train.jsonl").open("a") as handle:
                        handle.write(json.dumps(record) + "\n")
                    logger.info(
                        "MLM step %d/%d loss=%.4f tokens=%d ETA=%s",
                        step, total_steps, loss, processed_tokens,
                        format_duration(eta_state["eta_seconds"]),
                    )
                    tracker.log({
                        "train/step": step, "train/tokens": processed_tokens,
                        "pretrain/loss": loss, "pretrain/tokens_per_second": record["tokens_per_second"],
                        "pretrain/eta_seconds": eta_state["eta_seconds"],
                        "pretrain/learning_rate": record["learning_rate"],
                        "system/gpu_memory_allocated_bytes": (
                            int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0
                        ),
                    })
                now = time.monotonic()
                timed_save = now - last_checkpoint >= config.runtime.checkpoint_every_minutes * 60
                if step % config.train.save_every == 0 or timed_save or signals.requested:
                    _ensure_checkpoint_space(config, run_dir)
                    payload = _checkpoint_payload(
                        model, config, step, processed_tokens, optimizer, scheduler, scaler,
                        batches.state_dict(), status="interrupted" if signals.requested else "running",
                        training_seconds=prior_training_seconds + elapsed,
                    )
                    step_path = run_dir / f"checkpoint_step_{step}.pt"
                    atomic_torch_save(payload, step_path)
                    atomic_hardlink(step_path, run_dir / "checkpoint_latest.pt")
                    _prune_checkpoints(run_dir, config.runtime.keep_last_checkpoints)
                    last_checkpoint = now
                    events.write("checkpoint_saved", step=step, tokens=processed_tokens)
                while milestone_index < len(milestone_values) and processed_tokens >= milestone_values[milestone_index]:
                    _ensure_checkpoint_space(config, run_dir)
                    target = milestone_values[milestone_index]
                    payload = _checkpoint_payload(
                        model, config, step, processed_tokens, optimizer, scheduler, scaler,
                        batches.state_dict(),
                        status="interrupted" if signals.requested else "milestone",
                        training_seconds=prior_training_seconds + elapsed,
                    )
                    atomic_torch_save(payload, run_dir / f"checkpoint_tokens_{target}.pt")
                    atomic_hardlink(
                        run_dir / f"checkpoint_tokens_{target}.pt",
                        run_dir / "checkpoint_latest.pt",
                    )
                    metadata = {
                        "target_tokens": target, "processed_tokens": processed_tokens,
                        "step": step,
                        "training_seconds": prior_training_seconds + elapsed,
                        "gpu_hours": (prior_training_seconds + elapsed) / 3_600,
                        "seconds_this_session": elapsed,
                        "gpu_hours_this_session": elapsed / 3_600,
                    }
                    atomic_write_json(run_dir / f"checkpoint_tokens_{target}.json", metadata)
                    milestone_index += 1
                if now - last_heartbeat >= config.runtime.heartbeat_seconds:
                    write_heartbeat(
                        run_dir / "heartbeat.json", stage="pretrain", step=step,
                        total_steps=total_steps, tokens=processed_tokens,
                        eta_seconds=eta_state["eta_seconds"],
                    )
                    last_heartbeat = now
                if signals.requested:
                    progress.close()
                    logger.warning("Received %s; checkpoint saved", signals.signal_name)
                    raise GracefulStop(
                        f"Pretraining interrupted by {signals.signal_name}",
                        preempting=signals.signal_name in {"SIGTERM", "SIGUSR1"},
                    )
        progress.close()
        output = run_dir / "checkpoint_last.pt"
        _ensure_checkpoint_space(config, run_dir)
        final_payload = _checkpoint_payload(
            model, config, step, processed_tokens, optimizer, scheduler, scaler,
            batches.state_dict(), status="completed",
            training_seconds=prior_training_seconds + (time.perf_counter() - start),
        )
        atomic_torch_save(final_payload, output)
        atomic_hardlink(output, run_dir / "checkpoint_latest.pt")
        atomic_write_json(run_dir / "completion.json", {
            "status": "completed", "step": step, "processed_tokens": processed_tokens,
            "completed_at": time.time(),
        })
        write_heartbeat(
            run_dir / "heartbeat.json", stage="pretrain", status="completed",
            step=step, total_steps=total_steps, tokens=processed_tokens, eta_seconds=0,
        )
        tracker.summary({"final_step": step, "processed_tokens": processed_tokens})
        events.write("pretrain_completed", step=step, tokens=processed_tokens)
        return output


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: Iterable[dict[str, Any]],
    task_type: str,
    device: torch.device,
    dtype: torch.dtype | None,
) -> tuple[dict[str, float], dict[str, Any]]:
    model.eval()
    logits, labels, ids, positions = [], [], [], []
    coverage, source_lengths, challenges, distractor_counts = [], [], [], []
    diagnostic_values: dict[str, list[float]] = {}
    for batch in tqdm(loader, desc="eval", leave=False):
        ids.extend(batch["ids"])
        positions.extend(batch["positions"])
        coverage.extend(batch["coverage"].tolist())
        source_lengths.extend(batch["source_lengths"])
        challenges.extend(batch["challenges"])
        distractor_counts.extend(batch["distractor_counts"])
        moved = move_batch(batch, device)
        with autocast_context(device, dtype):
            output = model(moved["input_ids"], moved["attention_mask"])
        for key, value in output.diagnostics.items():
            if value.numel() == 1:
                diagnostic_values.setdefault(key, []).append(float(value.detach().cpu()))
        logits.append(output.logits.float().cpu().numpy())
        labels.append(batch["labels"].numpy())
    logits_array = np.concatenate(logits)
    labels_array = np.concatenate(labels)
    metrics = compute_metrics(logits_array, labels_array, task_type)
    metrics["retained_token_coverage"] = float(np.mean(coverage))
    metrics["length_buckets"] = length_bucket_metrics(
        logits_array, labels_array, source_lengths, task_type
    )
    for key, values in diagnostic_values.items():
        metrics[f"diagnostic/{key}"] = float(np.mean(values))
    raw = {
        "ids": ids,
        "logits": logits_array,
        "labels": labels_array,
        "positions": positions,
        "coverage": coverage,
        "source_lengths": source_lengths,
        "challenges": challenges,
        "distractor_counts": distractor_counts,
    }
    return metrics, raw


def _task_info(task: str) -> tuple[str, int]:
    if task.startswith(("synthetic_", "stress_")):
        return "binary", 2
    info = get_dataset_info(task)
    return info.task_type, info.num_labels


def finetune(
    config: RunConfig,
    run_dir: Path,
    encoder_checkpoint: Path,
    device_name: str = "auto",
    tracking_group: str | None = None,
) -> list[Path]:
    task_type, num_labels = _task_info(config.task)
    configure_token_dependent_fields(config.model, config.data_dir)
    outputs = []
    for seed in config.train.finetune_seeds:
        seed_dir = run_dir / config.task / f"seed_{seed}"
        best_path = seed_dir / "checkpoint_best.pt"
        completion_path = seed_dir / "completion.json"
        saved_config_path = seed_dir / "config.yaml"
        completion_valid = False
        if completion_path.exists() and saved_config_path.exists():
            try:
                completion_valid = (
                    json.loads(completion_path.read_text()).get("status") == "completed"
                )
                if completion_valid:
                    _validate_saved_config(
                        yaml.safe_load(saved_config_path.read_text()), config, saved_config_path
                    )
            except (json.JSONDecodeError, OSError, yaml.YAMLError):
                completion_valid = False
        if (
            completion_valid
            and (seed_dir / "test_metrics.json").exists()
            and (seed_dir / "test_predictions.npz").exists()
            and best_path.exists()
            and config.runtime.resume != "never"
        ):
            outputs.append(best_path)
            continue
        seed_everything(seed)
        device = resolve_device(device_name)
        configure_accelerator(device)
        dtype = amp_dtype(device, config.train.amp)
        model = build_model(config.model, "finetune", num_labels, task_type)
        missing, unexpected = load_encoder_checkpoint(model, str(encoder_checkpoint))
        allowed_missing = ("scorer", "band_logits")
        bad_missing = [name for name in missing if not any(token in name for token in allowed_missing)]
        if unexpected or bad_missing:
            raise RuntimeError(f"Checkpoint mismatch; missing={bad_missing}, unexpected={unexpected}")
        model.encoder.gradient_checkpointing = config.train.gradient_checkpointing
        model.to(device)
        if config.train.compile and hasattr(torch, "compile"):
            model = torch.compile(model)
        extrapolation = config.task == "stress_retrieve_extrapolation"
        if extrapolation and config.model.max_length < 4_096:
            raise ValueError("stress_retrieve_extrapolation requires model.max_length >= 4096")
        train_length = config.train.finetune_train_length or (1_024 if extrapolation else config.model.max_length)
        validation_length = config.train.finetune_validation_length or train_length
        eval_length = config.train.finetune_eval_length or config.model.max_length
        is_stress = config.task.startswith(("synthetic_", "stress_"))
        shared_lengths = [length for length in (1_024, 2_048, 4_096) if length <= train_length] or [train_length]
        train_synthetic_lengths = (
            [train_length]
            if extrapolation or config.train.finetune_train_length is not None
            else shared_lengths
        )
        validation_synthetic_lengths = (
            [validation_length]
            if extrapolation or config.train.finetune_validation_length is not None
            else ([length for length in (1_024, 2_048, 4_096) if length <= validation_length]
                  or [validation_length])
        )
        train_loader = task_dataloader(
            config.data_dir, config.task, "train", train_length,
            config.train.batch_tokens, config.train.num_workers, seed, True,
            synthetic_lengths=train_synthetic_lengths if is_stress else None,
            dataset_seed=config.seed,
        )
        validation_loader = task_dataloader(
            config.data_dir, config.task, "validation", validation_length,
            config.train.eval_batch_tokens, config.train.num_workers, seed, False,
            synthetic_lengths=validation_synthetic_lengths if is_stress else None,
            dataset_seed=config.seed,
        )
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * config.train.epochs
        if config.train.max_steps is not None:
            total_steps = min(total_steps, config.train.max_steps)
        optimizer = _optimizer(model, config)
        scheduler = _scheduler(optimizer, total_steps, config.train.warmup_ratio)
        scaler = _scaler(device, dtype)
        seed_dir.mkdir(parents=True, exist_ok=True)
        config.save(seed_dir / "config.yaml")
        best_score = -math.inf
        global_step = 0
        start_epoch = 0
        start_batch = 0
        resume_path = _resume_checkpoint(config, seed_dir)
        if resume_path is not None:
            state = _restore_training_state(
                resume_path, model, optimizer, scheduler, scaler, config
            )
            global_step = int(state.get("step", 0))
            start_epoch = int(state.get("epoch", 0))
            start_batch = int(state.get("batch_in_epoch", 0))
            best_score = float(state.get("best_score", -math.inf))
            if state.get("status") == "completed":
                start_epoch = config.train.epochs
                start_batch = 0
            if start_epoch >= config.train.epochs and not best_path.exists():
                atomic_hardlink(resume_path, best_path)
        group = tracking_group or f"standalone/{config.experiment}"
        run_name = f"{config.experiment}-{config.task}-seed-{seed}"
        with tracking_run(config, seed_dir, run_name, group, "finetune") as (logger, tracker, events):
            if resume_path:
                logger.info(
                    "Resumed fine-tuning from %s at epoch %d batch %d step %d",
                    resume_path, start_epoch + 1, start_batch, global_step,
                )
            events.write(
                "finetune_started", task=config.task, seed=seed, step=global_step,
                epoch=start_epoch, batch=start_batch,
            )
            eta = ETA(total_steps, global_step)
            last_checkpoint = time.monotonic()
            last_heartbeat = 0.0
            loss: float | None = None
            with SignalController() as signals:
                for epoch in range(start_epoch, config.train.epochs):
                    if hasattr(train_loader.batch_sampler, "set_epoch"):
                        train_loader.batch_sampler.set_epoch(epoch)
                    epoch_start_batch = start_batch if epoch == start_epoch else 0
                    sampler_handles_offset = hasattr(train_loader.batch_sampler, "set_start_batch")
                    if sampler_handles_offset:
                        train_loader.batch_sampler.set_start_batch(epoch_start_batch)
                    model.train()
                    progress_total = min(
                        len(train_loader),
                        epoch_start_batch + max(0, total_steps - global_step),
                    )
                    progress = tqdm(
                        total=progress_total, initial=epoch_start_batch,
                        desc=f"seed {seed} epoch {epoch + 1}", dynamic_ncols=True,
                    )
                    enumerate_start = epoch_start_batch if sampler_handles_offset else 0
                    batch_source = train_loader if global_step < total_steps else ()
                    for batch_index, batch in enumerate(batch_source, start=enumerate_start):
                        if not sampler_handles_offset and batch_index < epoch_start_batch:
                            continue
                        loss = _step(model, batch, optimizer, scaler, device, dtype)
                        scheduler.step()
                        global_step += 1
                        progress.update(1)
                        eta_state = eta.update(global_step)
                        if global_step % config.train.log_every == 0 or global_step == 1:
                            progress.set_postfix(loss=f"{loss:.4f}", eta=format_duration(eta_state["eta_seconds"]))
                            record = {
                                "step": global_step, "epoch": epoch + 1, "batch": batch_index + 1,
                                "loss": loss, "eta_seconds": eta_state["eta_seconds"],
                                "learning_rate": scheduler.get_last_lr()[0],
                            }
                            with (seed_dir / "train.jsonl").open("a") as handle:
                                handle.write(json.dumps(record) + "\n")
                            logger.info(
                                "task=%s seed=%d step=%d/%d loss=%.4f ETA=%s",
                                config.task, seed, global_step, total_steps, loss,
                                format_duration(eta_state["eta_seconds"]),
                            )
                            tracker.log({
                                "train/step": global_step, "train/loss": loss,
                                "train/eta_seconds": eta_state["eta_seconds"],
                                "train/learning_rate": record["learning_rate"],
                                "system/gpu_memory_allocated_bytes": (
                                    int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0
                                ),
                            })
                        now = time.monotonic()
                        timed_save = now - last_checkpoint >= config.runtime.checkpoint_every_minutes * 60
                        if global_step % config.train.save_every == 0 or timed_save or signals.requested:
                            _ensure_checkpoint_space(config, seed_dir)
                            payload = _checkpoint_payload(
                                model, config, global_step, optimizer=optimizer,
                                scheduler=scheduler, scaler=scaler, epoch=epoch,
                                batch_in_epoch=batch_index + 1, best_score=best_score,
                                status="interrupted" if signals.requested else "running",
                            )
                            step_path = seed_dir / f"checkpoint_step_{global_step}.pt"
                            atomic_torch_save(payload, step_path)
                            atomic_hardlink(step_path, seed_dir / "checkpoint_latest.pt")
                            _prune_checkpoints(seed_dir, config.runtime.keep_last_checkpoints)
                            last_checkpoint = now
                            events.write("checkpoint_saved", step=global_step, epoch=epoch, batch=batch_index + 1)
                        if now - last_heartbeat >= config.runtime.heartbeat_seconds:
                            write_heartbeat(
                                seed_dir / "heartbeat.json", stage="finetune", task=config.task,
                                seed=seed, step=global_step, total_steps=total_steps,
                                eta_seconds=eta_state["eta_seconds"],
                            )
                            last_heartbeat = now
                        if signals.requested:
                            progress.close()
                            logger.warning("Received %s; checkpoint saved", signals.signal_name)
                            raise GracefulStop(
                                f"Fine-tuning interrupted by {signals.signal_name}",
                                preempting=signals.signal_name in {"SIGTERM", "SIGUSR1"},
                            )
                        if global_step >= total_steps:
                            break
                    progress.close()
                    metrics, _ = evaluate(model, validation_loader, task_type, device, dtype)
                    score = metrics.get("micro_f1", metrics.get("accuracy", -math.inf))
                    validation_record = {
                        "epoch": epoch + 1, "step": global_step, "loss": loss, **metrics
                    }
                    with (seed_dir / "validation.jsonl").open("a") as handle:
                        handle.write(json.dumps(validation_record) + "\n")
                    tracker.log({
                        "epoch": epoch + 1,
                        **{f"validation/{key}": value for key, value in metrics.items()
                           if isinstance(value, (int, float))},
                    }, force=True)
                    improved = score > best_score
                    if improved:
                        best_score = score
                        _ensure_checkpoint_space(config, seed_dir)
                        atomic_torch_save(
                            _checkpoint_payload(
                                model, config, global_step, optimizer=optimizer,
                                scheduler=scheduler, scaler=scaler, epoch=epoch + 1,
                                best_score=best_score, status="best",
                            ),
                            best_path,
                        )
                        atomic_hardlink(best_path, seed_dir / "checkpoint_latest.pt")
                    else:
                        _ensure_checkpoint_space(config, seed_dir)
                        atomic_torch_save(
                            _checkpoint_payload(
                                model, config, global_step, optimizer=optimizer,
                                scheduler=scheduler, scaler=scaler, epoch=epoch + 1,
                                batch_in_epoch=0, best_score=best_score,
                            ),
                            seed_dir / "checkpoint_latest.pt",
                        )
                    start_batch = 0
                    if signals.requested:
                        logger.warning("Received %s during validation; epoch checkpoint saved", signals.signal_name)
                        raise GracefulStop(
                            f"Fine-tuning interrupted by {signals.signal_name}",
                            preempting=signals.signal_name in {"SIGTERM", "SIGUSR1"},
                        )
                    if global_step >= total_steps:
                        break

            best_state = torch.load(best_path, map_location=device, weights_only=False)
            getattr(model, "_orig_mod", model).load_state_dict(best_state["model"])
            test_loader = task_dataloader(
                config.data_dir, config.task, "test", eval_length,
                config.train.eval_batch_tokens, config.train.num_workers, seed, False,
                synthetic_lengths=([length for length in (1_024, 2_048, 4_096) if length <= eval_length]
                                   or [eval_length]) if is_stress else None,
                dataset_seed=config.seed,
            )
            metrics, raw = evaluate(model, test_loader, task_type, device, dtype)
            metrics["confidence_95"] = bootstrap_confidence_intervals(
                raw["logits"], raw["labels"], task_type, seed=seed
            )
            if config.task.startswith(("synthetic_", "stress_")):
                metrics["evidence_heatmap"] = evidence_breakdown(
                    raw["logits"], raw["labels"], raw["positions"], raw["source_lengths"]
                )
                metrics["stress_breakdown"] = stress_breakdown(
                    raw["logits"], raw["labels"], raw["positions"], raw["source_lengths"],
                    raw["challenges"], raw["distractor_counts"],
                )
            atomic_write_json(seed_dir / "test_metrics.json", metrics)
            predictions_path = seed_dir / "test_predictions.npz"
            predictions_temp = predictions_path.with_suffix(".npz.tmp")
            with predictions_temp.open("wb") as handle:
                np.savez_compressed(
                    handle, ids=np.asarray(raw["ids"]), logits=raw["logits"],
                    labels=raw["labels"], coverage=np.asarray(raw["coverage"]),
                    source_lengths=np.asarray(raw["source_lengths"]),
                    challenges=np.asarray(raw["challenges"], dtype=str),
                    distractor_counts=np.asarray(raw["distractor_counts"]),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(predictions_temp, predictions_path)
            # checkpoint_last is the selected model and its matching optimizer
            # state, marked terminal. It is never a mismatched best-model/last-
            # optimizer hybrid.
            final_payload = dict(best_state)
            final_payload.update({
                "step": global_step,
                "epoch": config.train.epochs,
                "batch_in_epoch": 0,
                "best_score": best_score,
                "status": "completed",
                "rng_state": capture_rng_state(),
            })
            _ensure_checkpoint_space(config, seed_dir)
            atomic_torch_save(final_payload, seed_dir / "checkpoint_last.pt")
            atomic_hardlink(seed_dir / "checkpoint_last.pt", seed_dir / "checkpoint_latest.pt")
            atomic_write_json(seed_dir / "completion.json", {
                "status": "completed", "task": config.task, "seed": seed,
                "step": global_step, "completed_at": time.time(),
            })
            write_heartbeat(
                seed_dir / "heartbeat.json", stage="finetune", status="completed",
                task=config.task, seed=seed, step=global_step,
                total_steps=total_steps, eta_seconds=0,
            )
            tracker.summary({
                **{f"test/{key}": value for key, value in metrics.items()
                   if isinstance(value, (int, float))},
                "best_validation_score": best_score,
            })
            events.write("finetune_completed", task=config.task, seed=seed, step=global_step)
            outputs.append(best_path)
    return outputs
