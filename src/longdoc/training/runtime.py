from __future__ import annotations

import contextlib
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_accelerator(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def amp_dtype(device: torch.device, setting: str) -> torch.dtype | None:
    if setting == "none" or device.type == "cpu":
        return None
    if setting == "bf16" or (
        setting == "auto" and device.type == "cuda" and torch.cuda.is_bf16_supported()
    ):
        return torch.bfloat16
    if setting in {"auto", "fp16"} and device.type == "cuda":
        return torch.float16
    return None


def autocast_context(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def atomic_torch_save(payload: dict, path: Path) -> None:
    free = shutil.disk_usage(path.parent if path.parent.exists() else path.parent.parent).free
    if free < 256 * 1024 * 1024:
        raise OSError(f"Refusing checkpoint write: less than 256 MiB free near {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_hardlink(source: Path, destination: Path) -> None:
    """Atomically point a second checkpoint name at immutable checkpoint bytes."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
        os.replace(temporary, destination)
    except OSError:
        # Network filesystems may not support hard links; fall back to a copy.
        import shutil

        temporary.unlink(missing_ok=True)
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
