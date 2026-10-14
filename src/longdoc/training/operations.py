from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import shutil
import signal
import socket
import time
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import torch

from longdoc.config import RunConfig


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def configure_logging(log_dir: Path, name: str = "longdoc") -> logging.Logger:
    """Create console, rotating text, and machine-readable event logs once per path."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"{name}.{hashlib.sha1(str(log_dir).encode()).hexdigest()[:10]}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    file_handler = RotatingFileHandler(
        log_dir / "run.log", maxBytes=20 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


class EventWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **fields: Any) -> None:
        record = {"time": time.time(), "event": event, **fields}
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
            handle.flush()


class Tracker:
    """A failure-tolerant W&B adapter; local artifacts remain authoritative."""

    def __init__(
        self,
        config: RunConfig,
        run_dir: Path,
        run_name: str,
        group: str,
        job_type: str,
        logger: logging.Logger,
    ):
        self.run = None
        self.logger = logger
        self.required = config.monitoring.required
        self.last_log = 0.0
        self.log_interval = config.monitoring.log_every_seconds
        if not config.monitoring.enabled or config.monitoring.mode == "disabled":
            return
        try:
            try:
                from dotenv import load_dotenv

                # Loads WANDB_API_KEY without ever reading or emitting its value here.
                load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)
                load_dotenv(Path.cwd() / ".env", override=False)
            except ImportError:
                pass
            import wandb

            wandb_dir = run_dir / "wandb"
            wandb_dir.mkdir(parents=True, exist_ok=True)
            stable_id = hashlib.sha1(
                f"{group}/{run_name}/{run_dir.resolve()}".encode()
            ).hexdigest()[:16]
            self.run = wandb.init(
                project=config.monitoring.project,
                entity=config.monitoring.entity,
                name=run_name,
                id=stable_id,
                resume="allow",
                group=group,
                job_type=job_type,
                tags=[*config.monitoring.tags, config.model.architecture],
                mode=config.monitoring.mode,
                dir=str(wandb_dir),
                config=config.to_dict(),
            )
            self.run.define_metric("train/step")
            self.run.define_metric("train/*", step_metric="train/step")
            self.run.define_metric("train/tokens")
            self.run.define_metric("pretrain/*", step_metric="train/tokens")
            self.run.define_metric("epoch")
            self.run.define_metric("validation/*", step_metric="epoch")
        except Exception:  # Optional third-party failures must not stop training.
            self.logger.exception("W&B initialization failed")
            if config.monitoring.required:
                raise

    def log(self, values: dict[str, Any], force: bool = False) -> None:
        if self.run is None:
            return
        now = time.monotonic()
        if not force and now - self.last_log < self.log_interval:
            return
        try:
            self.run.log(values)
            self.last_log = now
        except Exception:
            self.logger.exception("W&B logging failed; continuing with authoritative local logs")
            self.run = None
            if self.required:
                raise

    def summary(self, values: dict[str, Any]) -> None:
        if self.run is not None:
            try:
                self.run.summary.update(values)
            except Exception:
                self.logger.exception("W&B summary update failed")
                self.run = None
                if self.required:
                    raise

    def mark_preempting(self) -> None:
        if self.run is not None:
            try:
                self.run.mark_preempting()
            except Exception as error:  # noqa: BLE001 - W&B implementations vary by version
                self.logger.warning("Could not mark W&B run as preempting: %s", error)

    def finish(self, exit_code: int = 0) -> None:
        if self.run is not None:
            run, self.run = self.run, None
            try:
                run.finish(exit_code=exit_code)
            except Exception:
                self.logger.exception("W&B shutdown failed")
                if self.required:
                    raise


@dataclass
class ETA:
    total: int
    completed: int = 0
    rate: float | None = None
    smoothing: float = 0.15
    _last_time: float = 0.0

    def __post_init__(self) -> None:
        self._last_time = time.monotonic()

    def update(self, completed: int) -> dict[str, float | None]:
        now = time.monotonic()
        delta_count = completed - self.completed
        delta_time = now - self._last_time
        if delta_count > 0 and delta_time > 0:
            current = delta_count / delta_time
            self.rate = current if self.rate is None else (
                self.smoothing * current + (1 - self.smoothing) * self.rate
            )
        self.completed = completed
        self._last_time = now
        eta = (self.total - completed) / self.rate if self.rate and completed < self.total else 0.0
        return {"rate": self.rate, "eta_seconds": eta}


class SignalController:
    """Turn scheduler/preemption signals into a checkpoint request."""

    def __init__(self):
        self.requested = False
        self.signal_name: str | None = None
        self._previous: dict[int, Any] = {}

    def _handler(self, signum: int, _frame: Any) -> None:
        self.requested = True
        self.signal_name = signal.Signals(signum).name

    def __enter__(self) -> SignalController:  # noqa: PYI034 - Python 3.10 has no typing.Self
        signals = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGUSR1"):
            signals.append(signal.SIGUSR1)
        for item in signals:
            self._previous[item] = signal.getsignal(item)
            signal.signal(item, self._handler)
        return self

    def __exit__(self, *_: object) -> None:
        for item, handler in self._previous.items():
            signal.signal(item, handler)


class GracefulStop(RuntimeError):
    def __init__(self, message: str, preempting: bool = True):
        super().__init__(message)
        self.preempting = preempting


def write_heartbeat(path: Path, **status: Any) -> None:
    gpu_memory = None
    if torch.cuda.is_available():
        gpu_memory = int(torch.cuda.memory_allocated())
    process_memory = None
    try:
        import psutil

        process_memory = int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        pass
    disk_free = shutil.disk_usage(path.parent if path.parent.exists() else path.parent.parent).free
    atomic_write_json(path, {
        "updated_at": time.time(),
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "gpu_memory_bytes": gpu_memory,
        "process_memory_bytes": process_memory,
        "disk_free_bytes": disk_free,
        **status,
    })


def format_duration(seconds: float | None) -> str:
    if seconds is None or not isinstance(seconds, (int, float)):
        return "estimating"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3_600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@contextlib.contextmanager
def tracking_run(
    config: RunConfig,
    run_dir: Path,
    run_name: str,
    group: str,
    job_type: str,
):
    logger = configure_logging(run_dir / "logs", run_name)
    tracker = Tracker(config, run_dir, run_name, group, job_type, logger)
    try:
        yield logger, tracker, EventWriter(run_dir / "logs" / "events.jsonl")
    except GracefulStop as error:
        if error.preempting:
            tracker.mark_preempting()
        tracker.finish(exit_code=1)
        raise
    except BaseException:
        tracker.finish(exit_code=1)
        raise
    else:
        tracker.finish(exit_code=0)
