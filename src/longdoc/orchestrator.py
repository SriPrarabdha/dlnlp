from __future__ import annotations

import copy
import fcntl
import json
import logging
import os
import re
import socket
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from longdoc.config import RunConfig, stable_config_id
from longdoc.plotting import write_result_plot, write_suite_reports
from longdoc.registry import apply_experiment, get_experiment
from longdoc.status import write_run_status
from longdoc.suites import Analysis, PaperSuite, get_suite
from longdoc.training.analysis import analyze_checkpoint
from longdoc.training.baseline import run_tfidf_baseline
from longdoc.training.benchmark import benchmark
from longdoc.training.operations import (
    EventWriter,
    GracefulStop,
    atomic_write_json,
    configure_logging,
    format_duration,
    tracking_run,
)
from longdoc.training.runtime import atomic_hardlink
from longdoc.training.trainer import finetune, pretrain


@dataclass(frozen=True)
class Job:
    id: str
    kind: str
    cell: str | None = None
    task: str | None = None
    analysis: Analysis | None = None
    dependencies: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def serializable(self) -> dict[str, Any]:
        result = asdict(self)
        return result


def _source_cell(cell: str) -> str:
    spec = get_experiment(cell)
    return spec.reuse_from if spec.pretraining == "reuse" and spec.reuse_from else cell


def build_jobs(suite: PaperSuite) -> list[Job]:
    jobs: list[Job] = []
    pretrain_ids: dict[str, str] = {}
    for cell in suite.cells:
        source = _source_cell(cell)
        if source not in pretrain_ids:
            identifier = f"pretrain:{source}"
            pretrain_ids[source] = identifier
            jobs.append(Job(identifier, "pretrain", cell=source))
    for cell in suite.cells:
        dependency = (pretrain_ids[_source_cell(cell)],)
        jobs.append(Job(f"benchmark:{cell}", "benchmark", cell=cell, dependencies=dependency))
        for task in suite.tasks[cell]:
            jobs.append(Job(
                f"finetune:{cell}:{task}", "finetune", cell=cell, task=task,
                dependencies=dependency,
            ))
    curve_cells = suite.curve_cells or ((suite.baseline,) if suite.curve_tasks else ())
    for cell in curve_cells:
        dependency = (pretrain_ids[_source_cell(cell)],)
        for task in suite.curve_tasks:
            jobs.append(Job(
                f"curve:{cell}:{task}", "curve", cell=cell, task=task,
                dependencies=dependency,
            ))
    for task in suite.tfidf_tasks:
        jobs.append(Job(f"tfidf:{task}", "tfidf", task=task))
    for item in suite.analyses:
        jobs.append(Job(
            f"analysis:{item.cell}:{item.task}:{item.name}", "analysis", cell=item.cell,
            task=item.task, analysis=item,
            dependencies=(f"finetune:{item.cell}:{item.task}",),
        ))
    non_report = tuple(job.id for job in jobs)
    jobs.append(Job("report", "report", dependencies=non_report))
    return jobs


def _set_override(config: RunConfig, dotted: str, value: Any) -> None:
    target: Any = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


class SuiteLock:
    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"This suite is already running (lock: {self.path})") from error
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "time": time.time()}))
        self.handle.flush()
        return self

    def __exit__(self, *_: object):
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


class SuiteRunner:
    def __init__(
        self,
        config: RunConfig,
        paper: str,
        device: str = "auto",
        repeats: int = 10,
        run_id: str | None = None,
        new_run: bool = False,
        inspect_only: bool = False,
        reuse_pretraining_from: Path | None = None,
    ):
        self.config = copy.deepcopy(config)
        self.config.experiment = paper
        self.config.task = "suite"
        self.config.checkpoint = None
        self.paper = paper
        self.device = device
        self.repeats = repeats
        self.new_run = new_run
        self.inspect_only = inspect_only
        self.reuse_pretraining_from = reuse_pretraining_from.resolve() if reuse_pretraining_from else None
        self.suite = get_suite(paper, config.model.max_length)
        base_id = run_id or stable_config_id(self.config)
        if new_run:
            base_id = f"{base_id}-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", base_id):
            raise ValueError("run_id may contain only letters, numbers, dot, underscore, and dash")
        self.run_id = base_id
        self.root = Path(config.output_dir) / paper / self.run_id
        self.jobs = build_jobs(self.suite)
        self.manifest_path = self.root / "manifest.json"
        if (
            self.config.runtime.resume == "never"
            and self.manifest_path.exists()
            and not self.new_run
        ):
            raise FileExistsError(
                f"A suite already exists at {self.root}. Use --resume auto to continue it or "
                "combine --resume never with --new-run for a clean run."
            )
        self.logger = logging.getLogger(f"longdoc.suite.{paper}.preview")
        self.events: EventWriter | None = None
        self.manifest = self._load_manifest()
        if self.reuse_pretraining_from is not None:
            self._validate_reuse_source()

    def _validate_reuse_source(self) -> None:
        assert self.reuse_pretraining_from is not None
        source_config_path = self.reuse_pretraining_from / "resolved_config.yaml"
        if not source_config_path.exists():
            raise FileNotFoundError(f"Reusable suite has no resolved config: {source_config_path}")
        source = yaml.safe_load(source_config_path.read_text())
        target = self.config.to_dict()
        ignored = {"experiment", "task", "stage", "output_dir", "checkpoint", "monitoring", "runtime"}
        source_identity = {key: value for key, value in source.items() if key not in ignored}
        target_identity = {key: value for key, value in target.items() if key not in ignored}
        source_identity.get("train", {}).pop("gradient_checkpointing", None)
        target_identity.get("train", {}).pop("gradient_checkpointing", None)
        if source_identity != target_identity:
            raise ValueError(
                "--reuse-pretraining-from is scientifically incompatible with this run: "
                "data, seed, model, or training configuration differs"
            )

    def _load_manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text())
            known = manifest.setdefault("jobs", {})
            if not self.inspect_only:
                for state in known.values():
                    if state.get("status") == "running":
                        state["status"] = "pending"
            added = False
            for job in self.jobs:
                if job.id not in known:
                    known[job.id] = {"status": "pending", "attempts": 0}
                    added = True
            if added and "report" in known:
                known["report"]["status"] = "pending"
            return manifest
        if self.config.runtime.resume == "required":
            raise FileNotFoundError(
                f"Suite resume was required, but no manifest exists at {self.manifest_path}"
            )
        return {
            "format_version": 1,
            "paper": self.paper,
            "run_id": self.run_id,
            "status": "pending",
            "created_at": time.time(),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "config_id": stable_config_id(self.config),
            "suite": self.suite.to_dict(),
            "jobs": {job.id: {"status": "pending", "attempts": 0} for job in self.jobs},
        }

    def _save_manifest(self) -> None:
        self.manifest["updated_at"] = time.time()
        atomic_write_json(self.manifest_path, self.manifest)
        try:
            write_run_status(self.root, self.paper, self.run_id, self.jobs, self.manifest)
        except Exception:
            self.logger.exception("Could not refresh live checklist")

    def _cell_config(self, cell: str, task: str | None = None) -> RunConfig:
        config = copy.deepcopy(self.config)
        apply_experiment(config, cell)
        if config.runtime.resume == "required":
            # The suite manifest established that this is a resume. Jobs that had
            # not started yet must still be allowed to create their first checkpoint.
            config.runtime.resume = "auto"
        if task is not None:
            config.task = task
        config.validate()
        return config

    def _pretrain_path(self, cell: str) -> Path:
        return self.root / "cells" / _source_cell(cell) / "pretrain" / "checkpoint_last.pt"

    def _job_complete_on_disk(self, job: Job) -> bool:
        if job.kind == "pretrain":
            checkpoint = self._pretrain_path(job.cell or "")
            completion = checkpoint.parent / "completion.json"
            if not checkpoint.exists() or not completion.exists():
                return False
            try:
                return json.loads(completion.read_text()).get("status") == "completed"
            except (json.JSONDecodeError, OSError):
                return False
        if job.kind == "benchmark":
            base = self.root / "cells" / str(job.cell) / "benchmark"
            return (base / "metrics.json").exists() and (base / "plots" / "benchmark.svg").exists()
        if job.kind == "finetune":
            task_root = self.root / "cells" / str(job.cell) / "finetune" / str(job.task)
            for seed in self.config.train.finetune_seeds:
                seed_root = task_root / f"seed_{seed}"
                completion = seed_root / "completion.json"
                if not all((seed_root / name).exists() for name in (
                    "test_metrics.json", "test_predictions.npz", "checkpoint_best.pt"
                )) or not completion.exists():
                    return False
                try:
                    if json.loads(completion.read_text()).get("status") != "completed":
                        return False
                except (json.JSONDecodeError, OSError):
                    return False
            return True
        if job.kind == "curve":
            return (self.root / "curves" / str(job.cell) / str(job.task) / "complete.json").exists()
        if job.kind == "analysis":
            base = self.root / "analyses" / str(job.cell) / str(job.analysis.name)
            return all(
                (base / f"seed_{seed}" / "result.json").exists()
                and (base / f"seed_{seed}" / "plot.svg").exists()
                for seed in self.config.train.finetune_seeds
            )
        if job.kind == "tfidf":
            base = self.root / "baselines" / "tfidf" / str(job.task)
            return (base / "metrics.json").exists() and (base / "plot.svg").exists()
        if job.kind == "report":
            return (
                (self.root / "summaries" / "suite_summary.json").exists()
                and self.manifest["jobs"][job.id].get("status") == "completed"
            )
        return False

    def plan(self) -> dict[str, Any]:
        return {
            "paper": self.paper, "run_id": self.run_id, "root": str(self.root),
            "reuse_pretraining_from": str(self.reuse_pretraining_from) if self.reuse_pretraining_from else None,
            "jobs": [job.serializable() for job in self.jobs],
        }

    def status(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"No run manifest exists at {self.manifest_path}. Run --stage plan or --stage suite first."
            )
        self.manifest = json.loads(self.manifest_path.read_text())
        return write_run_status(self.root, self.paper, self.run_id, self.jobs, self.manifest)

    def _execute(self, job: Job) -> dict[str, Any]:
        group = f"{self.paper}/{self.run_id}"
        if job.kind == "pretrain":
            if self.reuse_pretraining_from is not None:
                source = self.reuse_pretraining_from / "cells" / str(job.cell) / "pretrain" / "checkpoint_last.pt"
                if not source.exists():
                    raise FileNotFoundError(f"Reusable MLM checkpoint not found: {source}")
                source_completion = source.parent / "completion.json"
                if not source_completion.exists() or json.loads(source_completion.read_text()).get("status") != "completed":
                    raise RuntimeError(f"Reusable MLM checkpoint is not marked complete: {source}")
                destination = self.root / "cells" / str(job.cell) / "pretrain" / "checkpoint_last.pt"
                atomic_hardlink(source, destination)
                atomic_hardlink(source, destination.parent / "checkpoint_latest.pt")
                milestones = []
                for milestone in source.parent.glob("checkpoint_tokens_*.pt"):
                    target = destination.parent / milestone.name
                    atomic_hardlink(milestone, target)
                    metadata = milestone.with_suffix(".json")
                    if metadata.exists():
                        atomic_write_json(
                            target.with_suffix(".json"), json.loads(metadata.read_text())
                        )
                    milestones.append(milestone.name)
                completion = source.parent / "completion.json"
                if completion.exists():
                    atomic_write_json(
                        destination.parent / "completion.json",
                        json.loads(completion.read_text()),
                    )
                return {
                    "checkpoint": str(destination), "reused_from": str(source),
                    "milestone_checkpoints": sorted(milestones),
                }
            config = self._cell_config(str(job.cell))
            path = pretrain(config, self.root / "cells" / str(job.cell) / "pretrain", self.device, group)
            return {"checkpoint": str(path)}
        if job.kind == "finetune":
            config = self._cell_config(str(job.cell), str(job.task))
            paths = finetune(
                config, self.root / "cells" / str(job.cell) / "finetune",
                self._pretrain_path(str(job.cell)), self.device, group,
            )
            return {"checkpoints": [str(path) for path in paths]}
        if job.kind == "benchmark":
            config = self._cell_config(str(job.cell))
            output = self.root / "cells" / str(job.cell) / "benchmark" / "metrics.json"
            with tracking_run(config, output.parent, f"{job.cell}-benchmark", group, "benchmark") as (_, tracker, _):
                result = benchmark(config, output, self.device, self.repeats)
                write_result_plot(result, output.parent / "plots" / "benchmark.svg", f"{job.cell}: throughput and memory")
                length_result = result.get("lengths", {}).get("4096", {})
                tracker.log({f"benchmark/{key}": value for key, value in length_result.items() if isinstance(value, (int, float))}, force=True)
            return {"output": str(output)}
        if job.kind == "tfidf":
            config = copy.deepcopy(self.config)
            config.task = str(job.task)
            output = self.root / "baselines" / "tfidf" / str(job.task) / "metrics.json"
            with tracking_run(config, output.parent, f"tfidf-{job.task}", group, "baseline") as (_, tracker, _):
                result = run_tfidf_baseline(config.data_dir, str(job.task), output, config.seed)
                write_result_plot(result, output.parent / "plot.svg", f"TF-IDF: {job.task}")
                tracker.log({
                    f"test/{key}": value for key, value in result.items()
                    if isinstance(value, (int, float))
                }, force=True)
            return {"output": str(output)}
        if job.kind == "curve":
            config = self._cell_config(str(job.cell), str(job.task))
            source_dir = self._pretrain_path(str(job.cell)).parent
            checkpoints = sorted(
                source_dir.glob("checkpoint_tokens_*.pt"),
                key=lambda path: int(path.stem.rsplit("_", 1)[1]),
            )
            curve_root = self.root / "curves" / str(job.cell) / str(job.task)
            completed = []
            for checkpoint in checkpoints:
                tokens = checkpoint.stem.rsplit("_", 1)[1]
                output_dir = curve_root / f"tokens_{tokens}" / "finetune"
                finetune(config, output_dir, checkpoint, self.device, group)
                completed.append(int(tokens))
            atomic_write_json(curve_root / "complete.json", {
                "status": "completed" if completed else "no_milestones_available",
                "token_milestones": completed,
            })
            return {"token_milestones": completed}
        if job.kind == "analysis" and job.analysis is not None:
            config = self._cell_config(str(job.cell), str(job.task))
            for dotted, value in job.analysis.overrides.items():
                _set_override(config, dotted, value)
            config.validate()
            outputs = []
            for seed in config.train.finetune_seeds:
                checkpoint = self.root / "cells" / str(job.cell) / "finetune" / str(job.task) / f"seed_{seed}" / "checkpoint_best.pt"
                output = self.root / "analyses" / str(job.cell) / job.analysis.name / f"seed_{seed}" / "result.json"
                with tracking_run(config, output.parent, f"{job.cell}-{job.analysis.name}-seed-{seed}", group, "analysis") as (_, tracker, _):
                    result = analyze_checkpoint(
                        config, checkpoint, output, job.analysis.kind, self.device, job.analysis.value,
                    )
                    write_result_plot(
                        result, output.parent / "plot.svg",
                        f"{job.cell}: {job.analysis.name} (seed {seed})",
                    )
                    flat = {f"analysis/{key}": value for key, value in result.items() if isinstance(value, (int, float))}
                    tracker.log(flat, force=True)
                outputs.append(str(output))
            return {"outputs": outputs}
        if job.kind == "report":
            summary = write_suite_reports(self.root, self.paper)
            return {"summary": str(self.root / "summaries" / "suite_summary.json"), "cells": len(summary["cells"])}
        raise ValueError(f"Unsupported suite job: {job.kind}")

    def _remaining_eta(self) -> float | None:
        durations: dict[str, list[float]] = {}
        all_durations: list[float] = []
        by_id = {job.id: job for job in self.jobs}
        for identifier, state in self.manifest["jobs"].items():
            if identifier not in by_id:
                continue
            if state.get("status") == "completed" and state.get("duration_seconds") is not None:
                durations.setdefault(by_id[identifier].kind, []).append(float(state["duration_seconds"]))
                all_durations.append(float(state["duration_seconds"]))
        if not all_durations:
            return None
        fallback = sum(all_durations) / len(all_durations)
        estimates = []
        for job in self.jobs:
            if self.manifest["jobs"][job.id]["status"] in {"completed", "skipped"}:
                continue
            samples = durations.get(job.kind)
            estimates.append(sum(samples) / len(samples) if samples else fallback)
        return sum(estimates)

    def run(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        self.logger = configure_logging(self.root / "logs", f"suite-{self.paper}")
        self.events = EventWriter(self.root / "logs" / "events.jsonl")
        self.config.save(self.root / "resolved_config.yaml")
        plan_path = self.root / "suite_plan.yaml"
        plan_temporary = plan_path.with_suffix(".yaml.tmp")
        plan_temporary.write_text(yaml.safe_dump(self.plan(), sort_keys=False))
        plan_temporary.replace(plan_path)
        with SuiteLock(self.root / ".suite.lock"):
            self.manifest["status"] = "running"
            self.manifest["pid"] = os.getpid()
            if any(
                job.kind != "report" and not self._job_complete_on_disk(job)
                for job in self.jobs
            ):
                self.manifest["jobs"]["report"]["status"] = "pending"
            self._save_manifest()
            failures = []
            try:
                for index, job in enumerate(self.jobs, 1):
                    state = self.manifest["jobs"][job.id]
                    if self.config.runtime.resume != "never" and self._job_complete_on_disk(job):
                        if state.get("status") != "completed":
                            state.update({
                                "status": "completed", "recovered_from_artifacts": True,
                                "finished_at": state.get("finished_at", time.time()),
                            })
                            self._save_manifest()
                        self.logger.info("[%d/%d] already complete: %s", index, len(self.jobs), job.id)
                        continue
                    failed_dependencies = [
                        dependency for dependency in job.dependencies
                        if self.manifest["jobs"][dependency]["status"] in {"failed", "blocked"}
                    ]
                    if failed_dependencies:
                        state.update({"status": "blocked", "blocked_by": failed_dependencies})
                        self._save_manifest()
                        continue
                    state.update({
                        "status": "running", "started_at": time.time(),
                        "attempts": int(state.get("attempts", 0)) + 1,
                    })
                    eta = self._remaining_eta()
                    self.manifest["eta_seconds"] = eta
                    self._save_manifest()
                    self.logger.info(
                        "[%d/%d] starting %s | suite ETA %s", index, len(self.jobs),
                        job.id, format_duration(eta),
                    )
                    self.events.write("job_started", job=job.id, index=index, total=len(self.jobs), eta_seconds=eta)
                    started = time.perf_counter()
                    try:
                        outputs = self._execute(job)
                    except GracefulStop:
                        state.update({
                            "status": "interrupted", "finished_at": time.time(),
                            "duration_seconds": time.perf_counter() - started,
                        })
                        self.manifest["status"] = "interrupted"
                        self._save_manifest()
                        try:
                            write_suite_reports(self.root, self.paper)
                        except Exception:
                            self.logger.exception("Could not refresh partial reports during shutdown")
                        raise
                    except Exception as error:
                        state.update({
                            "status": "failed", "finished_at": time.time(),
                            "duration_seconds": time.perf_counter() - started,
                            "error": repr(error), "traceback": traceback.format_exc(),
                        })
                        failures.append(job.id)
                        self.logger.exception("Job failed: %s", job.id)
                        self.events.write("job_failed", job=job.id, error=repr(error))
                        self._save_manifest()
                        if self.config.runtime.fail_fast:
                            break
                        continue
                    state.update({
                        "status": "completed", "finished_at": time.time(),
                        "duration_seconds": time.perf_counter() - started, "outputs": outputs,
                    })
                    state.pop("error", None)
                    state.pop("traceback", None)
                    self._save_manifest()
                    self.events.write("job_completed", job=job.id, duration_seconds=state["duration_seconds"])
                    self.logger.info(
                        "[%d/%d] completed %s in %s",
                        index, len(self.jobs), job.id,
                        format_duration(state["duration_seconds"]),
                    )
                # Reports are still useful after independent failures.
                if failures:
                    write_suite_reports(self.root, self.paper)
                self.manifest["status"] = "partial" if failures else "completed"
                self.manifest["eta_seconds"] = 0.0
                self.manifest["finished_at"] = time.time()
                self.manifest["failed_jobs"] = failures
                self._save_manifest()
            except BaseException:
                if self.manifest.get("status") != "interrupted":
                    self.manifest["status"] = "failed"
                    self._save_manifest()
                raise
        if failures:
            raise RuntimeError(
                f"Suite completed with {len(failures)} failed job(s). Resume after fixing the cause: "
                + ", ".join(failures)
            )
        return self.root
