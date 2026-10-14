from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from longdoc.config import RunConfig, load_config
from longdoc.data.catalog import CATALOG
from longdoc.models.modeling import build_model
from longdoc.orchestrator import SuiteRunner, build_jobs
from longdoc.registry import EXPERIMENTS, apply_experiment
from longdoc.reporting import build_failure_fingerprint, summarize_runs
from longdoc.suites import PAPERS, STUDIES, SUITES, get_suite
from longdoc.training.analysis import analyze_checkpoint
from longdoc.training.baseline import run_tfidf_baseline
from longdoc.training.benchmark import benchmark
from longdoc.training.operations import atomic_write_json
from longdoc.training.trainer import finetune, pretrain


def _apply_overrides(config: RunConfig, expressions: list[str]) -> None:
    for expression in expressions:
        if "=" not in expression:
            raise ValueError(f"Override must be key=value, got {expression!r}")
        path, raw = expression.split("=", 1)
        value = yaml.safe_load(raw)
        target: Any = config
        parts = path.split(".")
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise ValueError(f"Unknown override path: {path}")
            target = getattr(target, part)
        if not hasattr(target, parts[-1]):
            raise ValueError(f"Unknown override path: {path}")
        setattr(target, parts[-1], value)


def _checkpoint_for(config: RunConfig, spec, root: Path, device: str) -> Path:
    if config.checkpoint:
        path = Path(config.checkpoint)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    pretrain_name = spec.reuse_from if spec.pretraining == "reuse" else spec.name
    path = root / pretrain_name / "pretrain" / "checkpoint_last.pt"
    if path.exists():
        return path
    pretrain_config = copy.deepcopy(config)
    apply_experiment(pretrain_config, pretrain_name)
    print(f"No reusable MLM checkpoint found; pretraining {pretrain_name} first.")
    return pretrain(pretrain_config, path.parent, device)


def _smoke(config: RunConfig) -> None:
    smoke = copy.deepcopy(config.model)
    smoke.vocab_size = 257
    smoke.hidden_size = 32
    smoke.intermediate_size = 64
    smoke.num_layers = 2
    smoke.num_heads = 4
    smoke.max_length = 64
    smoke.attention_window = 16
    smoke.projection_rank = 8
    smoke.segment_size = 16
    smoke.memory_size = 16
    smoke.compressed_memory_size = 8
    model = build_model(smoke, "finetune", 3, "multiclass")
    ids = torch.randint(5, smoke.vocab_size, (2, smoke.max_length))
    mask = torch.ones_like(ids, dtype=torch.bool)
    labels = torch.tensor([0, 2])
    result = model(ids, mask, labels)
    result.loss.backward()
    print(json.dumps({
        "architecture": smoke.architecture,
        "logits_shape": list(result.logits.shape),
        "loss": float(result.loss.detach()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled long-document experiments")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--experiment", choices=SUITES,
        help="Study or paper suite to run end-to-end",
    )
    parser.add_argument("--cell", help="Low-level registry cell for a single debug run")
    parser.add_argument(
        "--stage", default="suite",
        choices=["suite", "plan", "status", "all", "pretrain", "finetune", "curve", "analyze", "benchmark", "tfidf", "fingerprint", "summarize", "smoke", "list"],
    )
    parser.add_argument(
        "--task", default="arxiv",
        help="Dataset name; a stress_* challenge; stress; or all real tasks",
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:1, or mps")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--repeats", type=int, default=10, help="Timed repetitions for benchmark stage")
    parser.add_argument(
        "--analysis", choices=["truncation", "prepend", "chunk_shuffle", "projection_spectrum", "selection_diagnostics", "ablation"],
        default="truncation",
    )
    parser.add_argument("--analysis-value", type=int, default=0)
    parser.add_argument("--run-id", help="Stable suite run ID; defaults to a config hash")
    parser.add_argument("--new-run", action="store_true", help="Start a separate timestamped run")
    parser.add_argument(
        "--reuse-pretraining-from", type=Path,
        help="Import compatible cell MLM checkpoints from another completed suite run",
    )
    parser.add_argument("--resume", choices=["auto", "never", "required"])
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"])
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.stage == "list":
        print("Shared study suites:")
        for study in STUDIES:
            suite = get_suite(study)
            print(f"  {study:12} {len(suite.cells)} cells, {len(build_jobs(suite))} scheduled jobs")
        print("\nPaper suites:")
        for paper in PAPERS:
            suite = get_suite(paper)
            print(f"  {paper:12} {len(suite.cells)} cells, {len(suite.analyses)} inference ablations")
        print("\nLow-level cells (use --cell):")
        for name, spec in sorted(EXPERIMENTS.items()):
            print(f"{name:42} {spec.pretraining:5} {spec.description}")
        return
    config = load_config(args.config)
    _apply_overrides(config, args.set)
    if args.data_dir:
        config.data_dir = str(args.data_dir)
    if args.output_dir:
        config.output_dir = str(args.output_dir)
    if args.checkpoint:
        config.checkpoint = str(args.checkpoint)
    if args.resume:
        config.runtime.resume = args.resume
    if args.no_wandb:
        config.monitoring.enabled = False
    if args.wandb_mode:
        config.monitoring.mode = args.wandb_mode
    if args.fail_fast:
        config.runtime.fail_fast = True
    config.task = args.task
    config.stage = args.stage
    config.train.lengths = [length for length in config.train.lengths if length <= config.model.max_length]
    if not config.train.lengths:
        config.train.lengths = [config.model.max_length]
    config.validate()
    root = Path(config.output_dir)

    if args.stage in {"suite", "plan", "status"}:
        if not args.experiment:
            raise ValueError("This stage requires --experiment with a study or paper suite name")
        if args.checkpoint:
            raise ValueError(
                "--checkpoint is only valid with a low-level --cell run; a paper suite manages "
                "multiple fresh and reused checkpoints from its manifest"
            )
        runner = SuiteRunner(
            config, args.experiment, args.device, args.repeats, args.run_id, args.new_run,
            inspect_only=args.stage == "status",
            reuse_pretraining_from=args.reuse_pretraining_from,
        )
        if args.stage == "plan":
            print(json.dumps(runner.plan(), indent=2))
            return
        if args.stage == "status":
            print(json.dumps(runner.status(), indent=2))
            return
        output = runner.run()
        print(f"Suite completed: {output}")
        return

    if args.stage == "summarize":
        result = summarize_runs(root)
        output = root / "summary.json"
        atomic_write_json(output, result)
        print(json.dumps(result, indent=2))
        return
    if args.stage == "fingerprint":
        result = build_failure_fingerprint(root, root / "reports")
        print(json.dumps(result, indent=2))
        return
    if args.stage == "smoke":
        cell = args.cell or "exact.baseline"
        apply_experiment(config, cell)
        _smoke(config)
        return
    if args.task == "all":
        tasks = list(CATALOG)
    elif args.task in {"stress", "stress_all"}:
        tasks = ["stress_retrieve", "stress_combine", "stress_order", "stress_distract"]
    else:
        tasks = [args.task]
    if args.stage == "tfidf":
        for task in tasks:
            metrics = run_tfidf_baseline(
                config.data_dir, task, root / "tfidf" / task / "metrics.json", config.seed
            )
            print(task, json.dumps(metrics))
        return
    experiment = args.cell or config.experiment
    if experiment in SUITES or experiment not in EXPERIMENTS:
        raise ValueError(
            "Low-level stages require --cell with a registry key. "
            "Use --stage list to see available cells."
        )
    spec = apply_experiment(config, experiment)
    if args.stage == "benchmark":
        result = benchmark(config, root / experiment / "benchmark.json", args.device, args.repeats)
        print(json.dumps(result, indent=2))
        return
    if args.stage == "analyze":
        if not args.checkpoint:
            raise ValueError("--stage analyze requires a fine-tuned --checkpoint")
        result = analyze_checkpoint(
            config, args.checkpoint, root / experiment / "analysis" / f"{args.analysis}.json",
            args.analysis, args.device, args.analysis_value,
        )
        print(json.dumps(result, indent=2))
        return
    if args.stage == "curve":
        final_checkpoint = _checkpoint_for(config, spec, root, args.device)
        checkpoints = sorted(
            final_checkpoint.parent.glob("checkpoint_tokens_*.pt"),
            key=lambda path: int(path.stem.rsplit("_", 1)[1]),
        )
        if not checkpoints:
            raise FileNotFoundError(
                f"No token-milestone checkpoints found beside {final_checkpoint}. "
                "Run pretraining with train.checkpoint_tokens configured."
            )
        for checkpoint in checkpoints:
            token_count = checkpoint.stem.rsplit("_", 1)[1]
            curve_dir = root / experiment / "compute_curve" / f"tokens_{token_count}"
            metadata_path = checkpoint.with_suffix(".json")
            if metadata_path.exists():
                atomic_write_json(
                    curve_dir / "pretraining.json", json.loads(metadata_path.read_text())
                )
            for task in tasks:
                task_config = copy.deepcopy(config)
                task_config.task = task
                finetune(
                    task_config,
                    curve_dir,
                    checkpoint,
                    args.device,
                )
        return

    checkpoint = None
    if args.stage in {"pretrain", "all"}:
        checkpoint = _checkpoint_for(config, spec, root, args.device)
        print(f"MLM checkpoint: {checkpoint}")
        if args.stage == "pretrain":
            return
    if args.stage == "finetune":
        checkpoint = _checkpoint_for(config, spec, root, args.device)
    assert checkpoint is not None
    for task in tasks:
        task_config = copy.deepcopy(config)
        task_config.task = task
        paths = finetune(task_config, root / experiment / "finetune", checkpoint, args.device)
        print(f"{task}: wrote {len(paths)} fine-tuned checkpoints")


if __name__ == "__main__":
    main()
