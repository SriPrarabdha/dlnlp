from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from longdoc.training.operations import atomic_write_json

FINGERPRINT_ROWS = {
    "Exact": "exact.baseline",
    "Longformer": "longformer.baseline",
    "Linformer": "linformer.fixed",
    "Long-Short": "longshort.baseline",
    "Compressive": "compressive.baseline",
    "RoFormer": "roformer.baseline",
}


def _average_descending_ranks(values: list[tuple[str, float]]) -> dict[str, float]:
    """Competition ranks with ties assigned their average occupied rank."""
    ordered = sorted(values, key=lambda item: item[1], reverse=True)
    result: dict[str, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and np.isclose(ordered[end][1], ordered[start][1]):
            end += 1
        average_rank = ((start + 1) + end) / 2
        for name, _ in ordered[start:end]:
            result[name] = float(average_rank)
        start = end
    return result


def summarize_runs(root: Path) -> dict:
    grouped: dict[str, dict[str, list[dict]]] = {}
    selected: dict[tuple[str, str, str], tuple[tuple[int, float, int], Path]] = {}
    candidates = [
        *root.glob("*/finetune/*/seed_*/test_metrics.json"),
        *root.glob("*/*/cells/*/finetune/*/seed_*/test_metrics.json"),
    ]
    for path in candidates:
        parts = path.relative_to(root).parts
        is_suite = len(parts) >= 8 and parts[2] == "cells"
        if is_suite:
            key = (parts[3], parts[5], parts[6])
        else:
            key = (parts[0], parts[2], parts[3])
        completion = path.parent / "completion.json"
        completed = 0
        if completion.exists():
            try:
                completed = int(
                    json.loads(completion.read_text()).get("status") == "completed"
                )
            except (json.JSONDecodeError, OSError):
                pass
        # A suite artifact wins an exact timestamp tie; otherwise the newest
        # completed artifact is authoritative for this experiment/task/seed.
        score = (completed, path.stat().st_mtime, int(is_suite))
        if key not in selected or score > selected[key][0]:
            selected[key] = (score, path)
    for _, path in selected.values():
        parts = path.relative_to(root).parts
        if len(parts) >= 8 and parts[2] == "cells":
            experiment, task = parts[3], parts[5]
        else:
            experiment, task = parts[0], parts[2]
        grouped.setdefault(experiment, {}).setdefault(task, []).append(json.loads(path.read_text()))
    summary = {}
    for experiment, tasks in grouped.items():
        summary[experiment] = {}
        for task, runs in tasks.items():
            scalar_keys = [key for key, value in runs[0].items() if isinstance(value, (int, float))]
            summary[experiment][task] = {
                key: {"mean": float(np.mean([run[key] for run in runs])), "std": float(np.std([run[key] for run in runs], ddof=1)) if len(runs) > 1 else 0.0}
                for key in scalar_keys
            }

    # Mean rank uses each task's primary metric; it never averages incompatible scores.
    task_metrics = {"arxiv": "accuracy", "scotus": "accuracy", "ecthr_a": "micro_f1", "imdb": "accuracy"}
    eligible = {
        experiment for experiment, values in summary.items()
        if all(task in values and metric in values[task] for task, metric in task_metrics.items())
    }
    ranks: dict[str, list[float]] = {experiment: [] for experiment in eligible}
    for task, metric in task_metrics.items():
        available = [(experiment, summary[experiment][task][metric]["mean"]) for experiment in eligible]
        for experiment, rank in _average_descending_ranks(available).items():
            ranks[experiment].append(rank)
    for experiment, experiment_summary in summary.items():
        values = ranks.get(experiment, [])
        experiment_summary["mean_rank"] = float(np.mean(values)) if values else None
    curves: dict[str, dict[str, dict]] = {}
    for path in root.glob("*/compute_curve/tokens_*/*/seed_*/test_metrics.json"):
        parts = path.relative_to(root).parts
        experiment, token_dir, task = parts[0], parts[2], parts[3]
        metrics = json.loads(path.read_text())
        metric_name = task_metrics.get(task, "accuracy")
        if metric_name in metrics:
            curves.setdefault(experiment, {}).setdefault(token_dir, {}).setdefault(task, []).append(
                float(metrics[metric_name])
            )
    curve_summary = {
        experiment: {
            token_dir: {
                task: {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}
                for task, values in tasks.items()
            }
            for token_dir, tasks in checkpoints.items()
        }
        for experiment, checkpoints in curves.items()
    }
    for experiment, checkpoints in curve_summary.items():
        for token_dir, values in checkpoints.items():
            metadata_path = root / experiment / "compute_curve" / token_dir / "pretraining.json"
            if metadata_path.exists():
                values["_pretraining"] = json.loads(metadata_path.read_text())
    efficiency = {}
    for experiment_dir in root.iterdir() if root.exists() else []:
        if not experiment_dir.is_dir():
            continue
        item = {}
        milestone_metadata = sorted(
            (experiment_dir / "pretrain").glob("checkpoint_tokens_*.json"),
            key=lambda path: int(path.stem.rsplit("_", 1)[1]),
        )
        if milestone_metadata:
            latest = json.loads(milestone_metadata[-1].read_text())
            item["pretraining_gpu_hours"] = latest.get("gpu_hours")
            item["pretraining_tokens"] = latest.get("processed_tokens")
        log_path = experiment_dir / "pretrain" / "train.jsonl"
        if log_path.exists() and not milestone_metadata:
            records = [json.loads(line) for line in log_path.read_text().splitlines() if line]
            if records:
                item["pretraining_gpu_hours"] = records[-1].get(
                    "gpu_hours", records[-1].get("gpu_hours_this_session")
                )
                item["pretraining_tokens"] = records[-1].get("tokens")
        benchmark_path = experiment_dir / "benchmark.json"
        if benchmark_path.exists():
            benchmark = json.loads(benchmark_path.read_text()).get("lengths", {}).get("4096", {})
            item["throughput_documents_per_second_4k"] = benchmark.get("documents_per_second")
            item["peak_memory_bytes_4k"] = benchmark.get("peak_memory_bytes")
        if item:
            efficiency[experiment_dir.name] = item
    summary["_compute_curves"] = curve_summary
    summary["_efficiency_frontier"] = efficiency
    return summary


def _experiment_dir(root: Path, experiment: str) -> Path:
    direct = root / experiment
    if direct.exists():
        return direct
    candidates = list(root.glob(f"*/*/cells/{experiment}"))
    if not candidates:
        return direct

    def candidate_key(path: Path) -> tuple[int, float]:
        manifest = path.parents[1] / "manifest.json"
        completed = 0
        if manifest.exists():
            try:
                completed = int(json.loads(manifest.read_text()).get("status") == "completed")
            except (json.JSONDecodeError, OSError):
                pass
        return completed, path.stat().st_mtime

    return max(candidates, key=candidate_key)


def _seed_metrics(root: Path, experiment: str, tasks: tuple[str, ...]) -> list[dict]:
    experiment_root = _experiment_dir(root, experiment)
    for task in tasks:
        paths = sorted((experiment_root / "finetune" / task).glob("seed_*/test_metrics.json"))
        if paths:
            return [json.loads(path.read_text()) for path in paths]
    return []


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _quality_cell(metrics: list[dict], path: tuple[str, ...]) -> tuple[float | None, float | None]:
    values = []
    for item in metrics:
        value = item
        try:
            for key in path:
                value = value[key]
            values.append(float(value))
        except (KeyError, TypeError):
            continue
    mean, std = _mean_std(values)
    if mean is None:
        return None, None
    # All stress tasks are balanced binary classifications; chance maps to zero.
    return float(np.clip((mean - 0.5) / 0.5, 0, 1)), float(std * 2)


def _icon_svg(kind: str, x: int, y: int) -> str:
    color = "#b9c7ff"
    if kind == "Exact":
        return "".join(
            f'<circle cx="{x + col * 4}" cy="{y + row * 4}" r="1.2" fill="{color}"/>'
            for row in range(4) for col in range(4)
        )
    if kind == "Longformer":
        return f'<path d="M{x},{y+14} L{x+14},{y} M{x+4},{y+16} L{x+16},{y+4}" stroke="{color}"/><circle cx="{x+2}" cy="{y+2}" r="2.5" fill="#ffd166"/>'
    if kind == "Linformer":
        return f'<path d="M{x},{y} L{x+16},{y+8} L{x},{y+16} Z" fill="none" stroke="{color}"/><path d="M{x+16},{y+4} V{y+12}" stroke="#ffd166" stroke-width="3"/>'
    if kind == "Long-Short":
        return f'<path d="M{x},{y+14} L{x+14},{y}" stroke="{color}"/><circle cx="{x+4}" cy="{y+3}" r="2" fill="#ffd166"/><circle cx="{x+12}" cy="{y+11}" r="2" fill="#ffd166"/>'
    if kind == "Compressive":
        return f'<rect x="{x}" y="{y}" width="8" height="16" fill="none" stroke="{color}"/><path d="M{x+9},{y+8} H{x+14}" stroke="{color}"/><rect x="{x+14}" y="{y+4}" width="4" height="8" fill="#ffd166"/>'
    return f'<path d="M{x},{y+8} C{x+4},{y-2} {x+8},{y+18} {x+12},{y+8} S{x+20},{y+8} {x+20},{y+8}" fill="none" stroke="{color}" stroke-width="2"/>'


def build_failure_fingerprint(root: Path, output_dir: Path) -> dict:
    columns = ("retrieve_late", "xor", "order", "distractors_32", "length_extrapolation", "throughput")
    result: dict[str, dict] = {}
    raw_throughput: dict[str, float] = {}
    for label, experiment in FINGERPRINT_ROWS.items():
        retrieve = _seed_metrics(root, experiment, ("stress_retrieve", "synthetic_retrieval"))
        extrapolation = _seed_metrics(root, experiment, ("stress_retrieve_extrapolation",))
        combine = _seed_metrics(root, experiment, ("stress_combine", "synthetic_xor"))
        order = _seed_metrics(root, experiment, ("stress_order",))
        distract = _seed_metrics(root, experiment, ("stress_distract",))
        cells = {
            "retrieve_late": _quality_cell(retrieve, ("stress_breakdown", "position_0.95")),
            "xor": _quality_cell(combine, ("accuracy",)),
            "order": _quality_cell(order, ("accuracy",)),
            "distractors_32": _quality_cell(distract, ("stress_breakdown", "distractors_32")),
            "length_extrapolation": _quality_cell(extrapolation, ("stress_breakdown", "length_4096")),
        }
        experiment_root = _experiment_dir(root, experiment)
        benchmark_path = experiment_root / "benchmark.json"
        if not benchmark_path.exists():
            benchmark_path = experiment_root / "benchmark" / "metrics.json"
        if benchmark_path.exists():
            benchmark = json.loads(benchmark_path.read_text())
            value = benchmark.get("lengths", {}).get("4096", {}).get("documents_per_second")
            if value is not None:
                raw_throughput[label] = float(value)
        result[label] = {"experiment": experiment, "cells": cells}
    fastest = max(raw_throughput.values(), default=0.0)
    for label in result:
        value = raw_throughput.get(label)
        result[label]["cells"]["throughput"] = (
            (value / fastest, 0.0) if value is not None and fastest else (None, None)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = {
        "normalization": "balanced-task quality: (accuracy - 0.5) / 0.5; throughput: fraction of fastest",
        "rows": result,
    }
    atomic_write_json(output_dir / "failure_fingerprint.json", serializable)

    width, height = 1_050, 410
    left, top, cell_w, cell_h = 180, 78, 135, 48
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#eef2ff}.small{font-size:11px}.label{font-size:14px;font-weight:600}.title{font-size:22px;font-weight:700}</style>',
        '<text x="24" y="32" class="title">Long-context failure fingerprint</text>',
        '<text x="24" y="52" class="small">normalized score ± seed standard deviation · blank = result not available</text>',
    ]
    headings = ("retrieve-late", "XOR", "order", "32 distractors", "1K→4K", "throughput")
    for col, heading in enumerate(headings):
        svg.append(f'<text x="{left + col * cell_w + cell_w/2}" y="68" text-anchor="middle" class="small">{heading}</text>')
    for row, (label, row_data) in enumerate(result.items()):
        y = top + row * cell_h
        svg.append(f'<text x="24" y="{y + 30}" class="label">{label}</text>')
        svg.append(_icon_svg(label, 142, y + 14))
        for col, column in enumerate(columns):
            x = left + col * cell_w
            score, uncertainty = row_data["cells"][column]
            if score is None:
                fill, text_value = "#222a40", "—"
            else:
                red = int(210 * (1 - score) + 40 * score)
                green = int(62 * (1 - score) + 180 * score)
                blue = int(72 * (1 - score) + 120 * score)
                fill = f"rgb({red},{green},{blue})"
                text_value = f"{score:.2f} ± {uncertainty:.2f}"
            svg.append(f'<rect x="{x+2}" y="{y+4}" width="{cell_w-6}" height="{cell_h-6}" rx="5" fill="{fill}"/>')
            svg.append(f'<text x="{x + cell_w/2}" y="{y+30}" text-anchor="middle" class="small">{text_value}</text>')
    svg.append('</svg>')
    svg_path = output_dir / "failure_fingerprint.svg"
    svg_temp = svg_path.with_suffix(".svg.tmp")
    svg_temp.write_text("\n".join(svg) + "\n")
    svg_temp.replace(svg_path)
    return serializable
