from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np

from longdoc.suites import HEADLINE_CELLS

PRIMARY_METRICS = {
    "arxiv": "accuracy", "scotus": "accuracy", "ecthr_a": "micro_f1", "imdb": "accuracy",
    "stress_retrieve": "accuracy", "stress_combine": "accuracy", "stress_order": "accuracy",
    "stress_distract": "accuracy", "stress_retrieve_extrapolation": "accuracy",
}
REAL_TASKS = ("arxiv", "scotus", "ecthr_a", "imdb")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def _average_descending_ranks(values: list[tuple[str, float]]) -> dict[str, float]:
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


def _line_plot(path: Path, title: str, series: dict[str, list[tuple[float, float]]], x_label: str, y_label: str) -> None:
    width, height = 920, 500
    left, right, top, bottom = 80, 30, 60, 70
    points = [point for values in series.values() for point in values]
    if not points:
        return
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmin == xmax:
        xmax += 1
    if ymin == ymax:
        ymin, ymax = ymin - 0.05, ymax + 0.05
    colors = ("#55c2ff", "#ffca66", "#70d69c", "#f58ca7", "#b69cff", "#ff8f5e")

    def sx(value: float) -> float:
        return left + (value - xmin) / (xmax - xmin) * (width - left - right)

    def sy(value: float) -> float:
        return top + (ymax - value) / (ymax - ymin) * (height - top - bottom)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#eef2ff}.small{font-size:12px}.title{font-size:22px;font-weight:700}</style>',
        f'<text x="{left}" y="34" class="title">{html.escape(title)}</text>',
        f'<path d="M{left},{top} V{height-bottom} H{width-right}" fill="none" stroke="#72809d"/>',
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle" class="small">{html.escape(x_label)}</text>',
        f'<text x="18" y="{height/2}" text-anchor="middle" class="small" transform="rotate(-90 18 {height/2})">{html.escape(y_label)}</text>',
    ]
    for index, (name, values) in enumerate(sorted(series.items())):
        color = colors[index % len(colors)]
        ordered = sorted(values)
        coordinates = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in ordered)
        svg.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x, y in ordered:
            svg.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.5" fill="{color}"/>')
        svg.append(f'<text x="{width-right-150}" y="{top+18*index}" class="small" fill="{color}">{html.escape(name)}</text>')
    for index in range(6):
        value = ymin + (ymax - ymin) * index / 5
        y = sy(value)
        svg.append(f'<text x="{left-10}" y="{y+4}" text-anchor="end" class="small">{value:.3f}</text>')
    svg.append("</svg>")
    _atomic_text(path, "\n".join(svg) + "\n")


def _bar_plot(path: Path, title: str, values: dict[str, float], y_label: str, lower_is_better: bool = False) -> None:
    if not values:
        return
    ordered = sorted(values.items(), key=lambda item: item[1], reverse=not lower_is_better)
    width, height, left, right, top, bottom = 940, 500, 75, 25, 60, 145
    high = max(value for _, value in ordered) * 1.12 or 1.0
    bar_width = (width - left - right) / len(ordered)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#eef2ff}.small{font-size:11px}.title{font-size:22px;font-weight:700}</style>',
        f'<text x="{left}" y="34" class="title">{html.escape(title)}</text>',
        f'<path d="M{left},{top} V{height-bottom} H{width-right}" fill="none" stroke="#72809d"/>',
        f'<text x="18" y="{height/2}" text-anchor="middle" class="small" transform="rotate(-90 18 {height/2})">{html.escape(y_label)}</text>',
    ]
    for index, (label, value) in enumerate(ordered):
        x = left + index * bar_width + 8
        bar_height = value / high * (height - top - bottom)
        y = height - bottom - bar_height
        svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width-16:.1f}" height="{bar_height:.1f}" rx="4" fill="#55c2ff"/>')
        svg.append(f'<text x="{x+(bar_width-16)/2:.1f}" y="{y-8:.1f}" text-anchor="middle" class="small">{value:.3f}</text>')
        svg.append(f'<text x="{x+(bar_width-16)/2:.1f}" y="{height-bottom+12}" class="small" transform="rotate(42 {x+(bar_width-16)/2:.1f} {height-bottom+12})">{html.escape(label)}</text>')
    svg.append("</svg>")
    _atomic_text(path, "\n".join(svg) + "\n")


def _heatmap_plot(
    path: Path,
    title: str,
    rows: list[str],
    columns: list[str],
    values: dict[tuple[str, str], tuple[float, float]],
) -> None:
    if not rows or not columns or not values:
        return
    cell_w, cell_h, left, top = 145, 48, 230, 90
    width, height = left + cell_w * len(columns) + 35, top + cell_h * len(rows) + 45
    available = [mean for mean, _ in values.values()]
    low, high = min(available), max(available)
    span = max(high - low, 1e-9)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#eef2ff}.small{font-size:11px}.label{font-size:13px;font-weight:600}.title{font-size:22px;font-weight:700}</style>',
        f'<text x="24" y="32" class="title">{html.escape(title)}</text>',
        '<text x="24" y="52" class="small">mean ± seed standard deviation · darker cells are better within this figure</text>',
    ]
    for col, label in enumerate(columns):
        svg.append(f'<text x="{left+col*cell_w+cell_w/2}" y="76" text-anchor="middle" class="small">{html.escape(label)}</text>')
    for row, label in enumerate(rows):
        y = top + row * cell_h
        svg.append(f'<text x="{left-12}" y="{y+29}" text-anchor="end" class="label">{html.escape(label)}</text>')
        for col, column in enumerate(columns):
            x = left + col * cell_w
            pair = values.get((label, column))
            if pair is None:
                fill, text_value = "#222a40", "—"
            else:
                mean, std = pair
                normalized = (mean - low) / span
                red = int(36 + 35 * normalized)
                green = int(53 + 125 * normalized)
                blue = int(83 + 68 * normalized)
                fill, text_value = f"rgb({red},{green},{blue})", f"{mean:.3f} ± {std:.3f}"
            svg.append(f'<rect x="{x+2}" y="{y+3}" width="{cell_w-6}" height="{cell_h-6}" rx="5" fill="{fill}"/>')
            svg.append(f'<text x="{x+cell_w/2}" y="{y+29}" text-anchor="middle" class="small">{text_value}</text>')
    svg.append("</svg>")
    _atomic_text(path, "\n".join(svg) + "\n")


def _seed_records(suite_root: Path, cell: str, task: str) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text())
        for path in sorted((suite_root / "cells" / cell / "finetune" / task).glob("seed_*/test_metrics.json"))
    ]


def _breakdown_series(
    suite_root: Path,
    task: str,
    prefix: str,
    cells: tuple[str, ...] = HEADLINE_CELLS,
) -> dict[str, list[tuple[float, float]]]:
    series: dict[str, list[tuple[float, float]]] = {}
    for cell in cells:
        records = _seed_records(suite_root, cell, task)
        keys = sorted({key for record in records for key in record.get("stress_breakdown", {}) if key.startswith(prefix)})
        points = []
        for ordinal, key in enumerate(keys):
            raw = key[len(prefix):]
            try:
                x = float(raw)
            except ValueError:
                x = float(ordinal)
            values = [float(record["stress_breakdown"][key]) for record in records if key in record.get("stress_breakdown", {})]
            if values:
                points.append((x, float(np.mean(values))))
        if points:
            series[cell] = points
    return series


def _breakdown_heatmap(
    suite_root: Path,
    output: Path,
    title: str,
    task: str,
    prefix: str,
    cells: tuple[str, ...],
) -> None:
    columns = sorted({
        key[len(prefix):]
        for cell in cells for record in _seed_records(suite_root, cell, task)
        for key in record.get("stress_breakdown", {}) if key.startswith(prefix)
    })
    values: dict[tuple[str, str], tuple[float, float]] = {}
    for cell in cells:
        records = _seed_records(suite_root, cell, task)
        for column in columns:
            key = prefix + column
            samples = [float(record["stress_breakdown"][key]) for record in records if key in record.get("stress_breakdown", {})]
            if samples:
                values[(cell, column)] = (
                    float(np.mean(samples)),
                    float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0,
                )
    _heatmap_plot(output, title, list(cells), columns, values)


def _write_presentation_plots(
    suite_root: Path,
    suite_name: str,
    summaries: dict[str, Any],
    mean_ranks: dict[str, float | None],
    efficiency: dict[str, Any],
) -> None:
    output = suite_root / "presentation"
    if suite_name == "head_to_head":
        rows = [cell for cell in HEADLINE_CELLS if cell in summaries]
        columns = list(REAL_TASKS)
        values = {
            (cell, task): (summaries[cell][task]["mean"], summaries[cell][task]["std"])
            for cell in rows for task in columns if task in summaries[cell]
        }
        tfidf = {}
        for task in columns:
            record_path = suite_root / "baselines" / "tfidf" / task / "metrics.json"
            if record_path.exists():
                record = json.loads(record_path.read_text())
                metric = PRIMARY_METRICS[task]
                if metric in record:
                    tfidf[("TF-IDF", task)] = (float(record[metric]), 0.0)
        heatmap_rows = [*rows, *(["TF-IDF"] if tfidf else [])]
        _heatmap_plot(output / "task_scores.svg", "Head-to-head downstream results", heatmap_rows, columns, {**values, **tfidf})
        ranks = {cell: float(mean_ranks[cell]) for cell in rows if mean_ranks.get(cell) is not None}
        _bar_plot(output / "mean_task_rank.svg", "Mean rank across four tasks", ranks, "mean rank (lower is better)", True)
        memory = {}
        throughput = {}
        parameters = {}
        frontier = {}
        for cell in rows:
            benchmark_result = efficiency.get(cell, {})
            lengths = benchmark_result.get("lengths", {})
            points = [
                (float(length), float(result["peak_memory_bytes"]) / (1024 ** 3))
                for length, result in lengths.items() if result.get("peak_memory_bytes") is not None
            ]
            if points:
                memory[cell] = points
            speed_points = [
                (float(length), float(result["documents_per_second"]))
                for length, result in lengths.items() if result.get("documents_per_second") is not None
            ]
            if speed_points:
                throughput[cell] = speed_points
                longest_speed = max(speed_points)[1]
                if cell in ranks:
                    frontier[cell] = [(longest_speed, ranks[cell])]
            if benchmark_result.get("parameters") is not None:
                parameters[cell] = float(benchmark_result["parameters"]) / 1_000_000
        _line_plot(output / "memory_by_length.svg", "Peak GPU memory by context length", memory, "context length", "peak allocated GiB")
        _line_plot(output / "throughput_by_length.svg", "Inference throughput by context length", throughput, "context length", "documents / second")
        _line_plot(output / "efficiency_frontier.svg", "Quality–speed efficiency frontier", frontier, "4K documents / second", "mean task rank (lower is better)")
        _bar_plot(output / "parameter_count.svg", "Model parameter counts", parameters, "millions of parameters")
        coverage_values = {
            (cell, task): (float(summaries[cell][task]["retained_token_coverage"]), 0.0)
            for cell in rows for task in columns
            if task in summaries[cell] and summaries[cell][task].get("retained_token_coverage") is not None
        }
        _heatmap_plot(output / "retained_token_coverage.svg", "Retained-token coverage", rows, columns, coverage_values)
    elif suite_name == "stress_test":
        # Reuse the canonical normalized heatmap with uncertainty and attention icons.
        from longdoc.reporting import build_failure_fingerprint

        build_failure_fingerprint(suite_root / "cells", output)
        _line_plot(
            output / "retrieval_by_position.svg", "Retrieval accuracy by evidence position",
            _breakdown_series(suite_root, "stress_retrieve", "position_"),
            "relative evidence position", "accuracy",
        )
        _line_plot(
            output / "distractor_robustness.svg", "Robustness to irrelevant markers",
            _breakdown_series(suite_root, "stress_distract", "distractors_"),
            "distractor markers", "accuracy",
        )
        for task, label in (
            ("stress_retrieve", "retrieve"), ("stress_combine", "XOR"),
            ("stress_order", "order"), ("stress_distract", "distract"),
        ):
            _line_plot(
                output / f"{task}_by_length.svg", f"{label} accuracy by sequence length",
                _breakdown_series(suite_root, task, "length_"), "sequence length", "accuracy",
            )
        _line_plot(
            output / "roformer_length_extrapolation.svg", "RoFormer train-1K/test-long extrapolation",
            _breakdown_series(suite_root, "stress_retrieve_extrapolation", "length_"),
            "test sequence length", "accuracy",
        )
    else:
        cells = tuple(summaries)
        tasks = sorted({task for cell in summaries.values() for task in cell})
        comparison = {
            (cell, task): (summaries[cell][task]["mean"], summaries[cell][task]["std"])
            for cell in cells for task in tasks if task in summaries[cell]
        }
        _heatmap_plot(
            output / "experiment_comparison.svg",
            f"{suite_name}: baseline, novelty, and ablation cells",
            list(cells), tasks, comparison,
        )
        if suite_name == "longformer":
            _line_plot(
                output / "evidence_position.svg", "Longformer accuracy by evidence position",
                _breakdown_series(suite_root, "stress_retrieve", "position_", cells),
                "relative evidence position", "accuracy",
            )
            _line_plot(
                output / "distractor_robustness.svg", "Longformer selector robustness",
                _breakdown_series(suite_root, "stress_distract", "distractors_", cells),
                "distractor markers", "accuracy",
            )
        elif suite_name == "linformer":
            _line_plot(
                output / "length_buckets.svg", "Linformer 2×2 factorial by document length",
                _breakdown_series(suite_root, "stress_retrieve", "length_", cells),
                "sequence length", "accuracy",
            )
            _line_plot(
                output / "position_shift.svg", "Linformer accuracy by cue position",
                _breakdown_series(suite_root, "stress_retrieve", "position_", cells),
                "relative evidence position", "accuracy",
            )
        elif suite_name == "longshort":
            for task, label in (("stress_combine", "XOR"), ("stress_order", "order")):
                _line_plot(
                    output / f"{task}_by_length.svg", f"Long-Short {label} by sequence length",
                    _breakdown_series(suite_root, task, "length_", cells),
                    "sequence length", "accuracy",
                )
                _breakdown_heatmap(
                    suite_root, output / f"{task}_segment_pairs.svg",
                    f"Long-Short {label} by evidence segment pair", task,
                    "length_4096/segment_pair_", cells,
                )
        elif suite_name == "compressive":
            _breakdown_heatmap(
                suite_root, output / "evidence_recency.svg",
                "Compressive Transformer by evidence recency", "stress_retrieve", "region_", cells,
            )
            _line_plot(
                output / "distractor_robustness.svg", "Compression robustness to distractors",
                _breakdown_series(suite_root, "stress_distract", "distractors_", cells),
                "distractor markers", "accuracy",
            )
        elif suite_name == "roformer":
            _line_plot(
                output / "order_by_length.svg", "RoFormer order accuracy by length",
                _breakdown_series(suite_root, "stress_order", "length_", cells),
                "sequence length", "accuracy",
            )
            _line_plot(
                output / "length_extrapolation.svg", "RoFormer train-1K/test-long extrapolation",
                _breakdown_series(suite_root, "stress_retrieve_extrapolation", "length_", cells),
                "test sequence length", "accuracy",
            )


def _numeric_leaves(value: Any, prefix: str = "", limit: int = 40) -> list[tuple[str, float]]:
    result: list[tuple[str, float]] = []
    if isinstance(value, (int, float)) and np.isfinite(value):
        return [(prefix or "value", float(value))]
    if isinstance(value, dict):
        for key, child in value.items():
            result.extend(_numeric_leaves(child, f"{prefix}/{key}" if prefix else str(key), limit))
            if len(result) >= limit:
                break
    elif isinstance(value, list):
        for index, child in enumerate(value[:limit]):
            result.extend(_numeric_leaves(child, f"{prefix}/{index}", limit))
            if len(result) >= limit:
                break
    return result[:limit]


def write_result_plot(result: dict[str, Any], output: Path, title: str) -> None:
    values = _numeric_leaves(result)
    if not values:
        return
    width = max(760, 34 * len(values) + 120)
    height, left, right, top, bottom = 480, 70, 25, 55, 155
    raw = [value for _, value in values]
    low, high = min(0.0, min(raw)), max(raw)
    if high == low:
        high = low + 1

    def sy(value: float) -> float:
        return top + (high - value) / (high - low) * (height - top - bottom)

    baseline = sy(0.0)
    bar_width = (width - left - right) / len(values)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#eef2ff}.small{font-size:10px}.title{font-size:21px;font-weight:700}</style>',
        f'<text x="{left}" y="32" class="title">{html.escape(title)}</text>',
        f'<path d="M{left},{top} V{height-bottom} H{width-right}" fill="none" stroke="#72809d"/>',
    ]
    for index, (label, value) in enumerate(values):
        x = left + index * bar_width + 3
        y = min(baseline, sy(value))
        bar_height = max(1.0, abs(sy(value) - baseline))
        svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(2, bar_width-6):.1f}" height="{bar_height:.1f}" fill="#55c2ff"/>')
        svg.append(f'<text x="{x+bar_width/2:.1f}" y="{height-bottom+8}" class="small" transform="rotate(55 {x+bar_width/2:.1f} {height-bottom+8})">{html.escape(label[-38:])}</text>')
    svg.append("</svg>")
    _atomic_text(output, "\n".join(svg) + "\n")


def write_suite_reports(suite_root: Path, suite_name: str | None = None) -> dict[str, Any]:
    suite_name = suite_name or suite_root.parent.name
    summaries: dict[str, Any] = {}
    score_series: dict[str, list[tuple[float, float]]] = {}
    validation_series: dict[str, list[tuple[float, float]]] = {}
    task_order: list[str] = []
    cells_root = suite_root / "cells"
    for cell_dir in sorted(cells_root.iterdir()) if cells_root.exists() else []:
        if not cell_dir.is_dir():
            continue
        cell_summary: dict[str, Any] = {}
        for task_dir in sorted((cell_dir / "finetune").iterdir()) if (cell_dir / "finetune").exists() else []:
            metric_name = PRIMARY_METRICS.get(task_dir.name, "accuracy")
            values = []
            coverages = []
            task_records = []
            for metric_path in sorted(task_dir.glob("seed_*/test_metrics.json")):
                metrics = json.loads(metric_path.read_text())
                task_records.append(metrics)
                if metric_name in metrics:
                    values.append(float(metrics[metric_name]))
                if "retained_token_coverage" in metrics:
                    coverages.append(float(metrics["retained_token_coverage"]))
                validation_path = metric_path.parent / "validation.jsonl"
                if validation_path.exists():
                    seed_points: dict[float, float] = {}
                    for line in validation_path.read_text().splitlines():
                        row = json.loads(line)
                        if metric_name in row:
                            seed_points[float(row["step"])] = float(row[metric_name])
                    validation_series.setdefault(
                        f"{cell_dir.name}:{task_dir.name}", []
                    ).extend(seed_points.items())
            if values:
                cell_summary[task_dir.name] = {
                    "metric": metric_name,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "seeds": len(values),
                    "retained_token_coverage": float(np.mean(coverages)) if coverages else None,
                }
                if task_dir.name not in task_order:
                    task_order.append(task_dir.name)
                score_series.setdefault(cell_dir.name, []).append(
                    (float(task_order.index(task_dir.name)), float(np.mean(values)))
                )
                plot_payload: dict[str, Any] = {"summary": cell_summary[task_dir.name]}
                breakdown_keys = sorted({
                    key for record in task_records
                    for key in record.get("stress_breakdown", {})
                })
                if breakdown_keys:
                    plot_payload["stress_breakdown_mean"] = {
                        key: float(np.mean([
                            record["stress_breakdown"][key]
                            for record in task_records if key in record.get("stress_breakdown", {})
                        ]))
                        for key in breakdown_keys
                    }
                write_result_plot(
                    plot_payload, cell_dir / "plots" / f"{task_dir.name}_test.svg",
                    f"{cell_dir.name}: {task_dir.name} test results",
                )
        summaries[cell_dir.name] = cell_summary
        _atomic_json(suite_root / "summaries" / "cells" / f"{cell_dir.name}.json", cell_summary)
        local_scores = {
            cell_dir.name: [
                (float(index), float(cell_summary[task]["mean"]))
                for index, task in enumerate(cell_summary)
            ]
        }
        _line_plot(
            cell_dir / "plots" / "task_scores.svg", f"{cell_dir.name}: task scores",
            local_scores, "task index (see cell summary)", "task-specific score",
        )
    rank_eligible = {
        cell for cell, values in summaries.items() if all(task in values for task in REAL_TASKS)
    }
    if suite_name == "head_to_head":
        rank_eligible &= set(HEADLINE_CELLS)
    task_ranks: dict[str, list[float]] = {cell: [] for cell in rank_eligible}
    for task in REAL_TASKS:
        ranked = [
            (cell, summaries[cell][task]["mean"]) for cell in rank_eligible
        ]
        for cell, rank in _average_descending_ranks(ranked).items():
            task_ranks[cell].append(rank)
    mean_ranks = {
        cell: (float(np.mean(task_ranks[cell])) if cell in task_ranks else None)
        for cell in summaries
    }
    baseline_summary: dict[str, Any] = {}
    for metric_path in sorted((suite_root / "baselines" / "tfidf").glob("*/metrics.json")):
        task = metric_path.parent.name
        record = json.loads(metric_path.read_text())
        metric = PRIMARY_METRICS.get(task, "accuracy")
        if metric in record:
            baseline_summary[task] = {"metric": metric, "value": float(record[metric])}
    suite_summary = {
        "cells": summaries, "mean_task_rank": mean_ranks, "task_axis": task_order,
        "baselines": {"tfidf": baseline_summary},
        "note": "Raw metrics are never averaged across tasks. Mean rank is reported only for cells with all four real tasks.",
    }
    _atomic_json(suite_root / "summaries" / "suite_summary.json", suite_summary)
    _line_plot(suite_root / "plots" / "task_scores.svg", "Task scores by experiment cell", score_series, "task index (see suite_summary.json)", "task-specific score")
    validation_series = {
        name: [
            (step, float(np.mean([value for candidate, value in points if candidate == step])))
            for step in sorted({candidate for candidate, _ in points})
        ]
        for name, points in validation_series.items()
    }
    _line_plot(suite_root / "plots" / "validation_curves.svg", "Validation curves", validation_series, "optimizer step", "task-specific score")

    compute_summary: dict[str, dict[str, Any]] = {}
    compute_series: dict[str, list[tuple[float, float]]] = {}
    for token_dir in sorted(suite_root.glob("curves/*/*/tokens_*")):
        cell, task = token_dir.parts[-3], token_dir.parts[-2]
        metric_name = PRIMARY_METRICS.get(task, "accuracy")
        values = []
        for metric_path in token_dir.glob(f"finetune/{task}/seed_*/test_metrics.json"):
            metrics = json.loads(metric_path.read_text())
            if metric_name in metrics:
                values.append(float(metrics[metric_name]))
        if not values:
            continue
        tokens = int(token_dir.name.rsplit("_", 1)[1])
        key = f"{cell}:{task}"
        compute_summary.setdefault(key, {})[str(tokens)] = {
            "metric": metric_name,
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
        compute_series.setdefault(key, []).append((float(tokens), float(np.mean(values))))
    _atomic_json(suite_root / "summaries" / "compute_curves.json", compute_summary)
    _line_plot(
        suite_root / "plots" / "accuracy_vs_pretraining_compute.svg",
        "Quality versus MLM token budget", compute_series, "non-padding MLM tokens",
        "task-specific score",
    )

    efficiency: dict[str, Any] = {}
    throughput_series: dict[str, list[tuple[float, float]]] = {}
    for benchmark_path in cells_root.glob("*/benchmark/metrics.json"):
        cell = benchmark_path.parents[1].name
        result = json.loads(benchmark_path.read_text())
        efficiency[cell] = result
        throughput_series[cell] = [
            (float(length), float(values["documents_per_second"]))
            for length, values in result.get("lengths", {}).items()
            if values.get("documents_per_second") is not None
        ]
    _atomic_json(suite_root / "summaries" / "efficiency.json", efficiency)
    _line_plot(
        suite_root / "plots" / "throughput_by_length.svg", "Inference throughput",
        throughput_series, "context length", "documents / second",
    )

    analysis_summary: dict[str, Any] = {}
    for analysis_dir in sorted(suite_root.glob("analyses/*/*")):
        if not analysis_dir.is_dir():
            continue
        records = [
            json.loads(path.read_text())
            for path in sorted(analysis_dir.glob("seed_*/result.json"))
        ]
        if not records:
            continue
        collected: dict[str, list[float]] = {}
        for record in records:
            for key, value in _numeric_leaves(record):
                collected.setdefault(key, []).append(value)
        name = f"{analysis_dir.parent.name}:{analysis_dir.name}"
        analysis_summary[name] = {
            "seeds": len(records),
            "scalars": {
                key: {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                }
                for key, values in collected.items()
            },
        }
    _atomic_json(suite_root / "summaries" / "analyses.json", analysis_summary)

    pretrain_series: dict[str, list[tuple[float, float]]] = {}
    for log_path in sorted(cells_root.glob("*/pretrain/train.jsonl")):
        values = []
        for line in log_path.read_text().splitlines():
            row = json.loads(line)
            values.append((float(row["tokens"]), float(row["loss"])))
        if values:
            pretrain_series[log_path.parents[1].name] = values
            _line_plot(
                log_path.parents[1] / "plots" / "pretraining_loss.svg",
                f"{log_path.parents[1].name}: MLM loss",
                {log_path.parents[1].name: values}, "non-padding tokens", "loss",
            )
    _line_plot(suite_root / "plots" / "pretraining_loss.svg", "MLM pretraining loss", pretrain_series, "non-padding tokens", "loss")
    suite_summary["compute_curves_file"] = "compute_curves.json"
    suite_summary["efficiency_file"] = "efficiency.json"
    suite_summary["analyses_file"] = "analyses.json"
    _write_presentation_plots(suite_root, suite_name, summaries, mean_ranks, efficiency)
    suite_summary["presentation_plots"] = sorted(
        str(path.relative_to(suite_root)) for path in (suite_root / "presentation").glob("*.svg")
    )
    _atomic_json(suite_root / "summaries" / "suite_summary.json", suite_summary)
    return suite_summary
