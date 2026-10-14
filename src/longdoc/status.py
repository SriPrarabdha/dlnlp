from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from longdoc.plotting import PRIMARY_METRICS
from longdoc.training.operations import atomic_write_json, format_duration

TERMINAL = {"completed", "failed", "blocked", "skipped"}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _job_result(root: Path, job: Any) -> dict[str, Any]:
    if job.kind == "finetune":
        metric = PRIMARY_METRICS.get(str(job.task), "accuracy")
        paths = sorted((root / "cells" / str(job.cell) / "finetune" / str(job.task)).glob("seed_*/test_metrics.json"))
        values = [float(record[metric]) for path in paths if (record := _read_json(path)) and metric in record]
        if values:
            return {
                "metric": metric,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "seeds": len(values),
            }
    elif job.kind == "tfidf":
        record = _read_json(root / "baselines" / "tfidf" / str(job.task) / "metrics.json")
        metric = PRIMARY_METRICS.get(str(job.task), "accuracy")
        if record and metric in record:
            return {"metric": metric, "value": float(record[metric])}
    elif job.kind == "benchmark":
        record = _read_json(root / "cells" / str(job.cell) / "benchmark" / "metrics.json")
        if record and record.get("lengths"):
            length = max(record["lengths"], key=lambda value: int(value))
            values = record["lengths"][length]
            return {
                "length": int(length),
                "documents_per_second": values.get("documents_per_second"),
                "peak_memory_mb": (
                    float(values["peak_memory_bytes"]) / (1024 * 1024)
                    if values.get("peak_memory_bytes") is not None else None
                ),
            }
    elif job.kind == "curve":
        record = _read_json(root / "curves" / str(job.cell) / str(job.task) / "complete.json")
        if record:
            return {"token_milestones": record.get("token_milestones", [])}
    elif job.kind == "analysis" and job.analysis is not None:
        paths = sorted((root / "analyses" / str(job.cell) / job.analysis.name).glob("seed_*/result.json"))
        if paths:
            return {"seeds": len(paths), "kind": job.analysis.kind}
    elif job.kind == "pretrain":
        checkpoint = root / "cells" / str(job.cell) / "pretrain" / "checkpoint_last.pt"
        if checkpoint.exists():
            return {"checkpoint": str(checkpoint.relative_to(root))}
    return {}


def _compact_result(result: dict[str, Any]) -> str:
    if not result:
        return "—"
    if "mean" in result:
        return f"{result['metric']}={result['mean']:.4f} ± {result['std']:.4f} ({result['seeds']} seeds)"
    if "value" in result:
        return f"{result['metric']}={result['value']:.4f}"
    if "documents_per_second" in result:
        speed = result.get("documents_per_second")
        memory = result.get("peak_memory_mb")
        speed_text = f"{speed:.2f} docs/s" if isinstance(speed, (int, float)) else "n/a"
        memory_text = f", {memory:.0f} MB" if isinstance(memory, (int, float)) else ""
        return f"{result['length']} tokens: {speed_text}{memory_text}"
    if "token_milestones" in result:
        return "milestones=" + ",".join(map(str, result["token_milestones"]))
    if "seeds" in result:
        return f"{result['kind']} ({result['seeds']} seeds)"
    if "checkpoint" in result:
        return result["checkpoint"]
    return "—"


def write_run_status(root: Path, suite: str, run_id: str, jobs: Iterable[Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """Write a machine-readable snapshot and a human-readable live checklist."""
    job_list = list(jobs)
    rows = []
    for job in job_list:
        state = manifest.get("jobs", {}).get(job.id, {"status": "pending"})
        result = _job_result(root, job)
        rows.append({
            "id": job.id,
            "kind": job.kind,
            "cell": job.cell,
            "task": job.task,
            "status": state.get("status", "pending"),
            "attempts": int(state.get("attempts", 0)),
            "duration_seconds": (
                state.get("duration_seconds")
                if state.get("duration_seconds") is not None
                else time.time() - float(state["started_at"])
                if state.get("status") == "running" and state.get("started_at") is not None
                else None
            ),
            "started_at": state.get("started_at"),
            "finished_at": state.get("finished_at"),
            "blocked_by": state.get("blocked_by", []),
            "error": state.get("error"),
            "result": result,
        })
    counts = Counter(row["status"] for row in rows)
    completed = counts.get("completed", 0) + counts.get("skipped", 0)
    summary = {
        "format_version": 1,
        "suite": suite,
        "run_id": run_id,
        "run_status": manifest.get("status", "pending"),
        "eta_seconds": manifest.get("eta_seconds"),
        "updated_at": time.time(),
        "progress": {
            "completed": completed,
            "total": len(rows),
            "percent": 100.0 * completed / len(rows) if rows else 100.0,
            "by_status": dict(sorted(counts.items())),
        },
        "jobs": rows,
    }
    status_dir = root / "status"
    atomic_write_json(status_dir / "status_summary.json", summary)

    lines = [
        f"# {suite} — live experiment checklist",
        "",
        f"Run: `{run_id}`  ",
        f"State: **{summary['run_status']}**  ",
        f"Progress: **{completed}/{len(rows)} ({summary['progress']['percent']:.1f}%)**  ",
        f"Suite ETA: **{format_duration(summary['eta_seconds'])}**  ",
        f"Updated: `{time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(summary['updated_at']))}`",
        "",
        "This file is regenerated whenever the suite changes state. `status_summary.json` is the machine-readable equivalent.",
        "",
    ]
    sections = (
        ("Running now", {"running"}),
        ("Completed", {"completed", "skipped"}),
        ("Pending", {"pending", "interrupted"}),
        ("Failed or blocked", {"failed", "blocked"}),
    )
    for title, statuses in sections:
        selected = [row for row in rows if row["status"] in statuses]
        lines.extend([f"## {title} ({len(selected)})", ""])
        if not selected:
            lines.extend(["None.", ""])
            continue
        lines.extend([
            "| Done | Sub-experiment | State | Attempts | Time | Result |",
            "|---:|---|---|---:|---:|---|",
        ])
        for row in selected:
            checked = "[x]" if row["status"] in {"completed", "skipped"} else "[ ]"
            duration = format_duration(row["duration_seconds"])
            result = _compact_result(row["result"]).replace("|", "\\|")
            error = str(row.get("error") or "").replace("|", "\\|").replace("\n", " ")[:160]
            if error:
                result = f"{result}; error: {error}"
            lines.append(
                f"| {checked} | `{row['id']}` | {row['status']} | {row['attempts']} | {duration} | {result} |"
            )
        lines.append("")
    checklist = status_dir / "CHECKLIST.md"
    checklist.parent.mkdir(parents=True, exist_ok=True)
    temporary = checklist.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(checklist)
    return summary
