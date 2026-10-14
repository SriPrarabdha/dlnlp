#!/usr/bin/env python3
"""Shrink checkpoints of runs that have finished, so a suite fits a small disk quota.

Only folders whose completion.json says "completed" are touched; unfinished runs
keep every file and can still resume.

- Fine-tuning seed folders: keep checkpoint_best.pt as model weights only
  (optimizer/scheduler/scaler state removed) and delete checkpoint_last.pt,
  checkpoint_latest.pt and checkpoint_step_*.pt. Analyses read only the
  weights in checkpoint_best.pt, and the suite treats the run as complete from
  completion.json, test_metrics.json, test_predictions.npz and checkpoint_best.pt.
- Pretraining folders: keep checkpoint_last.pt and checkpoint_tokens_*.pt as
  weights only and delete checkpoint_latest.pt and checkpoint_step_*.pt.

Usage (safe to run repeatedly, also while the suite is running):
    python scripts/compact_finished.py /path/to/results/linformer/main
"""
import json
import os
import sys
from pathlib import Path

import torch

DROP = ("optimizer", "scheduler", "scaler", "data_state")


def finished(folder: Path) -> bool:
    try:
        return json.loads((folder / "completion.json").read_text()).get("status") == "completed"
    except (OSError, json.JSONDecodeError):
        return False


def slim(path: Path) -> int:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not any(key in state for key in DROP):
        return 0
    before = path.stat().st_size
    temporary = path.with_suffix(".pt.slim")
    torch.save({key: value for key, value in state.items() if key not in DROP}, temporary)
    os.replace(temporary, path)
    return before - path.stat().st_size


def remove(path: Path) -> int:
    if not path.exists():
        return 0
    size = path.stat().st_size if path.stat().st_nlink == 1 else 0
    path.unlink()
    return size


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: compact_finished.py SUITE_RUN_DIR")
    freed = 0
    for completion in Path(sys.argv[1]).rglob("completion.json"):
        folder = completion.parent
        if not finished(folder):
            continue
        if folder.name.startswith("seed_"):
            steps = [path.name for path in folder.glob("checkpoint_step_*.pt")]
            for name in ("checkpoint_last.pt", "checkpoint_latest.pt", *steps):
                freed += remove(folder / name)
            if (folder / "checkpoint_best.pt").exists():
                freed += slim(folder / "checkpoint_best.pt")
        elif folder.name == "pretrain":
            steps = [path.name for path in folder.glob("checkpoint_step_*.pt")]
            for name in ("checkpoint_latest.pt", *steps):
                freed += remove(folder / name)
            for path in (folder / "checkpoint_last.pt", *folder.glob("checkpoint_tokens_*.pt")):
                if path.exists():
                    freed += slim(path)
    print(f"freed {freed / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
