from pathlib import Path

import pytest
import torch

from longdoc.config import RunConfig
from longdoc.training.trainer import _resume_checkpoint, _scheduler


def test_zero_warmup_does_not_zero_the_only_training_step():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    scheduler = _scheduler(optimizer, total_steps=1, warmup_ratio=0.0)
    assert scheduler.get_last_lr() == [1e-3]


def test_resume_rejects_a_changed_scientific_configuration(tmp_path: Path):
    original = RunConfig()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    torch.save(
        {"config": original.to_dict(), "status": "completed"},
        run_dir / "checkpoint_last.pt",
    )
    changed = RunConfig()
    changed.train.learning_rate *= 2
    with pytest.raises(ValueError, match="configuration does not match"):
        _resume_checkpoint(changed, run_dir)


def test_resume_allows_changing_activation_checkpointing(tmp_path: Path):
    original = RunConfig()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "checkpoint_last.pt"
    torch.save(
        {"config": original.to_dict(), "status": "completed"},
        checkpoint,
    )
    changed = RunConfig()
    changed.train.gradient_checkpointing = not original.train.gradient_checkpointing

    assert _resume_checkpoint(changed, run_dir) == checkpoint
