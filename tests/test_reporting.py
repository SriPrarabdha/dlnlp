import json
from pathlib import Path

from longdoc.reporting import summarize_runs


def _write_completed_metrics(path: Path, accuracy: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"accuracy": accuracy}))
    (path.parent / "completion.json").write_text('{"status": "completed"}')


def test_summary_does_not_count_the_same_seed_from_two_layouts(tmp_path: Path):
    direct = tmp_path / "exact.baseline/finetune/imdb/seed_17/test_metrics.json"
    suite = (
        tmp_path
        / "head_to_head/main/cells/exact.baseline/finetune/imdb/seed_17/test_metrics.json"
    )
    _write_completed_metrics(direct, 0.25)
    _write_completed_metrics(suite, 0.75)
    suite.touch()
    summary = summarize_runs(tmp_path)
    assert summary["exact.baseline"]["imdb"]["accuracy"]["mean"] == 0.75
    assert summary["exact.baseline"]["imdb"]["accuracy"]["std"] == 0.0
