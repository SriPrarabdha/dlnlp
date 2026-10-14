import json
from pathlib import Path

from longdoc.orchestrator import build_jobs
from longdoc.status import write_run_status
from longdoc.suites import get_suite


def test_status_snapshot_separates_running_and_pending_jobs(tmp_path: Path):
    jobs = build_jobs(get_suite("stress_test", max_length=512))
    manifest = {
        "status": "running",
        "jobs": {job.id: {"status": "pending", "attempts": 0} for job in jobs},
    }
    manifest["jobs"][jobs[0].id] = {"status": "running", "attempts": 1}

    summary = write_run_status(tmp_path, "stress_test", "test", jobs, manifest)

    assert summary["progress"]["by_status"]["running"] == 1
    assert (tmp_path / "status" / "CHECKLIST.md").exists()
    saved = json.loads((tmp_path / "status" / "status_summary.json").read_text())
    assert saved["jobs"][0]["status"] == "running"
