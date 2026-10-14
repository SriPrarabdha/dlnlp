from pathlib import Path

from longdoc.config import load_config
from longdoc.data.batching import EpochBatchSampler
from longdoc.orchestrator import SuiteRunner, build_jobs
from longdoc.suites import SUITES, get_suite


def test_every_paper_plan_has_unique_resolvable_dependencies():
    for paper in SUITES:
        jobs = build_jobs(get_suite(paper))
        identifiers = [job.id for job in jobs]
        assert len(identifiers) == len(set(identifiers))
        known = set(identifiers)
        assert all(set(job.dependencies) <= known for job in jobs)
        assert jobs[-1].kind == "report"


def test_shared_study_plans_include_expected_controls():
    head_jobs = build_jobs(get_suite("head_to_head"))
    assert {job.task for job in head_jobs if job.kind == "tfidf"} == {
        "arxiv", "scotus", "ecthr_a", "imdb"
    }
    assert len([job for job in head_jobs if job.kind == "curve"]) == 24
    assert any(job.cell == "exact.truncated512" for job in head_jobs)

    stress_jobs = build_jobs(get_suite("stress_test"))
    assert any(job.task == "stress_retrieve_extrapolation" for job in stress_jobs)
    assert len([job for job in stress_jobs if job.kind == "benchmark"]) == 6


def test_real_batch_order_can_be_recreated_at_epoch_boundary():
    sampler = EpochBatchSampler(size=19, batch_size=4, shuffle=True, seed=17)
    sampler.set_epoch(3)
    first = list(sampler)
    sampler.set_epoch(3)
    assert list(sampler) == first
    sampler.set_start_batch(2)
    assert list(sampler) == first[2:]
    sampler.set_start_batch(0)
    sampler.set_epoch(4)
    assert list(sampler) != first


def test_quick_roformer_plan_does_not_claim_4k_extrapolation(tmp_path: Path):
    config = load_config("configs/quick.yaml")
    config.output_dir = str(tmp_path)
    runner = SuiteRunner(config, "roformer")
    identifiers = [job["id"] for job in runner.plan()["jobs"]]
    assert not any("extrapolation" in identifier for identifier in identifiers)
