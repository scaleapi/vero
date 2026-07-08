"""Tests for vero.harbor.protocol — sidecar wire types + redaction/summary."""

from vero.core.budget import SplitBudget
from vero.core.dataset.base import SplitAccess, SplitAccessLevel
from vero.core.db.candidate import Candidate
from vero.core.db.dataset import DatasetSubset
from vero.core.db.result import (
    ExperimentResult,
    ExperimentResultStatus,
    SampleResult,
)
from vero.core.db.run import ExperimentRun
from vero.harbor.protocol import (
    build_status,
    summarize_experiment,
    tier_for_split,
)


def _experiment(scores: list[float]) -> "object":
    from vero.core.db.database import Experiment
    from vero.core.db.dataset import DatasetSample

    run = ExperimentRun(
        candidate=Candidate(commit="abc123", repo_name="r"),
        dataset_subset=DatasetSubset(split="validation", dataset_id="ds1"),
    )
    sample_results = {
        i: SampleResult(
            dataset_sample=DatasetSample(sample_id=i, split="validation", dataset_id="ds1"),
            score=s,
        )
        for i, s in enumerate(scores)
    }
    result = ExperimentResult(
        run_id=run.id, status=ExperimentResultStatus.SUCCESS, sample_results=sample_results
    )
    return Experiment(run=run, result=result)


class TestTier:
    def test_listed_split_tier(self):
        accesses = [SplitAccess.no_access("test"), SplitAccess.non_viewable("validation")]
        assert tier_for_split("test", accesses) == SplitAccessLevel.no_access
        assert tier_for_split("validation", accesses) == SplitAccessLevel.non_viewable

    def test_unlisted_fails_closed_to_no_access(self):
        # An undeclared split must fail CLOSED, never default to viewable.
        assert tier_for_split("train", []) == SplitAccessLevel.no_access
        assert (
            tier_for_split("train", [SplitAccess.no_access("test")])
            == SplitAccessLevel.no_access
        )


class TestSummarize:
    def test_aggregate_only_no_per_sample(self):
        exp = _experiment([1.0, 0.0, 1.0])
        summary = summarize_experiment(exp, result_path="/x/y")
        assert summary.commit == "abc123"
        assert summary.split == "validation"
        assert summary.dataset_id == "ds1"
        assert summary.n_samples == 3
        assert summary.mean_score is not None
        # no per-sample field exists on the summary at all
        assert not any("sample" in k for k in summary.to_dict() if k != "n_samples")

    def test_budget_serialized(self):
        exp = _experiment([1.0])
        b = SplitBudget(split="validation", dataset_id="ds1", total_run_budget=5)
        d = summarize_experiment(exp, result_path=None, budget_remaining=b).to_dict()
        assert d["budget_remaining"]["remaining_run_budget"] == 5
        assert d["result_path"] is None


class TestBuildStatus:
    def test_lists_budgeted_splits_with_tier(self):
        budget = {
            ("train", "ds1"): SplitBudget(split="train", dataset_id="ds1", total_run_budget=10),
            ("validation", "ds1"): SplitBudget(split="validation", dataset_id="ds1", total_run_budget=3),
        }
        accesses = [SplitAccess.non_viewable("validation")]  # train left unlisted -> fails closed
        status = build_status(submit_enabled=True, budget=budget, split_accesses=accesses)

        assert status.submit_enabled is True
        by_split = {s["split"]: s for s in status.splits}
        # An unlisted split now fails CLOSED (no_access), not open (viewable): a
        # budgeted split must be tiered explicitly or the sidecar denies it.
        assert by_split["train"]["tier"] == str(SplitAccessLevel.no_access)
        assert by_split["train"]["agent_evaluable"] is False
        assert by_split["validation"]["tier"] == str(SplitAccessLevel.non_viewable)
        assert by_split["validation"]["agent_evaluable"] is True
        assert by_split["validation"]["remaining_run_budget"] == 3

    def test_advertises_subset_floor_on_non_viewable_only(self):
        budget = {
            ("train", "ds1"): SplitBudget(split="train", dataset_id="ds1", total_run_budget=10),
            ("validation", "ds1"): SplitBudget(split="validation", dataset_id="ds1", total_run_budget=3),
        }
        accesses = [
            SplitAccess.viewable("train"),
            SplitAccess.non_viewable("validation"),
        ]
        status = build_status(
            submit_enabled=False,
            budget=budget,
            split_accesses=accesses,
            k_anonymity_floor=5,
        )
        by_split = {s["split"]: s for s in status.splits}
        assert by_split["validation"]["min_subset_samples"] == 5
        assert by_split["train"]["min_subset_samples"] == 1
