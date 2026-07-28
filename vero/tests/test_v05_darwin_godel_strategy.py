from __future__ import annotations

import pytest

from vero.candidate import Candidate
from vero.evaluation import (
    EvaluationSet,
    EvaluationStatus,
    EvaluationSummary,
    MetricSelector,
    ObjectiveResult,
    ObjectiveSpec,
)
from vero.optimization import DarwinGodelStrategy
from vero.optimization.models import OptimizationContext

OBJ_MAX = ObjectiveSpec(selector=MetricSelector(metric="score"), direction="maximize")


def _summary(candidate_id: str, value: float | None, *, feasible: bool = True) -> EvaluationSummary:
    return EvaluationSummary(
        evaluation_id=f"eval-{candidate_id}",
        candidate_id=candidate_id,
        candidate_version=f"{candidate_id}-v",
        backend_id="cmd",
        evaluation_set=EvaluationSet(name="perf"),
        status=EvaluationStatus.SUCCESS,
        metrics={"score": value} if value is not None else {},
        objective=ObjectiveResult(value=value, feasible=feasible),
        total_cases=1,
        successful_cases=1,
        errored_cases=0,
        skipped_cases=0,
    )


def _context(*, candidates, evaluations=(), best=None, baseline=None) -> OptimizationContext:
    baseline = baseline or next(iter(candidates.values()))
    return OptimizationContext(
        session_id="s",
        round=0,
        workspace=object(),  # unused by the strategy
        baseline=baseline,
        evaluations=tuple(evaluations),
        candidates=candidates,
        best=best,
    )


@pytest.mark.asyncio
async def test_dgm_seeds_from_baseline_when_archive_has_only_the_baseline():
    strat = DarwinGodelStrategy(objective=OBJ_MAX, num_offspring=3, seed=0)
    base = Candidate(id="base", version="base-v")
    ctx = _context(candidates={"base": base}, baseline=base)

    proposals = await strat.propose(ctx)

    assert len(proposals) == 3
    assert all(p.parent_id == "base" for p in proposals)
    assert all(p.metadata["strategy"] == "darwin-godel" for p in proposals)
    assert proposals[0].producer_id == "self"  # self-application producer by default


@pytest.mark.asyncio
async def test_dgm_keeps_the_whole_archive_reachable_including_low_and_unscored():
    # Unlike EvolutionaryStrategy's fittest-K population, every archived agent —
    # even a weak or not-yet-scored one — can be a parent (stepping stones).
    strat = DarwinGodelStrategy(objective=OBJ_MAX, num_offspring=300, seed=1)
    cands = {c: Candidate(id=c, version=f"{c}-v") for c in ["base", "strong", "weak", "unscored"]}
    evals = (_summary("base", 0.10), _summary("strong", 0.90), _summary("weak", 0.05))
    ctx = _context(candidates=cands, evaluations=evals, best=cands["strong"], baseline=cands["base"])

    parents = [p.parent_id for p in await strat.propose(ctx)]
    seen = set(parents)

    assert "weak" in seen and "unscored" in seen          # low + unscored stay reachable
    assert parents.count("strong") > parents.count("weak")  # performance still biases selection


@pytest.mark.asyncio
async def test_dgm_inverse_children_downweights_explored_lineages():
    # Equal performance, but the more-explored lineage gets a lower selection weight.
    strat = DarwinGodelStrategy(objective=OBJ_MAX, seed=0)
    strat._score = {"a": 0.8, "b": 0.8}
    strat._children = {"a": 3, "b": 0}

    assert strat._weight("a") == pytest.approx(0.8 / 4)
    assert strat._weight("b") == pytest.approx(0.8 / 1)
    assert strat._weight("a") < strat._weight("b")


@pytest.mark.asyncio
async def test_dgm_tracks_children_across_rounds():
    strat = DarwinGodelStrategy(objective=OBJ_MAX, num_offspring=4, seed=3)
    cands = {c: Candidate(id=c, version=f"{c}-v") for c in ["a", "b"]}
    ctx = _context(candidates=cands, evaluations=(_summary("a", 0.8), _summary("b", 0.8)), baseline=cands["a"])

    await strat.propose(ctx)
    await strat.propose(ctx)

    # Every selection increments the parent's child count; 2 rounds × 4 offspring = 8.
    assert sum(strat._children.values()) == 8
    assert strat._generation == 2


@pytest.mark.asyncio
async def test_dgm_minimize_direction_prefers_lower_values():
    strat = DarwinGodelStrategy(
        objective=ObjectiveSpec(selector=MetricSelector(metric="score"), direction="minimize"),
        num_offspring=200,
        base_weight=0.01,  # sharpen so the performance signal dominates
        seed=0,
    )
    cands = {c: Candidate(id=c, version=f"{c}-v") for c in ["hi", "lo"]}
    ctx = _context(candidates=cands, evaluations=(_summary("hi", 9.0), _summary("lo", 1.0)), baseline=cands["lo"])

    parents = [p.parent_id for p in await strat.propose(ctx)]

    # Under minimize, the lower value ("lo") is the higher performer → picked more.
    assert parents.count("lo") > parents.count("hi")
