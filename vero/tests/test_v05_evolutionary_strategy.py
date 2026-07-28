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
from vero.optimization import EvolutionaryStrategy
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
async def test_seeds_offspring_from_baseline_before_any_feasible_candidate():
    strat = EvolutionaryStrategy(objective=OBJ_MAX, num_offspring=3, seed=0)
    baseline = Candidate(id="base", version="base-v")
    ctx = _context(candidates={"base": baseline}, baseline=baseline)

    proposals = await strat.propose(ctx)

    assert len(proposals) == 3
    assert all(p.parent_id == "base" for p in proposals)
    assert len({p.id for p in proposals}) == 3  # unique proposal ids
    assert all(p.metadata["strategy"] == "evolutionary" for p in proposals)
    assert all(p.metadata["generation"] == 0 for p in proposals)


@pytest.mark.asyncio
async def test_selects_parents_from_the_fittest_population():
    strat = EvolutionaryStrategy(
        objective=OBJ_MAX,
        num_offspring=6,
        population_size=2,
        tournament_size=1,  # each pick is a uniform draw from the top-2
        seed=0,
    )
    candidates = {cid: Candidate(id=cid, version=f"{cid}-v") for cid in ["base", "a", "b", "c"]}
    evaluations = (
        _summary("base", 0.10),
        _summary("a", 0.90),
        _summary("b", 0.80),
        _summary("c", 0.20),
        _summary("d", None, feasible=False),  # infeasible: excluded from the population
    )
    ctx = _context(
        candidates=candidates,
        evaluations=evaluations,
        best=candidates["a"],
        baseline=candidates["base"],
    )

    proposals = await strat.propose(ctx)

    assert len(proposals) == 6
    # Population is the top-2 by value: {a: 0.9, b: 0.8}; the weaker/infeasible ones
    # and the unknown "d" never become parents.
    assert {p.parent_id for p in proposals} <= {"a", "b"}
    assert "c" not in {p.parent_id for p in proposals}


@pytest.mark.asyncio
async def test_minimize_direction_prefers_lower_values():
    strat = EvolutionaryStrategy(
        objective=ObjectiveSpec(
            selector=MetricSelector(metric="score"), direction="minimize"
        ),
        num_offspring=5,
        population_size=1,
        tournament_size=1,
        seed=0,
    )
    candidates = {cid: Candidate(id=cid, version=f"{cid}-v") for cid in ["hi", "lo"]}
    ctx = _context(
        candidates=candidates,
        evaluations=(_summary("hi", 9.0), _summary("lo", 1.0)),
        best=candidates["lo"],
    )

    proposals = await strat.propose(ctx)

    # Under minimize, the single-slot population is the lowest value ("lo").
    assert {p.parent_id for p in proposals} == {"lo"}
