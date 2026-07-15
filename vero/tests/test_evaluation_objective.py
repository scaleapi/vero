from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from vero.core.db.candidate import Candidate
from vero.evaluation import (
    BackendProvenance,
    CaseError,
    CaseResult,
    CaseStatus,
    ConstraintOperator,
    EvaluationRecord,
    EvaluationReport,
    EvaluationRequest,
    EvaluationSet,
    EvaluationStatus,
    MetricAggregation,
    MetricConstraint,
    MetricSelector,
    ObjectiveResult,
    ObjectiveSpec,
    compare_evaluation_records,
    evaluate_objective,
    resolve_metric,
    select_best_evaluation,
)


def _report(status=EvaluationStatus.SUCCESS):
    return EvaluationReport(
        status=status,
        metrics={"latency": 10.0, "correct": 1.0},
        cases=[
            CaseResult(
                case_id="1",
                status=CaseStatus.SUCCESS,
                metrics={"score": 1.0},
            ),
            CaseResult(
                case_id="2",
                status=CaseStatus.SUCCESS,
                metrics={"score": 3.0},
            ),
            CaseResult(
                case_id="3",
                status=CaseStatus.ERROR,
                metrics={"score": 100.0},
                errors=[CaseError(message="failed", terminal=True)],
            ),
            CaseResult(
                case_id="4",
                status=CaseStatus.SKIPPED,
                metrics={"score": 200.0},
            ),
        ],
    )


@pytest.mark.parametrize(
    ("aggregation", "expected"),
    [
        (MetricAggregation.MEAN, 2.0),
        (MetricAggregation.MEDIAN, 2.0),
        (MetricAggregation.MIN, 1.0),
        (MetricAggregation.MAX, 3.0),
    ],
)
def test_case_metric_aggregations_include_only_successful_cases(
    aggregation, expected
):
    selector = MetricSelector(metric="score", aggregation=aggregation)

    assert resolve_metric(_report(), selector) == expected


def test_report_metric_selector_reads_only_top_level_metrics():
    assert resolve_metric(
        _report(), MetricSelector(metric="latency", aggregation="report")
    ) == 10.0
    assert (
        resolve_metric(_report(), MetricSelector(metric="score", aggregation="report"))
        is None
    )


@pytest.mark.parametrize(
    ("operator", "threshold", "expected_feasible"),
    [
        (ConstraintOperator.EQ, 1.0, True),
        (ConstraintOperator.NE, 0.0, True),
        (ConstraintOperator.LT, 2.0, True),
        (ConstraintOperator.LTE, 1.0, True),
        (ConstraintOperator.GT, 0.0, True),
        (ConstraintOperator.GTE, 1.0, True),
        (ConstraintOperator.EQ, 0.0, False),
    ],
)
def test_objective_evaluates_every_constraint_operator(
    operator, threshold, expected_feasible
):
    spec = ObjectiveSpec(
        selector=MetricSelector(metric="latency"),
        direction="minimize",
        constraints=[
            MetricConstraint(
                selector=MetricSelector(metric="correct"),
                operator=operator,
                value=threshold,
            )
        ],
    )

    result = evaluate_objective(_report(), spec)

    assert result.value == 10.0
    assert result.feasible is expected_feasible
    assert len(result.violations) == (0 if expected_feasible else 1)


def test_objective_collects_all_constraint_violations():
    spec = ObjectiveSpec(
        selector=MetricSelector(metric="latency"),
        direction="minimize",
        constraints=[
            MetricConstraint(
                selector=MetricSelector(metric="correct"),
                operator="==",
                value=0.0,
            ),
            MetricConstraint(
                selector=MetricSelector(metric="missing"),
                operator=">",
                value=0.0,
            ),
        ],
    )

    result = evaluate_objective(_report(), spec)

    assert result.feasible is False
    assert len(result.violations) == 2
    assert result.violations[1].observed is None


@pytest.mark.parametrize("status", [EvaluationStatus.SUCCESS, EvaluationStatus.FAILED])
def test_failure_value_is_used_for_missing_metric_or_failed_report(status):
    spec = ObjectiveSpec(
        selector=MetricSelector(metric="missing" if status == "success" else "latency"),
        direction="maximize",
        failure_value=-1.0,
    )

    result = evaluate_objective(_report(status), spec)

    assert result == ObjectiveResult(value=-1.0, feasible=False)


def _record(
    *,
    commit: str,
    value: float,
    feasible: bool = True,
    direction: str = "minimize",
    candidate_created_at: datetime | None = None,
) -> EvaluationRecord:
    objective_spec = ObjectiveSpec(
        selector=MetricSelector(metric="score"),
        direction=direction,
    )
    created_at = datetime(2026, 1, 1)
    return EvaluationRecord(
        id=str(uuid4()),
        request=EvaluationRequest(
            candidate=Candidate(
                commit=commit,
                repo_name="repo",
                created_at=candidate_created_at or created_at,
            ),
            evaluation_set=EvaluationSet(),
        ),
        report=EvaluationReport(
            status=EvaluationStatus.SUCCESS,
            metrics={"score": value},
        ),
        backend_id="default",
        backend=BackendProvenance(
            name="fake", version="1", config_digest="0" * 64
        ),
        objective_spec=objective_spec,
        objective=ObjectiveResult(value=value, feasible=feasible),
        created_at=created_at,
        completed_at=created_at + timedelta(seconds=1),
    )


@pytest.mark.parametrize(
    ("direction", "first", "second", "winner"),
    [
        ("maximize", 2.0, 1.0, "a"),
        ("minimize", 2.0, 1.0, "b"),
    ],
)
def test_best_evaluation_respects_objective_direction(
    direction, first, second, winner
):
    records = [
        _record(commit="a", value=first, direction=direction),
        _record(commit="b", value=second, direction=direction),
    ]

    assert select_best_evaluation(records).request.candidate.commit == winner


def test_feasible_record_always_outranks_infeasible_record():
    feasible = _record(commit="feasible", value=100.0, feasible=True)
    infeasible = _record(commit="infeasible", value=1.0, feasible=False)

    assert compare_evaluation_records(feasible, infeasible) > 0
    assert select_best_evaluation([infeasible, feasible]) is feasible
    assert select_best_evaluation([infeasible]) is None


def test_equal_objectives_use_created_at_then_commit_tie_breaks():
    older = datetime(2026, 1, 1)
    newer = datetime(2026, 1, 2)
    assert (
        select_best_evaluation(
            [
                _record(commit="a", value=1.0, candidate_created_at=older),
                _record(commit="z", value=1.0, candidate_created_at=newer),
            ]
        ).request.candidate.commit
        == "z"
    )
    assert (
        select_best_evaluation(
            [
                _record(commit="b", value=1.0, candidate_created_at=older),
                _record(commit="a", value=1.0, candidate_created_at=older),
            ]
        ).request.candidate.commit
        == "a"
    )


def test_equal_objectives_normalize_mixed_timestamp_awareness():
    same_naive = datetime(2026, 1, 1)
    same_aware = datetime(2026, 1, 1, tzinfo=UTC)

    winner = select_best_evaluation(
        [
            _record(commit="b", value=1.0, candidate_created_at=same_naive),
            _record(commit="a", value=1.0, candidate_created_at=same_aware),
        ]
    )

    assert winner.request.candidate.commit == "a"
