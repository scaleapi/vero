from datetime import UTC, datetime, timedelta

import pytest

from vero.candidate import Candidate
from vero.evaluation import (
    BackendProvenance,
    BackendRegistry,
    CaseError,
    CaseResult,
    CaseStatus,
    ConstraintOperator,
    DisclosureLevel,
    EvaluationAcknowledgement,
    EvaluationCost,
    EvaluationRecord,
    EvaluationReport,
    EvaluationRequest,
    EvaluationSet,
    EvaluationStatus,
    EvaluationSummary,
    MetricAggregation,
    MetricConstraint,
    MetricSelector,
    ObjectiveResult,
    ObjectiveSpec,
    UnknownBackendError,
    compare_evaluation_records,
    evaluate_objective,
    project_evaluation,
    resolve_metric,
    select_best_evaluation,
)


def report(status=EvaluationStatus.SUCCESS):
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
def test_case_metric_aggregations_use_only_successful_cases(aggregation, expected):
    assert resolve_metric(
        report(), MetricSelector(metric="score", aggregation=aggregation)
    ) == expected


def test_case_metric_aggregation_penalizes_failed_and_missing_cases():
    selector = MetricSelector(
        metric="score",
        aggregation=MetricAggregation.MEAN,
        case_failure_value=0.0,
    )

    assert resolve_metric(report(), selector) == 1.0


@pytest.mark.parametrize(
    ("operator", "threshold", "feasible"),
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
def test_objective_supports_every_constraint_operator(operator, threshold, feasible):
    specification = ObjectiveSpec(
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

    result = evaluate_objective(report(), specification)

    assert result.value == 10.0
    assert result.feasible is feasible
    assert len(result.violations) == (0 if feasible else 1)


def make_record(
    candidate_id: str,
    value: float,
    *,
    feasible: bool = True,
    direction: str = "minimize",
    candidate_created_at: datetime | None = None,
) -> EvaluationRecord:
    specification = ObjectiveSpec(
        selector=MetricSelector(metric="score"),
        direction=direction,
    )
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    return EvaluationRecord(
        id=f"evaluation:{candidate_id}",
        request=EvaluationRequest(
            candidate=Candidate(
                id=candidate_id,
                version=f"version:{candidate_id}",
                created_at=candidate_created_at or created_at,
            )
        ),
        report=EvaluationReport(
            status=EvaluationStatus.SUCCESS,
            metrics={"score": value},
        ),
        backend_id="default",
        backend=BackendProvenance(
            name="fake",
            version="1",
            config_digest="0" * 64,
        ),
        objective_spec=specification,
        objective=ObjectiveResult(value=value, feasible=feasible),
        created_at=created_at,
        completed_at=created_at + timedelta(seconds=1),
    )


def test_selection_respects_direction_feasibility_and_tie_breaks():
    assert (
        select_best_evaluation(
            [
                make_record("a", 2.0, direction="maximize"),
                make_record("b", 1.0, direction="maximize"),
            ]
        ).request.candidate.id
        == "a"
    )
    feasible = make_record("feasible", 100.0)
    infeasible = make_record("infeasible", 1.0, feasible=False)
    assert compare_evaluation_records(feasible, infeasible) > 0
    assert select_best_evaluation([infeasible]) is None

    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    assert (
        select_best_evaluation(
            [
                make_record("b", 1.0, candidate_created_at=created_at),
                make_record("a", 1.0, candidate_created_at=created_at),
            ]
        ).request.candidate.id
        == "a"
    )


def test_disclosure_projections_exclude_case_details():
    record = make_record("a", 1.0)
    aggregate = project_evaluation(record, DisclosureLevel.AGGREGATE)
    hidden = project_evaluation(record, DisclosureLevel.NONE)

    assert isinstance(aggregate, EvaluationSummary)
    assert aggregate.candidate_id == "a"
    assert "cases" not in aggregate.model_dump()
    assert isinstance(hidden, EvaluationAcknowledgement)
    assert set(hidden.model_dump()) == {"evaluation_id", "status"}


class FakeBackend:
    provenance = BackendProvenance(
        name="fake",
        version="1",
        config_digest="0" * 64,
    )

    async def resolve_cost(self, evaluation_set: EvaluationSet) -> EvaluationCost:
        return EvaluationCost()

    async def evaluate(self, *, context, request):
        return EvaluationReport(status=EvaluationStatus.SUCCESS)


def test_backend_registry_is_explicit_and_rejects_unknown_ids():
    backend = FakeBackend()
    registry = BackendRegistry({"trusted": backend})

    assert registry.resolve("trusted") is backend
    with pytest.raises(UnknownBackendError):
        registry.resolve("missing")
    with pytest.raises(ValueError, match="already registered"):
        registry.register("trusted", backend)
