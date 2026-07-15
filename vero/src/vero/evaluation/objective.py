"""Metric resolution, objective evaluation, ranking, and disclosure."""

from __future__ import annotations

import operator
import statistics
from functools import cmp_to_key
from typing import Iterable

from vero.evaluation.models import (
    CaseStatus,
    ConstraintOperator,
    ConstraintViolation,
    DisclosureLevel,
    EvaluationAcknowledgement,
    EvaluationRecord,
    EvaluationReport,
    EvaluationStatus,
    EvaluationSummary,
    MetricAggregation,
    MetricSelector,
    ObjectiveResult,
    ObjectiveSpec,
)


_OPERATORS = {
    ConstraintOperator.EQ: operator.eq,
    ConstraintOperator.NE: operator.ne,
    ConstraintOperator.LT: operator.lt,
    ConstraintOperator.LTE: operator.le,
    ConstraintOperator.GT: operator.gt,
    ConstraintOperator.GTE: operator.ge,
}


def resolve_metric(report: EvaluationReport, selector: MetricSelector) -> float | None:
    """Resolve a report metric or aggregate it across successful cases."""
    if selector.aggregation == MetricAggregation.REPORT:
        return report.metrics.get(selector.metric)

    values = [
        case.metrics[selector.metric]
        for case in report.cases
        if case.status == CaseStatus.SUCCESS and selector.metric in case.metrics
    ]
    if not values:
        return None
    if selector.aggregation == MetricAggregation.MEAN:
        return float(statistics.fmean(values))
    if selector.aggregation == MetricAggregation.MEDIAN:
        return float(statistics.median(values))
    if selector.aggregation == MetricAggregation.MIN:
        return min(values)
    if selector.aggregation == MetricAggregation.MAX:
        return max(values)
    raise AssertionError(f"unsupported aggregation: {selector.aggregation}")


def evaluate_objective(
    report: EvaluationReport,
    specification: ObjectiveSpec,
) -> ObjectiveResult:
    """Evaluate the optimization objective and all feasibility constraints."""
    value = resolve_metric(report, specification.selector)
    violations: list[ConstraintViolation] = []

    for constraint in specification.constraints:
        observed = resolve_metric(report, constraint.selector)
        if observed is None:
            violations.append(
                ConstraintViolation(
                    constraint=constraint,
                    observed=None,
                    reason=f"metric {constraint.selector.metric!r} is unavailable",
                )
            )
            continue
        if not _OPERATORS[constraint.operator](observed, constraint.value):
            violations.append(
                ConstraintViolation(
                    constraint=constraint,
                    observed=observed,
                    reason=(
                        f"observed {observed} does not satisfy "
                        f"{constraint.operator.value} {constraint.value}"
                    ),
                )
            )

    if report.status != EvaluationStatus.SUCCESS or value is None:
        return ObjectiveResult(value=specification.failure_value, feasible=False)
    if violations:
        return ObjectiveResult(value=value, feasible=False, violations=violations)
    return ObjectiveResult(value=value, feasible=True)


def _compare_results(
    left: ObjectiveResult,
    right: ObjectiveResult,
    direction: str,
) -> int:
    if left.feasible != right.feasible:
        return 1 if left.feasible else -1
    if left.value is None and right.value is None:
        return 0
    if left.value is None:
        return -1
    if right.value is None:
        return 1
    if left.value == right.value:
        return 0
    if direction == "maximize":
        return 1 if left.value > right.value else -1
    return 1 if left.value < right.value else -1


def compare_evaluation_records(left: EvaluationRecord, right: EvaluationRecord) -> int:
    """Compare compatible records with deterministic candidate tie-breaks."""
    if left.objective_spec is None or left.objective is None:
        raise ValueError("left record does not contain an objective")
    if right.objective_spec is None or right.objective is None:
        raise ValueError("right record does not contain an objective")
    if left.objective_spec != right.objective_spec:
        raise ValueError("records use different objective specifications")

    comparison = _compare_results(
        left.objective,
        right.objective,
        left.objective_spec.direction,
    )
    if comparison:
        return comparison

    left_created = left.request.candidate.created_at
    right_created = right.request.candidate.created_at
    if left_created != right_created:
        return 1 if left_created > right_created else -1

    left_id = left.request.candidate.id
    right_id = right.request.candidate.id
    if left_id == right_id:
        return 0
    return 1 if left_id < right_id else -1


def select_best_evaluation(
    records: Iterable[EvaluationRecord],
) -> EvaluationRecord | None:
    """Return the best feasible record, or ``None`` if none are feasible."""
    feasible = [
        record
        for record in records
        if record.objective is not None and record.objective.feasible
    ]
    if not feasible:
        return None
    return max(feasible, key=cmp_to_key(compare_evaluation_records))


def project_evaluation(
    record: EvaluationRecord,
    disclosure: DisclosureLevel,
) -> EvaluationRecord | EvaluationSummary | EvaluationAcknowledgement:
    """Project a private record into an approved disclosure shape."""
    if disclosure == DisclosureLevel.FULL:
        return record
    if disclosure == DisclosureLevel.NONE:
        return EvaluationAcknowledgement(
            evaluation_id=record.id,
            status=record.report.status,
        )

    counts = {status: 0 for status in CaseStatus}
    for case in record.report.cases:
        counts[case.status] += 1
    return EvaluationSummary(
        evaluation_id=record.id,
        candidate_id=record.request.candidate.id,
        candidate_version=record.request.candidate.version,
        backend_id=record.backend_id,
        evaluation_set=record.request.evaluation_set,
        status=record.report.status,
        metrics=dict(record.report.metrics),
        objective=record.objective,
        total_cases=len(record.report.cases),
        successful_cases=counts[CaseStatus.SUCCESS],
        errored_cases=counts[CaseStatus.ERROR],
        skipped_cases=counts[CaseStatus.SKIPPED],
    )
