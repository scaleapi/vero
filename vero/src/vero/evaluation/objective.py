"""Objective evaluation, candidate ordering, and disclosure projections."""

from __future__ import annotations

import math
import statistics
from datetime import UTC, datetime
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
    EvaluationSummary,
    MetricAggregation,
    MetricSelector,
    ObjectiveResult,
    ObjectiveSpec,
)


def resolve_metric(report: EvaluationReport, selector: MetricSelector) -> float | None:
    """Resolve a report-level metric or aggregate it over successful cases."""
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
        return statistics.fmean(values)
    if selector.aggregation == MetricAggregation.MEDIAN:
        return float(statistics.median(values))
    if selector.aggregation == MetricAggregation.MIN:
        return min(values)
    if selector.aggregation == MetricAggregation.MAX:
        return max(values)
    raise AssertionError(f"unsupported metric aggregation: {selector.aggregation}")


def _satisfies(observed: float, operator: ConstraintOperator, expected: float) -> bool:
    if operator == ConstraintOperator.EQ:
        return observed == expected
    if operator == ConstraintOperator.NE:
        return observed != expected
    if operator == ConstraintOperator.LT:
        return observed < expected
    if operator == ConstraintOperator.LTE:
        return observed <= expected
    if operator == ConstraintOperator.GT:
        return observed > expected
    if operator == ConstraintOperator.GTE:
        return observed >= expected
    raise AssertionError(f"unsupported constraint operator: {operator}")


def evaluate_objective(
    report: EvaluationReport,
    spec: ObjectiveSpec,
) -> ObjectiveResult:
    """Compute the persisted objective result for a validated evaluation report."""
    violations: list[ConstraintViolation] = []
    for constraint in spec.constraints:
        observed = resolve_metric(report, constraint.selector)
        if observed is None:
            violations.append(
                ConstraintViolation(
                    constraint=constraint,
                    observed=None,
                    reason="constraint metric is missing",
                )
            )
        elif not _satisfies(observed, constraint.operator, constraint.value):
            violations.append(
                ConstraintViolation(
                    constraint=constraint,
                    observed=observed,
                    reason=(
                        f"observed value {observed} does not satisfy "
                        f"{constraint.operator.value} {constraint.value}"
                    ),
                )
            )

    objective_value = resolve_metric(report, spec.selector)
    report_succeeded = report.status.value == "success"
    feasible = report_succeeded and objective_value is not None and not violations
    if not report_succeeded or objective_value is None:
        objective_value = spec.failure_value

    return ObjectiveResult(
        value=objective_value,
        feasible=feasible,
        violations=violations,
    )


def _compare_objective_results(
    left: ObjectiveResult,
    right: ObjectiveResult,
    direction: str,
) -> int:
    if left.feasible != right.feasible:
        return 1 if left.feasible else -1
    if left.value is None or right.value is None:
        if left.value is None and right.value is None:
            return 0
        return -1 if left.value is None else 1
    if math.isclose(left.value, right.value, rel_tol=0.0, abs_tol=0.0):
        return 0
    if direction == "maximize":
        return 1 if left.value > right.value else -1
    return 1 if left.value < right.value else -1


def compare_evaluation_records(left: EvaluationRecord, right: EvaluationRecord) -> int:
    """Compare records using objective feasibility, direction, and stable tie-breaks."""
    if left.objective_spec is None or left.objective is None:
        raise ValueError("left record does not contain an objective")
    if right.objective_spec is None or right.objective is None:
        raise ValueError("right record does not contain an objective")
    if left.objective_spec != right.objective_spec:
        raise ValueError("records use different objective specifications")

    comparison = _compare_objective_results(
        left.objective,
        right.objective,
        left.objective_spec.direction,
    )
    if comparison:
        return comparison

    def normalized(timestamp: datetime) -> datetime:
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=UTC)
        return timestamp.astimezone(UTC)

    left_created = normalized(left.request.candidate.created_at)
    right_created = normalized(right.request.candidate.created_at)
    if left_created != right_created:
        return 1 if left_created > right_created else -1

    left_commit = left.request.candidate.commit
    right_commit = right.request.candidate.commit
    if left_commit == right_commit:
        return 0
    return 1 if left_commit < right_commit else -1


def select_best_evaluation(
    records: Iterable[EvaluationRecord],
) -> EvaluationRecord | None:
    """Return the best feasible record, or ``None`` when none are feasible."""
    feasible_records = [
        record
        for record in records
        if record.objective is not None and record.objective.feasible
    ]
    if not feasible_records:
        return None
    return max(feasible_records, key=cmp_to_key(compare_evaluation_records))


def project_evaluation(
    record: EvaluationRecord,
    disclosure: DisclosureLevel,
) -> EvaluationRecord | EvaluationSummary | EvaluationAcknowledgement:
    """Project a private evaluation record into an approved disclosure shape."""
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
        candidate_commit=record.request.candidate.commit,
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
