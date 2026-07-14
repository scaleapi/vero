"""Read-only conversion from schema-v1 experiment data to canonical records."""

from __future__ import annotations

from vero.core.constants import default_minimum_score
from vero.core.db.database import ExperimentDatabase
from vero.core.db.result import ExperimentResultStatus
from vero.evaluation.models import (
    BackendProvenance,
    CaseError,
    CaseIds,
    CaseResult,
    CaseStatus,
    DiagnosticSeverity,
    EvaluationDiagnostic,
    EvaluationLimits,
    EvaluationRecord,
    EvaluationReport,
    EvaluationRequest,
    EvaluationSet,
    EvaluationStatus,
    MetricAggregation,
    MetricSelector,
    ObjectiveResult,
    ObjectiveSpec,
)


def compatibility_objective() -> ObjectiveSpec:
    return ObjectiveSpec(
        selector=MetricSelector(
            metric="score",
            aggregation=MetricAggregation.REPORT,
        ),
        direction="maximize",
        failure_value=default_minimum_score,
    )


def _convert_case(sample_id: int, sample) -> tuple[CaseResult, list[EvaluationDiagnostic]]:
    errors: list[CaseError] = []
    if sample.error:
        errors.append(
            CaseError(
                message=sample.error,
                phase="execution",
                metadata={"traceback": sample.error_traceback}
                if sample.error_traceback
                else {},
            )
        )
    if sample.eval_error:
        errors.append(CaseError(message=sample.eval_error, phase="scoring"))

    is_error = sample.is_error()
    if is_error and not errors:
        errors.append(
            CaseError(
                message="Legacy sample failed without a recorded error message",
                code="legacy_missing_error",
                phase="legacy",
            )
        )
    if errors and is_error:
        errors[-1] = errors[-1].model_copy(update={"terminal": True})

    metrics = dict(sample.metrics)
    diagnostics = []
    if sample.score is not None:
        if "score" in metrics:
            diagnostics.append(
                EvaluationDiagnostic(
                    code="legacy_score_metric_collision",
                    message="Explicit legacy sample score replaced custom metric 'score'",
                    severity=DiagnosticSeverity.WARNING,
                    phase="legacy_conversion",
                    metadata={"sample_id": sample_id},
                )
            )
        metrics["score"] = sample.score

    return (
        CaseResult(
            case_id=str(sample_id),
            status=CaseStatus.ERROR if is_error else CaseStatus.SUCCESS,
            metrics=metrics,
            input=sample.input,
            output=sample.output,
            feedback=sample.feedback,
            errors=errors,
            execution_trace=list(sample.execution_trace)
            if sample.execution_trace is not None
            else None,
            evaluation_trace=list(sample.eval_trace)
            if sample.eval_trace is not None
            else None,
            metadata={"legacy_sample_id": sample_id},
        ),
        diagnostics,
    )


def convert_experiment_database(database: ExperimentDatabase):
    from vero.evaluation.persistence import EvaluationDatabase

    converted = EvaluationDatabase(id=database.id)
    converted.datasets = {
        key: value.model_dump(mode="json") for key, value in database.datasets.items()
    }
    objective_spec = compatibility_objective()

    for experiment in database.get_experiments():
        run = experiment.run
        result = experiment.result
        subset = run.dataset_subset
        selection = (
            CaseIds(ids=[str(sample_id) for sample_id in subset.sample_ids])
            if subset.sample_ids
            else None
        )
        evaluation_set = EvaluationSet(
            name=subset.dataset_id,
            partition=subset.split,
            **({"selection": selection} if selection is not None else {}),
        )
        cases = []
        diagnostics = []
        for sample_id, sample in result.sample_results.items():
            case, case_diagnostics = _convert_case(sample_id, sample)
            cases.append(case)
            diagnostics.extend(case_diagnostics)

        report_status = (
            EvaluationStatus.SUCCESS
            if result.status == ExperimentResultStatus.SUCCESS
            else EvaluationStatus.FAILED
        )
        if result.status == ExperimentResultStatus.UNKNOWN:
            diagnostics.append(
                EvaluationDiagnostic(
                    code="legacy_unknown_status",
                    message="Legacy evaluation had unknown status and was converted as failed",
                    severity=DiagnosticSeverity.WARNING,
                    phase="legacy_conversion",
                )
            )
        score = result.score(fill_score=default_minimum_score)
        metrics = {
            "error_rate": result.error_rate(),
            "num_results": float(len(result.sample_results)),
        }
        if score is not None:
            metrics["score"] = score
        report = EvaluationReport(
            status=report_status,
            metrics=metrics,
            cases=cases,
            diagnostics=diagnostics,
            error="Legacy evaluation failed"
            if report_status == EvaluationStatus.FAILED
            else None,
        )
        objective = ObjectiveResult(
            value=(
                score
                if score is not None and report_status == EvaluationStatus.SUCCESS
                else objective_spec.failure_value
            ),
            feasible=report_status == EvaluationStatus.SUCCESS and score is not None,
        )
        created_at = run.candidate.created_at
        converted.add_evaluation(
            EvaluationRecord(
                id=result.id,
                request=EvaluationRequest(
                    candidate=run.candidate,
                    evaluation_set=evaluation_set,
                    limits=EvaluationLimits(),
                ),
                report=report,
                backend_id="vero-task",
                backend=BackendProvenance.from_config(
                    name="vero-task",
                    version="legacy-v1",
                    config={
                        "dataset_id": subset.dataset_id,
                        "split": subset.split,
                    },
                ),
                objective_spec=objective_spec,
                objective=objective,
                created_at=created_at,
                completed_at=created_at,
            )
        )
    return converted


def deserialize_legacy_database(data: dict):
    return convert_experiment_database(ExperimentDatabase.deserialize(data))
