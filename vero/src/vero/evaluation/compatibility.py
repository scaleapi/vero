"""Deprecated dataset views backed by canonical evaluation records."""

from __future__ import annotations

import warnings
from datetime import datetime

from vero.core.db.candidate import Candidate
from vero.core.db.database import Experiment, ExperimentDatabase
from vero.core.db.dataset import DatasetSample, DatasetSubset
from vero.core.db.result import (
    ExperimentResult,
    ExperimentResultStatus,
    SampleResult,
)
from vero.core.db.run import ExperimentRun
from vero.core.evaluation import BaseEvaluationParameters
from vero.evaluation.engine import EvaluationEngine
from vero.evaluation.models import (
    AllCases,
    CaseIds,
    CaseRange,
    CaseStatus,
    EvaluationAuthorization,
    EvaluationLimits,
    EvaluationRecord,
    EvaluationRequest,
    EvaluationSet,
    EvaluationStatus,
    ObjectiveSpec,
)
from vero.workspace import GitWorkspace, Workspace


def _legacy_sample_ids(evaluation_set: EvaluationSet) -> list[int] | None:
    selection = evaluation_set.selection
    if isinstance(selection, AllCases):
        return None
    if isinstance(selection, CaseIds):
        return [int(case_id) for case_id in selection.ids]
    if isinstance(selection, CaseRange):
        return list(range(selection.start, selection.stop))
    raise AssertionError(f"unsupported case selection: {selection}")


def evaluation_record_to_experiment(record: EvaluationRecord) -> Experiment:
    """Project a VeroTask record into the deprecated Experiment API."""
    evaluation_set = record.request.evaluation_set
    if evaluation_set.partition is None:
        raise ValueError("dataset compatibility views require a partition")

    try:
        selected_ids = _legacy_sample_ids(evaluation_set)
    except ValueError as error:
        raise ValueError("dataset compatibility views require integer case IDs") from error

    run = ExperimentRun(
        candidate=record.request.candidate,
        dataset_subset=DatasetSubset(
            dataset_id=evaluation_set.name,
            split=evaluation_set.partition,
            sample_ids=selected_ids,
        ),
    )
    sample_results: dict[int, SampleResult] = {}
    for case in record.report.cases:
        try:
            sample_id = int(case.case_id)
        except ValueError as error:
            raise ValueError(
                "dataset compatibility views require integer case IDs"
            ) from error

        execution_error = None
        evaluation_error = None
        error_traceback = None
        if case.status == CaseStatus.ERROR:
            for case_error in case.errors:
                if case_error.phase == "scoring":
                    evaluation_error = case_error.message
                elif execution_error is None or case_error.terminal:
                    execution_error = case_error.message
                    traceback_value = case_error.metadata.get("traceback")
                    if isinstance(traceback_value, str):
                        error_traceback = traceback_value

        metrics = dict(case.metrics)
        score = metrics.pop("score", None)
        input_value = case.input if isinstance(case.input, dict) else None
        sample_results[sample_id] = SampleResult(
            dataset_sample=DatasetSample(
                dataset_id=evaluation_set.name,
                split=evaluation_set.partition,
                sample_id=sample_id,
            ),
            commit=record.request.candidate.commit,
            result_id=record.id,
            input=input_value,
            output=case.output,
            score=score,
            feedback=case.feedback,
            metrics=metrics,
            error=execution_error,
            eval_error=evaluation_error,
            error_traceback=error_traceback,
            execution_trace=case.execution_trace,
            eval_trace=case.evaluation_trace,
        )

    result = ExperimentResult(
        id=record.id,
        run_id=run.id,
        status=(
            ExperimentResultStatus.SUCCESS
            if record.report.status == EvaluationStatus.SUCCESS
            else ExperimentResultStatus.FAILED
        ),
        sample_results=sample_results,
    )
    return Experiment(run=run, result=result)


def evaluation_database_to_experiment_database(
    database,
    *,
    backend_id: str = "vero-task",
) -> ExperimentDatabase:
    """Build a read-only-compatible dataset view from canonical records."""
    converted = ExperimentDatabase(id=database.id)
    for record in database.evaluations.values():
        if record.backend_id != backend_id:
            continue
        try:
            converted.add_experiment(evaluation_record_to_experiment(record))
        except ValueError:
            continue
    return converted


class VeroTaskEvaluatorAdapter:
    """Preserve ``Evaluator.evaluate`` while routing work through the engine."""

    def __init__(
        self,
        *,
        engine: EvaluationEngine,
        workspace: Workspace,
        backend_id: str,
        objective: ObjectiveSpec,
        task: str | None,
        compatibility_db: ExperimentDatabase,
    ):
        self.engine = engine
        self.workspace = workspace
        self.backend_id = backend_id
        self.objective = objective
        self.task = task
        self.compatibility_db = compatibility_db

    async def evaluate(
        self,
        commit: str,
        dataset_id: str,
        split: str,
        task: str | None = None,
        sample_ids: list[int] | None = None,
        db: ExperimentDatabase | None = None,
        evaluation_parameters: BaseEvaluationParameters | None = None,
        use_copy: bool | None = None,
        *,
        meter_budget: bool = False,
        add_to_compatibility_db: bool = True,
    ) -> Experiment:
        warnings.warn(
            "Evaluator.evaluate(dataset_id=..., split=...) is deprecated; use "
            "EvaluationEngine.evaluate_record()",
            DeprecationWarning,
            stacklevel=2,
        )
        if task is not None and self.task is not None and task != self.task:
            raise ValueError(
                f"approved VeroTask backend is configured for {self.task!r}, not {task!r}"
            )
        if use_copy is not None and use_copy != self.engine.evaluator.use_copy:
            raise ValueError(
                "per-call use_copy overrides are unsupported by the canonical evaluator"
            )
        if isinstance(self.workspace, GitWorkspace):
            commit = await self.workspace.resolve_ref(commit)
        parameters = evaluation_parameters or BaseEvaluationParameters()
        candidate = self.engine.database.candidates.get((self.workspace.name, commit))
        if candidate is None:
            candidate = Candidate(
                commit=commit,
                repo_name=self.workspace.name,
                created_at=datetime.now(),
            )
        selection = (
            CaseIds(ids=[str(sample_id) for sample_id in sample_ids])
            if sample_ids is not None
            else AllCases()
        )
        request = EvaluationRequest(
            candidate=candidate,
            evaluation_set=EvaluationSet(
                name=dataset_id,
                partition=split,
                selection=selection,
            ),
            parameters=parameters.task_params,
            limits=EvaluationLimits(
                timeout_seconds=parameters.timeout,
                case_timeout_seconds=parameters.sample_timeout,
                max_concurrency=parameters.max_concurrency,
                retry_config=parameters.retry_config,
            ),
        )
        record = await self.engine.evaluate_record(
            backend_id=self.backend_id,
            request=request,
            objective_spec=self.objective,
            authorization=EvaluationAuthorization(
                may_evaluate=True,
                meter_budget=meter_budget,
            ),
        )
        experiment = evaluation_record_to_experiment(record)
        if db is not None:
            db.add_experiment(experiment)
        elif add_to_compatibility_db and self.compatibility_db is not None:
            self.compatibility_db.add_experiment(experiment)
        return experiment
