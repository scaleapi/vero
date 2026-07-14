from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from vero.core.db.candidate import Candidate
from vero.evaluation import (
    AllCases,
    BackendProvenance,
    CaseError,
    CaseIds,
    CaseRange,
    CaseResult,
    CaseStatus,
    DiagnosticSeverity,
    DisclosureLevel,
    EvaluationArtifact,
    EvaluationDiagnostic,
    EvaluationRecord,
    EvaluationReport,
    EvaluationRequest,
    EvaluationSet,
    EvaluationStatus,
    MetricAggregation,
    MetricSelector,
    ObjectiveResult,
    ObjectiveSpec,
    project_evaluation,
)


@pytest.mark.parametrize(
    "selection",
    [
        AllCases(),
        CaseIds(ids=["case-2", "case-7"]),
        CaseRange(stop=10),
        CaseRange(start=10, stop=20),
    ],
)
def test_evaluation_set_round_trips_case_selection(selection):
    evaluation_set = EvaluationSet(
        name="performance",
        partition="validation",
        selection=selection,
    )

    restored = EvaluationSet.model_validate_json(evaluation_set.model_dump_json())

    assert restored == evaluation_set
    assert type(restored.selection) is type(selection)
    assert restored.budget_key("command") == "command:performance:validation"


@pytest.mark.parametrize(
    ("model", "kwargs"),
    [
        (EvaluationSet, {"name": " "}),
        (EvaluationSet, {"partition": ""}),
        (CaseIds, {"ids": []}),
        (CaseIds, {"ids": ["same", "same"]}),
        (CaseIds, {"ids": [""]}),
        (CaseRange, {"start": -1, "stop": 2}),
        (CaseRange, {"start": 2, "stop": 2}),
        (CaseRange, {"start": 3, "stop": 2}),
    ],
)
def test_case_selection_rejects_invalid_values(model, kwargs):
    with pytest.raises(ValidationError):
        model(**kwargs)


def test_request_fingerprint_uses_canonical_json():
    candidate = Candidate(commit="abc", repo_name="demo")
    first = EvaluationRequest(
        candidate=candidate,
        evaluation_set=EvaluationSet(),
        parameters={"outer": {"z": 1, "a": 2}, "alpha": True},
    )
    second = EvaluationRequest(
        candidate=candidate,
        evaluation_set=EvaluationSet(),
        parameters={"alpha": True, "outer": {"a": 2, "z": 1}},
    )

    assert first.fingerprint() == second.fingerprint()


@pytest.mark.parametrize(
    "path",
    ["", "/absolute.log", "../escape.log", "logs/../escape.log", "logs//run.log", "logs\\run.log"],
)
def test_artifacts_reject_unsafe_paths(path):
    with pytest.raises(ValidationError):
        EvaluationArtifact(path=path)


def test_case_result_preserves_multiple_errors_in_order():
    case = CaseResult(
        case_id="1",
        status=CaseStatus.ERROR,
        errors=[
            CaseError(
                message="first attempt timed out",
                attempt=1,
                retryable=True,
            ),
            CaseError(
                message="second attempt failed",
                attempt=2,
                retryable=False,
                terminal=True,
            ),
        ],
    )

    assert [error.attempt for error in case.errors] == [1, 2]


def test_successful_case_may_have_non_terminal_retry_errors():
    case = CaseResult(
        case_id="1",
        status=CaseStatus.SUCCESS,
        metrics={"score": 1.0},
        errors=[CaseError(message="retry", retryable=True, terminal=False)],
    )

    assert case.status == CaseStatus.SUCCESS


@pytest.mark.parametrize(
    "case",
    [
        {"case_id": "1", "status": "error", "errors": []},
        {
            "case_id": "1",
            "status": "success",
            "errors": [{"message": "fatal", "terminal": True}],
        },
        {
            "case_id": "1",
            "status": "skipped",
            "errors": [{"message": "fatal", "terminal": True}],
        },
    ],
)
def test_case_status_and_terminal_errors_must_agree(case):
    with pytest.raises(ValidationError):
        CaseResult.model_validate(case)


def test_report_rejects_duplicate_cases_and_non_finite_metrics():
    case = CaseResult(case_id="same", status=CaseStatus.SUCCESS)
    with pytest.raises(ValidationError, match="case IDs must be unique"):
        EvaluationReport(
            status=EvaluationStatus.SUCCESS,
            cases=[case, case],
        )

    with pytest.raises(ValidationError, match="must be finite"):
        EvaluationReport(
            status=EvaluationStatus.SUCCESS,
            metrics={"latency": float("nan")},
        )


def test_diagnostics_are_structured_and_chronological():
    report = EvaluationReport(
        status=EvaluationStatus.SUCCESS,
        diagnostics=[
            EvaluationDiagnostic(
                code="compile_warning",
                message="unused variable",
                severity=DiagnosticSeverity.WARNING,
                phase="compile",
            ),
            EvaluationDiagnostic(
                code="retry",
                message="measurement retried",
                severity=DiagnosticSeverity.INFO,
                phase="measure",
            ),
        ],
    )

    assert [diagnostic.code for diagnostic in report.diagnostics] == [
        "compile_warning",
        "retry",
    ]


def test_backend_provenance_digest_is_stable_across_key_order():
    first = BackendProvenance.from_config(
        name="command", version="1", config={"command": ["run"], "timeout": 10}
    )
    second = BackendProvenance.from_config(
        name="command", version="1", config={"timeout": 10, "command": ["run"]}
    )

    assert first == second
    assert len(first.config_digest) == 64


def _record(*, commit: str = "abc", created_at: datetime | None = None) -> EvaluationRecord:
    objective_spec = ObjectiveSpec(
        selector=MetricSelector(
            metric="latency_ms", aggregation=MetricAggregation.REPORT
        ),
        direction="minimize",
    )
    return EvaluationRecord(
        id=str(uuid4()),
        request=EvaluationRequest(
            candidate=Candidate(
                commit=commit,
                repo_name="demo",
                created_at=created_at or datetime(2026, 1, 1),
            ),
            evaluation_set=EvaluationSet(name="performance"),
        ),
        report=EvaluationReport(
            status=EvaluationStatus.SUCCESS,
            metrics={"latency_ms": 2.5},
            cases=[
                CaseResult(
                    case_id="ok",
                    status=CaseStatus.SUCCESS,
                    metrics={"correct": 1.0},
                    input={"secret": "input"},
                    output={"secret": "output"},
                ),
                CaseResult(
                    case_id="bad",
                    status=CaseStatus.ERROR,
                    errors=[CaseError(message="failed", terminal=True)],
                ),
                CaseResult(case_id="skip", status=CaseStatus.SKIPPED),
            ],
            diagnostics=[
                EvaluationDiagnostic(
                    code="private",
                    message="sensitive detail",
                    severity=DiagnosticSeverity.INFO,
                )
            ],
            artifacts=[EvaluationArtifact(path="run.log")],
        ),
        backend_id="trusted-command",
        backend=BackendProvenance(
            name="command",
            version="1",
            config_digest="0" * 64,
        ),
        objective_spec=objective_spec,
        objective=ObjectiveResult(value=2.5, feasible=True),
        created_at=datetime(2026, 1, 1),
        completed_at=datetime(2026, 1, 1) + timedelta(seconds=1),
    )


def test_aggregate_projection_excludes_sensitive_report_fields():
    summary = project_evaluation(_record(), DisclosureLevel.AGGREGATE)
    payload = summary.model_dump(mode="json")

    assert payload["backend_id"] == "trusted-command"
    assert payload["total_cases"] == 3
    assert payload["successful_cases"] == 1
    assert payload["errored_cases"] == 1
    assert payload["skipped_cases"] == 1
    for private_field in (
        "cases",
        "diagnostics",
        "artifacts",
        "error",
        "input",
        "output",
        "feedback",
    ):
        assert private_field not in payload


def test_none_projection_contains_only_id_and_status():
    acknowledgement = project_evaluation(_record(), DisclosureLevel.NONE)

    assert set(acknowledgement.model_dump()) == {"evaluation_id", "status"}
