from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from vero.candidate import Candidate
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
    EvaluationAccessPolicy,
    EvaluationArtifact,
    EvaluationBudget,
    EvaluationDefinition,
    EvaluationDiagnostic,
    EvaluationPlan,
    EvaluationPrincipal,
    EvaluationRecord,
    EvaluationReport,
    EvaluationRequest,
    EvaluationSet,
    EvaluationStatus,
    MetricSelector,
    ObjectiveResult,
    ObjectiveSpec,
    RetryPolicy,
)


def candidate(candidate_id: str = "candidate-1") -> Candidate:
    return Candidate(
        id=candidate_id,
        version=f"snapshot:{candidate_id}",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_candidate_identity_is_workspace_neutral():
    value = Candidate.from_version(
        "remote-revision:42",
        candidate_id="idea-7",
        parent_id="baseline",
        metadata={"producer": "coding-agent"},
    )

    assert value.id == "idea-7"
    assert value.version == "remote-revision:42"
    assert value.parent_id == "baseline"
    assert not hasattr(value, "commit")
    assert not hasattr(value, "repo_name")


def test_candidate_rejects_naive_timestamps_and_self_parent():
    with pytest.raises(ValidationError, match="timezone-aware"):
        Candidate(id="a", version="1", created_at=datetime(2026, 1, 1))
    with pytest.raises(ValidationError, match="own parent"):
        Candidate(id="a", version="1", parent_id="a")


@pytest.mark.parametrize(
    "selection",
    [
        AllCases(),
        CaseIds(ids=["case-2", "case-7"]),
        CaseRange(stop=10),
        CaseRange(start=10, stop=20),
    ],
)
def test_evaluation_set_round_trips_selection(selection):
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
        (CaseRange, {"start": -1, "stop": 2}),
        (CaseRange, {"start": 2, "stop": 2}),
    ],
)
def test_selection_rejects_invalid_values(model, kwargs):
    with pytest.raises(ValidationError):
        model(**kwargs)


def test_request_fingerprint_ignores_candidate_display_metadata():
    first = EvaluationRequest(
        candidate=Candidate(
            id="same",
            version="version-1",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            description="first description",
        ),
        parameters={"outer": {"z": 1, "a": 2}, "alpha": True},
    )
    second = EvaluationRequest(
        candidate=Candidate(
            id="same",
            version="version-1",
            created_at=datetime(2026, 1, 2, tzinfo=UTC),
            description="updated description",
        ),
        parameters={"alpha": True, "outer": {"a": 2, "z": 1}},
    )

    assert first.fingerprint() == second.fingerprint()


def test_retry_policy_restores_transient_provider_defaults():
    policy = RetryPolicy()

    assert policy.max_attempts == 3
    assert policy.retry_on_timeout is True
    assert policy.retry_status_codes == [429, 503, 529]
    assert RetryPolicy.disabled().max_attempts == 1


@pytest.mark.parametrize(
    "values",
    [
        {"initial_delay_seconds": 2, "maximum_delay_seconds": 1},
        {"retry_exception_names": ["same", "same"]},
        {"retry_status_codes": [99]},
        {"retry_message_patterns": ["["]},
    ],
)
def test_retry_policy_rejects_invalid_configuration(values):
    with pytest.raises(ValidationError):
        RetryPolicy(**values)


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/absolute.log",
        "../escape.log",
        "logs/../escape.log",
        "logs//run.log",
        "logs\\run.log",
    ],
)
def test_artifacts_reject_unsafe_paths(path):
    with pytest.raises(ValidationError):
        EvaluationArtifact(path=path)


def test_case_result_preserves_multiple_attempt_errors():
    result = CaseResult(
        case_id="1",
        status=CaseStatus.ERROR,
        errors=[
            CaseError(message="timed out", attempt=1, retryable=True),
            CaseError(
                message="failed again",
                attempt=2,
                retryable=False,
                terminal=True,
            ),
        ],
    )

    assert [error.attempt for error in result.errors] == [1, 2]


@pytest.mark.parametrize(
    "value",
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
def test_case_status_and_terminal_errors_agree(value):
    with pytest.raises(ValidationError):
        CaseResult.model_validate(value)


def test_report_preserves_structured_diagnostics_and_rejects_duplicate_cases():
    case = CaseResult(case_id="same", status=CaseStatus.SUCCESS)
    with pytest.raises(ValidationError, match="case IDs must be unique"):
        EvaluationReport(
            status=EvaluationStatus.SUCCESS,
            cases=[case, case],
        )

    report = EvaluationReport(
        status=EvaluationStatus.FAILED,
        diagnostics=[
            EvaluationDiagnostic(
                code="compile_failed",
                message="compiler returned 1",
                severity=DiagnosticSeverity.ERROR,
                phase="compile",
            )
        ],
    )
    assert report.diagnostics[0].code == "compile_failed"
    assert not hasattr(report, "error")


def test_backend_provenance_digest_is_stable_across_key_order():
    first = BackendProvenance.from_config(
        name="command", version="1", config={"command": ["run"], "timeout": 10}
    )
    second = BackendProvenance.from_config(
        name="command", version="1", config={"timeout": 10, "command": ["run"]}
    )

    assert first == second


def test_record_is_schema_one_and_requires_aware_ordered_timestamps():
    specification = ObjectiveSpec(
        selector=MetricSelector(metric="score"),
        direction="maximize",
    )
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    record = EvaluationRecord(
        id="evaluation-1",
        request=EvaluationRequest(candidate=candidate()),
        report=EvaluationReport(
            status=EvaluationStatus.SUCCESS,
            metrics={"score": 1.0},
        ),
        backend_id="command",
        backend=BackendProvenance(
            name="command",
            version="1",
            config_digest="0" * 64,
        ),
        objective_spec=specification,
        objective=ObjectiveResult(value=1.0, feasible=True),
        created_at=created_at,
        completed_at=created_at + timedelta(seconds=1),
    )

    assert record.schema_version == 2
    assert record.request.candidate.version == "snapshot:candidate-1"

    with pytest.raises(ValidationError, match="must not be before"):
        record.model_copy(
            update={"completed_at": created_at - timedelta(seconds=1)}
        ).__class__.model_validate(
            record.model_copy(
                update={"completed_at": created_at - timedelta(seconds=1)}
            ).model_dump()
        )


def test_evaluation_plan_models_selection_visibility_and_principal_budgets():
    validation = EvaluationSet(name="validation", partition="validation")
    test = EvaluationSet(name="test", partition="test")
    plan = EvaluationPlan(
        evaluations=[
            EvaluationDefinition(
                evaluation_set=validation,
                access=EvaluationAccessPolicy(
                    disclosure=DisclosureLevel.AGGREGATE,
                ),
                agent_budget=EvaluationBudget(
                    backend_id="command",
                    evaluation_set_key=validation.budget_key("command"),
                    principal=EvaluationPrincipal.AGENT,
                    total_runs=10,
                ),
                system_budget=EvaluationBudget(
                    backend_id="command",
                    evaluation_set_key=validation.budget_key("command"),
                    principal=EvaluationPrincipal.SYSTEM,
                    total_runs=100,
                ),
            ),
            EvaluationDefinition(
                evaluation_set=test,
                access=EvaluationAccessPolicy(
                    agent_can_evaluate=False,
                    agent_visible=False,
                    disclosure=DisclosureLevel.NONE,
                ),
            ),
        ],
        selection_evaluation="validation",
        final_evaluation="test",
    )

    assert plan.selection.evaluation_set == validation
    assert plan.final.evaluation_set == test
    assert {budget.principal for budget in plan.budgets} == {
        EvaluationPrincipal.AGENT,
        EvaluationPrincipal.SYSTEM,
    }

    with pytest.raises(ValidationError, match="final evaluation must be"):
        EvaluationPlan(
            evaluations=[EvaluationDefinition(evaluation_set=test)],
            selection_evaluation="test",
            final_evaluation="test",
        )
