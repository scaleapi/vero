from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import pytest

from vero.candidate import Candidate
from vero.candidate_repository import GitCandidateRepository
from vero.evaluation import (
    BackendProvenance,
    BackendRegistry,
    BudgetLedger,
    CaseIds,
    CaseResult,
    CaseStatus,
    DisclosureLevel,
    EvaluationAccessPolicy,
    EvaluationAuthorization,
    EvaluationBudget,
    EvaluationCost,
    EvaluationDatabase,
    EvaluationDeniedError,
    EvaluationEngine,
    EvaluationLimits,
    EvaluationReport,
    EvaluationRequest,
    EvaluationSet,
    EvaluationStatus,
    Evaluator,
    MetricSelector,
    ObjectiveSpec,
)
from vero.runtime.context import context_digest
from vero.sandbox import LocalSandbox
from vero.sidecar import (
    EvaluationAccessError,
    EvaluationJobStatus,
    EvaluationSidecar,
    GitCandidateTransport,
    SidecarEvaluationPolicy,
    SidecarEvaluationRequest,
    SubmissionDisabledError,
)
from vero.workspace import GitWorkspace, Workspace


class StubWorkspace(Workspace):
    def __init__(self, root: Path, version: str):
        root.mkdir(parents=True, exist_ok=True)
        self._root = root
        self._version = version

    @property
    def sandbox(self):
        return None

    @property
    def root(self) -> str:
        return str(self._root)

    @property
    def project_path(self) -> str:
        return str(self._root)

    @property
    def name(self) -> str:
        return "stub"

    async def current_version(self) -> str:
        return self._version

    async def save(self, message="Save") -> str:
        return self._version

    async def restore(self, version_id, message=None) -> str:
        self._version = version_id
        return version_id

    async def diff(self, from_version=None, to_version=None) -> str:
        return ""

    async def log(self, max_count=10, since_version=None) -> str:
        return ""

    async def is_ancestor(self, version_a, version_b) -> bool:
        return True

    async def copy(self, name=None, from_version=None):
        return StubWorkspace(self._root, from_version or self._version)

    @asynccontextmanager
    async def temp_copy(self, from_version=None):
        yield StubWorkspace(self._root, from_version or self._version)

    @asynccontextmanager
    async def at(self, version_id):
        previous = self._version
        self._version = version_id
        try:
            yield
        finally:
            self._version = previous

    async def is_dirty(self) -> bool:
        return False


class StubCandidateRepository:
    family = "stub"

    def __init__(self, root: Path):
        self.root = root

    @asynccontextmanager
    async def checkout(self, candidate, *, sandbox, name=None):
        yield StubWorkspace(self.root, candidate.version)


class StubBackend:
    @property
    def provenance(self):
        return BackendProvenance(
            name="stub",
            version="1",
            config_digest="0" * 64,
        )

    async def resolve_cost(self, evaluation_set):
        selection = evaluation_set.selection
        if isinstance(selection, CaseIds):
            return EvaluationCost(cases=len(selection.ids))
        return EvaluationCost(cases=8)

    async def evaluate(self, *, context, request):
        selection = request.evaluation_set.selection
        ids = (
            selection.ids
            if isinstance(selection, CaseIds)
            else [f"case-{i}" for i in range(8)]
        )
        cases = [
            CaseResult(
                case_id=case_id,
                status=CaseStatus.SUCCESS,
                metrics={"score": 0.75},
            )
            for case_id in ids
        ]
        for case in cases:
            await context.case_store.save(case)
        return EvaluationReport(
            status=EvaluationStatus.SUCCESS,
            metrics={"score": 0.75},
            cases=cases,
        )


class StubTransport:
    def __init__(self, candidate: Candidate):
        self.candidate = candidate
        self.calls: list[str | None] = []

    async def import_candidate(self, version=None):
        self.calls.append(version)
        return self.candidate


def _sidecar(tmp_path: Path, *, submit_enabled=False, fixed_limits=False, disclose_budget=True):
    candidate = Candidate(
        id="candidate",
        version="candidate-version",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    workspace = StubWorkspace(tmp_path / "repo", candidate.version)
    backend = StubBackend()
    evaluation_set = EvaluationSet(name="benchmark", partition="validation")
    ledger = BudgetLedger(
        [
            EvaluationBudget(
                backend_id="primary",
                evaluation_set_key=evaluation_set.budget_key("primary"),
                total_runs=3,
                total_cases=20,
            )
        ]
    )
    engine = EvaluationEngine(
        evaluator=Evaluator(
            candidate_repository=StubCandidateRepository(tmp_path / "stub-workspace"),
            sandbox=workspace.sandbox,
            session_dir=tmp_path / "session",
        ),
        backends=BackendRegistry({"primary": backend, "secondary": StubBackend()}),
        database=EvaluationDatabase(id="session"),
        budget_ledger=ledger,
    )
    transport = StubTransport(candidate)
    sidecar = EvaluationSidecar(
        engine=engine,
        candidate_transport=transport,
        access_policies=[
            SidecarEvaluationPolicy(
                backend_id="primary",
                evaluation_set_name="benchmark",
                partition="validation",
                access=EvaluationAccessPolicy(
                    disclosure=DisclosureLevel.AGGREGATE,
                    min_aggregate_cases=5,
                ),
                objective=ObjectiveSpec(
                    selector=MetricSelector(metric="score"),
                    direction="maximize",
                ),
                limits=EvaluationLimits() if fixed_limits else None,
            ),
            SidecarEvaluationPolicy(
                backend_id="secondary",
                evaluation_set_name="public",
                access=EvaluationAccessPolicy(disclosure=DisclosureLevel.FULL),
            ),
        ],
        agent_volume=tmp_path / "agent-volume",
        admin_volume=tmp_path / "admin-volume",
        submit_enabled=submit_enabled,
        disclose_budget=disclose_budget,
    )
    return sidecar, transport, ledger


@pytest.mark.asyncio
async def test_sidecar_uses_canonical_disclosure_budget_and_multiple_backends(tmp_path):
    sidecar, transport, ledger = _sidecar(tmp_path)
    evaluation_set = EvaluationSet(
        name="benchmark",
        partition="validation",
        selection=CaseIds(ids=[f"case-{i}" for i in range(5)]),
    )

    response = await sidecar.evaluate(
        SidecarEvaluationRequest(
            backend_id="primary",
            evaluation_set=evaluation_set,
            version="HEAD",
        )
    )

    assert response.disclosure == DisclosureLevel.AGGREGATE
    assert response.receipt.result.metrics == {"score": 0.75}
    assert response.receipt.result.total_cases == 5
    assert transport.calls == ["HEAD"]
    assert response.receipt.result_path == (
        f".evals/results/{context_digest(response.receipt.evaluation_id)}/"
        "evaluation.json"
    )
    assert (
        tmp_path
        / "agent-volume"
        / "results"
        / context_digest(response.receipt.evaluation_id)
        / "evaluation.json"
    ).is_file()
    budget = ledger.get("primary", evaluation_set)
    assert budget.remaining_runs == 2
    assert budget.remaining_cases == 15
    status = sidecar.status()
    assert [item.backend_id for item in status.evaluation_access] == [
        "primary",
        "secondary",
    ]
    assert status.evaluation_access[0].expose_case_resources is False
    # The synchronous evaluation is now a tracked job, so it is visible in
    # status exactly like a detached one.
    assert len(status.evaluation_jobs) == 1
    assert status.evaluation_jobs[0].status == EvaluationJobStatus.COMPLETE
    assert status.evaluation_jobs[0].evaluation_id == response.receipt.evaluation_id


@pytest.mark.asyncio
async def test_sidecar_detached_job_returns_before_evaluation_and_persists_result(
    tmp_path,
):
    sidecar, _, _ = _sidecar(tmp_path)

    class BlockingBackend(StubBackend):
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def evaluate(self, *, context, request):
            self.started.set()
            await self.release.wait()
            return await super().evaluate(context=context, request=request)

    backend = BlockingBackend()
    sidecar.engine.backends._backends["primary"] = backend
    job = await sidecar.start_evaluation_job(
        SidecarEvaluationRequest(
            backend_id="primary",
            evaluation_set=EvaluationSet(
                name="benchmark",
                partition="validation",
                selection=CaseIds(ids=[f"case-{i}" for i in range(5)]),
            ),
            version="HEAD",
        )
    )

    assert job.status == EvaluationJobStatus.RUNNING
    assert job.version == "candidate-version"
    assert job.receipt is None
    await backend.started.wait()
    assert sidecar.status().evaluation_jobs[0].job_id == job.job_id
    drain = asyncio.create_task(
        sidecar.engine.quiesce_agent_evaluations(timeout_seconds=1.0)
    )
    await asyncio.sleep(0)
    assert not drain.done()

    backend.release.set()
    task = sidecar._evaluation_job_tasks[job.job_id]
    await task
    assert await drain == 1

    completed = sidecar.evaluation_job(job.job_id)
    assert completed.status == EvaluationJobStatus.COMPLETE
    assert completed.receipt is not None
    assert completed.receipt.result.metrics == {"score": 0.75}
    persisted = tmp_path / "session/evaluation-jobs" / f"{job.job_id}.json"
    assert json.loads(persisted.read_text())["status"] == "complete"


@pytest.mark.asyncio
async def test_sidecar_detached_job_reports_safe_admission_failure(tmp_path):
    sidecar, _, _ = _sidecar(tmp_path)

    job = await sidecar.start_evaluation_job(
        SidecarEvaluationRequest(
            backend_id="primary",
            evaluation_set=EvaluationSet(name="private", partition="validation"),
        )
    )

    assert job.status == EvaluationJobStatus.FAILED
    assert job.error == "evaluation denied"
    assert job.receipt is None


@pytest.mark.asyncio
async def test_sidecar_tracks_candidate_import_as_part_of_an_agent_evaluation(
    tmp_path,
):
    sidecar, original_transport, _ = _sidecar(tmp_path)

    class BlockingTransport(StubTransport):
        def __init__(self, candidate):
            super().__init__(candidate)
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def import_candidate(self, version=None):
            self.calls.append(version)
            self.started.set()
            await self.release.wait()
            return self.candidate

    transport = BlockingTransport(original_transport.candidate)
    sidecar.candidate_transport = transport
    evaluation = asyncio.create_task(
        sidecar.evaluate(
            SidecarEvaluationRequest(
                backend_id="primary",
                evaluation_set=EvaluationSet(
                    name="benchmark",
                    partition="validation",
                ),
                version="HEAD",
            )
        )
    )
    await transport.started.wait()

    drain = asyncio.create_task(
        sidecar.engine.quiesce_agent_evaluations(timeout_seconds=1.0)
    )
    await asyncio.sleep(0)
    assert not drain.done()
    transport.release.set()

    assert (await evaluation).receipt.status == EvaluationStatus.SUCCESS
    assert await drain == 1
    with pytest.raises(EvaluationDeniedError, match="finalization has started"):
        await sidecar.evaluate(
            SidecarEvaluationRequest(
                backend_id="primary",
                evaluation_set=EvaluationSet(
                    name="benchmark",
                    partition="validation",
                ),
            )
        )


def test_sidecar_status_reports_inference_usage_and_remaining_budget(tmp_path):
    sidecar, _, _ = _sidecar(tmp_path)
    usage_path = tmp_path / "inference/usage.json"
    usage_path.parent.mkdir()
    usage_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scopes": {
                    "producer": {
                        "requests": 3,
                        "total_tokens": 120,
                        "active_requests": 0,
                    }
                },
            }
        )
    )
    sidecar.inference_usage_path = usage_path
    sidecar.inference_limits = {
        "producer": {
            "allowed_models": ["gpt-test"],
            "max_requests": 10,
            "max_tokens": 1000,
            "max_concurrency": 2,
        }
    }

    usage = sidecar.status().inference_usage

    assert usage is not None
    assert usage["producer"]["requests"] == 3
    assert usage["producer"]["remaining_requests"] == 7
    assert usage["producer"]["remaining_tokens"] == 880


def test_sidecar_hides_budget_when_disclosure_disabled(tmp_path):
    # Budget-blind mode: enforcement is unchanged but /status must not reveal
    # per-set case budgets or inference remaining, closing the live leak.
    disclosed, _, _ = _sidecar(tmp_path)
    assert any(a.budget is not None for a in disclosed.status().evaluation_access)

    blind, _, _ = _sidecar(tmp_path, disclose_budget=False)
    usage_path = tmp_path / "inference/usage.json"
    usage_path.parent.mkdir()
    usage_path.write_text(json.dumps({"schema_version": 1, "scopes": {}}))
    blind.inference_usage_path = usage_path
    blind.inference_limits = {"producer": {"max_requests": 10, "max_tokens": 1000}}

    status = blind.status()
    assert all(a.budget is None for a in status.evaluation_access)
    assert status.inference_usage is None


@pytest.mark.asyncio
async def test_sidecar_context_survives_restart_without_disclosing_admin_runs(
    tmp_path: Path,
):
    sidecar, transport, _ = _sidecar(tmp_path)
    admin = await sidecar.engine.evaluate_record(
        backend_id="secondary",
        request=EvaluationRequest(
            candidate=transport.candidate,
            evaluation_set=EvaluationSet(name="public"),
        ),
        authorization=EvaluationAuthorization(
            may_evaluate=True,
            meter_budget=False,
            disclosure=DisclosureLevel.FULL,
        ),
    )
    response = await sidecar.evaluate(
        SidecarEvaluationRequest(
            backend_id="primary",
            evaluation_set=EvaluationSet(
                name="benchmark",
                partition="validation",
                selection=CaseIds(ids=[f"case-{index}" for index in range(5)]),
            ),
        )
    )

    restarted = EvaluationSidecar(
        engine=sidecar.engine,
        candidate_transport=transport,
        access_policies=list(sidecar._policies.values()),
        agent_volume=tmp_path / "agent-volume",
        admin_volume=tmp_path / "admin-volume",
    )
    await restarted.initialize_context()

    index = json.loads((tmp_path / "agent-volume/results/index.json").read_text())[
        "evaluations"
    ]
    assert [entry["evaluation_id"] for entry in index] == [
        response.receipt.evaluation_id
    ]
    assert admin.id not in json.dumps(index)


class CostReportingBackend(StubBackend):
    async def evaluate(self, *, context, request):
        report = await super().evaluate(context=context, request=request)
        cases = [
            case.model_copy(
                update={
                    "metrics": {
                        **case.metrics,
                        "agent_reported_input_tokens": 111.0,
                        "wall_seconds": 7.0,
                    }
                }
            )
            for case in report.cases
        ]
        return report.model_copy(
            update={
                "cases": cases,
                "metrics": {
                    **report.metrics,
                    "inference_total_tokens": 12345.0,
                    "inference_cached_input_tokens": 678.0,
                    "agent_reported_total_input_tokens": 555.0,
                    # distribution wrappers of the same budget signal
                    "mean_case_inference_input_tokens": 99.0,
                    "median_case_agent_reported_output_tokens": 33.0,
                    "mean_case_wall_seconds": 42.0,
                    "median_case_wall_seconds": 40.0,
                }
            }
        )


@pytest.mark.asyncio
async def test_budget_blind_sidecar_redacts_inference_metrics(tmp_path):
    # Cost telemetry is budget signal: in budget-blind mode the agent's receipt
    # and context must not carry inference_* metrics; latency stays visible and
    # the admin record keeps everything.
    blind, _, _ = _sidecar(tmp_path, disclose_budget=False)
    blind.engine.backends = BackendRegistry(
        {"primary": CostReportingBackend(), "secondary": CostReportingBackend()}
    )
    response = await blind.evaluate(
        SidecarEvaluationRequest(
            backend_id="primary",
            evaluation_set=EvaluationSet(
                name="benchmark",
                partition="validation",
                selection=CaseIds(ids=[f"case-{i}" for i in range(5)]),
            ),
        )
    )

    receipt_metrics = response.receipt.result.metrics
    assert "inference_total_tokens" not in receipt_metrics
    assert "inference_cached_input_tokens" not in receipt_metrics
    # per-case distribution wrappers are the same signal renamed, so they are
    # redacted too; latency distributions stay visible
    assert "mean_case_inference_input_tokens" not in receipt_metrics
    assert "median_case_agent_reported_output_tokens" not in receipt_metrics
    assert receipt_metrics["mean_case_wall_seconds"] == 42.0
    assert receipt_metrics["median_case_wall_seconds"] == 40.0

    document = json.loads(
        (
            tmp_path
            / "agent-volume/results"
            / context_digest(response.receipt.evaluation_id)
            / "evaluation.json"
        ).read_text()
    )
    context_metrics = document["result"]["metrics"]
    assert "inference_total_tokens" not in context_metrics
    assert "agent_reported_total_input_tokens" not in context_metrics
    assert context_metrics["mean_case_wall_seconds"] == 42.0

    # the trusted record is untouched, including per-case cost metrics
    record = blind.engine.database.get_evaluation(response.receipt.evaluation_id)
    assert record.report.metrics["inference_total_tokens"] == 12345.0
    assert record.report.metrics["agent_reported_total_input_tokens"] == 555.0
    assert record.report.cases[0].metrics["agent_reported_input_tokens"] == 111.0

    # full disclosure in budget-blind mode: per-case cost metrics are redacted
    # from the agent-visible case files too, while wall_seconds stays
    full = await blind.evaluate(
        SidecarEvaluationRequest(
            backend_id="secondary",
            evaluation_set=EvaluationSet(name="public"),
        )
    )
    full_doc = json.loads(
        (
            tmp_path
            / "agent-volume/results"
            / context_digest(full.receipt.evaluation_id)
            / "evaluation.json"
        ).read_text()
    )
    case_file = full_doc["result"]["case_files"][0]["path"]
    case_doc = json.loads(
        (
            tmp_path
            / "agent-volume/results"
            / context_digest(full.receipt.evaluation_id)
            / case_file
        ).read_text()
    )
    assert "agent_reported_input_tokens" not in case_doc["result"]["metrics"]
    assert case_doc["result"]["metrics"]["wall_seconds"] == 7.0

    # with budget disclosure on, the same metrics flow to the agent
    disclosed, _, _ = _sidecar(tmp_path / "disclosed", disclose_budget=True)
    disclosed.engine.backends = BackendRegistry(
        {"primary": CostReportingBackend(), "secondary": CostReportingBackend()}
    )
    open_response = await disclosed.evaluate(
        SidecarEvaluationRequest(
            backend_id="primary",
            evaluation_set=EvaluationSet(
                name="benchmark",
                partition="validation",
                selection=CaseIds(ids=[f"case-{i}" for i in range(5)]),
            ),
        )
    )
    assert open_response.receipt.result.metrics["inference_total_tokens"] == 12345.0


@pytest.mark.asyncio
async def test_sidecar_context_writes_plan_and_tasks(tmp_path):
    sidecar, _, _ = _sidecar(tmp_path)
    await sidecar.initialize_context()

    volume = tmp_path / "agent-volume"
    # Case resources live under tasks/ (the documented name), not cases/.
    tasks_index = json.loads((volume / "tasks/index.json").read_text())
    assert tasks_index["case_resources"] == []
    assert not (volume / "cases").exists()

    plan = json.loads((volume / "plan.json").read_text())
    by_name = {entry["name"]: entry for entry in plan["evaluations"]}
    assert set(by_name) == {"benchmark", "public"}
    assert by_name["benchmark"]["disclosure"] == "aggregate"
    assert by_name["benchmark"]["budget"]["remaining_runs"] == 3
    assert by_name["benchmark"]["budget"]["remaining_cases"] == 20

    # Each evaluation refreshes the plan with the depleted budget.
    await sidecar.evaluate(
        SidecarEvaluationRequest(
            backend_id="primary",
            evaluation_set=EvaluationSet(
                name="benchmark",
                partition="validation",
                selection=CaseIds(ids=[f"case-{i}" for i in range(5)]),
            ),
        )
    )
    plan = json.loads((volume / "plan.json").read_text())
    by_name = {entry["name"]: entry for entry in plan["evaluations"]}
    assert by_name["benchmark"]["budget"]["remaining_runs"] == 2
    assert by_name["benchmark"]["budget"]["remaining_cases"] == 15

    # Budget-blind mode withholds the budget column, nothing else.
    blind, _, _ = _sidecar(tmp_path, disclose_budget=False)
    await blind.initialize_context()
    plan = json.loads((volume / "plan.json").read_text())
    assert all(entry["budget"] is None for entry in plan["evaluations"])


@pytest.mark.asyncio
async def test_sidecar_fails_closed_before_transfer_or_budget(tmp_path):
    sidecar, transport, ledger = _sidecar(tmp_path)
    too_small = SidecarEvaluationRequest(
        backend_id="primary",
        evaluation_set=EvaluationSet(
            name="benchmark",
            partition="validation",
            selection=CaseIds(ids=["single"]),
        ),
    )

    with pytest.raises(EvaluationAccessError, match="at least 5"):
        await sidecar.evaluate(too_small)
    # The synchronous failure is recorded as a tracked job (visible in status)
    # even though the error is still re-raised to the caller.
    assert sidecar.status().evaluation_jobs[0].status == EvaluationJobStatus.FAILED
    assert sidecar.status().evaluation_jobs[0].error == "evaluation denied"
    with pytest.raises(EvaluationAccessError, match="not agent-evaluable"):
        await sidecar.evaluate(
            too_small.model_copy(
                update={"evaluation_set": EvaluationSet(name="hidden")}
            )
        )
    with pytest.raises(EvaluationAccessError, match="not agent-controllable"):
        await sidecar.evaluate(
            SidecarEvaluationRequest(
                backend_id="primary",
                evaluation_set=EvaluationSet(
                    name="benchmark",
                    partition="validation",
                ),
                parameters={"harbor_model_override": "untrusted-model"},
            )
        )

    assert transport.calls == []
    budget = ledger.get("primary", too_small.evaluation_set)
    assert budget.remaining_runs == 3
    assert budget.remaining_cases == 20


@pytest.mark.asyncio
async def test_sidecar_rejects_agent_limits_when_policy_fixes_them(tmp_path):
    sidecar, transport, ledger = _sidecar(tmp_path, fixed_limits=True)
    evaluation_set = EvaluationSet(
        name="benchmark",
        partition="validation",
        selection=CaseIds(ids=[f"case-{i}" for i in range(5)]),
    )

    with pytest.raises(EvaluationAccessError, match="limits are fixed"):
        await sidecar.evaluate(
            SidecarEvaluationRequest(
                backend_id="primary",
                evaluation_set=evaluation_set,
                limits=EvaluationLimits(timeout_seconds=10),
            )
        )

    assert transport.calls == []
    budget = ledger.get("primary", evaluation_set)
    assert budget.remaining_runs == 3
    assert budget.remaining_cases == 20


@pytest.mark.asyncio
async def test_sidecar_submission_is_explicit_and_durable(tmp_path):
    sidecar, _, _ = _sidecar(tmp_path)
    with pytest.raises(SubmissionDisabledError):
        await sidecar.submit()

    enabled, _, _ = _sidecar(tmp_path / "enabled", submit_enabled=True)
    submission = await enabled.submit("candidate-ref")

    assert submission.candidate.version == "candidate-version"
    assert (tmp_path / "enabled/admin-volume/submission.json").is_file()


def _git(path: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=path,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _init_repo(path: Path, content: str) -> str:
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.name", "VeRO Test")
    _git(path, "config", "user.email", "vero@example.test")
    (path / "program.txt").write_text(content, encoding="utf-8")
    _git(path, "add", "program.txt")
    _git(path, "commit", "-q", "-m", f"program {content}")
    return _git(path, "rev-parse", "HEAD")


@pytest.mark.asyncio
async def test_git_candidate_transport_fetches_to_stable_ref(tmp_path, monkeypatch):
    agent_repo = tmp_path / "agent repo"
    trusted_repo = tmp_path / "trusted"
    agent_commit = _init_repo(agent_repo, "candidate")
    _init_repo(trusted_repo, "baseline")
    sandbox = await LocalSandbox.create(root=tmp_path)
    workspace = await GitWorkspace.from_path(sandbox, str(trusted_repo))
    system_config = tmp_path / "trusted-system-gitconfig"
    _git(
        tmp_path,
        "config",
        "--file",
        str(system_config),
        "--add",
        "safe.directory",
        str(trusted_repo),
    )
    _git(
        tmp_path,
        "config",
        "--file",
        str(system_config),
        "--add",
        "safe.directory",
        str(agent_repo),
    )
    _git(
        tmp_path,
        "config",
        "--file",
        str(system_config),
        "--add",
        "safe.directory",
        str(agent_repo / ".git"),
    )
    original_run = sandbox.run

    async def run_as_different_owner(command, cwd=None, timeout=30, env=None):
        return await original_run(
            command,
            cwd=cwd,
            timeout=timeout,
            env={
                **(env or {}),
                "GIT_TEST_ASSUME_DIFFERENT_OWNER": "1",
                "GIT_CONFIG_SYSTEM": str(system_config),
            },
        )

    monkeypatch.setattr(sandbox, "run", run_as_different_owner)
    candidate_repository = await GitCandidateRepository.create(
        tmp_path / "candidate-store",
        workspace=workspace,
    )
    transport = GitCandidateTransport(
        workspace=workspace,
        candidate_repository=candidate_repository,
        agent_repo_path=str(agent_repo),
    )

    candidate = await transport.import_candidate()
    repeated = await transport.import_candidate(agent_commit)

    assert candidate == repeated
    assert candidate.version == agent_commit
    assert candidate.description == "program candidate"
    shutil.rmtree(agent_repo)
    checkout_sandbox = await LocalSandbox.create(root=tmp_path)
    async with candidate_repository.checkout(
        candidate,
        sandbox=checkout_sandbox,
    ) as checkout:
        assert await checkout.current_version() == agent_commit
    retained_refs = _git(
        candidate_repository.repository_path,
        "for-each-ref",
        "--format=%(refname)",
        "refs/vero/candidates",
    ).splitlines()
    assert len(retained_refs) == 1
    assert (
        _git(candidate_repository.repository_path, "rev-parse", retained_refs[0])
        == agent_commit
    )
    assert (
        _git(
            candidate_repository.repository_path,
            "for-each-ref",
            "--format=%(refname)",
            "refs/vero/incoming",
        )
        == ""
    )


def test_access_policy_defaults_to_safe_k_anonymity_floor():
    # The floor now lives on the core policy: omitted resolves to 5 under
    # aggregate disclosure and 1 otherwise.
    aggregate = EvaluationAccessPolicy(disclosure=DisclosureLevel.AGGREGATE)
    assert aggregate.min_aggregate_cases == 5
    assert EvaluationAccessPolicy().min_aggregate_cases == 1
