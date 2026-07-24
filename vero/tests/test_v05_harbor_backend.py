from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import ModuleType, SimpleNamespace

import pytest

from vero.candidate import Candidate
from vero.evaluation import (
    CaseCheckpointStore,
    CaseIds,
    CaseRange,
    CaseStatus,
    EvaluationContext,
    EvaluationLimits,
    EvaluationRequest,
    EvaluationSet,
    EvaluationStatus,
    RetryPolicy,
)
from vero.harbor import HarborBackend, HarborBackendConfig
from vero.sandbox import CommandResult, LocalSandbox


def _cases(path: Path) -> Path:
    path.write_text(
        json.dumps(
            [
                {"id": "case-a", "task_name": "example/alpha"},
                {"id": "case-b", "task_name": "example/beta"},
                {"id": "case-c", "task_name": "example/gamma"},
            ]
        ),
        encoding="utf-8",
    )
    return path


def _config(tmp_path: Path, **updates) -> HarborBackendConfig:
    values = {
        "task_source": "example/tasks@1.0",
        "agent_import_path": "candidate.agent:Agent",
        "cases_path": str(_cases(tmp_path / "cases.json")),
        "harbor_requirement": "harbor==0.1.17",
        "evaluation_set_name": "harbor-bench",
        "partition": "test",
        "uv_executable": sys.executable,
        "infrastructure_max_attempts": 1,
        "infrastructure_retry_delay_seconds": 0,
    }
    values.update(updates)
    return HarborBackendConfig(**values)


def _request(selection=None) -> EvaluationRequest:
    return EvaluationRequest(
        candidate=Candidate(
            id="candidate",
            version="version",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
        evaluation_set=EvaluationSet(
            name="harbor-bench",
            partition="test",
            **({"selection": selection} if selection is not None else {}),
        ),
        limits=EvaluationLimits(
            timeout_seconds=90,
            max_concurrency=7,
            retry=RetryPolicy.disabled(),
        ),
    )


def test_backend_accepts_pinned_environment_extra(tmp_path):
    config = _config(
        tmp_path,
        harbor_requirement="harbor[modal]==0.20.0",
    )

    assert config.harbor_requirement == "harbor[modal]==0.20.0"


def test_environment_default_routes_agent_through_gateway_on_openai(tmp_path):
    config = _config(
        tmp_path,
        inference_gateway_url="http://inference-gateway:8001",
        inference_gateway_token="gw-token",
    )
    env = HarborBackend(config)._environment("eval-1")

    assert env["OPENAI_API_KEY"] == "gw-token"
    assert env["OPENAI_BASE_URL"] == (
        "http://inference-gateway:8001/scopes/evaluation/eval-1/v1"
    )
    # litellm-style alias used by task verifier env templates
    assert env["OPENAI_API_BASE"] == env["OPENAI_BASE_URL"]
    assert "VERO_AGENT_INFERENCE_API_KEY" not in env


def test_environment_routes_task_services_to_upstream(tmp_path, monkeypatch):
    monkeypatch.setenv("VERO_INFERENCE_UPSTREAM_API_KEY", "upstream-key")
    monkeypatch.setenv("VERO_INFERENCE_UPSTREAM_BASE_URL", "https://upstream/v1")
    config = _config(
        tmp_path,
        inference_gateway_url="http://inference-gateway:8001",
        inference_gateway_token="gw-token",
        task_services_use_upstream=True,
        upstream_api_key_env="VERO_INFERENCE_UPSTREAM_API_KEY",
        upstream_base_url_env="VERO_INFERENCE_UPSTREAM_BASE_URL",
    )
    env = HarborBackend(config)._environment("eval-1")

    # task-owned eval services reach the real upstream via OPENAI_*
    assert env["OPENAI_API_KEY"] == "upstream-key"
    assert env["OPENAI_BASE_URL"] == "https://upstream/v1"
    assert env["OPENAI_API_BASE"] == "https://upstream/v1"
    # the candidate agent keeps the metered gateway on dedicated vars
    assert env["VERO_AGENT_INFERENCE_API_KEY"] == "gw-token"
    assert env["VERO_AGENT_INFERENCE_BASE_URL"] == (
        "http://inference-gateway:8001/scopes/evaluation/eval-1/v1"
    )


def test_task_services_use_upstream_requires_gateway_and_upstream_env(tmp_path):
    with pytest.raises(ValueError, match="requires an inference gateway"):
        _config(tmp_path, task_services_use_upstream=True)
    with pytest.raises(ValueError, match="requires upstream_api_key_env"):
        _config(
            tmp_path,
            task_services_use_upstream=True,
            inference_gateway_url="http://inference-gateway:8001",
            inference_gateway_token="gw-token",
        )


@pytest.mark.parametrize(
    ("request_factory", "message"),
    [
        (
            lambda: _request().model_copy(
                update={
                    "limits": _request().limits.model_copy(
                        update={"retry": RetryPolicy(max_attempts=2)}
                    )
                }
            ),
            "generic per-case retries",
        ),
        (
            lambda: _request().model_copy(
                update={
                    "limits": _request().limits.model_copy(
                        update={"case_timeout_seconds": 30}
                    )
                }
            ),
            "case timeout is fixed by the backend",
        ),
        (lambda: _request().model_copy(update={"seed": 7}), "evaluation seed"),
    ],
)
def test_harbor_backend_rejects_unsupported_generic_controls(
    tmp_path, request_factory, message
):
    backend = HarborBackend(_config(tmp_path))

    with pytest.raises(ValueError, match=message):
        backend.validate_request(request_factory())


class FakeSandbox(LocalSandbox):
    def __init__(
        self,
        root: Path,
        trials: dict[str, list[dict]],
        result: CommandResult | None = None,
    ):
        super().__init__(root)
        self.trials = trials
        self.result = result or CommandResult("harbor output", "", 0)
        self.command = None
        self.cwd = None
        self.timeout = None
        self.env = None
        self.run_as = None
        self.chown_commands: list[list[str]] = []
        self.chmod_commands: list[list[str]] = []
        self.probe_commands: list[tuple[list[str], str | None]] = []

    async def run(self, command, cwd=None, timeout=30, env=None, run_as=None):
        if isinstance(command, list) and command[:1] == ["chown"]:
            # Record the harness-isolation provisioning without touching real uids.
            self.chown_commands.append(command)
            return CommandResult("", "", 0)
        if isinstance(command, list) and command[:1] == ["chmod"]:
            self.chmod_commands.append(command)
            return CommandResult("", "", 0)
        if isinstance(command, list) and command[:1] == ["test"]:
            # The workspace-reachability probe, run as the dropped user.
            self.probe_commands.append((command, run_as))
            return CommandResult("", "", 0)
        if not isinstance(command, list) or "--jobs-dir" not in command:
            return await super().run(
                command, cwd=cwd, timeout=timeout, env=env, run_as=run_as
            )
        self.command = command
        self.cwd = cwd
        self.timeout = timeout
        self.env = env
        self.run_as = run_as
        jobs_dir = Path(command[command.index("--jobs-dir") + 1])
        for task_name, attempts in self.trials.items():
            for index, attempt in enumerate(attempts):
                trial_dir = (
                    jobs_dir / f"job-{index}" / f"trial-{task_name.split('/')[-1]}"
                )
                trial_dir.mkdir(parents=True, exist_ok=True)
                payload = {
                    "task_name": task_name,
                    "trial_name": f"trial-{index}",
                    "finished_at": f"2026-01-01T00:00:0{index}Z",
                    **attempt,
                }
                agent_files = payload.pop("_agent_files", {})
                (trial_dir / "result.json").write_text(json.dumps(payload))
                for relative_path, content in agent_files.items():
                    path = trial_dir / "agent" / relative_path
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if isinstance(content, bytes):
                        path.write_bytes(content)
                    else:
                        path.write_text(content, encoding="utf-8")
        return self.result


async def _context(tmp_path: Path, sandbox: FakeSandbox) -> EvaluationContext:
    target = tmp_path / "target"
    target.mkdir(exist_ok=True)
    result_dir = tmp_path / "result"
    artifact_dir = result_dir / "artifacts"
    artifact_dir.mkdir(parents=True)
    return EvaluationContext(
        workspace=SimpleNamespace(
            project_path=str(target), root=str(target), sandbox=sandbox
        ),
        session_id="session",
        evaluation_id="evaluation",
        result_dir=result_dir,
        artifact_dir=artifact_dir,
        case_store=CaseCheckpointStore(result_dir / "cases"),
    )


@pytest.mark.asyncio
async def test_harbor_backend_resolves_canonical_case_selections(tmp_path):
    backend = HarborBackend(_config(tmp_path))
    evaluation_set = EvaluationSet(name="harbor-bench", partition="test")

    assert (await backend.resolve_cost(evaluation_set)).cases == 3
    assert (
        await backend.resolve_cost(
            evaluation_set.model_copy(update={"selection": CaseRange(start=1, stop=3)})
        )
    ).cases == 2
    assert (
        await backend.resolve_cost(
            evaluation_set.model_copy(
                update={"selection": CaseIds(ids=["case-c", "case-a"])}
            )
        )
    ).cases == 2

    with pytest.raises(ValueError, match="unknown Harbor case IDs"):
        await backend.resolve_cost(
            evaluation_set.model_copy(update={"selection": CaseIds(ids=["missing"])})
        )


@pytest.mark.asyncio
async def test_harbor_backend_exports_complete_authorized_local_tasks(tmp_path):
    task_source = tmp_path / "tasks"
    task_source.mkdir()
    (task_source / "metric.py").write_text("def score(): return 1\n")
    cases_path = tmp_path / "local-cases.json"
    cases_path.write_text(
        json.dumps(
            [
                {"id": "case-a", "task_name": "alpha"},
                {"id": "case-b", "task_name": "beta"},
            ]
        )
    )
    for task_name in ("alpha", "beta"):
        task = task_source / task_name
        task.mkdir()
        (task / "task.toml").write_text(f'[task]\nname="example/{task_name}"\n')
        (task / "instruction.md").write_text(f"Question for {task_name}\n")
        (task / "attachment.txt").write_text(f"attachment for {task_name}\n")
    backend = HarborBackend(
        _config(
            tmp_path,
            task_source=str(task_source),
            cases_path=str(cases_path),
        )
    )
    destination = tmp_path / "context"
    destination.mkdir()

    await backend.export_case_resources(
        evaluation_set=EvaluationSet(
            name="harbor-bench",
            partition="test",
            selection=CaseIds(ids=["case-b"]),
        ),
        destination=str(destination),
        sandbox=await LocalSandbox.create(root=tmp_path),
    )

    index = json.loads((destination / "index.json").read_text())
    assert index["task_source_exposed"] is True
    assert [item["case_id"] for item in index["cases"]] == ["case-b"]
    exported = destination / index["cases"][0]["path"]
    assert (exported / "task.toml").is_file()
    assert (exported / "instruction.md").read_text() == "Question for beta\n"
    assert (exported / "attachment.txt").read_text() == "attachment for beta\n"
    assert (destination / "dataset-files/metric.py").is_file()
    assert not (destination / "tasks" / "alpha").exists()
    assert destination.stat().st_mode & 0o005 == 0o005
    assert exported.stat().st_mode & 0o005 == 0o005
    assert (exported / "attachment.txt").stat().st_mode & 0o004 == 0o004


@pytest.mark.asyncio
async def test_harbor_backend_maps_remote_download_results_by_request_order(
    tmp_path, monkeypatch
):
    class FakeTaskId:
        def __init__(self, name: str):
            self.name = name

        def get_name(self) -> str:
            return self.name

    async def get_dataset_metadata(_client, _source):
        return SimpleNamespace(
            task_ids=[
                FakeTaskId("example/alpha"),
                FakeTaskId("example/beta"),
                FakeTaskId("example/gamma"),
            ]
        )

    async def download_tasks(_client, *, task_ids, output_dir, export):
        assert export is True
        results = []
        for task_id in task_ids:
            path = output_dir / task_id.get_name().split("/")[-1]
            path.mkdir(parents=True)
            (path / "instruction.md").write_text(task_id.get_name())
            results.append(SimpleNamespace(path=path))
        return SimpleNamespace(results=results)

    async def download_dataset_files(_client, _metadata, *, output_dir):
        output_dir.mkdir()
        path = output_dir / "metric.py"
        path.write_text("def score(): return 1\n")
        return [path]

    class PackageDatasetClient:
        pass

    class TaskClient:
        pass

    PackageDatasetClient.get_dataset_metadata = get_dataset_metadata
    PackageDatasetClient.download_dataset_files = download_dataset_files
    TaskClient.download_tasks = download_tasks

    modules = {
        name: ModuleType(name)
        for name in (
            "harbor",
            "harbor.registry",
            "harbor.registry.client",
            "harbor.registry.client.package",
            "harbor.tasks",
            "harbor.tasks.client",
        )
    }
    modules[
        "harbor.registry.client.package"
    ].PackageDatasetClient = PackageDatasetClient
    modules["harbor.tasks.client"].TaskClient = TaskClient
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    backend = HarborBackend(_config(tmp_path))
    destination = tmp_path / "context"
    destination.mkdir()

    await backend.export_case_resources(
        evaluation_set=EvaluationSet(
            name="harbor-bench",
            partition="test",
            selection=CaseIds(ids=["case-c", "case-a"]),
        ),
        destination=str(destination),
        sandbox=await LocalSandbox.create(root=tmp_path),
    )

    index = json.loads((destination / "index.json").read_text())
    assert [item["case_id"] for item in index["cases"]] == ["case-c", "case-a"]
    assert [
        (destination / item["path"] / "instruction.md").read_text()
        for item in index["cases"]
    ] == ["example/gamma", "example/alpha"]
    assert (destination / "dataset-files/metric.py").is_file()


@pytest.mark.asyncio
async def test_harbor_backend_exposes_complete_successful_trial_records(tmp_path):
    secret = "evaluation-scope-secret"
    sandbox = FakeSandbox(
        tmp_path,
        {
            "example/alpha": [
                {
                    "verifier_result": {"rewards": {"reward": 1.0}},
                    "_agent_files": {
                        "trajectory.json": json.dumps(
                            {"steps": [{"message": f"used {secret}"}]}
                        ),
                        "gaia-trace.jsonl": (
                            '{"turn":1,"action":"search"}\n{"turn":2,"answer":"done"}\n'
                        ),
                    },
                }
            ]
        },
    )
    backend = HarborBackend(_config(tmp_path, environment={"EVALUATION_TOKEN": secret}))

    report = await backend.evaluate(
        context=await _context(tmp_path, sandbox),
        request=_request(CaseIds(ids=["case-a"])),
    )

    artifacts = report.cases[0].artifacts
    assert {Path(artifact.path).name for artifact in artifacts} == {
        "trajectory.json",
        "gaia-trace.jsonl",
        "result.json",
    }
    for artifact in artifacts:
        content = (tmp_path / "result/artifacts" / artifact.path).read_text()
        assert secret not in content
    trajectory = next(
        artifact
        for artifact in artifacts
        if Path(artifact.path).name == "trajectory.json"
    )
    assert "[REDACTED]" in (tmp_path / "result/artifacts" / trajectory.path).read_text()


@pytest.mark.asyncio
async def test_harbor_backend_exposes_exact_failed_trial_result(tmp_path):
    detail = "Invalid schema for function 'transcribe_audio': Missing 'language'."
    sandbox = FakeSandbox(
        tmp_path,
        {
            "example/alpha": [
                {
                    "verifier_result": None,
                    "exception_info": {
                        "exception_type": "BadRequestError",
                        "exception_message": detail,
                    },
                }
            ]
        },
    )
    backend = HarborBackend(_config(tmp_path))

    report = await backend.evaluate(
        context=await _context(tmp_path, sandbox),
        request=_request(CaseIds(ids=["case-a"])),
    )

    artifacts = report.cases[0].artifacts
    result_artifact = next(
        artifact for artifact in artifacts if Path(artifact.path).name == "result.json"
    )
    result = json.loads(
        (tmp_path / "result/artifacts" / result_artifact.path).read_text()
    )
    assert result["exception_info"]["exception_message"] == detail
    assert result_artifact.description.endswith("result.json")


@pytest.mark.asyncio
async def test_harbor_backend_scores_agent_crash_as_informative_task_failure(tmp_path):
    sandbox = FakeSandbox(
        tmp_path,
        {
            "example/alpha": [{"verifier_result": {"rewards": {"reward": 1.0}}}],
            "example/beta": [
                {
                    "verifier_result": None,
                    "exception_info": {"exception_type": "AgentCrash"},
                }
            ],
        },
    )
    backend = HarborBackend(_config(tmp_path))
    runtime_context = await _context(tmp_path, sandbox)

    report = await backend.evaluate(
        context=runtime_context,
        request=_request(CaseRange(stop=2)),
    )

    # A candidate whose own harness crashes is an informative task failure:
    # scored at the failure value, a real SUCCESS sample that counts toward the
    # mean, and NOT an infrastructure error.
    assert report.status == EvaluationStatus.SUCCESS
    assert report.metrics == {"score": 0.5, "error_rate": 0.0, "score_stddev": 0.5}
    assert [case.status for case in report.cases] == [
        CaseStatus.SUCCESS,
        CaseStatus.SUCCESS,
    ]
    assert report.cases[1].metrics["score"] == 0.0
    assert report.cases[1].output["error_category"] == "task_failure"
    assert sandbox.command[:15] == [
        sys.executable,
        "run",
        "--python",
        "3.12",
        "--no-config",
        "--no-env-file",
        "--default-index",
        "https://pypi.org/simple",
        "--index-strategy",
        "first-index",
        "--project",
        str(tmp_path / "target"),
        "--with",
        "harbor==0.1.17",
        "harbor",
    ]
    assert sandbox.command.count("-i") == 2
    assert sandbox.command[sandbox.command.index("-n") + 1] == "7"
    assert (
        sandbox.command[sandbox.command.index("--agent-timeout-multiplier") + 1]
        == "0.3"
    )
    assert sandbox.timeout == 90
    assert [artifact.path for artifact in report.artifacts] == [
        "harbor/stdout.log",
        "harbor/stderr.log",
    ]
    checkpoints = await runtime_context.case_store.load_all()
    assert [case.case_id for case in checkpoints] == ["case-a", "case-b"]


@pytest.mark.asyncio
async def test_harbor_backend_excludes_transient_infrastructure_from_aggregate(
    tmp_path,
):
    sandbox = FakeSandbox(
        tmp_path,
        {
            "example/alpha": [{"verifier_result": {"rewards": {"reward": 1.0}}}],
            "example/beta": [
                {
                    "verifier_result": None,
                    "exception_info": {"exception_type": "openai.RateLimitError"},
                }
            ],
        },
    )
    backend = HarborBackend(_config(tmp_path))

    report = await backend.evaluate(
        context=await _context(tmp_path, sandbox),
        request=_request(CaseRange(stop=2)),
    )

    # The rate-limited case is infrastructure: excluded from the mean (which is
    # the sole successful case, 1.0) and reported as the error rate.
    assert report.metrics["score"] == 1.0
    assert report.metrics["error_rate"] == 0.5
    assert [case.status for case in report.cases] == [
        CaseStatus.SUCCESS,
        CaseStatus.ERROR,
    ]
    assert report.cases[1].output["error_category"] == "transient_infra"
    assert report.cases[1].errors[0].code == "transient_infrastructure"


@pytest.mark.asyncio
async def test_harbor_backend_marks_inference_budget_exhaustion_invalid(tmp_path):
    sandbox = FakeSandbox(
        tmp_path,
        {
            "example/alpha": [{"verifier_result": {"rewards": {"reward": 1.0}}}],
            "example/beta": [
                {
                    "verifier_result": None,
                    "exception_info": {
                        "exception_type": "openai.RateLimitError",
                        # The gateway's distinct code survives in the body message
                        # even though the client collapses the type to a 429.
                        "message": "Error code: 429 - inference token budget "
                        "exhausted (code: budget_exhausted)",
                    },
                }
            ],
        },
    )
    backend = HarborBackend(_config(tmp_path))

    report = await backend.evaluate(
        context=await _context(tmp_path, sandbox),
        request=_request(CaseRange(stop=2)),
    )

    # Budget exhaustion is terminating: the whole evaluation is INVALID with a
    # distinct, non-retryable code — never a rate-limit, never a score of zero.
    assert report.status == EvaluationStatus.INVALID
    assert "score" not in report.metrics
    assert any(
        diagnostic.code == "inference_budget_exhausted"
        for diagnostic in report.diagnostics
    )


@pytest.mark.asyncio
async def test_harbor_backend_treats_missing_coverage_as_infrastructure(tmp_path):
    # Only alpha produces a trial; beta is dropped entirely by the sub-run.
    sandbox = FakeSandbox(
        tmp_path,
        {"example/alpha": [{"verifier_result": {"rewards": {"reward": 1.0}}}]},
    )
    backend = HarborBackend(_config(tmp_path))

    report = await backend.evaluate(
        context=await _context(tmp_path, sandbox),
        request=_request(CaseRange(stop=2)),
    )

    # The dropped case is infrastructure (excluded from the mean, counted as
    # error rate) rather than a silent zero, and coverage loss is logged.
    assert report.metrics["score"] == 1.0
    assert report.metrics["error_rate"] == 0.5
    assert [case.status for case in report.cases] == [
        CaseStatus.SUCCESS,
        CaseStatus.ERROR,
    ]
    assert report.cases[1].output["error_category"] == "transient_infra"
    assert any(
        diagnostic.code == "harbor_incomplete_coverage"
        for diagnostic in report.diagnostics
    )


@pytest.mark.asyncio
async def test_harbor_backend_runs_harness_as_unprivileged_user(tmp_path):
    sandbox = FakeSandbox(
        tmp_path,
        {"example/alpha": [{"verifier_result": {"rewards": {"reward": 1.0}}}]},
    )
    backend = HarborBackend(_config(tmp_path, harness_user="harness"))

    await backend.evaluate(
        context=await _context(tmp_path, sandbox),
        request=_request(CaseRange(stop=1)),
    )

    # `harbor run` drops to the unprivileged user, and its work dirs (workspace +
    # staging) are handed to that user first so it never needs the trusted state.
    assert sandbox.run_as == "harness"
    assert sandbox.chown_commands, "expected the harness work dirs to be chowned"
    for command in sandbox.chown_commands:
        assert command[:3] == ["chown", "-R", "harness:harness"]
    # The checkout's mktemp parent (0700 root) is made traversable so the
    # dropped-uid harness can resolve the editable candidate package's absolute
    # path — otherwise `import <agent>` fails with "No module named ...".
    checkout_parent = str(PurePosixPath(str(tmp_path / "target")).parent)
    assert ["chmod", "o+x", checkout_parent] in sandbox.chmod_commands
    # And it probes, as the dropped user, that the workspace is actually reachable
    # before running harbor — so a permission gap fails fast and clearly here.
    assert (["test", "-r", str(tmp_path / "target")], "harness") in sandbox.probe_commands


def test_harbor_config_rejects_harness_isolation_with_upstream_task_services(tmp_path):
    # uid isolation cannot hide env vars, so the raw upstream key would still
    # reach the isolated harness; the combination must be refused.
    with pytest.raises(ValueError, match="harness_user cannot be combined"):
        _config(
            tmp_path,
            inference_gateway_url="http://inference-gateway:8001",
            inference_gateway_token="gw-token",
            task_services_use_upstream=True,
            upstream_api_key_env="VERO_INFERENCE_UPSTREAM_API_KEY",
            harness_user="harness",
        )


@pytest.mark.asyncio
async def test_harbor_backend_routes_candidate_inference_through_scoped_gateway(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "raw-provider-key")
    sandbox = FakeSandbox(
        tmp_path,
        {"example/alpha": [{"verifier_result": {"rewards": {"reward": 1.0}}}]},
    )
    backend = HarborBackend(
        _config(
            tmp_path,
            inference_gateway_url="http://inference-gateway:8001",
            inference_gateway_token="evaluation-scope-token",
        )
    )

    await backend.evaluate(
        context=await _context(tmp_path, sandbox),
        request=_request(CaseIds(ids=["case-a"])),
    )

    assert sandbox.env["OPENAI_API_KEY"] == "evaluation-scope-token"
    assert sandbox.env["OPENAI_BASE_URL"] == (
        "http://inference-gateway:8001/scopes/evaluation/evaluation/v1"
    )
    assert "raw-provider-key" not in sandbox.env.values()


@pytest.mark.asyncio
async def test_harbor_backend_matches_canonical_result_task_name(tmp_path):
    cases_path = tmp_path / "canonical-cases.json"
    cases_path.write_text(
        json.dumps(
            [
                {
                    "id": "local-task",
                    "task_name": "local-task",
                    "result_task_name": "org/local-task",
                }
            ]
        ),
        encoding="utf-8",
    )
    sandbox = FakeSandbox(
        tmp_path,
        {"org/local-task": [{"verifier_result": {"rewards": {"reward": 1.0}}}]},
    )
    backend = HarborBackend(_config(tmp_path, cases_path=str(cases_path)))

    report = await backend.evaluate(
        context=await _context(tmp_path, sandbox),
        request=_request(),
    )

    assert report.status == EvaluationStatus.SUCCESS
    assert report.metrics["score"] == 1.0
    assert report.cases[0].output["task_name"] == "local-task"
    assert report.cases[0].output["result_task_name"] == "org/local-task"


@pytest.mark.asyncio
async def test_harbor_backend_mean_counts_dead_attempts_as_failures(tmp_path):
    sandbox = FakeSandbox(
        tmp_path,
        {
            "example/alpha": [
                {"verifier_result": {"rewards": {"pass": 1.0}}},
                {
                    "verifier_result": None,
                    "exception_info": {"exception_type": "TimeoutError"},
                },
            ]
        },
    )
    backend = HarborBackend(_config(tmp_path, aggregate_attempts="mean"))

    report = await backend.evaluate(
        context=await _context(tmp_path, sandbox),
        request=_request(CaseIds(ids=["case-a"])),
    )

    assert report.metrics == {"score": 0.5, "error_rate": 0.0, "score_stddev": 0.0}
    assert report.cases[0].metrics == {
        "score": 0.5,
        "n_attempts": 2.0,
        "n_scored": 1.0,
        # the dead attempt was a TimeoutError -> infra dilution, not clean signal
        "n_dead_infra": 1.0,
        "n_clean": 1.0,
    }


@pytest.mark.asyncio
async def test_harbor_backend_fails_when_no_requested_trials_match(tmp_path):
    secret = "sensitive-token"
    sandbox = FakeSandbox(
        tmp_path,
        {},
        CommandResult("", f"runner failed with {secret}", 1),
    )
    backend = HarborBackend(_config(tmp_path, environment={"TOKEN": secret}))

    report = await backend.evaluate(
        context=await _context(tmp_path, sandbox),
        request=_request(CaseIds(ids=["case-a"])),
    )

    assert report.status == EvaluationStatus.FAILED
    assert report.diagnostics[0].code == "infrastructure_failure"
    assert secret not in report.diagnostics[0].message
    assert secret not in (tmp_path / "result/artifacts/harbor/stderr.log").read_text()


def test_harbor_backend_rejects_controlled_extra_flags(tmp_path):
    with pytest.raises(ValueError, match="backend-controlled"):
        _config(tmp_path, extra_args=["--jobs-dir=/tmp/forged"])


def test_environment_routes_finalization_to_reserved_scope(tmp_path):
    config = _config(
        tmp_path,
        inference_gateway_url="http://inference-gateway:8001",
        inference_gateway_token="gw-eval",
        inference_gateway_finalization_token="gw-fin",
    )
    backend = HarborBackend(config)

    search = backend._environment("eval-1", finalization=False)
    assert search["OPENAI_API_KEY"] == "gw-eval"
    assert "/scopes/evaluation/eval-1/" in search["OPENAI_BASE_URL"]

    final = backend._environment("eval-1", finalization=True)
    assert final["OPENAI_API_KEY"] == "gw-fin"
    assert "/scopes/finalization/eval-1/" in final["OPENAI_BASE_URL"]


def test_environment_finalization_falls_back_when_no_reserved_token(tmp_path):
    config = _config(
        tmp_path,
        inference_gateway_url="http://inference-gateway:8001",
        inference_gateway_token="gw-eval",
    )
    final = HarborBackend(config)._environment("eval-1", finalization=True)
    assert final["OPENAI_API_KEY"] == "gw-eval"
    assert "/scopes/evaluation/eval-1/" in final["OPENAI_BASE_URL"]


def test_retry_config_json_carries_editable_backoff(tmp_path):
    backend = HarborBackend(
        _config(
            tmp_path,
            max_retries=5,
            retry_wait_multiplier=3.0,
            retry_min_wait_seconds=2.0,
            retry_max_wait_seconds=45.0,
        )
    )

    payload = json.loads(backend._retry_config_json())

    assert payload == {
        "retry": {
            "max_retries": 5,
            "wait_multiplier": 3.0,
            "min_wait_sec": 2.0,
            "max_wait_sec": 45.0,
        }
    }


def test_command_forwards_retry_config_and_max_retries(tmp_path):
    backend = HarborBackend(_config(tmp_path, max_retries=5))

    command = backend._command(
        workspace="/work/agent",
        request=_request(),
        cases=[],
        jobs_dir="/staging/jobs",
        task_source="example/tasks@1.0",
        local_task_source=False,
        retry_config_path="/staging/retry-config.json",
    )

    # backoff arrives via the JobConfig snippet ...
    assert "--config" in command
    assert command[command.index("--config") + 1] == "/staging/retry-config.json"
    # ... while the count stays a flag (harbor applies it over the --config base).
    assert command[command.index("--max-retries") + 1] == "5"
