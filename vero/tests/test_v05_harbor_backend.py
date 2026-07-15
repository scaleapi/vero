from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

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
)
from vero.harbor import HarborBackend, HarborBackendConfig
from vero.sandbox import CommandResult


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
            case_timeout_seconds=30,
            max_concurrency=7,
        ),
    )


class FakeSandbox:
    def __init__(self, trials: dict[str, list[dict]], result: CommandResult | None = None):
        self.trials = trials
        self.result = result or CommandResult("harbor output", "", 0)
        self.command = None
        self.cwd = None
        self.timeout = None
        self.env = None

    async def run(self, command, *, cwd, timeout, env):
        self.command = command
        self.cwd = cwd
        self.timeout = timeout
        self.env = env
        jobs_dir = Path(command[command.index("--jobs-dir") + 1])
        for task_name, attempts in self.trials.items():
            for index, attempt in enumerate(attempts):
                trial_dir = jobs_dir / f"job-{index}" / f"trial-{task_name.split('/')[-1]}"
                trial_dir.mkdir(parents=True, exist_ok=True)
                payload = {
                    "task_name": task_name,
                    "trial_name": f"trial-{index}",
                    "finished_at": f"2026-01-01T00:00:0{index}Z",
                    **attempt,
                }
                (trial_dir / "result.json").write_text(json.dumps(payload))
        return self.result


async def _context(tmp_path: Path, sandbox: FakeSandbox) -> EvaluationContext:
    target = tmp_path / "target"
    target.mkdir(exist_ok=True)
    result_dir = tmp_path / "result"
    artifact_dir = result_dir / "artifacts"
    artifact_dir.mkdir(parents=True)
    return EvaluationContext(
        workspace=SimpleNamespace(project_path=str(target), sandbox=sandbox),
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
            evaluation_set.model_copy(
                update={"selection": CaseIds(ids=["missing"])}
            )
        )


@pytest.mark.asyncio
async def test_harbor_backend_runs_and_zero_fills_missing_rewards(tmp_path):
    sandbox = FakeSandbox(
        {
            "example/alpha": [
                {"verifier_result": {"rewards": {"reward": 1.0}}}
            ],
            "example/beta": [
                {
                    "verifier_result": None,
                    "exception_info": {"exception_type": "AgentCrash"},
                }
            ],
        }
    )
    backend = HarborBackend(_config(tmp_path))
    runtime_context = await _context(tmp_path, sandbox)

    report = await backend.evaluate(
        context=runtime_context,
        request=_request(CaseRange(stop=2)),
    )

    assert report.status == EvaluationStatus.SUCCESS
    assert report.metrics == {"score": 0.5, "error_rate": 0.5}
    assert [case.status for case in report.cases] == [
        CaseStatus.SUCCESS,
        CaseStatus.ERROR,
    ]
    assert report.cases[1].metrics["score"] == 0.0
    assert report.cases[1].errors[0].code == "harbor_no_reward"
    assert sandbox.command[:7] == [
        sys.executable,
        "run",
        "--project",
        str(tmp_path / "target"),
        "--with",
        "harbor==0.1.17",
        "harbor",
    ]
    assert sandbox.command.count("-i") == 2
    assert sandbox.command[sandbox.command.index("-n") + 1] == "7"
    assert sandbox.timeout == 90
    assert [artifact.path for artifact in report.artifacts] == [
        "harbor/stdout.log",
        "harbor/stderr.log",
    ]
    checkpoints = await runtime_context.case_store.load_all()
    assert [case.case_id for case in checkpoints] == ["case-a", "case-b"]


@pytest.mark.asyncio
async def test_harbor_backend_mean_counts_dead_attempts_as_failures(tmp_path):
    sandbox = FakeSandbox(
        {
            "example/alpha": [
                {"verifier_result": {"rewards": {"pass": 1.0}}},
                {
                    "verifier_result": None,
                    "exception_info": {"exception_type": "TimeoutError"},
                },
            ]
        }
    )
    backend = HarborBackend(_config(tmp_path, aggregate_attempts="mean"))

    report = await backend.evaluate(
        context=await _context(tmp_path, sandbox),
        request=_request(CaseIds(ids=["case-a"])),
    )

    assert report.metrics == {"score": 0.5, "error_rate": 0.0}
    assert report.cases[0].metrics == {
        "score": 0.5,
        "n_attempts": 2.0,
        "n_scored": 1.0,
    }


@pytest.mark.asyncio
async def test_harbor_backend_fails_when_no_requested_trials_match(tmp_path):
    secret = "sensitive-token"
    sandbox = FakeSandbox(
        {},
        CommandResult("", f"runner failed with {secret}", 1),
    )
    backend = HarborBackend(_config(tmp_path, environment={"TOKEN": secret}))

    report = await backend.evaluate(
        context=await _context(tmp_path, sandbox),
        request=_request(CaseIds(ids=["case-a"])),
    )

    assert report.status == EvaluationStatus.FAILED
    assert report.diagnostics[0].code == "harbor_no_trials"
    assert secret not in report.diagnostics[0].message
    assert secret not in (tmp_path / "result/artifacts/harbor/stderr.log").read_text()


def test_harbor_backend_rejects_controlled_extra_flags(tmp_path):
    with pytest.raises(ValueError, match="backend-controlled"):
        _config(tmp_path, extra_args=["--jobs-dir=/tmp/forged"])
