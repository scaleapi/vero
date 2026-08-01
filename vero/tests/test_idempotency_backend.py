"""Idempotency regressions for the Harbor backend's sub-run plumbing.

Every test here pins work that a killed or restarted run must not have to redo:
the trials a dying sub-run already finished, a trial record half way through
redaction, and the case-resource tree a previous attempt already staged. None of
them touch scoring; the scores they assert are incidental.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from vero.candidate import Candidate
from vero.evaluation import (
    CaseCheckpointStore,
    CaseIds,
    EvaluationContext,
    EvaluationLimits,
    EvaluationRequest,
    EvaluationSet,
    RetryPolicy,
)
from vero.harbor import HarborBackend, HarborBackendConfig
from vero.sandbox import CommandResult, LocalSandbox

TASK_NAME = "example/alpha"


def _cases(path: Path) -> Path:
    path.write_text(
        json.dumps(
            [
                {"id": "case-a", "task_name": TASK_NAME},
                {"id": "case-b", "task_name": "example/beta"},
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
            max_concurrency=4,
            retry=RetryPolicy.disabled(),
        ),
    )


async def _context(tmp_path: Path, sandbox: LocalSandbox) -> EvaluationContext:
    target = tmp_path / "target"
    target.mkdir(parents=True, exist_ok=True)
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


class TrialWritingSandbox(LocalSandbox):
    """Writes one finished trial into the sub-run's jobs directory.

    ``dies`` reproduces the sub-run that goes away part way through: the trial it
    already finished is on disk in the sandbox, and the run call never returns.
    """

    def __init__(self, root: Path, *, agent_files: dict[str, str], dies: bool = False):
        super().__init__(root)
        self.agent_files = agent_files
        self.dies = dies
        self.download_attempts: list[str] = []
        self.download_fails = False
        self.download_error: BaseException | None = None

    async def run(self, command, cwd=None, timeout=30, env=None, run_as=None):
        if not isinstance(command, list) or "--jobs-dir" not in command:
            return await super().run(
                command, cwd=cwd, timeout=timeout, env=env, run_as=run_as
            )
        jobs_dir = Path(command[command.index("--jobs-dir") + 1])
        trial_dir = jobs_dir / "job-0" / "trial-alpha"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_name": TASK_NAME,
                    "trial_name": "trial-0",
                    "finished_at": "2026-01-01T00:00:00Z",
                    "verifier_result": {"rewards": {"reward": 1.0}},
                }
            ),
            encoding="utf-8",
        )
        for relative_path, content in self.agent_files.items():
            path = trial_dir / "agent" / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            # A distinctive mode, so a test can tell whether redaction preserved
            # the record's own permissions.
            path.chmod(0o640)
        if self.dies:
            raise RuntimeError("sandbox went away mid sub-run")
        return CommandResult("harbor output", "", 0)

    async def download(self, remote_path: str, local_path: str) -> None:
        self.download_attempts.append(remote_path)
        if self.download_error is not None:
            raise self.download_error
        if self.download_fails:
            raise RuntimeError("docker cp: no such container")
        await super().download(remote_path, local_path)


@pytest.mark.asyncio
async def test_dying_sub_run_still_salvages_the_trials_it_finished(tmp_path):
    sandbox = TrialWritingSandbox(tmp_path, agent_files={}, dies=True)
    backend = HarborBackend(_config(tmp_path))
    context = await _context(tmp_path, sandbox)

    with pytest.raises(RuntimeError, match="sandbox went away"):
        await backend.evaluate(
            context=context, request=_request(CaseIds(ids=["case-a"]))
        )

    # The staging area is gone by now, so this local copy is the only surviving
    # record of the trial the sub-run did finish before it died.
    salvaged = sorted((context.artifact_dir / "harbor" / "jobs").rglob("result.json"))
    assert len(salvaged) == 1
    assert json.loads(salvaged[0].read_text())["task_name"] == TASK_NAME


@pytest.mark.asyncio
async def test_failed_salvage_does_not_replace_the_sub_run_failure(tmp_path):
    sandbox = TrialWritingSandbox(tmp_path, agent_files={}, dies=True)
    sandbox.download_fails = True
    backend = HarborBackend(_config(tmp_path))
    context = await _context(tmp_path, sandbox)

    # A sandbox that has gone away fails the salvage copy too; the caller must
    # still see the sub-run's own failure, which is the diagnosable one.
    with pytest.raises(RuntimeError, match="sandbox went away"):
        await backend.evaluate(
            context=context, request=_request(CaseIds(ids=["case-a"]))
        )

    assert sandbox.download_attempts, "the salvage download was never attempted"


@pytest.mark.asyncio
async def test_cancelled_salvage_does_not_replace_the_sub_run_failure(tmp_path):
    """A cancellation landing on the salvage must not eat the real failure.

    The salvage sits in a ``finally`` and is shielded so a cancelled task still
    gets its trials off the sandbox, which means a second cancellation can arrive
    while that shielded copy is parked. Catching only ``Exception`` there let the
    ``CancelledError`` out of the ``finally`` and it replaced the sub-run's own
    error, so the caller was told the run was cancelled when what actually
    happened was a dead sandbox. The cancellation is not lost by being swallowed:
    a genuinely cancelling task sees it again at its next await.
    """
    sandbox = TrialWritingSandbox(tmp_path, agent_files={}, dies=True)
    sandbox.download_error = asyncio.CancelledError()
    backend = HarborBackend(_config(tmp_path))
    context = await _context(tmp_path, sandbox)

    with pytest.raises(RuntimeError, match="sandbox went away"):
        await backend.evaluate(
            context=context, request=_request(CaseIds(ids=["case-a"]))
        )

    assert sandbox.download_attempts, "the salvage download was never attempted"


@pytest.mark.asyncio
async def test_interrupted_redaction_leaves_the_original_trial_record(
    tmp_path, monkeypatch
):
    secret = "evaluation-scope-secret"
    original = json.dumps({"steps": [{"message": f"used {secret}"}]})
    sandbox = TrialWritingSandbox(tmp_path, agent_files={"trajectory.json": original})
    backend = HarborBackend(_config(tmp_path, environment={"EVALUATION_TOKEN": secret}))
    context = await _context(tmp_path, sandbox)
    real_replace = os.replace

    def fail_the_rename(source, destination, *args, **kwargs):
        if str(destination).endswith("trajectory.json"):
            raise OSError("no space left on device")
        return real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "replace", fail_the_rename)

    report = await backend.evaluate(
        context=context, request=_request(CaseIds(ids=["case-a"]))
    )

    trajectory = next(
        context.artifact_dir / artifact.path
        for artifact in report.cases[0].artifacts
        if Path(artifact.path).name == "trajectory.json"
    )
    # The redaction that could not complete leaves the record exactly as Harbor
    # wrote it, not truncated to nothing: the report the caller sees is sanitized
    # on its own path, so an intact original is recoverable while a half-written
    # one is neither the original nor the redacted version.
    assert trajectory.read_text(encoding="utf-8") == original


@pytest.mark.asyncio
async def test_completed_redaction_leaves_no_half_written_sibling(tmp_path):
    secret = "evaluation-scope-secret"
    sandbox = TrialWritingSandbox(
        tmp_path,
        agent_files={
            "trajectory.json": json.dumps({"steps": [{"message": f"used {secret}"}]})
        },
    )
    backend = HarborBackend(_config(tmp_path, environment={"EVALUATION_TOKEN": secret}))
    context = await _context(tmp_path, sandbox)

    report = await backend.evaluate(
        context=context, request=_request(CaseIds(ids=["case-a"]))
    )

    names = {Path(artifact.path).name for artifact in report.cases[0].artifacts}
    assert names == {"result.json", "trajectory.json"}
    trajectory = next(
        context.artifact_dir / artifact.path
        for artifact in report.cases[0].artifacts
        if Path(artifact.path).name == "trajectory.json"
    )
    assert "[REDACTED]" in trajectory.read_text(encoding="utf-8")
    assert secret not in trajectory.read_text(encoding="utf-8")
    # The rename must not hand the agent context a record with this process's
    # umask in place of the record's own mode.
    assert trajectory.stat().st_mode & 0o777 == 0o640
    # The sibling the rename consumed must not survive as a second copy of the
    # record next to it.
    assert sorted(path.name for path in trajectory.parent.iterdir()) == [
        "trajectory.json"
    ]


@pytest.mark.asyncio
async def test_case_resource_staging_isolates_attempts_from_each_other(
    tmp_path, monkeypatch
):
    """A retry must stage somewhere the previous attempt cannot be holding.

    A stable staging path would be the more idempotent choice on its own, since
    a retry could reclaim the tree a killed attempt abandoned. It is not the safe
    one: nothing serializes two processes over one configured cache path, and the
    stage clears its directory before using it, so a shared name means one
    attempt can delete a tree another is still materializing into. Distinct paths
    cost a stale directory after a hard kill and buy that isolation.
    """

    task_source = tmp_path / "tasks"
    task_source.mkdir()
    for task_name in ("alpha", "beta"):
        task = task_source / task_name
        task.mkdir()
        (task / "task.toml").write_text(f'[task]\nname="example/{task_name}"\n')
    cases_path = tmp_path / "local-cases.json"
    cases_path.write_text(
        json.dumps(
            [
                {"id": "case-a", "task_name": "alpha"},
                {"id": "case-b", "task_name": "beta"},
            ]
        ),
        encoding="utf-8",
    )
    cache = tmp_path / "case-resources" / "test"
    backend = HarborBackend(
        _config(
            tmp_path,
            task_source=str(task_source),
            cases_path=str(cases_path),
            case_resources_cache_path=str(cache),
        )
    )
    evaluation_set = EvaluationSet(
        name="harbor-bench", partition="test", selection=CaseIds(ids=["case-b"])
    )
    sandbox = await LocalSandbox.create(root=tmp_path)
    staged: list[Path] = []
    materialize = backend._materialize_case_resources

    async def record_then_fail_once(root, cases, evaluation_set):
        staged.append(root)
        if len(staged) == 1:
            raise OSError("dataset download died")
        await materialize(root, cases, evaluation_set)

    monkeypatch.setattr(backend, "_materialize_case_resources", record_then_fail_once)

    first = tmp_path / "context-first"
    first.mkdir()
    with pytest.raises(OSError, match="dataset download died"):
        await backend.export_case_resources(
            evaluation_set=evaluation_set, destination=str(first), sandbox=sandbox
        )

    # Stand in for the tree a run killed outright leaves behind, which never
    # reaches the cleanup below.
    (staged[0] / "tasks").mkdir(parents=True)
    (staged[0] / "tasks" / "leftover").write_text("partial\n", encoding="utf-8")

    second = tmp_path / "context-second"
    second.mkdir()
    await backend.export_case_resources(
        evaluation_set=evaluation_set, destination=str(second), sandbox=sandbox
    )

    # Two attempts, two paths: neither can clear the other's work out from under
    # it, and the abandoned partial cannot leak into what the retry exports.
    assert staged[0] != staged[1]
    assert staged[0].parent == staged[1].parent == cache.parent
    index = json.loads((second / "index.json").read_text())
    assert [item["case_id"] for item in index["cases"]] == ["case-b"]
    assert not (second / "tasks" / "leftover").exists()
    # The attempt that ran to completion cleans up after itself; only the killed
    # one leaves a directory behind, which is the accepted cost of the isolation.
    assert not staged[1].exists()
