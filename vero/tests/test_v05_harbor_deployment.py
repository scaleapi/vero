from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from vero.evaluation import (
    DisclosureLevel,
    EvaluationAccessPolicy,
    EvaluationBudget,
    EvaluationSet,
    MetricSelector,
    ObjectiveSpec,
)
from vero.harbor import (
    HarborBackendConfig,
    build_harbor_components,
)
from vero.report import generate_experiment_report
from vero.sidecar import (
    SidecarEvaluationPolicy,
    VerificationTarget,
)


def _git(path: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=path,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _repo(path: Path, content: str) -> str:
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.name", "VeRO Test")
    _git(path, "config", "user.email", "vero@example.test")
    (path / "program.py").write_text(content, encoding="utf-8")
    _git(path, "add", "program.py")
    _git(path, "commit", "-q", "-m", "baseline")
    return _git(path, "rev-parse", "HEAD")


@pytest.mark.asyncio
async def test_standard_deployment_factory_builds_one_canonical_runtime(tmp_path):
    trusted = tmp_path / "trusted"
    agent = tmp_path / "agent"
    baseline = _repo(trusted, "VALUE = 1\n")
    _repo(agent, "VALUE = 1\n")
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        json.dumps({"id": "task", "task_name": "org/task"}) + "\n",
        encoding="utf-8",
    )
    evaluation_set = EvaluationSet(name="benchmark", partition="validation")
    objective = ObjectiveSpec(
        selector=MetricSelector(metric="score"),
        direction="maximize",
    )
    backend_config = HarborBackendConfig(
        task_source="org/benchmark@1.0",
        agent_import_path="program:Agent",
        cases_path=str(cases),
        harbor_requirement="harbor==0.1.17",
        evaluation_set_name="benchmark",
        partition="validation",
        uv_executable=sys.executable,
    )
    budget = EvaluationBudget(
        backend_id="validation",
        evaluation_set_key=evaluation_set.budget_key("validation"),
        total_runs=4,
        total_cases=10,
    )
    config = {
        "repo_path": str(trusted),
        "agent_repo_path": str(agent),
        "session_dir": str(tmp_path / "state/session"),
        "session_id": "trial",
        "backends": {"validation": backend_config.model_dump(mode="json")},
        "access_policies": [
            SidecarEvaluationPolicy(
                backend_id="validation",
                evaluation_set_name="benchmark",
                partition="validation",
                objective=objective,
                access=EvaluationAccessPolicy(
                    disclosure=DisclosureLevel.AGGREGATE
                ),
            ).model_dump(mode="json")
        ],
        "budgets": [budget.model_dump(mode="json")],
        "selection": {
            "mode": "auto_best",
            "backend_id": "validation",
            "evaluation_set": evaluation_set.model_dump(mode="json"),
            "objective": objective.model_dump(mode="json"),
            "baseline_version": "HEAD",
        },
        "targets": [
            VerificationTarget(
                reward_key="reward",
                backend_id="validation",
                evaluation_set=evaluation_set,
                objective=objective,
            ).model_dump(mode="json")
        ],
        "agent_volume": str(tmp_path / "state/agent"),
        "admin_volume": str(tmp_path / "state/admin"),
    }

    components = await build_harbor_components(config)

    assert components.sidecar.engine is components.verifier.engine
    assert components.verifier.selection.baseline_candidate.version == baseline
    assert components.sidecar.status().evaluation_access[0].budget.remaining_runs == 4
    assert (tmp_path / "state/session/budgets.json").is_file()
    # The trusted session dir is locked to its owner so the unprivileged harness
    # cannot read the held-out records, budgets, or candidates it contains.
    assert ((tmp_path / "state/session").stat().st_mode & 0o777) == 0o700
    manifest = json.loads((tmp_path / "state/session/harbor-session.json").read_text())
    assert manifest["id"] == "trial"
    assert manifest["selection"]["evaluation_set"] == {
        "name": "benchmark",
        "partition": "validation",
        "selection": {"kind": "all"},
    }
    assert (tmp_path / "state/agent/manifest.json").is_file()
    assert json.loads(
        (tmp_path / "state/agent/results/index.json").read_text()
    ) == {"schema_version": 1, "evaluations": []}
    report = await generate_experiment_report(
        tmp_path / "state/session",
        tmp_path / "experiment.html",
    )
    assert baseline in report.read_text()


@pytest.mark.asyncio
async def test_standard_deployment_fails_closed_on_corrupt_budget_state(tmp_path):
    trusted = tmp_path / "trusted"
    agent = tmp_path / "agent"
    _repo(trusted, "VALUE = 1\n")
    _repo(agent, "VALUE = 1\n")
    cases = tmp_path / "cases.json"
    cases.write_text('[{"id":"task","task_name":"org/task"}]')
    session = tmp_path / "state/session"
    session.mkdir(parents=True)
    (session / "budgets.json").write_text("not json", encoding="utf-8")
    evaluation_set = EvaluationSet(name="benchmark")
    objective = ObjectiveSpec(
        selector=MetricSelector(metric="score"),
        direction="maximize",
    )
    config = {
        "repo_path": str(trusted),
        "agent_repo_path": str(agent),
        "session_dir": str(session),
        "backends": {
            "backend": {
                "task_source": "org/benchmark@1.0",
                "agent_import_path": "program:Agent",
                "cases_path": str(cases),
                "harbor_requirement": "harbor==0.1.17",
                "uv_executable": sys.executable,
            }
        },
        "access_policies": [],
        "budgets": [],
        "selection": {
            "mode": "auto_best",
            "backend_id": "backend",
            "evaluation_set": evaluation_set.model_dump(mode="json"),
            "objective": objective.model_dump(mode="json"),
            "baseline_version": "HEAD",
            "baseline_floor": False,
        },
        "targets": [
            {
                "reward_key": "reward",
                "backend_id": "backend",
                "evaluation_set": evaluation_set.model_dump(mode="json"),
                "objective": objective.model_dump(mode="json"),
            }
        ],
        "admin_volume": str(tmp_path / "state/admin"),
    }

    with pytest.raises(ValueError, match="invalid durable budget ledger"):
        await build_harbor_components(config)


@pytest.mark.asyncio
async def test_deployment_finalize_hook_archives_gateway_state(tmp_path):
    # Even without W&B configured, finalize must copy the gateway's usage
    # ledger and request log into the session so /session/export carries them.
    trusted = tmp_path / "trusted"
    agent = tmp_path / "agent"
    _repo(trusted, "VALUE = 1\n")
    _repo(agent, "VALUE = 1\n")
    cases = tmp_path / "cases.json"
    cases.write_text('[{"id":"task","task_name":"org/task"}]')
    usage_path = tmp_path / "state/inference/usage.json"
    usage_path.parent.mkdir(parents=True)
    usage_path.write_text('{"schema_version": 1, "scopes": {}}')
    requests_dir = tmp_path / "state/inference/requests"
    requests_dir.mkdir()
    (requests_dir / "requests-00001.jsonl").write_text('{"scope":"producer"}\n')
    evaluation_set = EvaluationSet(name="benchmark")
    objective = ObjectiveSpec(
        selector=MetricSelector(metric="score"),
        direction="maximize",
    )
    session_dir = tmp_path / "state/session"
    config = {
        "repo_path": str(trusted),
        "agent_repo_path": str(agent),
        "session_dir": str(session_dir),
        "backends": {
            "backend": {
                "task_source": "org/benchmark@1.0",
                "agent_import_path": "program:Agent",
                "cases_path": str(cases),
                "harbor_requirement": "harbor==0.1.17",
                "uv_executable": sys.executable,
            }
        },
        "access_policies": [],
        "budgets": [],
        "selection": {
            "mode": "auto_best",
            "backend_id": "backend",
            "evaluation_set": evaluation_set.model_dump(mode="json"),
            "objective": objective.model_dump(mode="json"),
            "baseline_version": "HEAD",
        },
        "targets": [
            {
                "reward_key": "reward",
                "backend_id": "backend",
                "evaluation_set": evaluation_set.model_dump(mode="json"),
                "objective": objective.model_dump(mode="json"),
            }
        ],
        "admin_volume": str(tmp_path / "state/admin"),
        "inference_usage_path": str(usage_path),
        "inference_request_log_dir": str(requests_dir),
    }

    components = await build_harbor_components(config)
    # No W&B -> no poller, but the finalize hook is still installed.
    assert components.telemetry is None
    assert components.verifier._on_finalized is not None

    class Result:
        shipped = True
        rewards = {}

    components.verifier._on_finalized(Result())

    exported = session_dir / "artifacts/inference"
    assert (exported / "usage.json").read_text().startswith('{"schema_version"')
    assert (
        exported / "requests/requests-00001.jsonl"
    ).read_text() == '{"scope":"producer"}\n'
