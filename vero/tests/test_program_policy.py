import json
import subprocess
import sys
from pathlib import Path

import pytest
import toml
from click.testing import CliRunner

from vero.config import build_program_runtime, load_config
from vero.core.cli import main
from vero.evaluation import (
    CommandBackend,
    CommandBackendConfig,
    EvaluationDatabase,
)
from vero.optimization import (
    CommandCandidateProducer,
    CommandCandidateProducerConfig,
)
from vero.policy import Policy
from vero.tools import EvaluationRunnerTool, EvaluationViewer


def _init_git_repository(path: Path) -> str:
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "add", "--all"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=vero",
            "-c",
            "user.email=vero@localhost",
            "commit",
            "-m",
            "baseline",
        ],
        cwd=path,
        check=True,
        capture_output=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _program_fixture(tmp_path: Path, *, with_optimizer: bool = True):
    target = tmp_path / "target"
    target.mkdir()
    (target / "program.txt").write_text("slow\n")
    baseline_commit = _init_git_repository(target)

    harness = tmp_path / "harness"
    harness.mkdir()
    harness_script = harness / "evaluate.py"
    harness_script.write_text(
        """
import json
import sys
from pathlib import Path

workspace, request_path, report_path = map(Path, sys.argv[1:])
request = json.loads(request_path.read_text())
program = (workspace / "program.txt").read_text().strip()
latency = 1.0 if program == "fast" else 10.0
Path(report_path).write_text(json.dumps({
    "schema_version": "1",
    "status": "success",
    "metrics": {"latency_ms": latency, "correct": 1.0},
}))
"""
    )

    optimizer = tmp_path / "optimizer"
    optimizer.mkdir()
    optimizer_script = optimizer / "optimize.py"
    optimizer_script.write_text(
        """
import sys
from pathlib import Path

workspace = Path(sys.argv[1])
(workspace / "program.txt").write_text("fast\\n")
"""
    )

    config = {
        "target": {"root": "./target", "ref": "HEAD"},
        "evaluation": {
            "backend": "command",
            "harness_root": "./harness",
            "command": [
                sys.executable,
                str(harness_script),
                "{workspace}",
                "{request}",
                "{report}",
            ],
            "evaluation_set": "performance",
            "use_copy": True,
        },
        "objective": {
            "metric": "latency_ms",
            "direction": "minimize",
            "constraints": [
                {"metric": "correct", "operator": "==", "value": 1.0}
            ],
        },
        "session": {
            "id": "test-session",
            "vero_home": str(tmp_path / "vero-home"),
        },
    }
    if with_optimizer:
        config["optimizer"] = {
            "root": "./optimizer",
            "command": [sys.executable, str(optimizer_script), "{workspace}"],
            "commit_message": "Use fast implementation",
            "max_candidates": 1,
        }
    config_path = tmp_path / "vero.toml"
    config_path.write_text(toml.dumps(config))
    return config_path, target, baseline_commit


def test_load_config_resolves_trusted_roots_relative_to_vero_toml(tmp_path: Path):
    config_path, target, _ = _program_fixture(tmp_path)

    config = load_config(config_path)

    assert config.target.root == str(target.resolve())
    assert config.evaluation.harness_root == str((tmp_path / "harness").resolve())
    assert config.optimizer.root == str((tmp_path / "optimizer").resolve())


@pytest.mark.asyncio
async def test_program_policy_optimizes_and_selects_a_generic_target(tmp_path: Path):
    config_path, target, baseline_commit = _program_fixture(tmp_path)
    runtime = await build_program_runtime(
        load_config(config_path),
        require_optimizer=True,
    )

    result = await runtime.policy.run()

    assert result.baseline.request.candidate.commit == baseline_commit
    assert result.baseline.objective.value == 10.0
    assert len(result.evaluations) == 2
    candidate = result.evaluations[1]
    assert candidate.request.candidate.parent_commit == baseline_commit
    assert candidate.objective.feasible is True
    assert candidate.objective.value == 1.0
    assert result.best == candidate
    assert result.best.request.candidate.commit != baseline_commit
    assert (target / "program.txt").read_text() == "fast\n"

    database = EvaluationDatabase.load_from_file(runtime.database_path)
    assert set(database.evaluations) == {
        result.baseline.id,
        candidate.id,
    }


@pytest.mark.asyncio
async def test_evaluation_only_config_requires_no_optimizer_or_dataset(tmp_path: Path):
    config_path, _, baseline_commit = _program_fixture(
        tmp_path,
        with_optimizer=False,
    )
    runtime = await build_program_runtime(load_config(config_path))

    record = await runtime.policy.evaluate_version(baseline_commit)

    assert record.report.metrics["latency_ms"] == 10.0
    assert record.objective.value == 10.0


@pytest.mark.asyncio
async def test_policy_constructor_supports_dataset_free_program_optimization(
    tmp_path: Path,
):
    config_path, target, baseline_commit = _program_fixture(tmp_path)
    config = load_config(config_path)
    backend = CommandBackend(
        CommandBackendConfig(
            harness_root=config.evaluation.harness_root,
            command=config.evaluation.command,
            working_directory=config.evaluation.working_directory,
        )
    )
    producer = CommandCandidateProducer(
        CommandCandidateProducerConfig(
            root=config.optimizer.root,
            command=config.optimizer.command,
            working_directory=config.optimizer.working_directory,
            commit_message=config.optimizer.commit_message,
        )
    )
    policy = Policy(
        project_path=target,
        optimizer=producer,
        backends={"command": backend},
        evaluation_set=config.evaluation.to_evaluation_set(),
        objective=config.objective.to_model(),
        max_candidates=1,
        ref="main",
        vero_home=tmp_path / "policy-vero-home",
        session_id="policy-session",
        use_default_logging=False,
        enable_console=False,
    )

    best = await policy.run()

    assert best.commit is not None
    assert best.commit != baseline_commit
    assert best.score == 1.0
    assert policy.dataset is None
    assert policy.program_run.baseline.objective.value == 10.0


@pytest.mark.asyncio
async def test_evaluation_runner_tool_uses_canonical_program_policy(tmp_path: Path):
    config_path, _, baseline_commit = _program_fixture(
        tmp_path,
        with_optimizer=False,
    )
    runtime = await build_program_runtime(load_config(config_path))
    tool = EvaluationRunnerTool()
    tool.bind(type("Session", (), {"program_policy": runtime.policy})())

    payload = json.loads(await tool.evaluate_candidate(baseline_commit))

    assert payload["candidate_commit"] == baseline_commit
    assert payload["metrics"]["latency_ms"] == 10.0
    assert "cases" not in payload
    assert await tool.evaluation_budget() == "No evaluation budget is configured."


@pytest.mark.asyncio
async def test_evaluation_viewer_exposes_summary_report_and_artifacts(tmp_path: Path):
    config_path, _, baseline_commit = _program_fixture(
        tmp_path,
        with_optimizer=False,
    )
    runtime = await build_program_runtime(load_config(config_path))
    record = await runtime.policy.evaluate_version(baseline_commit)
    viewer = EvaluationViewer()
    viewer.bind(
        type(
            "Session",
            (),
            {
                "evaluation_database": runtime.policy.engine.database,
                "split_accesses": None,
            },
        )()
    )

    summaries = json.loads(viewer.list_evaluations())
    report = json.loads(viewer.view_evaluation_report(record.id))
    artifacts = json.loads(viewer.view_evaluation_artifacts(record.id))

    assert summaries[0]["evaluation_id"] == record.id
    assert "cases" not in summaries[0]
    assert report["metrics"]["latency_ms"] == 10.0
    assert artifacts["report"][0]["path"] == "command/stdout.log"


def test_cli_evaluate_and_run_accept_vero_toml(tmp_path: Path):
    config_path, _, baseline_commit = _program_fixture(tmp_path)
    runner = CliRunner()

    evaluate_result = runner.invoke(main, ["evaluate", "--config", str(config_path)])
    assert evaluate_result.exit_code == 0, evaluate_result.output
    assert "Status: success" in evaluate_result.output
    assert "Objective: 10.0" in evaluate_result.output

    run_result = runner.invoke(main, ["run", "--config", str(config_path)])
    assert run_result.exit_code == 0, run_result.output
    assert f"Baseline commit: {baseline_commit}" in run_result.output
    assert "Baseline objective: 10.0" in run_result.output
    assert "Best objective: 1.0" in run_result.output


def test_cli_defaults_to_vero_toml_for_generic_evaluation(
    tmp_path: Path, monkeypatch
):
    _program_fixture(tmp_path, with_optimizer=False)
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(main, ["evaluate"])

    assert result.exit_code == 0, result.output
    assert "Objective: 10.0" in result.output
