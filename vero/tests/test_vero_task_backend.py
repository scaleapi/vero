import json
import subprocess
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from datasets import Dataset, DatasetDict

from vero.agents.base import BaseAgent
from vero.artifacts import TracesArtifact
from vero.core.db.candidate import Candidate
from vero.core.db.database import Experiment
from vero.core.db.dataset import DatasetSample, DatasetSubset
from vero.core.db.result import (
    ExperimentResult,
    ExperimentResultStatus,
    SampleResult,
)
from vero.core.db.run import ExperimentRun
from vero.evaluation import (
    CaseIds,
    CaseRange,
    EvaluationContext,
    EvaluationRecord,
    EvaluationReport,
    EvaluationRequest,
    EvaluationSet,
    EvaluationStatus,
    BackendProvenance,
    CaseError,
    CaseResult,
    CaseStatus,
    VeroTaskBackend,
    VeroTaskBackendConfig,
    compatibility_objective,
    evaluate_objective,
    evaluation_record_to_experiment,
)
from vero.policy import Policy


class _NoOpAgent(BaseAgent):
    def init(self, session):
        self.session = session

    async def step(self, input, max_turns=200, on_event=None, **kwargs):
        return None

    def serialize_trace(self):
        return None

    def serialize_state(self):
        return None

    def deserialize_state(self, state):
        return None


def _backend(tmp_path: Path) -> VeroTaskBackend:
    return VeroTaskBackend(
        VeroTaskBackendConfig(
            session_id="session",
            vero_home=str(tmp_path),
            task="main",
        )
    )


def test_vero_task_backend_resolves_ids_and_truncated_ranges(tmp_path: Path, monkeypatch):
    backend = _backend(tmp_path)
    monkeypatch.setattr(backend, "_split_size", lambda _: 5)

    assert backend._sample_ids(
        EvaluationSet(partition="test", selection=CaseIds(ids=["0", "4"]))
    ) == [0, 4]
    assert backend._sample_ids(
        EvaluationSet(
            partition="test",
            selection=CaseRange(start=2, stop=20),
        )
    ) == [2, 3, 4]
    with pytest.raises(ValueError, match="canonical non-negative integer"):
        backend._sample_ids(
            EvaluationSet(partition="test", selection=CaseIds(ids=["01"]))
        )
    with pytest.raises(ValueError, match="outside partition"):
        backend._sample_ids(
            EvaluationSet(partition="test", selection=CaseRange(start=5, stop=6))
        )


@pytest.mark.asyncio
async def test_vero_task_backend_converts_legacy_experiment_to_report(
    tmp_path: Path, monkeypatch
):
    backend = _backend(tmp_path)
    monkeypatch.setattr(backend, "_split_size", lambda _: 2)
    candidate = Candidate(
        commit="candidate",
        repo_name="repo",
        created_at=datetime(2026, 1, 1),
    )
    run = ExperimentRun(
        candidate=candidate,
        dataset_subset=DatasetSubset(
            dataset_id="dataset",
            split="test",
        ),
    )
    experiment = Experiment(
        run=run,
        result=ExperimentResult(
            run_id=run.id,
            status=ExperimentResultStatus.SUCCESS,
            sample_results={
                0: SampleResult(
                    dataset_sample=DatasetSample(
                        dataset_id="dataset", split="test", sample_id=0
                    ),
                    score=1.0,
                ),
                1: SampleResult(
                    dataset_sample=DatasetSample(
                        dataset_id="dataset", split="test", sample_id=1
                    ),
                    error="failed",
                ),
            },
        ),
    )
    evaluate = AsyncMock(return_value=experiment)
    monkeypatch.setattr("vero.evaluation.vero_task.LegacyEvaluator.evaluate", evaluate)

    class Workspace:
        name = "repo"
        root = str(tmp_path)
        project_path = str(tmp_path)
        sandbox = SimpleNamespace()

        async def current_version(self):
            return "candidate"

        async def is_dirty(self):
            return False

    context = EvaluationContext(
        workspace=Workspace(),
        session_id="session",
        evaluation_id="evaluation",
        result_dir=tmp_path / "result",
        artifact_dir=tmp_path / "result" / "artifacts",
        case_store=SimpleNamespace(),
    )
    request = EvaluationRequest(
        candidate=candidate,
        evaluation_set=EvaluationSet(name="dataset", partition="test"),
        parameters={"temperature": 0.0},
    )

    report = await backend.evaluate(context=context, request=request)

    assert report.status == EvaluationStatus.SUCCESS
    assert report.metrics["score"] == 0.5
    assert report.metrics["error_rate"] == 0.5
    assert [case.case_id for case in report.cases] == ["0", "1"]
    assert report.cases[1].errors[0].terminal is True
    call = evaluate.await_args.kwargs
    assert call["dataset_id"] == "dataset"
    assert call["split"] == "test"
    assert call["task"] == "main"
    assert call["evaluation_parameters"].task_params == {"temperature": 0.0}
    assert call["use_copy"] is False


def test_canonical_vero_task_record_projects_to_legacy_experiment_view():
    candidate = Candidate(
        commit="candidate",
        repo_name="repo",
        created_at=datetime(2026, 1, 1),
    )
    request = EvaluationRequest(
        candidate=candidate,
        evaluation_set=EvaluationSet(
            name="dataset",
            partition="test",
            selection=CaseIds(ids=["4", "9"]),
        ),
    )
    report = EvaluationReport(
        status=EvaluationStatus.SUCCESS,
        metrics={"score": 0.5},
        cases=[
            CaseResult(
                case_id="4",
                status=CaseStatus.SUCCESS,
                metrics={"score": 1.0, "latency": 2.0},
                output={"answer": 1},
            ),
            CaseResult(
                case_id="9",
                status=CaseStatus.ERROR,
                errors=[
                    CaseError(message="inference failed", phase="execution"),
                    CaseError(
                        message="scoring failed",
                        phase="scoring",
                        terminal=True,
                    ),
                ],
            ),
        ],
    )
    objective_spec = compatibility_objective()
    record = EvaluationRecord(
        id="evaluation",
        request=request,
        report=report,
        backend_id="vero-task",
        backend=BackendProvenance.from_config(
            name="vero-task",
            version="1",
            config={},
        ),
        objective_spec=objective_spec,
        objective=evaluate_objective(report, objective_spec),
        created_at=datetime(2026, 1, 1),
        completed_at=datetime(2026, 1, 1),
    )

    experiment = evaluation_record_to_experiment(record)

    assert experiment.id == "evaluation"
    assert experiment.run.dataset_subset.sample_ids == [4, 9]
    assert experiment.result.sample_results[4].score == 1.0
    assert experiment.result.sample_results[4].metrics == {"latency": 2.0}
    assert experiment.result.sample_results[9].error == "inference failed"
    assert experiment.result.sample_results[9].eval_error == "scoring failed"


@pytest.mark.asyncio
async def test_dataset_policy_routes_evaluation_through_vero_task_backend(
    tmp_path: Path,
    monkeypatch,
):
    target = tmp_path / "target"
    target.mkdir()
    (target / "program.txt").write_text("candidate\n")
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=target,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "add", "--all"],
        cwd=target,
        check=True,
        capture_output=True,
    )
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
        cwd=target,
        check=True,
        capture_output=True,
    )
    dataset_path = tmp_path / "dataset"
    DatasetDict({"test": Dataset.from_dict({"input": ["one"]})}).save_to_disk(
        str(dataset_path)
    )

    calls = []

    async def evaluate_backend(self, *, context, request):
        calls.append((context, request))
        return EvaluationReport(
            status=EvaluationStatus.SUCCESS,
            metrics={"score": 0.75, "error_rate": 0.0, "num_results": 1.0},
            cases=[
                CaseResult(
                    case_id="0",
                    status=CaseStatus.SUCCESS,
                    metrics={"score": 0.75},
                )
            ],
        )

    monkeypatch.setattr(VeroTaskBackend, "evaluate", evaluate_backend)
    policy = Policy(
        project_path=target,
        dataset=dataset_path,
        agent=_NoOpAgent(),
        task="score",
        use_copy=False,
        vero_home=tmp_path / "vero-home",
        artifacts=[TracesArtifact()],
        split_accesses=[],
        use_default_logging=False,
        enable_console=False,
    )
    await policy.init()

    experiment = await policy.evaluate_version(
        policy.session.base_version,
        split="test",
    )

    assert len(calls) == 1
    assert calls[0][1].evaluation_set == EvaluationSet(
        name=dataset_path.stem,
        partition="test",
    )
    assert experiment.result.score() == 0.75
    assert len(policy.evaluation_db.evaluations) == 1
    record = next(iter(policy.evaluation_db.evaluations.values()))
    assert record.backend_id == "vero-task"
    assert record.schema_version == 2
    assert policy.session.policy is policy
    assert policy.session.engine.database is policy.session.database
    assert policy.session.engine.budget_ledger is policy.session.budget_ledger
    assert {
        "evaluator",
        "db",
        "budget",
        "evaluation_parameters",
        "program_policy",
        "evaluation_engine",
        "evaluation_database",
        "evaluation_backend_id",
        "evaluation_objective",
    }.isdisjoint(vars(policy.session))
    compatibility_db = policy.session.db
    assert compatibility_db is not None
    assert [item.id for item in compatibility_db.get_experiments()] == [record.id]
    result_dir = (
        Path(policy._vero_home)
        / "sessions"
        / policy.session_id
        / "experiments"
        / record.id
    )
    assert (result_dir / "evaluation.json").exists()
    assert not (result_dir / "evaluation_parameters.json").exists()
    trace_dir = (
        target
        / "_vero"
        / "traces"
        / f"test__{record.request.candidate.commit[:8]}"
    )
    trace_summary = json.loads((trace_dir / "summary.json").read_text())
    assert trace_summary["evaluation_id"] == record.id
    assert trace_summary["objective"]["value"] == 0.75
    case_files = [
        path for path in trace_dir.glob("*.json") if path.name != "summary.json"
    ]
    assert len(case_files) == 1
    assert json.loads(case_files[0].read_text())["case_id"] == "0"
    assert policy.get_best_version(["test"]).evaluation_id == record.id
    policy.finish()
    database_payload = json.loads((
        Path(policy._vero_home)
        / "sessions"
        / policy.session_id
        / "database.json"
    ).read_text())
    assert database_payload["schema_version"] == 2
