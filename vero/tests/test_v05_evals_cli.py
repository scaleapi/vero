from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio
from click.testing import CliRunner

from vero.candidate import Candidate
from vero.evals_cli import _enrich_job, evals
from vero.evaluation import (
    BackendProvenance,
    CaseResult,
    CaseStatus,
    DisclosureLevel,
    EvaluationRecord,
    EvaluationReport,
    EvaluationRequest,
    EvaluationSet,
    EvaluationStatus,
    MetricSelector,
    ObjectiveResult,
    ObjectiveSpec,
    project_evaluation,
)
from vero.runtime.context import AgentContextDirectory
from vero.sandbox import LocalSandbox


def _record(evaluation_id: str, scores: dict[str, float], value: float):
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    return EvaluationRecord(
        id=evaluation_id,
        request=EvaluationRequest(
            candidate=Candidate(
                id=f"candidate:{evaluation_id}",
                version="a" * 40,
                created_at=created_at,
            ),
            evaluation_set=EvaluationSet(name="validation", partition="validation"),
        ),
        report=EvaluationReport(
            status=EvaluationStatus.SUCCESS,
            metrics={"score": value},
            cases=[
                CaseResult(
                    case_id=case_id,
                    status=CaseStatus.SUCCESS,
                    metrics={"score": score},
                    input={"task_name": f"task-{case_id}"},
                    output={"answer": "answer"},
                    execution_trace=[
                        {"turn": 1, "message": f"marker-{case_id}"},
                        {"turn": 2, "tool": "run", "result": {"stdout": "ok"}},
                    ],
                )
                for case_id, score in scores.items()
            ],
        ),
        backend_id="backend",
        backend=BackendProvenance(name="test", version="1", config_digest="0" * 64),
        objective_spec=ObjectiveSpec(
            selector=MetricSelector(metric="score"), direction="maximize"
        ),
        objective=ObjectiveResult(value=value, feasible=True),
        created_at=created_at,
        completed_at=created_at + timedelta(seconds=1),
    )


@pytest_asyncio.fixture
async def context_dir(tmp_path: Path) -> Path:
    baseline = _record("evaluation:baseline", {"one": 0.25, "two": 1.0}, 0.625)
    candidate = _record("evaluation:candidate", {"one": 0.75, "two": 0.5}, 0.7)
    aggregate = _record("evaluation:aggregate", {"one": 0.5}, 0.5)
    project = tmp_path / "project"
    project.mkdir()
    directory = AgentContextDirectory(
        sandbox=await LocalSandbox.create(root=tmp_path),
        root=str(project / ".evals"),
        session_dir=tmp_path / "session",
    )
    await directory.reset()
    await directory.write_header(
        session_id="session",
        round_number=1,
        proposal_id="proposal",
        parent_candidate_id="parent",
    )
    await directory.write_evaluations(
        [
            (item, disclosure, project_evaluation(item, disclosure))
            for item, disclosure in (
                (baseline, DisclosureLevel.FULL),
                (candidate, DisclosureLevel.FULL),
                (aggregate, DisclosureLevel.AGGREGATE),
            )
        ]
    )
    root = project / ".evals"
    (root / "plan.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evaluations": [
                    {
                        "name": "validation",
                        "partition": "validation",
                        "base_selection": {"kind": "all"},
                        "agent_can_evaluate": True,
                        "agent_selection": "arbitrary",
                        "disclosure": "full",
                        "expose_case_resources": True,
                        "cases": 12,
                        "budget": {"remaining_runs": 3, "remaining_cases": 40},
                    }
                ],
            }
        )
    )
    resources = root / "tasks" / "digest" / "resources"
    resources.mkdir(parents=True)
    (resources / "one.json").write_text('{"prompt": "task one"}')
    (resources / "index.json").write_text(
        json.dumps({"schema_version": 1, "cases": [{"case_id": "one", "path": "one.json"}]})
    )
    (root / "tasks" / "index.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "case_resources": [
                    {
                        "backend_id": "backend",
                        "evaluation_set": {"name": "validation", "partition": "validation"},
                        "path": "digest",
                    }
                ],
            }
        )
    )
    return root


def _invoke(context_dir: Path, *arguments: str):
    result = CliRunner().invoke(
        evals, [*arguments, "--context", str(context_dir)], catch_exceptions=False
    )
    return result


@pytest.mark.asyncio
async def test_list_shows_every_result_and_sorts(context_dir: Path):
    result = _invoke(context_dir, "list", "--json")
    assert result.exit_code == 0
    rows = json.loads(result.output)
    assert {row["id"] for row in rows} == {
        "evaluation:baseline",
        "evaluation:candidate",
        "evaluation:aggregate",
    }
    by_id = {row["id"]: row for row in rows}
    assert by_id["evaluation:baseline"]["score"] == 0.625
    assert by_id["evaluation:baseline"]["cases"] == 2
    assert by_id["evaluation:aggregate"]["disclosure"] == "aggregate"
    assert by_id["evaluation:aggregate"]["cases"] == 1  # summary count only

    ordered = json.loads(
        _invoke(context_dir, "list", "--json", "--sort", "score", "--desc").output
    )
    assert ordered[0]["id"] == "evaluation:candidate"


@pytest.mark.asyncio
async def test_show_summarizes_case_files(context_dir: Path):
    result = _invoke(context_dir, "show", "evaluation:baseline")
    assert result.exit_code == 0
    assert "2 cases" in result.output
    assert "evals cases" in result.output


@pytest.mark.asyncio
async def test_cases_lists_per_case_results_and_respects_disclosure(context_dir: Path):
    result = _invoke(context_dir, "cases", "evaluation:baseline", "--json")
    assert result.exit_code == 0
    rows = json.loads(result.output)
    assert [(row["case_id"], row["score"]) for row in rows] == [
        ("one", 0.25),
        ("two", 1.0),
    ]
    assert all(row["trace"] for row in rows)

    denied = CliRunner().invoke(
        evals, ["cases", "evaluation:aggregate", "--context", str(context_dir)]
    )
    assert denied.exit_code != 0
    assert "disclosure" in denied.output


@pytest.mark.asyncio
async def test_trace_summary_and_span_window(context_dir: Path):
    summary = _invoke(context_dir, "trace", "evaluation:baseline", "one")
    assert summary.exit_code == 0
    payload = json.loads(summary.output)
    assert payload["execution_trace"]["spans"] == 2
    assert any("turn" in shape for shape in payload["execution_trace"]["shapes"])

    span = _invoke(context_dir, "trace", "evaluation:baseline", "one", "--span", "0")
    assert span.exit_code == 0
    assert "marker-one" in span.output


@pytest.mark.asyncio
async def test_diff_reports_per_case_verdicts(context_dir: Path):
    result = _invoke(
        context_dir, "diff", "evaluation:baseline", "evaluation:candidate", "--json"
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    verdicts = {row["case_id"]: row["verdict"] for row in payload["cases"]}
    assert verdicts == {"one": "improved", "two": "regressed"}
    assert payload["summary"] == {"improved": 1, "regressed": 1}


@pytest.mark.asyncio
async def test_plan_shows_budget(context_dir: Path):
    result = _invoke(context_dir, "plan", "--json")
    assert result.exit_code == 0
    rows = json.loads(result.output)
    assert rows == [
        {
            "evaluation": "validation",
            "partition": "validation",
            "cases": 12,
            "can_evaluate": True,
            "selection": "arbitrary",
            "disclosure": "full",
            "runs_left": 3,
            "cases_left": 40,
        }
    ]


@pytest.mark.asyncio
async def test_tasks_lists_sets_then_task_paths(context_dir: Path):
    sets = json.loads(_invoke(context_dir, "tasks", "--json").output)
    assert sets == [
        {
            "evaluation": "validation",
            "partition": "validation",
            "tasks": 1,
            "path": "tasks/digest/resources/",
        }
    ]
    tasks = json.loads(_invoke(context_dir, "tasks", "validation", "--json").output)
    assert tasks == [{"case_id": "one", "path": "tasks/digest/resources/one.json"}]


@pytest.mark.asyncio
async def test_context_is_discovered_from_workspace(context_dir: Path, monkeypatch):
    monkeypatch.chdir(context_dir.parent)
    monkeypatch.delenv("VERO_CONTEXT_PATH", raising=False)
    result = CliRunner().invoke(evals, ["list", "--json"], catch_exceptions=False)
    assert result.exit_code == 0
    assert len(json.loads(result.output)) == 3


def test_enrich_job_adds_elapsed_and_requested_cases():
    # terminal job: elapsed from created->completed, exactly.
    done = _enrich_job(
        {
            "job_id": "j",
            "status": "complete",
            "created_at": "2026-01-01T00:00:00Z",
            "completed_at": "2026-01-01T00:05:00Z",
        }
    )
    assert done["elapsed_seconds"] == 300.0
    # subset range run: requested_cases = stop - start.
    ranged = _enrich_job(
        {
            "job_id": "j",
            "status": "running",
            "created_at": "2026-01-01T00:00:00Z",
            "evaluation_set": {"selection": {"start": 0, "stop": 8}},
        }
    )
    assert ranged["requested_cases"] == 8
    assert ranged["elapsed_seconds"] >= 0
    # explicit case ids -> len; whole-partition run -> no requested_cases.
    assert _enrich_job({"evaluation_set": {"selection": {"ids": ["a", "b"]}}})[
        "requested_cases"
    ] == 2
    assert "requested_cases" not in _enrich_job(
        {"status": "running", "evaluation_set": {"name": "v"}}
    )


def test_status_job_output_is_enriched(monkeypatch):
    import vero.harbor.cli as harbor_cli

    monkeypatch.setattr(
        harbor_cli,
        "_request",
        lambda method, path, **kw: {
            "job_id": "j",
            "status": "running",
            "created_at": "2026-01-01T00:00:00Z",
            "evaluation_set": {"selection": {"start": 0, "stop": 10}},
        },
    )
    result = CliRunner().invoke(evals, ["status", "j"], catch_exceptions=False)
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["requested_cases"] == 10
    assert "elapsed_seconds" in payload


def test_wait_blocks_until_complete_then_prints_result(monkeypatch):
    import vero.harbor.cli as harbor_cli

    statuses = iter(["running", "running", "complete"])
    calls: list[str] = []

    def fake_request(method, path, **kw):
        calls.append(path)
        if path.endswith("/result"):
            return {"result": {"objective": {"value": 0.5}}}
        return {"job_id": "j", "status": next(statuses), "created_at": "2026-01-01T00:00:00Z"}

    monkeypatch.setattr(harbor_cli, "_request", fake_request)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    result = CliRunner().invoke(
        evals, ["wait", "j", "--poll-interval", "1"], catch_exceptions=False
    )
    assert result.exit_code == 0
    assert json.loads(result.output)["result"]["objective"]["value"] == 0.5
    assert calls[-1] == "/eval/jobs/j/result"  # result fetched only after terminal


def test_wait_timeout_returns_still_running_and_enriched(monkeypatch):
    import vero.harbor.cli as harbor_cli

    monkeypatch.setattr(
        harbor_cli,
        "_request",
        lambda method, path, **kw: {
            "job_id": "j",
            "status": "running",
            "created_at": "2026-01-01T00:00:00Z",
            "evaluation_set": {"selection": {"ids": ["a", "b", "c"]}},
        },
    )
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    result = CliRunner().invoke(
        evals, ["wait", "j", "--timeout", "0"], catch_exceptions=False
    )
    assert result.exit_code == 0  # clean exit, not Exit 143
    payload = json.loads(result.output)
    assert payload["status"] == "running"
    assert payload["requested_cases"] == 3
