from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from vero.candidate import Candidate
from vero.evaluation import (
    BackendProvenance,
    CaseResult,
    CaseStatus,
    EvaluationPrincipal,
    EvaluationRecord,
    EvaluationReport,
    EvaluationRequest,
    EvaluationSet,
    EvaluationStatus,
    MetricSelector,
    ObjectiveResult,
    ObjectiveSpec,
)
from vero.runtime import RuntimeEvent, WandbEventSink
from vero.runtime.wandb import SidecarWandbSink


class FakeArtifact:
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.files = []

    def add_file(self, path, name):
        self.files.append((path, name))


class FakeRun:
    def __init__(self):
        self.logged = []
        self.summary = {}
        self.finished = []
        self.artifacts = []

    def log(self, payload, *, step):
        self.logged.append((payload, step))

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)

    def finish(self, *, exit_code=0):
        self.finished.append(exit_code)


class FakeWandb:
    def __init__(self):
        self.kwargs = None
        self.run = FakeRun()

    def init(self, **kwargs):
        self.kwargs = kwargs
        return self.run

    def Artifact(self, *, name, type):
        return FakeArtifact(name=name, type=type)


def test_wandb_sink_tracks_canonical_runtime_events(tmp_path: Path):
    client = FakeWandb()
    sink = WandbEventSink(
        project="vero-tests",
        session_id="session/with:unsafe-run-id-characters",
        session_dir=tmp_path / "session",
        mode="offline",
        config={"vero/objective_metric": "latency_ms"},
        client=client,
    )

    assert client.kwargs["project"] == "vero-tests"
    assert client.kwargs["id"].startswith("vero-")
    assert client.kwargs["resume"] == "allow"
    assert client.kwargs["mode"] == "offline"
    assert client.kwargs["config"]["vero/session_id"].startswith("session/")

    sink(
        RuntimeEvent(
            session_id="session",
            kind="evaluation_completed",
            payload={
                "step": 2,
                "candidate_id": "candidate",
                "evaluation_id": "evaluation-1",
                "metrics/latency_ms": 1.25,
                "objective/value": 1.25,
                "objective/feasible": True,
            },
        )
    )
    assert client.run.logged == [
        (
            {
                "candidate_id": "candidate",
                "evaluation_id": "evaluation-1",
                "metrics/latency_ms": 1.25,
                "objective/value": 1.25,
                "objective/feasible": True,
            },
            0,
        )
    ]
    assert json.loads(
        (tmp_path / "session" / "artifacts" / "wandb" / "state.json").read_text()
    ) == {"evaluation_ids": ["evaluation-1"], "next_step": 1}

    # Replayed history on resume is not sent twice or logged with a stale step.
    sink(
        RuntimeEvent(
            session_id="session",
            kind="evaluation_completed",
            payload={"step": 0, "evaluation_id": "evaluation-1"},
        )
    )
    assert len(client.run.logged) == 1

    sink(
        RuntimeEvent(
            session_id="session",
            kind="session_completed",
            payload={"status": "completed", "best_objective": 1.25},
        )
    )
    assert client.run.summary["best_objective"] == 1.25
    assert client.run.finished == [0]


def _evaluation_record(
    *, status=EvaluationStatus.SUCCESS, score=0.75
) -> EvaluationRecord:
    created = datetime(2026, 1, 1, tzinfo=UTC)
    return EvaluationRecord(
        id="eval-1",
        request=EvaluationRequest(
            candidate=Candidate(id="cand", version="v1", created_at=created),
            evaluation_set=EvaluationSet(name="benchmark", partition="validation"),
        ),
        report=EvaluationReport(
            status=status,
            metrics={"score": score, "score_stddev": 0.1, "error_rate": 0.0},
            cases=[
                CaseResult(
                    case_id="c1", status=CaseStatus.SUCCESS, metrics={"score": score}
                )
            ],
        ),
        backend_id="primary",
        backend=BackendProvenance(name="harbor", version="1", config_digest="0" * 64),
        principal=EvaluationPrincipal.SYSTEM,
        objective_spec=ObjectiveSpec(
            selector=MetricSelector(metric="score"), direction="maximize"
        ),
        objective=ObjectiveResult(value=score, feasible=True),
        created_at=created,
        completed_at=created + timedelta(seconds=3),
    )


def test_sidecar_wandb_sink_logs_evaluation_records(tmp_path: Path):
    client = FakeWandb()
    sink = SidecarWandbSink(
        project="vero",
        session_id="s",
        session_dir=tmp_path / "session",
        client=client,
    )

    sink(_evaluation_record())

    payload, step = client.run.logged[0]
    assert step == 0
    assert payload["evaluation_id"] == "eval-1"
    assert payload["status"] == "success"
    assert payload["principal"] == "system"
    assert payload["partition"] == "validation"
    # Quality metrics are scoped by partition/principal so dev, validation, and
    # the admin held-out re-score do not share an axis.
    assert payload["validation/system/score"] == 0.75
    assert payload["validation/system/metric/score_stddev"] == 0.1
    assert payload["validation/system/cases/success"] == 1
    # num_cases records the evaluation's sample size (number of trials).
    assert payload["validation/system/num_cases"] == 1
    # The bare, partition-blind keys are gone — nothing conflates partitions.
    assert "score" not in payload
    assert "cases/success" not in payload

    # Same record is not logged twice.
    sink(_evaluation_record())
    assert len(client.run.logged) == 1


def test_sidecar_wandb_run_id_is_unique_per_invocation_but_stable_on_restart(
    tmp_path: Path,
):
    # Distinct session volumes must get distinct W&B runs (not all resume one
    # shared run), while a restart against the same volume resumes its run.
    a = FakeWandb()
    SidecarWandbSink(project="v", session_id="trial", session_dir=tmp_path / "a", client=a)
    b = FakeWandb()
    SidecarWandbSink(project="v", session_id="trial", session_dir=tmp_path / "b", client=b)
    assert a.kwargs["id"] != b.kwargs["id"]

    # Same session dir (persisted state) -> same run id on restart.
    a2 = FakeWandb()
    SidecarWandbSink(project="v", session_id="trial", session_dir=tmp_path / "a", client=a2)
    assert a2.kwargs["id"] == a.kwargs["id"]


def test_sidecar_wandb_run_dir_is_outside_the_archived_session_dir(tmp_path: Path):
    # Regression: W&B's working directory accumulates symlinks (`latest-run`,
    # debug logs, an absolute cache link). If it lived under session_dir it would
    # be swept into create_harbor_session_archive, whose symlink refusal 500s the
    # entire /session/export and loses the run's eval data. The run dir must be
    # outside session_dir; only symlink-free resume state may live in artifacts.
    client = FakeWandb()
    session_dir = tmp_path / "session"
    SidecarWandbSink(
        project="vero",
        session_id="s",
        session_dir=session_dir,
        client=client,
    )

    wandb_dir = Path(client.kwargs["dir"]).resolve()
    resolved_session = session_dir.resolve()
    assert not wandb_dir.is_relative_to(resolved_session)


def test_sidecar_wandb_sink_logs_full_trace_including_per_case_artifacts(
    tmp_path: Path,
):
    from vero.evaluation import EvaluationArtifact

    client = FakeWandb()
    session_dir = tmp_path / "session"
    sink = SidecarWandbSink(
        project="vero",
        session_id="s",
        session_dir=session_dir,
        log_traces=True,
        client=client,
    )

    # Lay out an evaluation's on-disk artifacts: report-level harbor logs plus
    # the full per-trial record attached to the case.
    eval_dir = session_dir / "evaluations" / "eval-1" / "artifacts"
    (eval_dir / "harbor").mkdir(parents=True)
    (eval_dir / "harbor" / "stdout.log").write_text("out", encoding="utf-8")
    trial = eval_dir / "harbor" / "jobs" / "trial-0" / "agent"
    trial.mkdir(parents=True)
    (trial.parent / "result.json").write_text("{}", encoding="utf-8")
    (trial / "trajectory.json").write_text("[]", encoding="utf-8")

    record = _evaluation_record()
    record = record.model_copy(
        update={
            "report": record.report.model_copy(
                update={
                    "artifacts": [
                        EvaluationArtifact(path="harbor/stdout.log"),
                    ],
                    "cases": [
                        record.report.cases[0].model_copy(
                            update={
                                "artifacts": [
                                    EvaluationArtifact(
                                        path="harbor/jobs/trial-0/result.json"
                                    ),
                                    EvaluationArtifact(
                                        path="harbor/jobs/trial-0/agent/trajectory.json"
                                    ),
                                ]
                            }
                        )
                    ],
                }
            )
        }
    )

    sink(record)

    assert len(client.run.artifacts) == 1
    artifact = client.run.artifacts[0]
    assert artifact.type == "evaluation_trace"
    uploaded = {name for _, name in artifact.files}
    # Both the report-level harbor log AND the full per-case trial records land.
    assert uploaded == {
        "harbor/stdout.log",
        "harbor/jobs/trial-0/result.json",
        "harbor/jobs/trial-0/agent/trajectory.json",
    }


def _gateway_state(tmp_path: Path, requests: int = 3) -> tuple[Path, Path]:
    usage_path = tmp_path / "inference" / "usage.json"
    requests_dir = tmp_path / "inference" / "requests"
    requests_dir.mkdir(parents=True, exist_ok=True)
    usage_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scopes": {
                    "producer": {
                        "requests": requests,
                        "upstream_errors": 0,
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                        "active_requests": 1,
                        "attributions": {},
                    }
                },
            }
        )
    )
    return usage_path, requests_dir


def test_sidecar_wandb_mirrors_gateway_usage_and_request_logs(tmp_path: Path):
    from vero.runtime.wandb import InferenceTelemetryPoller

    client = FakeWandb()
    sink = SidecarWandbSink(
        project="v", session_id="s", session_dir=tmp_path / "session", client=client
    )
    usage_path, requests_dir = _gateway_state(tmp_path)
    (requests_dir / "requests-00001.jsonl").write_text('{"a":1}\n')
    (requests_dir / "requests-00002.jsonl").write_text('{"a":2}\n')

    poller = InferenceTelemetryPoller(
        sink=sink,
        usage_path=usage_path,
        request_log_dir=requests_dir,
        interval_seconds=5,
    )
    poller.poll_once()

    payload, step = client.run.logged[0]
    assert step == 0
    assert payload["inference/producer/requests"] == 3
    assert payload["inference/producer/total_tokens"] == 15
    assert payload["inference/producer/active_requests"] == 1
    # Only the rotated file ships; the highest-numbered one is still growing.
    assert len(client.run.artifacts) == 1
    assert client.run.artifacts[0].type == "inference_request_log"
    assert [name for _, name in client.run.artifacts[0].files] == [
        "requests-00001.jsonl"
    ]

    # Unchanged gateway state: no duplicate series point, no new artifact.
    poller.poll_once()
    assert len(client.run.logged) == 1
    assert len(client.run.artifacts) == 1

    # Usage moved -> a new point; eval records share the same step counter.
    _gateway_state(tmp_path, requests=4)
    poller.poll_once()
    assert client.run.logged[1][1] == 1
    sink(_evaluation_record())
    assert client.run.logged[2][1] == 2

    # The final flush ships the active file too.
    poller.poll_once(final=True)
    assert [name for _, name in client.run.artifacts[-1].files] == [
        "requests-00001.jsonl",
        "requests-00002.jsonl",
    ]

    # A restarted sink remembers what shipped and does not re-upload.
    restarted = FakeWandb()
    resumed = SidecarWandbSink(
        project="v", session_id="s", session_dir=tmp_path / "session", client=restarted
    )
    resumed.ship_request_logs(requests_dir, final=True)
    assert restarted.run.artifacts == []


def test_sidecar_wandb_telemetry_is_best_effort(tmp_path: Path):
    from vero.runtime.wandb import InferenceTelemetryPoller

    client = FakeWandb()
    sink = SidecarWandbSink(
        project="v", session_id="s", session_dir=tmp_path / "session", client=client
    )
    usage_path = tmp_path / "inference" / "usage.json"
    usage_path.parent.mkdir(parents=True)
    usage_path.write_text("{corrupt")

    poller = InferenceTelemetryPoller(
        sink=sink,
        usage_path=usage_path,
        request_log_dir=tmp_path / "inference" / "missing",
        interval_seconds=5,
    )
    poller.poll_once()  # must not raise
    assert client.run.logged == []
    assert client.run.artifacts == []


def test_scheme_less_wandb_base_url_is_repaired_before_init(tmp_path: Path, monkeypatch):
    """A self-hosted host written without a scheme must not silently kill reporting.

    W&B parses `base_url` as a URL, so `WANDB_BASE_URL=wandb.example.com` raises
    out of `wandb.init()` and the sidecar disables W&B for the whole run.
    """
    from vero.runtime.wandb import normalize_wandb_base_url

    monkeypatch.setenv("WANDB_BASE_URL", "wandb.example.com")
    assert normalize_wandb_base_url() == "https://wandb.example.com"
    assert os.environ["WANDB_BASE_URL"] == "https://wandb.example.com"

    # Already-qualified values, including plain http, are left exactly as given.
    monkeypatch.setenv("WANDB_BASE_URL", "http://localhost:8080")
    assert normalize_wandb_base_url() == "http://localhost:8080"
    assert os.environ["WANDB_BASE_URL"] == "http://localhost:8080"

    monkeypatch.delenv("WANDB_BASE_URL", raising=False)
    assert normalize_wandb_base_url() is None

    # And the repair happens before a sink opens its run.
    monkeypatch.setenv("WANDB_BASE_URL", "wandb.example.com")
    SidecarWandbSink(
        project="v", session_id="s", session_dir=tmp_path / "session", client=FakeWandb()
    )
    assert os.environ["WANDB_BASE_URL"] == "https://wandb.example.com"
