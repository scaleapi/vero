"""Crash-safety of the W&B sinks' step counter and resume state.

The sinks own two pieces of durable bookkeeping: the monotonic W&B step and the
dedupe keys that say what has already been sent. Both are written to
``artifacts/wandb/state.json``, and a run that dies mid-flight (the common case,
runs take hours) resumes from exactly that file.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

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


class FakeRun:
    def __init__(self, *, log_raises: bool = False):
        self.logged: list[tuple[dict, int]] = []
        self.summary: dict = {}
        self.log_raises = log_raises

    def log(self, payload, *, step):
        if self.log_raises:
            # Stands in for the process dying between the state write and the
            # point actually reaching W&B, which is the window under test.
            raise RuntimeError("simulated crash while logging")
        self.logged.append((payload, step))

    def log_artifact(self, artifact):  # pragma: no cover - not exercised here
        pass

    def finish(self, *, exit_code=0):
        pass


class FakeWandb:
    def __init__(self, run: FakeRun | None = None):
        self.kwargs: dict | None = None
        self.run = run if run is not None else FakeRun()

    def init(self, **kwargs):
        self.kwargs = kwargs
        return self.run

    def Artifact(self, *, name, type):  # pragma: no cover - not exercised here
        raise AssertionError("no artifact should be built by these tests")


def _state(session_dir: Path) -> dict:
    return json.loads(
        (session_dir / "artifacts" / "wandb" / "state.json").read_text(encoding="utf-8")
    )


def _record(evaluation_id: str = "eval-1") -> EvaluationRecord:
    created = datetime(2026, 1, 1, tzinfo=UTC)
    return EvaluationRecord(
        id=evaluation_id,
        request=EvaluationRequest(
            candidate=Candidate(id="cand", version="v1", created_at=created),
            evaluation_set=EvaluationSet(name="benchmark", partition="validation"),
        ),
        report=EvaluationReport(
            status=EvaluationStatus.SUCCESS,
            metrics={"score": 0.5},
            cases=[
                CaseResult(
                    case_id="c1", status=CaseStatus.SUCCESS, metrics={"score": 0.5}
                )
            ],
        ),
        backend_id="primary",
        backend=BackendProvenance(name="harbor", version="1", config_digest="0" * 64),
        principal=EvaluationPrincipal.SYSTEM,
        objective_spec=ObjectiveSpec(
            selector=MetricSelector(metric="score"), direction="maximize"
        ),
        objective=ObjectiveResult(value=0.5, feasible=True),
        created_at=created,
        completed_at=created + timedelta(seconds=3),
    )


def _evaluation_event(evaluation_id: str) -> RuntimeEvent:
    return RuntimeEvent(
        session_id="session",
        kind="evaluation_completed",
        payload={"step": 0, "evaluation_id": evaluation_id, "objective/value": 0.5},
    )


def test_runtime_event_sink_persists_the_step_before_spending_it(tmp_path: Path):
    """A crash at the log boundary must not leave the step free for reuse."""
    session_dir = tmp_path / "session"
    crashing = FakeWandb(FakeRun(log_raises=True))
    sink = WandbEventSink(
        project="v", session_id="s", session_dir=session_dir, client=crashing
    )

    with pytest.raises(RuntimeError):
        sink(_evaluation_event("evaluation-1"))

    # The step was durable before it was spent, so the restart below cannot hand
    # W&B step 0 a second time.
    assert _state(session_dir) == {
        "evaluation_ids": ["evaluation-1"],
        "next_step": 1,
    }

    resumed_client = FakeWandb()
    resumed = WandbEventSink(
        project="v", session_id="s", session_dir=session_dir, client=resumed_client
    )
    resumed(_evaluation_event("evaluation-2"))
    assert [step for _, step in resumed_client.run.logged] == [1]


def test_sidecar_sink_persists_the_step_before_spending_it(tmp_path: Path):
    """Same window on the sidecar's evaluation stream."""
    session_dir = tmp_path / "session"
    crashing = FakeWandb(FakeRun(log_raises=True))
    sink = SidecarWandbSink(
        project="v", session_id="s", session_dir=session_dir, client=crashing
    )

    with pytest.raises(RuntimeError):
        sink(_record())

    state = _state(session_dir)
    assert state["next_step"] == 1
    assert state["evaluation_ids"] == ["eval-1"]

    resumed_client = FakeWandb()
    resumed = SidecarWandbSink(
        project="v", session_id="s", session_dir=session_dir, client=resumed_client
    )
    resumed(_record("eval-2"))
    assert [step for _, step in resumed_client.run.logged] == [1]


def test_sidecar_inference_usage_persists_the_step_before_spending_it(tmp_path: Path):
    """And on the gateway usage series the poller drives."""
    session_dir = tmp_path / "session"
    crashing = FakeWandb(FakeRun(log_raises=True))
    sink = SidecarWandbSink(
        project="v", session_id="s", session_dir=session_dir, client=crashing
    )

    with pytest.raises(RuntimeError):
        sink.log_inference_usage({"producer": {"requests": 3, "total_tokens": 15}})

    assert _state(session_dir)["next_step"] == 1

    resumed_client = FakeWandb()
    resumed = SidecarWandbSink(
        project="v", session_id="s", session_dir=session_dir, client=resumed_client
    )
    resumed(_record())
    assert [step for _, step in resumed_client.run.logged] == [1]


def test_sidecar_inference_usage_ledger_survives_a_restart(tmp_path: Path):
    """The usage dedupe key belongs in state.json, not only in memory.

    Held in memory only, a restarted sidecar had no idea what cumulative usage it
    had already reported and re-logged a byte-identical point at a fresh step.
    """
    session_dir = tmp_path / "session"
    first = FakeWandb()
    sink = SidecarWandbSink(
        project="v", session_id="s", session_dir=session_dir, client=first
    )
    scopes = {"producer": {"requests": 3, "total_tokens": 15}}
    sink.log_inference_usage(scopes)
    assert len(first.run.logged) == 1
    assert _state(session_dir)["inference_usage"] == {
        "inference/producer/requests": 3,
        "inference/producer/total_tokens": 15,
    }

    # A restart against the same session volume resumes the ledger: unchanged
    # gateway counters are recognized as already reported.
    restarted = FakeWandb()
    resumed = SidecarWandbSink(
        project="v", session_id="s", session_dir=session_dir, client=restarted
    )
    resumed.log_inference_usage(scopes)
    assert restarted.run.logged == []

    # Movement still logs, on the step the previous process left behind.
    resumed.log_inference_usage({"producer": {"requests": 4, "total_tokens": 20}})
    assert [step for _, step in restarted.run.logged] == [1]


class BlockingRun(FakeRun):
    """A run whose first ``log`` parks, so two callers really do overlap."""

    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def log(self, payload, *, step):
        if not self.entered.is_set():
            self.entered.set()
            assert self.release.wait(10)
        self.logged.append((payload, step))


def test_sidecar_sink_serializes_the_poller_against_the_evaluation_listener(
    tmp_path: Path,
):
    """The poller thread and the loop thread must not be inside the body at once.

    ``InferenceTelemetryPoller`` runs ``poll_once`` under ``asyncio.to_thread``
    while the engine calls the sink as a listener on the loop thread, so both
    reach the same ``next_step`` and the same state.json.
    """
    run = BlockingRun()
    client = FakeWandb(run)
    sink = SidecarWandbSink(
        project="v", session_id="s", session_dir=tmp_path / "session", client=client
    )

    listener = threading.Thread(target=sink, args=(_record(),))
    listener.start()
    assert run.entered.wait(10)

    poller_finished = threading.Event()

    def poll() -> None:
        sink.log_inference_usage({"producer": {"requests": 1}})
        poller_finished.set()

    poller = threading.Thread(target=poll)
    poller.start()
    try:
        # The listener is parked inside the guarded body, so the poller cannot
        # get in. Released in `finally` so a failure here does not leave the two
        # threads parked until interpreter exit.
        assert not poller_finished.wait(0.3)
    finally:
        run.release.set()
    listener.join(10)
    poller.join(10)
    assert poller_finished.is_set()
    assert [step for _, step in run.logged] == [0, 1]


class UploadingRun(FakeRun):
    """A run whose artifact upload parks, standing in for a slow transfer."""

    def __init__(self):
        super().__init__()
        self.uploading = threading.Event()
        self.release = threading.Event()
        self.uploaded: list[object] = []

    def log_artifact(self, artifact):
        self.uploading.set()
        assert self.release.wait(10)
        self.uploaded.append(artifact)


class UploadingWandb(FakeWandb):
    def Artifact(self, *, name, type):
        class _Artifact:
            def __init__(self) -> None:
                self.files: list[str] = []

            def add_file(self, path, *, name):
                self.files.append(name)

        return _Artifact()


def test_shipping_request_logs_does_not_block_the_evaluation_listener(tmp_path: Path):
    """The shared-state lock must not span the artifact upload.

    ``ship_request_logs`` runs on the telemetry poller's worker thread and
    ``__call__`` on the sidecar's event loop, and they share the same lock over
    ``next_step`` and state.json. Holding it across ``log_artifact``, which is a
    file transfer, would let a slow W&B upload stall the loop that answers the
    agent's requests.
    """
    run = UploadingRun()
    client = UploadingWandb(run)
    session_dir = tmp_path / "session"
    sink = SidecarWandbSink(
        project="v", session_id="s", session_dir=session_dir, client=client
    )
    log_dir = tmp_path / "requests"
    log_dir.mkdir()
    (log_dir / "requests-0.jsonl").write_text('{"model": "m"}\n', encoding="utf-8")

    shipper = threading.Thread(
        target=sink.ship_request_logs, args=(log_dir,), kwargs={"final": True}
    )
    shipper.start()
    try:
        assert run.uploading.wait(10)
        # The upload is parked and the loop thread's listener still gets through.
        sink(_record())
        assert [step for _, step in run.logged] == [0]
        # The snapshot is not claimed until the bytes are actually shipped, so a
        # crash right here re-ships next poll rather than losing the logs.
        assert _state(session_dir)["request_log_files"] == {}
    finally:
        run.release.set()
    shipper.join(10)

    assert run.uploaded, "the artifact was never uploaded"
    assert _state(session_dir)["request_log_files"] == {"requests-0.jsonl": 15}
