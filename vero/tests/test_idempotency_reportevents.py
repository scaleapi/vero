"""Regressions for surviving a process death mid-write of the event log and report."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from vero.evaluation import (
    BackendProvenance,
    EvaluationPlan,
    EvaluationSet,
    MetricSelector,
    ObjectiveSpec,
)
from vero.report import build_experiment_report_data, generate_experiment_report
from vero.runtime import (
    JsonlEventSink,
    OptimizationComponentSpec,
    OptimizationRunSpec,
    RuntimeEvent,
    SessionManifest,
    SessionStatus,
)

SESSION_ID = "idempotency-reportevents"


def write_session(root: Path) -> Path:
    """Write the smallest session a report can be generated from.

    The candidate repository family is deliberately not ``git`` so these tests
    exercise the event log and the report write without paying for a real
    candidate repository; the events and the HTML output are what is under test.
    """
    created = datetime(2026, 1, 1, tzinfo=UTC)
    component = OptimizationComponentSpec(type="test", config_digest="0" * 64)
    manifest = SessionManifest(
        id=SESSION_ID,
        status=SessionStatus.COMPLETED,
        backend_id="test",
        backend=BackendProvenance.from_config(name="test", version="1", config={}),
        candidate_repository_family="memory",
        candidate_repository_format_version=1,
        evaluation_plan=EvaluationPlan.single(EvaluationSet(name="development")),
        objective=ObjectiveSpec(
            selector=MetricSelector(metric="score"), direction="maximize"
        ),
        run=OptimizationRunSpec(
            max_proposals=1,
            max_rounds=1,
            max_concurrency=1,
            strategy=component,
            producers={"test": component},
        ),
        created_at=created,
        updated_at=created,
    )
    session_dir = root / "session"
    session_dir.mkdir(parents=True)
    (session_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    return session_dir


def event_line(kind: str) -> str:
    return RuntimeEvent(
        session_id=SESSION_ID,
        kind=kind,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        payload={"note": "kept"},
    ).model_dump_json()


def test_report_survives_a_torn_last_event_line(tmp_path: Path):
    session_dir = write_session(tmp_path)
    intact = event_line("evaluation_completed")
    torn = event_line("verification_completed")
    # A SIGKILL mid-append leaves exactly this shape: complete records, then one
    # half-written record with no trailing newline.
    (session_dir / "events.jsonl").write_text(
        f"{intact}\n{torn[: len(torn) // 2]}", encoding="utf-8"
    )

    data = asyncio.run(build_experiment_report_data(session_dir))

    assert [event["kind"] for event in data["events"]] == ["evaluation_completed"]
    assert data["skipped_event_lines"] == 1


def test_report_survives_a_torn_multibyte_character_in_the_event_log(tmp_path: Path):
    session_dir = write_session(tmp_path)
    intact = event_line("evaluation_completed")
    # Events are written with ensure_ascii=False, so a death in the middle of a
    # multi-byte character leaves undecodable bytes rather than bad JSON.
    (session_dir / "events.jsonl").write_bytes(
        intact.encode("utf-8") + b"\n" + '{"payload": "café'.encode()[:-1]
    )

    data = asyncio.run(build_experiment_report_data(session_dir))

    assert [event["kind"] for event in data["events"]] == ["evaluation_completed"]
    assert data["skipped_event_lines"] == 1


def test_report_counts_no_skipped_lines_for_an_intact_event_log(tmp_path: Path):
    session_dir = write_session(tmp_path)
    (session_dir / "events.jsonl").write_text(
        f"{event_line('evaluation_completed')}\n", encoding="utf-8"
    )

    data = asyncio.run(build_experiment_report_data(session_dir))

    assert [event["kind"] for event in data["events"]] == ["evaluation_completed"]
    assert data["skipped_event_lines"] == 0


def test_experiment_html_write_that_dies_keeps_the_previous_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    session_dir = write_session(tmp_path)
    destination = asyncio.run(generate_experiment_report(session_dir))
    previous = destination.read_text(encoding="utf-8")

    def failing_replace(*arguments: object, **keywords: object) -> None:
        raise OSError("simulated death while publishing the report")

    monkeypatch.setattr(os, "replace", failing_replace)

    with pytest.raises(OSError):
        asyncio.run(generate_experiment_report(session_dir))

    monkeypatch.undo()
    assert destination.read_text(encoding="utf-8") == previous
    assert [path.name for path in session_dir.glob("*.tmp")] == []


def test_experiment_html_closes_its_descriptor_when_fdopen_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A failure between mkstemp and fdopen must close the raw descriptor once.

    Staging the report through a temporary file opens a window where the bare
    descriptor is nobody's responsibility but this function's. Closing it from a
    shared failure path instead would be a double close once fdopen has
    succeeded, and a second close can land on a descriptor the interpreter has
    since handed to something else.
    """

    session_dir = write_session(tmp_path)
    closed: list[int] = []
    real_close = os.close

    def fail_fdopen(*_arguments: object, **_keywords: object) -> None:
        raise RuntimeError("fdopen failed")

    def tracked_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(os, "fdopen", fail_fdopen)
    monkeypatch.setattr(os, "close", tracked_close)

    with pytest.raises(RuntimeError, match="fdopen failed"):
        asyncio.run(generate_experiment_report(session_dir))

    monkeypatch.undo()
    assert len(closed) == 1
    assert [path.name for path in session_dir.glob("*.tmp")] == []


class RecordingHandle:
    """Delegate to a real file handle while recording every write it receives."""

    def __init__(self, handle: Any, writes: list[Any]):
        self._handle = handle
        self._writes = writes

    def write(self, payload: Any) -> int:
        self._writes.append(payload)
        return self._handle.write(payload)

    def flush(self) -> None:
        self._handle.flush()

    def fileno(self) -> int:
        return self._handle.fileno()

    def __enter__(self) -> RecordingHandle:
        self._handle.__enter__()
        return self

    def __exit__(self, *arguments: object) -> None:
        self._handle.__exit__(*arguments)


def test_event_sink_appends_record_and_newline_in_one_fsynced_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "session" / "events.jsonl"
    sink = JsonlEventSink(path)
    modes: list[str] = []
    writes: list[Any] = []
    fsynced: list[int] = []
    real_open = Path.open
    real_fsync = os.fsync

    def recording_open(
        self: Path, mode: str = "r", *arguments: object, **keywords: object
    ) -> Any:
        handle = real_open(self, mode, *arguments, **keywords)  # type: ignore[arg-type]
        if self != path:
            return handle
        modes.append(mode)
        return RecordingHandle(handle, writes)

    def recording_fsync(descriptor: int) -> None:
        fsynced.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr(Path, "open", recording_open)
    monkeypatch.setattr(os, "fsync", recording_fsync)

    # A payload well past any buffer is the case that used to flush the record
    # without its newline and fuse it with the next record.
    event = RuntimeEvent(
        session_id=SESSION_ID, kind="agent", payload={"text": "x" * 200_000}
    )
    asyncio.run(sink(event))

    monkeypatch.undo()
    assert modes == ["ab"]
    expected = json.dumps(event.model_dump(mode="json"), ensure_ascii=False)
    assert writes == [f"{expected}\n".encode()]
    assert fsynced
    assert json.loads(path.read_text(encoding="utf-8"))["id"] == event.id
