from __future__ import annotations

import json
from pathlib import Path

from vero.runtime import RuntimeEvent, WandbEventSink


class FakeRun:
    def __init__(self):
        self.logged = []
        self.summary = {}
        self.finished = []

    def log(self, payload, *, step):
        self.logged.append((payload, step))

    def finish(self, *, exit_code=0):
        self.finished.append(exit_code)


class FakeWandb:
    def __init__(self):
        self.kwargs = None
        self.run = FakeRun()

    def init(self, **kwargs):
        self.kwargs = kwargs
        return self.run


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
