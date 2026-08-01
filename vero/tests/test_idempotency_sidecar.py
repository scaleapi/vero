"""Restart- and crash-survival regressions for the sidecar process.

Every test here stands for one way a run that had already paid for its work lost
it to a restart: an export that was named before its bytes were durable, an admin
token the outer agent could no longer use, an orphaned job whose evaluation could
not be found again, export scratch directories that filled the volume, and a
baseline measurement that existed only in one HTTP response.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from vero.candidate import Candidate
from vero.evaluation import (
    BackendProvenance,
    BackendRegistry,
    EvaluationCost,
    EvaluationDatabase,
    EvaluationRecord,
    EvaluationReport,
    EvaluationRequest,
    EvaluationSet,
    EvaluationStatus,
    MetricSelector,
    ObjectiveResult,
    ObjectiveSpec,
    RunningEvaluationManifest,
)
from vero.sidecar import (
    CanonicalVerifier,
    EvaluationJobStatus,
    EvaluationSidecar,
    HarborSessionManifest,
    SidecarEvaluationJob,
    VerificationResult,
    VerificationSelection,
    VerificationTarget,
)
from vero.sidecar.app import create_app
from vero.sidecar.auth import read_admin_token, write_admin_token
from vero.sidecar.serve import SidecarComponents, build_app
from vero.sidecar.session import (
    create_harbor_session_archive,
    extract_harbor_session_archive,
)

OBJECTIVE = ObjectiveSpec(
    selector=MetricSelector(metric="score"),
    direction="maximize",
)
PROVENANCE = BackendProvenance(name="stub", version="1", config_digest="0" * 64)
EVALUATION_SET = EvaluationSet(name="benchmark", partition="validation")


def _candidate(version: str) -> Candidate:
    return Candidate(
        id=version,
        version=version,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


class _StubBackend:
    @property
    def provenance(self) -> BackendProvenance:
        return PROVENANCE

    async def resolve_cost(self, evaluation_set):
        return EvaluationCost(cases=1)

    async def evaluate(self, *, context, request):
        raise AssertionError("the fake engine answers evaluations directly")


class _StubSidecar:
    """Only what the FastAPI transport touches on the export path."""

    def __init__(self, session_dir: Path):
        self.engine = SimpleNamespace(
            evaluator=SimpleNamespace(session_dir=session_dir),
            database=EvaluationDatabase(id="session"),
        )


class _StubVerifier:
    async def finalize(self) -> VerificationResult:
        return VerificationResult(rewards={"reward": 0.75})


# ITEM 3c: the session archive is the only durable copy of a finished run.


def _session_manifest() -> HarborSessionManifest:
    return HarborSessionManifest(
        id="trial",
        task_name="org/optimize",
        created_at=datetime(2026, 7, 16, tzinfo=UTC),
        backends={"validation": PROVENANCE},
        selection=VerificationSelection(mode="submit"),
        targets=[
            VerificationTarget(
                reward_key="reward",
                backend_id="validation",
                evaluation_set=EVALUATION_SET,
                objective=OBJECTIVE,
            )
        ],
    )


def test_session_archive_is_flushed_before_and_after_it_is_named(
    tmp_path,
    monkeypatch,
):
    # The export must not publish the archive's final name over bytes that are
    # still only in the page cache: fsync the archive, then rename, then fsync
    # the directory that now carries the name.
    session = tmp_path / "source"
    session.mkdir()
    (session / "harbor-session.json").write_text(
        _session_manifest().model_dump_json(indent=2) + "\n"
    )
    (session / "database.json").write_text('{"id":"trial"}\n')
    destination = tmp_path / "out" / "session.tar.gz"

    events: list[str] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def record_fsync(descriptor):
        kind = "directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "archive"
        events.append(f"fsync-{kind}")
        return real_fsync(descriptor)

    def record_replace(source, target):
        events.append("replace")
        return real_replace(source, target)

    monkeypatch.setattr(os, "fsync", record_fsync)
    monkeypatch.setattr(os, "replace", record_replace)

    create_harbor_session_archive(session, destination)

    assert events == ["fsync-archive", "replace", "fsync-directory"]
    # The archive itself must still be a readable session, not just durable.
    extracted = extract_harbor_session_archive(destination, tmp_path / "extracted")
    assert (extracted / "database.json").read_text() == '{"id":"trial"}\n'


# ITEM 38: a restart must not invalidate the token the outer agent already holds.


async def _stub_components(*, factory_path, config_path) -> SidecarComponents:
    return SidecarComponents(
        sidecar=_StubSidecar(Path(config_path).parent),
        verifier=_StubVerifier(),
    )


@pytest.mark.asyncio
async def test_build_app_reuses_an_admin_token_the_agent_already_holds(
    tmp_path,
    monkeypatch,
):
    # Minting a fresh token on every start meant a mid-run sidecar restart 401'd
    # the outer agent's next admin call even though every other piece of run state
    # had survived.
    monkeypatch.setattr("vero.sidecar.serve.build_components", _stub_components)
    token_path = tmp_path / "admin" / "token"
    write_admin_token(token_path, "token-the-agent-is-holding")

    app = await build_app(
        factory_path="module:attribute",
        config_path=tmp_path / "config.json",
        admin_token_path=token_path,
    )

    client = TestClient(app)
    response = client.post(
        "/finalize",
        headers={"Authorization": "Bearer token-the-agent-is-holding"},
    )
    assert response.status_code == 200
    assert response.json()["rewards"] == {"reward": 0.75}
    assert read_admin_token(token_path) == "token-the-agent-is-holding"


@pytest.mark.asyncio
async def test_build_app_still_mints_an_admin_token_on_a_first_start(
    tmp_path,
    monkeypatch,
):
    # Companion to the reuse test above: reuse must not break the case the volume
    # has no token yet, which is every run's first start.
    monkeypatch.setattr("vero.sidecar.serve.build_components", _stub_components)
    token_path = tmp_path / "admin" / "token"

    app = await build_app(
        factory_path="module:attribute",
        config_path=tmp_path / "config.json",
        admin_token_path=token_path,
    )

    minted = read_admin_token(token_path)
    assert minted
    client = TestClient(app)
    assert (
        client.post(
            "/finalize",
            headers={"Authorization": f"Bearer {minted}"},
        ).status_code
        == 200
    )


# ITEM 45: an orphaned job must keep pointing at the evaluation it was driving.


def _write_job(
    session_dir: Path,
    job_id: str,
    *,
    version: str | None,
    created_at: datetime,
    status: EvaluationJobStatus = EvaluationJobStatus.RUNNING,
) -> Path:
    job = SidecarEvaluationJob(
        job_id=job_id,
        status=status,
        backend_id="primary",
        evaluation_set=EVALUATION_SET,
        version=version,
        created_at=created_at,
    )
    path = session_dir / "evaluation-jobs" / f"{job_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(job.model_dump_json(indent=2) + "\n")
    return path


def _write_running_evaluation(
    session_dir: Path,
    evaluation_id: str,
    *,
    version: str,
    created_at: datetime,
) -> None:
    manifest = RunningEvaluationManifest(
        id=evaluation_id,
        request=EvaluationRequest(
            candidate=_candidate(version),
            evaluation_set=EVALUATION_SET,
        ),
        backend_id="primary",
        backend=PROVENANCE,
        created_at=created_at,
    )
    path = session_dir / "evaluations" / evaluation_id / "evaluation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2) + "\n")


def _restarted_sidecar(session_dir: Path) -> EvaluationSidecar:
    return EvaluationSidecar(
        engine=SimpleNamespace(
            evaluator=SimpleNamespace(
                session_dir=session_dir,
                evaluations_dir=session_dir / "evaluations",
            ),
            backends={},
            database=EvaluationDatabase(id="session"),
            budget_ledger=None,
        ),
        candidate_transport=None,
        access_policies=[],
    )


def test_interrupted_job_keeps_the_evaluation_it_was_driving(tmp_path):
    # Without the evaluation_id the interrupted job is a dead end: the budget its
    # evaluation reserved can never be reconciled against it, and the cases it had
    # already checkpointed belong to nothing.
    session = tmp_path / "session"
    started = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)
    job_path = _write_job(session, "job-1", version="candidate-1", created_at=started)
    _write_running_evaluation(
        session,
        "evaluation-1",
        version="candidate-1",
        created_at=started,
    )

    job = _restarted_sidecar(session).evaluation_job("job-1")

    assert job.status == EvaluationJobStatus.FAILED
    assert job.error == "evaluation job was interrupted by a sidecar restart"
    assert job.evaluation_id == "evaluation-1"
    assert json.loads(job_path.read_text())["evaluation_id"] == "evaluation-1"


def test_interrupted_job_declines_an_evaluation_it_may_not_own(tmp_path):
    # Two indistinguishable mid-flight evaluations, and one that started before the
    # job existed: none of them can be attributed to this job, and guessing would
    # send the reconciler after a reservation another job still owns.
    session = tmp_path / "session"
    started = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)
    _write_job(session, "job-1", version="candidate-1", created_at=started)
    _write_running_evaluation(
        session,
        "evaluation-1",
        version="candidate-1",
        created_at=started,
    )
    _write_running_evaluation(
        session,
        "evaluation-2",
        version="candidate-1",
        created_at=started,
    )
    _write_running_evaluation(
        session,
        "evaluation-earlier",
        version="candidate-1",
        created_at=datetime(2026, 7, 16, 11, 0, tzinfo=UTC),
    )

    job = _restarted_sidecar(session).evaluation_job("job-1")

    assert job.status == EvaluationJobStatus.FAILED
    assert job.evaluation_id is None


# ITEM 47c: export scratch directories must not accumulate for a whole run.


def test_session_export_sweeps_stale_scratch_directories(tmp_path, monkeypatch):
    # A crashed export never runs its cleanup background task, and the sidecar
    # lives for the whole run, so the leftovers fill the volume until every later
    # export fails with the session still unexported.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    stale = tmp_path / "vero-harbor-export-crashed"
    (stale / "leftovers").mkdir(parents=True)
    fresh = tmp_path / "vero-harbor-export-inflight"
    fresh.mkdir()
    unrelated = tmp_path / "vero-inference-usage"
    unrelated.mkdir()
    long_ago = time.time() - 7200
    for path in (stale, unrelated):
        os.utime(path, (long_ago, long_ago))

    def create_archive(_session_dir, destination):
        destination.write_bytes(b"portable-session")
        return destination

    monkeypatch.setattr(
        "vero.sidecar.app.create_harbor_session_archive",
        create_archive,
    )
    client = TestClient(
        create_app(
            sidecar=_StubSidecar(tmp_path / "session"),
            verifier=_StubVerifier(),
            admin_token="admin-secret",
        )
    )

    exported = client.get(
        "/session/export",
        headers={"Authorization": "Bearer admin-secret"},
    )

    assert exported.content == b"portable-session"
    assert not stale.exists()
    # An export that may still be streaming to a concurrent caller, and anything
    # outside the export prefix, are both left alone.
    assert fresh.is_dir()
    assert unrelated.is_dir()


# ITEM 36: a baseline measurement must survive a dropped response.


class _FakeEngine:
    """Answers admin evaluations from a fixed score table."""

    def __init__(self, scores: dict[tuple[str, str], list[float]]):
        self.backends = BackendRegistry({"backend": _StubBackend()})
        self.database = EvaluationDatabase(id="session")
        self.scores = scores
        self.calls: list[tuple[str, str]] = []
        self._sequence = 0

    async def evaluate_record(
        self,
        *,
        backend_id,
        request,
        objective_spec,
        authorization,
        principal,
    ) -> EvaluationRecord:
        key = (request.candidate.version, request.evaluation_set.name)
        self.calls.append(key)
        score = self.scores[key].pop(0)
        self._sequence += 1
        now = datetime(2026, 2, 1, tzinfo=UTC)
        record = EvaluationRecord(
            id=f"admin-{self._sequence}",
            request=request,
            report=EvaluationReport(
                status=EvaluationStatus.SUCCESS,
                metrics={"score": score},
            ),
            backend_id=backend_id,
            backend=PROVENANCE,
            objective_spec=objective_spec,
            objective=ObjectiveResult(value=score, feasible=True),
            created_at=now,
            completed_at=now,
        )
        self.database.add_evaluation(record)
        return record


@pytest.mark.asyncio
async def test_measure_baseline_persists_the_aggregate_it_returns(tmp_path):
    # Held-out replicates are the most expensive scoring a run does, and the HTTP
    # response used to be the only copy: a dropped connection meant paying for the
    # whole measurement again.
    baseline = _candidate("baseline")
    engine = _FakeEngine(
        {
            ("baseline", "selection"): [0.5, 0.6],
            ("baseline", "test"): [0.4, 0.5],
        }
    )
    verifier = CanonicalVerifier(
        engine=engine,
        selection=VerificationSelection(
            mode="auto_best",
            backend_id="backend",
            evaluation_set=EvaluationSet(name="selection"),
            objective=OBJECTIVE,
            baseline_candidate=baseline,
        ),
        targets=[
            VerificationTarget(
                reward_key="reward",
                backend_id="backend",
                evaluation_set=EvaluationSet(name="test"),
                objective=OBJECTIVE,
                max_attempts=1,
            )
        ],
        admin_volume=tmp_path,
    )

    result = await verifier.measure_baseline(replicates=2)

    stored = json.loads((tmp_path / "baseline.json").read_text())
    assert stored == result
    # The aggregate itself is untouched by being persisted.
    assert result["selection"]["mean"] == 0.55
    assert result["selection"]["n"] == 2
    assert result["targets"]["reward"]["mean"] == 0.45
    assert engine.calls == [
        ("baseline", "selection"),
        ("baseline", "selection"),
        ("baseline", "test"),
        ("baseline", "test"),
    ]
