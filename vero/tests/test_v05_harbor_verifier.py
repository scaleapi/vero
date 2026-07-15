from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from vero.candidate import Candidate
from vero.evaluation import (
    BackendProvenance,
    BackendRegistry,
    EvaluationDatabase,
    EvaluationRecord,
    EvaluationReport,
    EvaluationSet,
    EvaluationStatus,
    MetricSelector,
    ObjectiveResult,
    ObjectiveSpec,
)
from vero.harbor import (
    CanonicalVerifier,
    Submission,
    VerificationSelection,
    VerificationTarget,
)


OBJECTIVE = ObjectiveSpec(
    selector=MetricSelector(metric="score"),
    direction="maximize",
)


class StubBackend:
    @property
    def provenance(self):
        return BackendProvenance(
            name="stub",
            version="1",
            config_digest="0" * 64,
        )

    async def resolve_cost(self, evaluation_set):
        raise AssertionError("fake engine handles evaluation directly")

    async def evaluate(self, *, context, request):
        raise AssertionError("fake engine handles evaluation directly")


class FakeEngine:
    def __init__(self, scores):
        self.backends = BackendRegistry({"backend": StubBackend()})
        self.database = EvaluationDatabase(id="session")
        self.scores = scores
        self.calls = []
        self._sequence = 0

    async def evaluate_record(
        self,
        *,
        backend_id,
        request,
        objective_spec,
        authorization,
    ):
        self.calls.append((request.candidate.version, request.evaluation_set.name))
        key = (request.candidate.version, request.evaluation_set.name)
        value = self.scores[key]
        if isinstance(value, list):
            value = value.pop(0)
        if isinstance(value, Exception):
            raise value
        self._sequence += 1
        record = _record(
            f"admin-{self._sequence}",
            request.candidate,
            request.evaluation_set,
            float(value),
            objective_spec,
            backend_id=backend_id,
        )
        self.database.add_evaluation(record)
        assert authorization.meter_budget is False
        return record


def _candidate(name: str, *, content: str | None = None, seconds: int = 0):
    return Candidate(
        id=name,
        version=name,
        created_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds),
        metadata={"content_digest": content or name},
    )


def _record(
    record_id: str,
    candidate: Candidate,
    evaluation_set: EvaluationSet,
    score: float,
    objective: ObjectiveSpec = OBJECTIVE,
    *,
    backend_id: str = "backend",
):
    now = datetime(2026, 2, 1, tzinfo=UTC)
    from vero.evaluation import EvaluationRequest

    return EvaluationRecord(
        id=record_id,
        request=EvaluationRequest(
            candidate=candidate,
            evaluation_set=evaluation_set,
        ),
        report=EvaluationReport(
            status=EvaluationStatus.SUCCESS,
            metrics={"score": score},
        ),
        backend_id=backend_id,
        backend=StubBackend().provenance,
        objective_spec=objective,
        objective=ObjectiveResult(value=score, feasible=True),
        created_at=now,
        completed_at=now,
    )


def _verifier(
    tmp_path: Path,
    engine: FakeEngine,
    *,
    baseline: Candidate,
    top_k: int = 1,
    score_baseline: bool = True,
):
    return CanonicalVerifier(
        engine=engine,
        selection=VerificationSelection(
            mode="auto_best",
            backend_id="backend",
            evaluation_set=EvaluationSet(name="selection"),
            objective=OBJECTIVE,
            baseline_candidate=baseline,
            rescore_top_k=top_k,
            rescore_attempts=1,
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
        score_baseline=score_baseline,
    )


@pytest.mark.asyncio
async def test_verifier_pools_repeats_then_admin_rescores_and_scores_baseline(tmp_path):
    baseline = _candidate("baseline")
    farmed = _candidate("farmed", content="same-code", seconds=1)
    duplicate = _candidate("duplicate", content="same-code", seconds=2)
    steady = _candidate("steady", seconds=3)
    engine = FakeEngine(
        {
            ("steady", "selection"): 0.65,
            ("steady", "test"): 0.8,
            ("baseline", "selection"): 0.6,
            ("baseline", "test"): 0.5,
        }
    )
    for index, (candidate, score) in enumerate(
        [(farmed, 0.95), (duplicate, 0.05), (steady, 0.7)]
    ):
        engine.database.add_evaluation(
            _record(f"record-{index}", candidate, EvaluationSet(name="selection"), score)
        )

    result = await _verifier(tmp_path, engine, baseline=baseline).finalize()

    assert result.candidate == steady
    assert result.rewards == {"reward": 0.8}
    assert result.baseline_rewards == {"reward": 0.5}
    assert engine.calls == [
        ("steady", "selection"),
        ("baseline", "selection"),
        ("steady", "test"),
        ("baseline", "test"),
    ]


@pytest.mark.asyncio
async def test_verifier_baseline_floor_prevents_shipping_a_regression(tmp_path):
    baseline = _candidate("baseline")
    candidate = _candidate("candidate", seconds=1)
    engine = FakeEngine(
        {
            ("candidate", "selection"): 0.55,
            ("baseline", "selection"): 0.6,
            ("baseline", "test"): 0.5,
        }
    )
    engine.database.add_evaluation(
        _record("selection", candidate, EvaluationSet(name="selection"), 0.9)
    )

    result = await _verifier(tmp_path, engine, baseline=baseline).finalize()

    assert result.candidate == baseline
    assert result.rewards == {"reward": 0.5}
    assert result.baseline_rewards == result.rewards
    assert engine.calls.count(("baseline", "test")) == 1


@pytest.mark.asyncio
async def test_submit_finalization_is_durable_and_idempotent(tmp_path):
    candidate = _candidate("submitted")
    engine = FakeEngine({("submitted", "test"): 0.9})
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "submission.json").write_text(
        Submission(candidate=candidate).model_dump_json(),
        encoding="utf-8",
    )
    selection = VerificationSelection(mode="submit", baseline_floor=False)
    target = VerificationTarget(
        reward_key="reward",
        backend_id="backend",
        evaluation_set=EvaluationSet(name="test"),
        objective=OBJECTIVE,
        max_attempts=1,
    )

    first = await CanonicalVerifier(
        engine=engine,
        selection=selection,
        targets=[target],
        admin_volume=tmp_path,
        score_baseline=False,
    ).finalize()
    engine.scores[("submitted", "test")] = 0.1
    replayed = await CanonicalVerifier(
        engine=engine,
        selection=selection,
        targets=[target],
        admin_volume=tmp_path,
        score_baseline=False,
    ).finalize()

    assert first == replayed
    assert replayed.rewards == {"reward": 0.9}
    assert engine.calls == [("submitted", "test")]


@pytest.mark.asyncio
async def test_verifier_floors_rewards_when_no_candidate_exists(tmp_path):
    baseline = _candidate("baseline")
    engine = FakeEngine({})
    result = await _verifier(
        tmp_path,
        engine,
        baseline=baseline,
        score_baseline=False,
    ).finalize()

    assert result.candidate is None
    assert result.rewards == {"reward": 0.0}
    assert "selection" in result.errors
