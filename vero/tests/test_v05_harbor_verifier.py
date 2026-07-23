from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from vero.candidate import Candidate
from vero.evaluation import (
    BackendProvenance,
    BackendRegistry,
    CaseResult,
    CaseStatus,
    EvaluationDatabase,
    EvaluationRecord,
    EvaluationReport,
    EvaluationRequest,
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
        self.drain_calls = []
        self.on_drain = None
        self._sequence = 0

    async def quiesce_agent_evaluations(self, *, timeout_seconds):
        self.drain_calls.append(timeout_seconds)
        if self.on_drain is not None:
            self.on_drain()
        return 0

    async def evaluate_record(
        self,
        *,
        backend_id,
        request,
        objective_spec,
        authorization,
        principal,
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
        assert principal.value == "admin"
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
    baseline_floor: bool = False,
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
            baseline_floor=baseline_floor,
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
            _record(
                f"record-{index}", candidate, EvaluationSet(name="selection"), score
            )
        )

    result = await _verifier(
        tmp_path, engine, baseline=baseline, baseline_floor=True
    ).finalize()

    assert result.candidate == steady
    assert result.rewards == {"reward": 0.8}
    assert result.baseline_rewards == {"reward": 0.5}
    assert engine.calls == [
        ("steady", "selection"),
        ("baseline", "selection"),
        ("steady", "test"),
        ("baseline", "test"),
    ]
    assert engine.drain_calls == [600.0]


@pytest.mark.asyncio
async def test_verifier_waits_for_an_inflight_selection_evaluation(tmp_path):
    baseline = _candidate("baseline")
    candidate = _candidate("candidate", seconds=1)
    engine = FakeEngine(
        {
            ("candidate", "selection"): 0.8,
            ("baseline", "selection"): 0.5,
            ("candidate", "test"): 0.9,
        }
    )

    def complete_inflight():
        engine.database.add_evaluation(
            _record(
                "agent-selection",
                candidate,
                EvaluationSet(name="selection"),
                0.75,
            )
        )

    engine.on_drain = complete_inflight

    result = await _verifier(
        tmp_path,
        engine,
        baseline=baseline,
        score_baseline=False,
    ).finalize()

    assert result.candidate == candidate
    assert result.rewards == {"reward": 0.9}
    assert engine.drain_calls == [600.0]


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

    result = await _verifier(
        tmp_path, engine, baseline=baseline, baseline_floor=True
    ).finalize()

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
    assert replayed.shipped is True
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
    # "Nothing shipped" is an explicit, distinct outcome — not a real zero.
    assert result.shipped is False
    assert result.rewards == {"reward": 0.0}
    assert "selection" in result.errors


@pytest.mark.asyncio
async def test_verifier_transforms_minimization_objective_into_higher_reward(tmp_path):
    candidate = _candidate("submitted")
    minimize = ObjectiveSpec(
        selector=MetricSelector(metric="score"),
        direction="minimize",
    )
    engine = FakeEngine({("submitted", "latency"): 2.5})
    (tmp_path / "submission.json").write_text(
        Submission(candidate=candidate).model_dump_json(),
        encoding="utf-8",
    )
    verifier = CanonicalVerifier(
        engine=engine,
        selection=VerificationSelection(mode="submit", baseline_floor=False),
        targets=[
            VerificationTarget(
                reward_key="latency_reward",
                backend_id="backend",
                evaluation_set=EvaluationSet(name="latency"),
                objective=minimize,
                reward_scale=-1.0,
                max_attempts=1,
            )
        ],
        admin_volume=tmp_path,
        score_baseline=False,
    )

    result = await verifier.finalize()

    assert result.rewards == {"latency_reward": -2.5}


def _record_with_cases(record_id, candidate, evaluation_set, score, case_ids):
    now = datetime(2026, 2, 1, tzinfo=UTC)
    return EvaluationRecord(
        id=record_id,
        request=EvaluationRequest(candidate=candidate, evaluation_set=evaluation_set),
        report=EvaluationReport(
            status=EvaluationStatus.SUCCESS,
            metrics={"score": score},
            cases=[
                CaseResult(
                    case_id=cid, status=CaseStatus.SUCCESS, metrics={"score": score}
                )
                for cid in case_ids
            ],
        ),
        backend_id="backend",
        backend=StubBackend().provenance,
        objective_spec=OBJECTIVE,
        objective=ObjectiveResult(value=score, feasible=True),
        created_at=now,
        completed_at=now,
    )


@pytest.mark.asyncio
async def test_verifier_prefers_agent_submission_over_auto_best(tmp_path):
    baseline = _candidate("baseline")
    picked = _candidate("picked", seconds=5)
    engine = FakeEngine({("picked", "test"): 0.9, ("baseline", "test"): 0.5})
    engine.database.add_evaluation(
        _record("r", _candidate("other", seconds=1), EvaluationSet(name="selection"), 0.7)
    )
    (tmp_path / "submission.json").write_text(
        Submission(candidate=picked).model_dump_json()
    )

    result = await _verifier(tmp_path, engine, baseline=baseline).finalize()

    assert result.candidate == picked
    assert result.shipped is True
    assert result.rewards == {"reward": 0.9}
    # The submitted candidate wins outright; auto_best never re-scored 'other'.
    assert ("other", "selection") not in engine.calls


@pytest.mark.asyncio
async def test_verifier_falls_back_to_last_candidate_when_no_selection_eval(tmp_path):
    baseline = _candidate("baseline")
    last = _candidate("last", seconds=9)
    engine = FakeEngine({("last", "test"): 0.42, ("baseline", "test"): 0.5})
    # Only a non-selection-partition eval exists -> no qualifying selection records.
    engine.database.add_evaluation(
        _record("r", last, EvaluationSet(name="development"), 0.7)
    )

    result = await _verifier(tmp_path, engine, baseline=baseline).finalize()

    # pick-last ships the most recent candidate instead of shipping nothing.
    assert result.candidate == last
    assert result.shipped is True
    assert result.rewards == {"reward": 0.42}


@pytest.mark.asyncio
async def test_verifier_coverage_threshold_excludes_undermeasured_candidates(tmp_path):
    baseline = _candidate("baseline")
    well = _candidate("well", seconds=1)
    thin = _candidate("thin", seconds=2)
    engine = FakeEngine(
        {
            ("well", "selection"): 0.6,
            ("well", "test"): 0.7,
            ("baseline", "selection"): 0.5,
            ("baseline", "test"): 0.4,
        }
    )
    engine.database.add_evaluation(
        _record_with_cases(
            "rw", well, EvaluationSet(name="selection"), 0.6, [f"c{i}" for i in range(10)]
        )
    )
    # Higher score but only 2/10 coverage -> below the 0.9 threshold -> excluded.
    engine.database.add_evaluation(
        _record_with_cases("rt", thin, EvaluationSet(name="selection"), 0.99, ["c0", "c1"])
    )

    result = await _verifier(tmp_path, engine, baseline=baseline, top_k=5).finalize()

    assert result.candidate == well
    assert ("thin", "selection") not in engine.calls
    assert ("well", "selection") in engine.calls


@pytest.mark.asyncio
async def test_verifier_uses_pinned_baseline_reward_without_scoring(tmp_path):
    # A pinned target baseline_reward is used verbatim; the seed is never scored.
    baseline = _candidate("baseline")
    cand = _candidate("cand", seconds=1)
    engine = FakeEngine({("cand", "selection"): 0.8, ("cand", "test"): 0.7})
    engine.database.add_evaluation(
        _record("r", cand, EvaluationSet(name="selection"), 0.8)
    )
    verifier = CanonicalVerifier(
        engine=engine,
        selection=VerificationSelection(
            mode="auto_best",
            backend_id="backend",
            evaluation_set=EvaluationSet(name="selection"),
            objective=OBJECTIVE,
            baseline_candidate=baseline,
            rescore_top_k=1,
            rescore_attempts=1,
        ),
        targets=[
            VerificationTarget(
                reward_key="reward",
                backend_id="backend",
                evaluation_set=EvaluationSet(name="test"),
                objective=OBJECTIVE,
                max_attempts=1,
                baseline_reward=0.55,
            )
        ],
        admin_volume=tmp_path,
        score_baseline=True,
    )

    result = await verifier.finalize()

    assert result.candidate == cand
    assert result.rewards == {"reward": 0.7}
    assert result.baseline_rewards == {"reward": 0.55}
    assert ("baseline", "test") not in engine.calls  # seed never scored


@pytest.mark.asyncio
async def test_verifier_uses_pinned_baseline_selection_score(tmp_path):
    # With a pinned selection score, the floor compares without re-scoring the seed.
    baseline = _candidate("baseline")
    cand = _candidate("cand", seconds=1)
    engine = FakeEngine({("cand", "selection"): 0.4, ("baseline", "test"): 0.3})
    engine.database.add_evaluation(
        _record("r", cand, EvaluationSet(name="selection"), 0.4)
    )
    verifier = CanonicalVerifier(
        engine=engine,
        selection=VerificationSelection(
            mode="auto_best",
            backend_id="backend",
            evaluation_set=EvaluationSet(name="selection"),
            objective=OBJECTIVE,
            baseline_candidate=baseline,
            rescore_top_k=1,
            rescore_attempts=1,
            baseline_floor=True,
            baseline_selection_score=0.6,
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
        score_baseline=False,
    )

    result = await verifier.finalize()

    # best (0.4) does not beat the pinned seed (0.6) → floor keeps the seed.
    assert result.candidate == baseline
    assert ("baseline", "selection") not in engine.calls  # seed never re-scored


@pytest.mark.asyncio
async def test_verifier_floor_fails_safe_when_seed_unmeasurable(tmp_path):
    # Floor on, unpinned, seed re-score fails → ship nothing (inconclusive),
    # not the best candidate unverified.
    baseline = _candidate("baseline")
    cand = _candidate("cand", seconds=1)
    engine = FakeEngine(
        {("cand", "selection"): 0.9, ("baseline", "selection"): RuntimeError("outage")}
    )
    engine.database.add_evaluation(
        _record("r", cand, EvaluationSet(name="selection"), 0.9)
    )

    result = await _verifier(
        tmp_path, engine, baseline=baseline, baseline_floor=True
    ).finalize()

    assert result.shipped is False
    assert "baseline floor" in result.errors.get("selection", "")


@pytest.mark.asyncio
async def test_verifier_score_baseline_produces_replicated_means(tmp_path):
    baseline = _candidate("baseline")
    engine = FakeEngine(
        {
            ("baseline", "selection"): [0.5, 0.6],
            ("baseline", "test"): [0.4, 0.5],
        }
    )

    out = await _verifier(tmp_path, engine, baseline=baseline).measure_baseline(
        replicates=2
    )

    assert out["candidate_version"] == "baseline"
    assert out["replicates"] == 2
    assert out["selection"]["n"] == 2
    assert out["selection"]["mean"] == 0.55
    assert out["targets"]["reward"]["n"] == 2
    assert out["targets"]["reward"]["mean"] == 0.45
    # 2 selection re-scores + 2 target scores, all admin.
    assert engine.calls == [
        ("baseline", "selection"),
        ("baseline", "selection"),
        ("baseline", "test"),
        ("baseline", "test"),
    ]
