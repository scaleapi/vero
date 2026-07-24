"""Admin-side candidate selection and final evaluation for Harbor tasks."""

from __future__ import annotations

import asyncio
import logging
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Callable, Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from vero.candidate import Candidate
from vero.evaluation import (
    EvaluationAuthorization,
    EvaluationLimits,
    EvaluationModel,
    EvaluationPrincipal,
    EvaluationRecord,
    EvaluationRequest,
    EvaluationSet,
    ObjectiveSpec,
)
from vero.evaluation.engine import EvaluationEngine
from vero.evaluation.persistence import _atomic_write_json
from vero.harbor.sidecar import Submission

logger = logging.getLogger(__name__)


class NoCandidateError(RuntimeError):
    """Raised when finalization has no submitted or evaluated candidate."""


class VerificationTarget(EvaluationModel):
    """One trusted final evaluation projected to one Harbor reward key."""

    reward_key: str
    backend_id: str
    evaluation_set: EvaluationSet
    objective: ObjectiveSpec
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    limits: EvaluationLimits = Field(default_factory=EvaluationLimits)
    failure_value: float = 0.0
    reward_scale: float = 1.0
    reward_offset: float = 0.0
    # Pin the seed's reward on this target (post scale/offset) to skip scoring
    # the fixed seed each run — reproducibility + one fewer eval per run.
    baseline_reward: float | None = None
    # Keep one attempt by default for adversarial candidates. Retrying an
    # unmeasurable candidate-controlled run creates a one-sided re-roll.
    max_attempts: int = Field(default=1, ge=1)

    @field_validator("reward_key", "backend_id")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("verification target identity must not be empty")
        return value

    @field_validator("failure_value", "reward_scale", "reward_offset")
    @classmethod
    def validate_failure_value(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("verification reward values must be finite")
        return value

    @field_validator("baseline_reward")
    @classmethod
    def validate_baseline_reward(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("baseline_reward must be finite")
        return value


class VerificationSelection(EvaluationModel):
    """How finalization chooses a candidate before scoring its targets."""

    mode: Literal["submit", "auto_best"] = "auto_best"
    backend_id: str | None = None
    evaluation_set: EvaluationSet | None = None
    objective: ObjectiveSpec | None = None
    baseline_candidate: Candidate | None = None
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    limits: EvaluationLimits = Field(default_factory=EvaluationLimits)
    rescore_top_k: int = Field(default=3, ge=1)
    rescore_attempts: int = Field(default=1, ge=1)
    # Off by default: the floor gates a ship on a validation comparison while
    # the reward is on the (possibly differently-distributed) target.
    baseline_floor: bool = False
    # Pin the seed's selection-partition score to skip re-scoring it each run.
    baseline_selection_score: float | None = None
    selection_coverage_threshold: float = Field(default=0.9, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_auto_best(self) -> VerificationSelection:
        fields = (self.backend_id, self.evaluation_set, self.objective)
        if self.mode == "auto_best" and any(value is None for value in fields):
            raise ValueError(
                "auto_best selection requires backend_id, evaluation_set, and objective"
            )
        if self.backend_id is not None and not self.backend_id.strip():
            raise ValueError("selection backend_id must not be empty")
        if (
            self.baseline_floor
            and self.mode == "auto_best"
            and self.baseline_candidate is None
        ):
            raise ValueError("baseline_floor requires baseline_candidate")
        return self


class VerificationResult(EvaluationModel):
    """Durable, idempotent output consumed by Harbor's verifier."""

    candidate: Candidate | None = None
    #: Whether a candidate was actually shipped. False means selection produced
    #: nothing (the rewards are failure values recording "no candidate"), which
    #: a benchmark must treat as a distinct outcome from a candidate that
    #: shipped and legitimately scored the failure value.
    shipped: bool = True
    rewards: dict[str, float]
    evaluation_ids: dict[str, str] = Field(default_factory=dict)
    baseline_rewards: dict[str, float] = Field(default_factory=dict)
    #: Per reward key: the scoring evaluation's full report metrics (accuracy
    #: plus cost/latency telemetry such as inference_*_tokens and
    #: mean_case_wall_seconds). Informational — the reward itself remains the
    #: objective value alone.
    reward_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    baseline_reward_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    errors: dict[str, str] = Field(default_factory=dict)

    @field_validator("rewards", "baseline_rewards")
    @classmethod
    def validate_rewards(cls, value: dict[str, float]) -> dict[str, float]:
        for key, reward in value.items():
            if not key.strip() or not math.isfinite(reward):
                raise ValueError("verification rewards require names and finite values")
        return value


class CanonicalVerifier:
    """Select, re-measure, and score candidates through one evaluation engine."""

    def __init__(
        self,
        *,
        engine: EvaluationEngine,
        selection: VerificationSelection,
        targets: list[VerificationTarget],
        admin_volume: Path,
        score_baseline: bool = True,
        evaluation_drain_timeout_seconds: float = 600.0,
        on_finalized: Callable[[VerificationResult], None] | None = None,
    ):
        if not targets:
            raise ValueError("at least one verification target is required")
        reward_keys = [target.reward_key for target in targets]
        if len(reward_keys) != len(set(reward_keys)):
            raise ValueError("verification reward keys must be unique")
        for target in targets:
            if target.backend_id not in engine.backends:
                raise ValueError(
                    f"verification target references unknown backend {target.backend_id!r}"
                )
        if (
            selection.backend_id is not None
            and selection.backend_id not in engine.backends
        ):
            raise ValueError(
                f"selection references unknown backend {selection.backend_id!r}"
            )
        self.engine = engine
        self.selection = selection
        self.targets = targets
        self.admin_volume = Path(admin_volume)
        self.score_baseline = score_baseline
        if evaluation_drain_timeout_seconds <= 0:
            raise ValueError("evaluation drain timeout must be positive")
        self.evaluation_drain_timeout_seconds = evaluation_drain_timeout_seconds
        self._on_finalized = on_finalized
        self._lock = asyncio.Lock()
        self._result: VerificationResult | None = None

    @property
    def result_path(self) -> Path:
        return self.admin_volume / "finalize.json"

    @property
    def submission_path(self) -> Path:
        return self.admin_volume / "submission.json"

    def _try_load_submission(self) -> Candidate | None:
        if not self.submission_path.exists():
            return None
        try:
            submission = Submission.model_validate_json(
                self.submission_path.read_text(encoding="utf-8")
            )
        except Exception as error:
            raise ValueError(f"invalid durable submission: {error}") from error
        return submission.candidate

    def _selection_records(self) -> list[EvaluationRecord]:
        assert self.selection.backend_id is not None
        assert self.selection.evaluation_set is not None
        assert self.selection.objective is not None
        baseline_version = (
            self.selection.baseline_candidate.version
            if self.selection.baseline_candidate is not None
            else None
        )
        # Match on backend, partition, and objective (not exact evaluation_set
        # equality) so a range/subset that samples the set still counts.
        selection_set = self.selection.evaluation_set
        matching: list[EvaluationRecord] = []
        for record in self.engine.database.evaluations.values():
            objective = record.objective
            evaluation_set = record.request.evaluation_set
            if (
                record.backend_id != self.selection.backend_id
                or evaluation_set.name != selection_set.name
                or evaluation_set.partition != selection_set.partition
                or record.objective_spec != self.selection.objective
                or objective is None
                or not objective.feasible
                or objective.value is None
                or record.request.candidate.version == baseline_version
            ):
                continue
            matching.append(record)
        if not matching:
            return []
        # Coverage threshold relative to the best-covered eval seen — the closest
        # proxy for the full set without re-enumerating it. An under-measured
        # eval is too noisy to rank on; the admin re-score on the full selection
        # set (in _auto_best) is the actual ship gate.
        coverage = {
            id(record): len({case.case_id for case in record.report.cases})
            for record in matching
        }
        best = max(coverage.values())
        min_cases = math.ceil(best * self.selection.selection_coverage_threshold)
        return [record for record in matching if coverage[id(record)] >= min_cases]

    def _shortlist(self) -> list[Candidate]:
        records = self._selection_records()
        if not records:
            return []

        by_content: dict[str, list[EvaluationRecord]] = defaultdict(list)
        for record in records:
            candidate = record.request.candidate
            content = candidate.metadata.get("content_digest")
            key = str(content) if content is not None else candidate.version
            by_content[key].append(record)

        pooled: list[tuple[float, Candidate]] = []
        for group in by_content.values():
            values = [record.objective.value for record in group]
            representative = min(
                (record.request.candidate for record in group),
                key=lambda candidate: candidate.id,
            )
            pooled.append((statistics.fmean(values), representative))
        pooled.sort(key=lambda item: item[1].id)
        pooled.sort(
            key=lambda item: item[0],
            reverse=self.selection.objective.direction == "maximize",
        )
        return [candidate for _, candidate in pooled[: self.selection.rescore_top_k]]

    async def _evaluate(
        self,
        *,
        candidate: Candidate,
        backend_id: str,
        evaluation_set: EvaluationSet,
        objective: ObjectiveSpec,
        parameters: dict[str, JsonValue],
        limits: EvaluationLimits,
        max_attempts: int,
    ) -> tuple[EvaluationRecord | None, str | None]:
        last_error: str | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                record = await self.engine.evaluate_record(
                    backend_id=backend_id,
                    request=EvaluationRequest(
                        candidate=candidate,
                        evaluation_set=evaluation_set,
                        parameters=parameters,
                        limits=limits,
                    ),
                    objective_spec=objective,
                    authorization=EvaluationAuthorization(
                        may_evaluate=True,
                        meter_budget=False,
                        disclosure="full",
                    ),
                    principal=EvaluationPrincipal.ADMIN,
                )
                if (
                    record.objective is not None
                    and record.objective.feasible
                    and record.objective.value is not None
                ):
                    return record, None
                last_error = "evaluation did not produce a feasible objective"
            except Exception as error:
                last_error = str(error) or type(error).__name__
            if attempt == max_attempts:
                break
        return None, last_error

    async def _rescore_candidate(
        self,
        candidate: Candidate,
    ) -> tuple[EvaluationRecord | None, str | None]:
        assert self.selection.backend_id is not None
        assert self.selection.evaluation_set is not None
        assert self.selection.objective is not None
        return await self._evaluate(
            candidate=candidate,
            backend_id=self.selection.backend_id,
            evaluation_set=self.selection.evaluation_set,
            objective=self.selection.objective,
            parameters=self.selection.parameters,
            limits=self.selection.limits,
            max_attempts=self.selection.rescore_attempts,
        )

    def _strictly_beats(
        self,
        candidate: EvaluationRecord,
        baseline: EvaluationRecord,
    ) -> bool:
        assert candidate.objective is not None and candidate.objective.value is not None
        assert baseline.objective is not None and baseline.objective.value is not None
        if not candidate.objective.feasible:
            return False
        if not baseline.objective.feasible:
            return True
        if self.selection.objective.direction == "maximize":
            return candidate.objective.value > baseline.objective.value
        return candidate.objective.value < baseline.objective.value

    def _beats_value(self, candidate: EvaluationRecord, baseline_value: float) -> bool:
        assert candidate.objective is not None and candidate.objective.value is not None
        if not candidate.objective.feasible:
            return False
        if self.selection.objective.direction == "maximize":
            return candidate.objective.value > baseline_value
        return candidate.objective.value < baseline_value

    def _pick_last(self) -> Candidate | None:
        """Last-resort fallback: the most recently measured candidate."""
        records = list(self.engine.database.evaluations.values())
        if not records:
            return None
        latest = max(records, key=lambda record: record.created_at)
        return latest.request.candidate

    async def _auto_best(self) -> Candidate | None:
        rescored: list[EvaluationRecord] = []
        for candidate in self._shortlist():
            record, _ = await self._rescore_candidate(candidate)
            if record is not None:
                rescored.append(record)
        if not rescored:
            return None
        assert self.selection.objective is not None
        rescored.sort(key=lambda record: record.request.candidate.id)
        rescored.sort(
            key=lambda record: record.objective.value,
            reverse=self.selection.objective.direction == "maximize",
        )
        best = rescored[0]

        baseline_candidate = self.selection.baseline_candidate
        if self.selection.baseline_floor and baseline_candidate is not None:
            pinned = self.selection.baseline_selection_score
            if pinned is not None:
                if not self._beats_value(best, pinned):
                    return baseline_candidate
            else:
                baseline, _ = await self._rescore_candidate(baseline_candidate)
                if baseline is None:
                    # Fail safe: can't verify the seed → don't ship it unverified.
                    raise NoCandidateError(
                        "baseline floor could not measure the seed (infrastructure)"
                    )
                if not self._strictly_beats(best, baseline):
                    return baseline_candidate
        return best.request.candidate

    async def _select_candidate(self) -> Candidate:
        # Chain: the agent's explicit submission wins; else auto_best over
        # coverage-qualified evals; else the current/last candidate. Only when
        # there is no candidate at all do we fail (and ship nothing).
        submission = self._try_load_submission()
        if submission is not None:
            return submission
        candidate = await self._auto_best()
        if candidate is not None:
            return candidate
        last = self._pick_last()
        if last is not None:
            return last
        raise NoCandidateError("no submitted, selectable, or prior candidate to ship")

    async def _score_target(
        self,
        candidate: Candidate,
        target: VerificationTarget,
    ) -> tuple[float, str | None, dict[str, float], str | None]:
        record, error = await self._evaluate(
            candidate=candidate,
            backend_id=target.backend_id,
            evaluation_set=target.evaluation_set,
            objective=target.objective,
            parameters=target.parameters,
            limits=target.limits,
            max_attempts=target.max_attempts,
        )
        if record is None:
            return target.failure_value, None, {}, error
        assert record.objective is not None and record.objective.value is not None
        reward = (
            target.reward_scale * float(record.objective.value) + target.reward_offset
        )
        return reward, record.id, dict(record.report.metrics), None

    async def measure_baseline(self, *, replicates: int = 1) -> dict[str, JsonValue]:
        """Admin-score the fixed seed to produce pinnable baseline numbers.

        Runs `replicates` trusted evaluations of the baseline on the selection
        partition and each target and returns per-key mean/stddev, so a stable
        value can be pinned (baseline_selection_score / target baseline_reward)
        instead of re-scoring the seed every run."""
        if replicates < 1:
            raise ValueError("replicates must be >= 1")
        baseline = self.selection.baseline_candidate
        if baseline is None:
            raise NoCandidateError("no baseline candidate to score")

        def _aggregate(values: list[float | None]) -> dict[str, JsonValue]:
            clean = [value for value in values if value is not None]
            return {
                "values": values,
                "n": len(clean),
                "mean": statistics.fmean(clean) if clean else None,
                "stddev": statistics.pstdev(clean) if len(clean) > 1 else 0.0,
            }

        result: dict[str, JsonValue] = {
            "candidate_version": baseline.version,
            "replicates": replicates,
        }
        if (
            self.selection.backend_id is not None
            and self.selection.evaluation_set is not None
            and self.selection.objective is not None
        ):
            selection_values: list[float | None] = []
            for _ in range(replicates):
                record, _ = await self._rescore_candidate(baseline)
                selection_values.append(
                    record.objective.value
                    if record is not None and record.objective is not None
                    else None
                )
            result["selection"] = _aggregate(selection_values)
        targets: dict[str, JsonValue] = {}
        for target in self.targets:
            target_values: list[float | None] = []
            for _ in range(replicates):
                reward, _, _, error = await self._score_target(baseline, target)
                target_values.append(None if error is not None else reward)
            targets[target.reward_key] = _aggregate(target_values)
        result["targets"] = targets
        return result

    async def _finalize(self) -> VerificationResult:
        try:
            candidate = await self._select_candidate()
        except NoCandidateError as error:
            return VerificationResult(
                shipped=False,
                rewards={
                    target.reward_key: target.failure_value for target in self.targets
                },
                errors={"selection": str(error)},
            )

        rewards: dict[str, float] = {}
        evaluation_ids: dict[str, str] = {}
        reward_metrics: dict[str, dict[str, float]] = {}
        errors: dict[str, str] = {}
        for target in self.targets:
            reward, evaluation_id, metrics, error = await self._score_target(
                candidate, target
            )
            rewards[target.reward_key] = reward
            if evaluation_id is not None:
                evaluation_ids[target.reward_key] = evaluation_id
            if metrics:
                reward_metrics[target.reward_key] = metrics
            if error is not None:
                errors[target.reward_key] = error

        baseline_rewards: dict[str, float] = {}
        baseline_reward_metrics: dict[str, dict[str, float]] = {}
        baseline = self.selection.baseline_candidate
        if self.score_baseline and baseline is not None:
            if baseline.version == candidate.version:
                baseline_rewards = dict(rewards)
                baseline_reward_metrics = dict(reward_metrics)
            else:
                for target in self.targets:
                    if target.baseline_reward is not None:
                        baseline_rewards[target.reward_key] = target.baseline_reward
                        continue
                    reward, _, metrics, error = await self._score_target(
                        baseline, target
                    )
                    baseline_rewards[target.reward_key] = reward
                    if metrics:
                        baseline_reward_metrics[target.reward_key] = metrics
                    if error is not None:
                        errors[f"baseline:{target.reward_key}"] = error

        return VerificationResult(
            candidate=candidate,
            rewards=rewards,
            evaluation_ids=evaluation_ids,
            baseline_rewards=baseline_rewards,
            reward_metrics=reward_metrics,
            baseline_reward_metrics=baseline_reward_metrics,
            errors=errors,
        )

    async def finalize(self) -> VerificationResult:
        """Return the first durable finalization result on every invocation."""
        async with self._lock:
            if self._result is not None:
                return self._result
            if self.result_path.exists():
                try:
                    self._result = VerificationResult.model_validate_json(
                        self.result_path.read_text(encoding="utf-8")
                    )
                except Exception as error:
                    raise ValueError(
                        f"invalid durable finalization result: {error}"
                    ) from error
                return self._result
            drained = await self.engine.quiesce_agent_evaluations(
                timeout_seconds=self.evaluation_drain_timeout_seconds,
            )
            if drained:
                logger.info(
                    "Finalization drained %d accepted agent evaluation(s)",
                    drained,
                )
            result = await self._finalize()
            await asyncio.to_thread(
                _atomic_write_json,
                self.result_path,
                result.model_dump(mode="json"),
            )
            self._result = result
            if self._on_finalized is not None:
                # Best-effort session-end hook (e.g. finish the W&B run); must
                # never affect the durable finalization result.
                try:
                    self._on_finalized(result)
                except Exception:
                    logger.warning("finalization hook failed", exc_info=True)
            return result
