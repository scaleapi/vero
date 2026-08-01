"""Generic optimization session lifecycle and durable manifest."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, JsonValue, field_validator

from vero.candidate import Candidate
from vero.candidate_repository import CandidateRepository
from vero.evaluation import (
    BackendProvenance,
    CaseStatus,
    EvaluationLimits,
    EvaluationPlan,
    EvaluationRecord,
    ObjectiveSpec,
)
from vero.evaluation.store.persistence import _atomic_write_json
from vero.models import StrictModel
from vero.optimization import OptimizationResult, Optimizer
from vero.runtime.artifacts import ArtifactStore
from vero.runtime.events import EventBus, JsonlEventSink, agent_event_emitter
from vero.workspace import Workspace


class SessionStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SessionFailure(StrictModel):
    type: str
    message: str

    @field_validator("type", "message")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("session failure fields must not be empty")
        return value


class OptimizationComponentSpec(StrictModel):
    """Stable type and configuration identity for a protocol component."""

    type: str
    config_digest: str

    @field_validator("type", "config_digest")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("component identity must not be empty")
        return value

    @field_validator("config_digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError("component config_digest must be a SHA-256 hex digest")
        return value


class OptimizationRunSpec(StrictModel):
    """Execution choices that must remain stable when a session resumes."""

    max_proposals: int = Field(ge=0)
    max_rounds: int = Field(ge=1)
    max_concurrency: int = Field(ge=1)
    strategy: OptimizationComponentSpec
    producers: dict[str, OptimizationComponentSpec]
    # Optional so manifests written before this field can still load; a resume
    # under a changed selection policy is then caught as a run-protocol mismatch.
    selection: OptimizationComponentSpec | None = None


class SessionManifest(StrictModel):
    schema_version: Literal[3] = 3
    id: str
    status: SessionStatus
    backend_id: str
    backend: BackendProvenance
    candidate_repository_family: str
    candidate_repository_format_version: int
    evaluation_plan: EvaluationPlan
    objective: ObjectiveSpec
    run: OptimizationRunSpec
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    limits: EvaluationLimits = Field(default_factory=EvaluationLimits)
    seed: int | None = None
    baseline: Candidate | None = None
    best_candidate_id: str | None = None
    best_evaluation_id: str | None = None
    final_baseline_evaluation_id: str | None = None
    final_evaluation_id: str | None = None
    created_at: datetime
    updated_at: datetime
    failure: SessionFailure | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("id", "backend_id", "candidate_repository_family")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("session identity must not be empty")
        return value

    @field_validator("created_at", "updated_at")
    @classmethod
    def normalize_timestamps(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("session timestamps must be timezone-aware")
        return value.astimezone(UTC)


@contextmanager
def _exclusive_session_lock(session_dir: Path) -> Iterator[None]:
    """Hold an advisory lock so two processes cannot drive one session.

    Nothing stopped a second `vero run` from attaching to a session directory
    that already had a live process in it, and an operator has every reason to
    try: a run takes hours, and when it goes quiet there is no way from the
    outside to tell a wedged process from a dead one. Both processes then derive
    the same pending work from the same manifest, and each holds its own
    in-memory copy of the budget ledger, so the same budget is spent twice, the
    manifest writes overwrite one another, and the recorded history describes
    neither run.

    An advisory ``flock`` is the right instrument precisely because the kernel
    drops it when the holder exits, however it exits. A relaunch after a real
    crash (the case this is all meant to support) acquires it immediately, and
    there is no cleanup step anyone can forget. The pid written into the file is
    only a breadcrumb for the refusal message: the lock itself lives in the open
    file description, never in the file's contents.
    """

    lock_path = session_dir / "run.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        # Only a would-block means contention. Any other OSError (a filesystem
        # that cannot lock, say) propagates as itself rather than being reported
        # as a second process, which would be a false diagnosis of the one thing
        # this lock exists to diagnose.
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            holder = os.read(descriptor, 64).decode(errors="replace").strip()
            raise RuntimeError(
                f"session {session_dir} is already being run by another process"
                f" (pid {holder or 'unknown'} holds {lock_path}), and two"
                " processes over one session double-spend its budgets and"
                " interleave their history. If that process is gone, the kernel"
                " has already released the lock and this run can be retried."
            ) from error
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"{os.getpid()}\n".encode())
        yield
    finally:
        # Closing the descriptor is what releases the lock. The file itself is
        # left behind deliberately: an unheld lock file is inert, and deleting
        # it would race with the next acquisition it is supposed to guard.
        os.close(descriptor)


@dataclass
class OptimizationSession:
    """Own the durable state and lifecycle of one optimization run."""

    id: str
    session_dir: Path
    optimizer: Optimizer
    baseline: Candidate | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    events: EventBus | None = None
    run_spec: OptimizationRunSpec | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("session ID must not be empty")
        expected = self.session_dir.resolve()
        evaluator_session = self.optimizer.engine.evaluator.session_dir.resolve()
        if evaluator_session != expected:
            raise ValueError(
                "optimizer evaluator session directory must match OptimizationSession"
            )
        optimizer_session_id = self.optimizer.session_id
        if optimizer_session_id is not None and optimizer_session_id != self.id:
            raise ValueError("optimizer session ID does not match OptimizationSession")
        self.optimizer.session_id = self.id
        evaluator_session_id = getattr(
            self.optimizer.engine.evaluator, "session_id", None
        )
        if evaluator_session_id is not None and evaluator_session_id != self.id:
            raise ValueError("evaluator session ID does not match OptimizationSession")
        self.optimizer.engine.evaluator.session_id = self.id
        if (
            self.optimizer.engine.evaluator.candidate_repository
            is not self.optimizer.candidate_repository
        ):
            raise ValueError(
                "optimizer and evaluator must share one candidate repository"
            )
        self.session_dir.mkdir(parents=True, exist_ok=True)
        if self.events is None:
            self.events = EventBus([JsonlEventSink(self.events_path)])
        self.artifacts = ArtifactStore(self.session_dir / "artifacts")
        self._event_step = len(self.optimizer.engine.database.evaluations)
        self.optimizer.engine.listeners.append(self._on_evaluation_completed)
        self._wire_agent_events()

    def _wire_agent_events(self) -> None:
        """Publish agent activity (tool calls, reasoning, messages) to the bus.

        Agent producers that expose a settable ``on_event`` and a normalizing
        agent get their events routed onto the session event bus. A
        caller-provided ``on_event`` is left untouched.
        """
        assert self.events is not None
        for producer in self.optimizer.producers.values():
            if getattr(producer, "on_event", None) is not None:
                continue
            if not hasattr(producer, "on_event"):
                continue
            agent = getattr(producer, "agent", None)
            if agent is None:
                continue
            emitter = agent_event_emitter(self.events, self.id, agent)
            if emitter is not None:
                producer.on_event = emitter

    @property
    def manifest_path(self) -> Path:
        return self.session_dir / "manifest.json"

    @property
    def events_path(self) -> Path:
        return self.session_dir / "events.jsonl"

    @property
    def database(self):
        return self.optimizer.engine.database

    @property
    def budget_ledger(self):
        return self.optimizer.engine.budget_ledger

    @property
    def candidate_repository(self) -> CandidateRepository:
        return self.optimizer.candidate_repository

    @property
    def workspace(self) -> Workspace:
        """The original workspace supplied when the session was created."""

        return self.optimizer.workspace

    def _initial_manifest(self, baseline: Candidate) -> SessionManifest:
        now = datetime.now(UTC)
        return SessionManifest(
            id=self.id,
            status=SessionStatus.CREATED,
            backend_id=self.optimizer.backend_id,
            backend=self.optimizer.engine.backends.resolve(
                self.optimizer.backend_id
            ).provenance,
            candidate_repository_family=self.candidate_repository.family,
            candidate_repository_format_version=(
                self.candidate_repository.format_version
            ),
            evaluation_plan=self.optimizer.evaluation_plan,
            objective=self.optimizer.objective,
            run=self._run_spec(),
            parameters=self.optimizer.parameters,
            limits=self.optimizer.limits,
            seed=self.optimizer.seed,
            baseline=baseline,
            created_at=now,
            updated_at=now,
            metadata=self.metadata,
        )

    @staticmethod
    def _component_spec(value: object) -> OptimizationComponentSpec:
        kind = type(value)
        type_name = f"{kind.__module__}.{kind.__qualname__}"
        payload: dict[str, object] = {}
        config = getattr(value, "config", None)
        if isinstance(config, BaseModel):
            payload["config"] = config.model_dump(mode="json")
        elif isinstance(config, dict):
            payload["config"] = config
        for name in ("producer_id", "instruction", "prompt", "max_turns"):
            if hasattr(value, name):
                payload[name] = getattr(value, name)
        agent = getattr(value, "agent", None)
        serialize_agent = getattr(agent, "dict", None)
        if callable(serialize_agent):
            payload["agent"] = serialize_agent()
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode()
        return OptimizationComponentSpec(
            type=type_name,
            config_digest=hashlib.sha256(encoded).hexdigest(),
        )

    @staticmethod
    def _differing_fields(persisted: BaseModel, current: BaseModel) -> list[str]:
        """Name the top-level fields that differ, to make a mismatch diagnosable."""

        left = persisted.model_dump(mode="json")
        right = current.model_dump(mode="json")
        return sorted(
            name for name in set(left) | set(right) if left.get(name) != right.get(name)
        ) or ["(no top-level field differs; a nested value does)"]

    def _run_spec(self) -> OptimizationRunSpec:
        if self.run_spec is not None:
            return self.run_spec
        return OptimizationRunSpec(
            max_proposals=self.optimizer.max_proposals,
            max_rounds=self.optimizer.max_rounds,
            max_concurrency=self.optimizer.max_concurrency,
            strategy=self._component_spec(self.optimizer.strategy),
            producers={
                producer_id: self._component_spec(producer)
                for producer_id, producer in sorted(self.optimizer.producers.items())
            },
            selection=self._component_spec(self.optimizer.selection),
        )

    async def _on_evaluation_completed(self, record: EvaluationRecord) -> None:
        """Publish evaluations as they finish, rather than replaying them at exit."""

        assert self.events is not None
        step = self._event_step
        self._event_step += 1
        await self.events.emit(
            session_id=self.id,
            kind="evaluation_completed",
            payload=self._evaluation_event_payload(record, step=step),
        )

    async def _save_manifest(self, manifest: SessionManifest) -> None:
        await asyncio.to_thread(
            _atomic_write_json,
            self.manifest_path,
            manifest.model_dump(mode="json"),
        )

    @staticmethod
    def _evaluation_event_payload(
        record: EvaluationRecord,
        *,
        step: int,
    ) -> dict[str, JsonValue]:
        counts = {status: 0 for status in CaseStatus}
        for case in record.report.cases:
            counts[case.status] += 1
        payload: dict[str, JsonValue] = {
            "step": step,
            "evaluation_id": record.id,
            "candidate_id": record.request.candidate.id,
            "candidate_version": record.request.candidate.version,
            "evaluation": record.request.evaluation_set.name,
            "partition": record.request.evaluation_set.partition,
            "principal": record.principal.value,
            "status": record.report.status.value,
            "cases/total": len(record.report.cases),
            "cases/success": counts[CaseStatus.SUCCESS],
            "cases/error": counts[CaseStatus.ERROR],
            "cases/skipped": counts[CaseStatus.SKIPPED],
        }
        payload.update(
            {f"metrics/{name}": value for name, value in record.report.metrics.items()}
        )
        if record.objective is not None:
            payload["objective/value"] = record.objective.value
            payload["objective/feasible"] = record.objective.feasible
        return payload

    def load_manifest(self) -> SessionManifest:
        return SessionManifest.model_validate_json(
            self.manifest_path.read_text(encoding="utf-8")
        )

    async def run(
        self,
        *,
        baseline: Candidate | None = None,
        skip_baseline_evaluation: bool = False,
        max_proposals: int | None = None,
    ) -> OptimizationResult:
        # Advisory-locked for the whole of run(): see _exclusive_session_lock
        # for why a second process on one session directory is refused here
        # rather than left to corrupt the budgets and the history.
        with _exclusive_session_lock(self.session_dir):
            manifest = self.load_manifest() if self.manifest_path.exists() else None
            if baseline is None:
                if manifest is not None and manifest.baseline is not None:
                    baseline = manifest.baseline
                elif self.baseline is not None:
                    baseline = self.baseline
                else:
                    baseline = Candidate.from_version(
                        await self.optimizer.workspace.current_version()
                    )
            if manifest is None:
                manifest = self._initial_manifest(baseline)
            if manifest.id != self.id:
                raise ValueError("session manifest ID does not match runtime session")
            if manifest.backend_id != self.optimizer.backend_id:
                raise ValueError("session backend does not match the persisted manifest")
            if manifest.candidate_repository_family != self.candidate_repository.family:
                raise ValueError(
                    "session candidate repository does not match the persisted manifest"
                )
            if (
                manifest.candidate_repository_format_version
                != self.candidate_repository.format_version
            ):
                raise ValueError(
                    "session candidate repository format does not match the "
                    "persisted manifest"
                )
            backend = self.optimizer.engine.backends.resolve(self.optimizer.backend_id)
            if manifest.backend != backend.provenance:
                raise ValueError(
                    "session backend configuration does not match the persisted manifest"
                )
            if manifest.evaluation_plan != self.optimizer.evaluation_plan:
                raise ValueError(
                    "session evaluation plan does not match the persisted manifest"
                )
            if manifest.objective != self.optimizer.objective:
                raise ValueError("session objective does not match the persisted manifest")
            if manifest.run != self._run_spec():
                # A session directory is bound to the protocol it was created with,
                # on purpose: reusing it under another would let its recorded history
                # misdescribe what produced the candidates. The usual cause is an
                # edited config -- changing `optimizer.model` between two commands is
                # enough, since the model is part of the producer's identity -- and
                # the bare message named neither the field nor the remedy.
                differing = ", ".join(self._differing_fields(manifest.run, self._run_spec()))
                raise ValueError(
                    "session run protocol does not match the persisted manifest"
                    f" (differs in: {differing}). A session records the protocol it was"
                    " created with, so a changed strategy, producer, model, or proposal"
                    " count needs a fresh session directory -- or discard the existing"
                    f" state with `vero session clear {self.session_dir} --yes`."
                )
            if manifest.parameters != self.optimizer.parameters:
                raise ValueError(
                    "session evaluation parameters do not match the persisted manifest"
                )
            if manifest.limits != self.optimizer.limits:
                raise ValueError(
                    "session evaluation limits do not match the persisted manifest"
                )
            if manifest.seed != self.optimizer.seed:
                raise ValueError(
                    "session evaluation seed does not match the persisted manifest"
                )
            if manifest.baseline is None or (
                manifest.baseline.id,
                manifest.baseline.version,
            ) != (baseline.id, baseline.version):
                raise ValueError("session baseline does not match the persisted manifest")

            # Deliberately not clearing `failure` here. This write is a rerun's very
            # first durable act, so clearing it destroyed the only recorded
            # explanation of the previous death before anyone had read it: the run
            # that lost 67 held-out trials to an upstream budget cap had its
            # diagnosis erased by the relaunch that was supposed to recover it. The
            # clear happens on the COMPLETED write below instead. The tradeoff is
            # that a stale failure now lingers on the manifest of a run that is
            # mid-flight, which is the right way round: `status` already says
            # RUNNING, so the failure reads as "how the last attempt ended", and
            # keeping a stale explanation costs nothing next to losing a real one.
            manifest = manifest.model_copy(
                update={
                    "status": SessionStatus.RUNNING,
                    "updated_at": datetime.now(UTC),
                }
            )
            await self._save_manifest(manifest)
            assert self.events is not None
            await self.events.emit(
                session_id=self.id,
                kind="session_started",
                payload={"baseline_candidate_id": baseline.id},
            )

            try:
                result = await self.optimizer.run(
                    baseline=baseline,
                    skip_baseline_evaluation=skip_baseline_evaluation,
                    max_proposals=max_proposals,
                )
            except BaseException as error:
                failure = SessionFailure(
                    type=f"{type(error).__module__}.{type(error).__name__}",
                    message=str(error) or type(error).__name__,
                )
                await self._save_manifest(
                    manifest.model_copy(
                        update={
                            "status": SessionStatus.FAILED,
                            "updated_at": datetime.now(UTC),
                            "failure": failure,
                        }
                    )
                )
                await self.events.emit(
                    session_id=self.id,
                    kind="session_failed",
                    payload={"error_type": failure.type, "message": failure.message},
                )
                raise

            best = result.best
            completed = manifest.model_copy(
                update={
                    "status": SessionStatus.COMPLETED,
                    "updated_at": datetime.now(UTC),
                    # Only a completed run may drop the previous attempt's failure:
                    # by here the session has a result, so the old explanation is
                    # genuinely obsolete rather than merely inconvenient.
                    "failure": None,
                    "best_candidate_id": (
                        best.request.candidate.id if best is not None else None
                    ),
                    "best_evaluation_id": best.id if best is not None else None,
                    "final_baseline_evaluation_id": (
                        result.final_baseline.id
                        if result.final_baseline is not None
                        else None
                    ),
                    "final_evaluation_id": (
                        result.final.id if result.final is not None else None
                    ),
                }
            )
            await self._save_manifest(completed)
            await self.events.emit(
                session_id=self.id,
                kind="session_completed",
                payload={
                    "best_candidate_id": completed.best_candidate_id,
                    "best_evaluation_id": completed.best_evaluation_id,
                    "evaluation_count": len(result.evaluations),
                    "status": "completed",
                    "baseline_candidate_id": result.baseline.request.candidate.id,
                    "baseline_objective": (
                        result.baseline.objective.value
                        if result.baseline.objective is not None
                        else None
                    ),
                    "best_objective": (
                        best.objective.value
                        if best is not None and best.objective is not None
                        else None
                    ),
                    "final_objective": (
                        result.final.objective.value
                        if result.final is not None
                        and result.final.objective is not None
                        else None
                    ),
                },
            )
            return result
