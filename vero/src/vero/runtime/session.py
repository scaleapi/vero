"""Generic optimization session lifecycle and durable manifest."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from vero.candidate import Candidate
from vero.evaluation import EvaluationSet, ObjectiveSpec
from vero.evaluation.persistence import _atomic_write_json
from vero.optimization import OptimizationResult, Optimizer
from vero.runtime.artifacts import ArtifactStore
from vero.runtime.events import EventBus, JsonlEventSink


class SessionStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SessionFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    message: str

    @field_validator("type", "message")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("session failure fields must not be empty")
        return value


class SessionManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    id: str
    status: SessionStatus
    backend_id: str
    evaluation_set: EvaluationSet
    objective: ObjectiveSpec
    baseline: Candidate | None = None
    best_candidate_id: str | None = None
    best_evaluation_id: str | None = None
    created_at: datetime
    updated_at: datetime
    failure: SessionFailure | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("id", "backend_id")
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


@dataclass
class OptimizationSession:
    """Own the durable state and lifecycle of one optimization run."""

    id: str
    session_dir: Path
    optimizer: Optimizer
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    events: EventBus | None = None

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
        evaluator_session_id = getattr(self.optimizer.engine.evaluator, "session_id", None)
        if evaluator_session_id is not None and evaluator_session_id != self.id:
            raise ValueError("evaluator session ID does not match OptimizationSession")
        self.optimizer.engine.evaluator.session_id = self.id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        if self.events is None:
            self.events = EventBus([JsonlEventSink(self.events_path)])
        self.artifacts = ArtifactStore(self.session_dir / "artifacts")

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

    def _initial_manifest(self, baseline: Candidate) -> SessionManifest:
        now = datetime.now(UTC)
        return SessionManifest(
            id=self.id,
            status=SessionStatus.CREATED,
            backend_id=self.optimizer.backend_id,
            evaluation_set=self.optimizer.evaluation_set,
            objective=self.optimizer.objective,
            baseline=baseline,
            created_at=now,
            updated_at=now,
            metadata=self.metadata,
        )

    def _save_manifest(self, manifest: SessionManifest) -> None:
        _atomic_write_json(
            self.manifest_path,
            manifest.model_dump(mode="json"),
        )

    def load_manifest(self) -> SessionManifest:
        return SessionManifest.model_validate_json(
            self.manifest_path.read_text(encoding="utf-8")
        )

    async def run(
        self,
        *,
        baseline: Candidate | None = None,
        skip_baseline_evaluation: bool = False,
    ) -> OptimizationResult:
        manifest = self.load_manifest() if self.manifest_path.exists() else None
        current_version = await self.optimizer.workspace.current_version()
        if baseline is None:
            if manifest is not None and manifest.baseline is not None:
                baseline = manifest.baseline
            else:
                baseline = Candidate.from_version(current_version)
        if baseline.version != current_version:
            raise ValueError("session baseline does not match the current workspace version")
        if manifest is None:
            manifest = self._initial_manifest(baseline)
        if manifest.id != self.id:
            raise ValueError("session manifest ID does not match runtime session")
        if manifest.baseline is None or (
            manifest.baseline.id,
            manifest.baseline.version,
        ) != (baseline.id, baseline.version):
            raise ValueError("session baseline does not match the persisted manifest")

        manifest = manifest.model_copy(
            update={
                "status": SessionStatus.RUNNING,
                "updated_at": datetime.now(UTC),
                "failure": None,
            }
        )
        self._save_manifest(manifest)
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
            )
        except BaseException as error:
            failure = SessionFailure(
                type=f"{type(error).__module__}.{type(error).__name__}",
                message=str(error) or type(error).__name__,
            )
            self._save_manifest(
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
                "best_candidate_id": (
                    best.request.candidate.id if best is not None else None
                ),
                "best_evaluation_id": best.id if best is not None else None,
            }
        )
        self._save_manifest(completed)
        await self.events.emit(
            session_id=self.id,
            kind="session_completed",
            payload={
                "best_candidate_id": completed.best_candidate_id,
                "best_evaluation_id": completed.best_evaluation_id,
                "evaluation_count": len(result.evaluations),
            },
        )
        return result
