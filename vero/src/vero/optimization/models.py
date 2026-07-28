"""Optimization proposals, context, and results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from uuid import uuid4

from pydantic import Field, JsonValue, field_validator, model_validator

from vero.candidate import Candidate
from vero.evaluation import (
    EvaluationAcknowledgement,
    EvaluationRecord,
    EvaluationSummary,
)
from vero.models import StrictModel
from vero.workspace import Workspace


class CandidateProposal(StrictModel):
    """A strategy's request for one producer to explore a parent candidate."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    producer_id: str = "default"
    parent_id: str | None = None
    instruction: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("id", "producer_id")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("proposal identity must not be empty")
        return value

    @field_validator("parent_id", "instruction")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("optional proposal text must not be empty")
        return value

    @model_validator(mode="after")
    def validate_parent(self) -> CandidateProposal:
        if self.parent_id == self.id:
            raise ValueError("a proposal cannot name itself as its parent")
        return self


class CandidateChange(StrictModel):
    """Producer metadata returned after it edits a supplied workspace."""

    description: str = "Optimize candidate"
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("candidate description must not be empty")
        return value


@dataclass(frozen=True)
class OptimizationContext:
    """Authorization-projected history supplied to an optimization strategy."""

    session_id: str
    round: int
    workspace: Workspace
    baseline: Candidate
    evaluations: tuple[
        EvaluationRecord | EvaluationSummary | EvaluationAcknowledgement,
        ...,
    ]
    candidates: Mapping[str, Candidate]
    best: Candidate | None


@dataclass(frozen=True)
class CandidateProductionContext:
    """Non-sensitive control context supplied to a candidate producer.

    Authorized evaluation details live in the read-only ``.evals`` tree and are
    returned through the evaluation gateway; they are intentionally not
    duplicated here as full records.
    """

    session_id: str
    round: int
    baseline: Candidate
    candidates: Mapping[str, Candidate]
    best: Candidate | None


@dataclass(frozen=True)
class GenerationOutcome:
    """Candidates and generation-time feedback from executing one proposal.

    Produced by a :class:`~vero.optimization.protocols.GenerationBackend` (the
    native in-process producer by default, or a Harbor run). ``candidate`` is the
    produced candidate (``None`` if the producer made no change);
    ``trial_candidates`` are the intermediate checkpoints it captured. Crucially,
    ``trial_evaluations`` are the **generation-time feedback** evaluations the
    producer *observed while iterating* (mid-run self-eval, or a Harbor sidecar's
    disclosed-partition scores) — distinct from the orchestrator's later
    selection and target scoring, which the Optimizer performs separately on the
    returned candidate.
    """

    candidate: Candidate | None
    trial_candidates: tuple[Candidate, ...]
    trial_evaluations: tuple[EvaluationRecord, ...]

    @property
    def candidates(self) -> tuple[Candidate, ...]:
        if self.candidate is None:
            return self.trial_candidates
        return (*self.trial_candidates, self.candidate)


@dataclass(frozen=True)
class OptimizationResult:
    baseline: EvaluationRecord
    evaluations: tuple[EvaluationRecord, ...]
    candidates: tuple[Candidate, ...]
    best: EvaluationRecord | None
    final_baseline: EvaluationRecord | None = None
    final: EvaluationRecord | None = None
