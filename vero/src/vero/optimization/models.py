"""Optimization proposals, context, and results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from uuid import uuid4

from pydantic import Field, JsonValue, field_validator, model_validator

from vero.candidate import Candidate
from vero.evaluation import EvaluationModel, EvaluationRecord
from vero.workspace import Workspace


class CandidateProposal(EvaluationModel):
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


class CandidateChange(EvaluationModel):
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
    round: int
    workspace: Workspace
    baseline: EvaluationRecord
    evaluations: tuple[EvaluationRecord, ...]
    candidates: Mapping[str, Candidate]
    best: EvaluationRecord | None


@dataclass(frozen=True)
class OptimizationResult:
    baseline: EvaluationRecord
    evaluations: tuple[EvaluationRecord, ...]
    candidates: tuple[Candidate, ...]
    best: EvaluationRecord | None
