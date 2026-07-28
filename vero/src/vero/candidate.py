"""Canonical identity for a versioned program candidate."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import (
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from vero.models import StrictModel


class Candidate(StrictModel):
    """A materialized version of the program being optimized.

    ``version`` is interpreted by the session's candidate repository. A
    session uses one repository/workspace family, so the identifier remains
    stable when the candidate is materialized in different sandboxes.
    """

    id: str
    version: str
    parent_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    description: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("id", "version")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("candidate identity must not be empty")
        return value

    @field_validator("parent_id", "description")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("optional candidate text must not be empty")
        return value

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("candidate timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_parent(self) -> Candidate:
        if self.parent_id == self.id:
            raise ValueError("a candidate cannot be its own parent")
        return self

    @classmethod
    def from_version(
        cls,
        version: str,
        *,
        candidate_id: str | None = None,
        parent_id: str | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Candidate:
        """Construct the usual one-candidate-per-workspace-version identity."""
        return cls(
            id=candidate_id or version,
            version=version,
            parent_id=parent_id,
            description=description,
            metadata=metadata or {},
        )
