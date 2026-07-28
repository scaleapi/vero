"""Durable candidate storage and compatible workspace materialization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import AsyncIterator, Generic, Sequence, TypeVar

from vero.candidate import Candidate
from vero.sandbox import Sandbox
from vero.workspace import Workspace

WorkspaceT = TypeVar("WorkspaceT", bound=Workspace)


class CandidateRepositoryError(RuntimeError):
    """Raised when durable candidate state is invalid or cannot be transferred."""


class CandidateRepository(ABC, Generic[WorkspaceT]):
    """Own durable candidates for one workspace family within a session."""

    @property
    @abstractmethod
    def family(self) -> str:
        """Stable workspace/repository family identifier."""
        ...

    @property
    @abstractmethod
    def format_version(self) -> int:
        """Persisted repository format version."""
        ...

    @abstractmethod
    def supports(self, workspace: Workspace) -> bool:
        """Whether this repository can capture the supplied workspace."""
        ...

    @abstractmethod
    async def capture(
        self,
        candidate: Candidate,
        workspace: WorkspaceT,
    ) -> Candidate:
        """Durably record a clean workspace at the candidate's version."""
        ...

    @abstractmethod
    async def materialize_agent_history(
        self,
        candidates: Sequence[Candidate],
        *,
        workspace: WorkspaceT,
        destination: str,
    ) -> None:
        """Expose a repository-native view of the visible candidates.

        ``destination`` is a sandbox path reserved for generated agent context.
        Implementations may install disposable native references in the supplied
        workspace, but must never mutate durable candidate state.
        """
        ...

    @abstractmethod
    def get(self, candidate_id: str) -> Candidate | None:
        """Return one durable candidate by identity."""
        ...

    @abstractmethod
    def list(self) -> tuple[Candidate, ...]:
        """Return all durable candidates in deterministic order."""
        ...

    @asynccontextmanager
    @abstractmethod
    async def checkout(
        self,
        candidate: Candidate,
        *,
        sandbox: Sandbox,
        name: str | None = None,
    ) -> AsyncIterator[WorkspaceT]:
        """Materialize a temporary isolated workspace for one candidate."""
        yield  # pragma: no cover
