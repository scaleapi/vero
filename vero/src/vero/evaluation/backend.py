"""Backend protocol, execution context, and approved backend registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from vero.evaluation.models import (
    BackendProvenance,
    EvaluationCost,
    EvaluationReport,
    EvaluationRequest,
    EvaluationSet,
)
from vero.evaluation.persistence import CaseCheckpointStore
from vero.workspace import Workspace


@dataclass(frozen=True)
class EvaluationContext:
    workspace: Workspace
    session_id: str
    evaluation_id: str
    result_dir: Path
    artifact_dir: Path
    case_store: CaseCheckpointStore


@runtime_checkable
class EvaluationBackend(Protocol):
    @property
    def provenance(self) -> BackendProvenance: ...

    async def resolve_cost(self, evaluation_set: EvaluationSet) -> EvaluationCost: ...

    async def evaluate(
        self,
        *,
        context: EvaluationContext,
        request: EvaluationRequest,
    ) -> EvaluationReport: ...


class BackendRegistry:
    """Immutable-by-default registry of trusted, preconfigured backend instances."""

    def __init__(self, backends: dict[str, EvaluationBackend] | None = None):
        self._backends: dict[str, EvaluationBackend] = {}
        for backend_id, backend in (backends or {}).items():
            self.register(backend_id, backend)

    def register(self, backend_id: str, backend: EvaluationBackend) -> None:
        if not backend_id.strip():
            raise ValueError("backend ID must not be empty")
        if backend_id in self._backends:
            raise ValueError(f"backend ID {backend_id!r} is already registered")
        if not isinstance(backend, EvaluationBackend):
            raise TypeError("backend does not implement EvaluationBackend")
        self._backends[backend_id] = backend

    def resolve(self, backend_id: str) -> EvaluationBackend:
        from vero.evaluation.exceptions import UnknownBackendError

        try:
            return self._backends[backend_id]
        except KeyError as error:
            raise UnknownBackendError(f"unknown evaluation backend: {backend_id!r}") from error

    def __contains__(self, backend_id: str) -> bool:
        return backend_id in self._backends

    def __iter__(self):
        return iter(self._backends)
