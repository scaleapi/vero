"""Extension protocols for optimization strategies and candidate producers."""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence, runtime_checkable

from vero.evaluation import (
    EvaluationBudget,
    CaseSelection,
    EvaluationReceipt,
    EvaluationRecord,
    ObjectiveSpec,
)
from vero.optimization.models import (
    CandidateChange,
    CandidateProductionContext,
    CandidateProposal,
    OptimizationContext,
)
from vero.workspace import Workspace


@runtime_checkable
class OptimizationStrategy(Protocol):
    async def propose(
        self,
        context: OptimizationContext,
    ) -> Sequence[CandidateProposal]: ...


@runtime_checkable
class CandidateEvaluationGateway(Protocol):
    """Evaluation capability scoped to one producer workspace."""

    async def evaluate(
        self,
        *,
        evaluation: str,
        selection: CaseSelection | None = None,
        candidate_id: str | None = None,
        description: str = "Evaluate agent checkpoint",
    ) -> EvaluationReceipt: ...

    def budgets(self) -> Mapping[str, EvaluationBudget | None]: ...


@runtime_checkable
class CandidateProducer(Protocol):
    async def produce(
        self,
        *,
        proposal: CandidateProposal,
        context: CandidateProductionContext,
        workspace: Workspace,
        evaluation: CandidateEvaluationGateway,
    ) -> CandidateChange | None: ...


@runtime_checkable
class SelectionPolicy(Protocol):
    def select(
        self,
        records: Sequence[EvaluationRecord],
        objective: ObjectiveSpec,
    ) -> EvaluationRecord | None: ...
