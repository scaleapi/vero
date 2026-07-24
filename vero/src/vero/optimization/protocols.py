"""Extension protocols for optimization strategies and candidate producers."""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence, runtime_checkable

from vero.candidate import Candidate
from vero.evaluation import (
    CaseSelection,
    EvaluationBudget,
    EvaluationReceipt,
    EvaluationRecord,
    ObjectiveSpec,
)
from vero.optimization.models import (
    CandidateChange,
    CandidateProductionContext,
    CandidateProposal,
    GenerationOutcome,
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
class GenerationBackend(Protocol):
    """Turn a parent + proposal into candidate(s) plus generation-time feedback.

    This is the swappable unit the Optimizer drives. The default is the
    Optimizer's built-in native production (an agent or operator editing a
    checkout in a sandbox, with mid-run self-eval). A Harbor implementation
    delegates to a ``harbor run`` instead. Either way, the orchestrator performs
    selection and target scoring *separately* on the returned candidate; the
    ``GenerationOutcome.trial_evaluations`` are only the generation-time feedback
    the producer observed while iterating.
    """

    async def generate(
        self,
        *,
        proposal: CandidateProposal,
        parent: Candidate,
        context: OptimizationContext,
        evaluation_records: Sequence[EvaluationRecord],
    ) -> GenerationOutcome: ...


@runtime_checkable
class SelectionPolicy(Protocol):
    def select(
        self,
        records: Sequence[EvaluationRecord],
        objective: ObjectiveSpec,
    ) -> EvaluationRecord | None: ...
